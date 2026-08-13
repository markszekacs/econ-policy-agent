"""Synthetic query generation with quote-grounded, chunking-neutral spans.

The circularity trap this design avoids: if gold spans were simply "the
parent chunk this question was generated from", the label set would inherit
the section strategy's chunk boundaries -- and the chunking ablation would
be rigged in favor of the section arm. Instead:

  1. The generator LLM reads a parent's text and produces a question PLUS
     verbatim answer quotes (the exact sentences that answer it).
  2. The quotes are located in the CANONICAL text by string search; the gold
     span is the located quote span itself -- typically one or two sentences,
     independent of any strategy's chunk boundaries.
  3. Questions must paraphrase (no >4 consecutive words copied from the
     source), so lexical-overlap shortcuts do not inflate dense OR keyword
     retrieval. Generated items that violate this are discarded.
  4. Generation is stratified over documents and content types so tables and
     boxes are represented, not just prose.

Wire call_llm() to your provider before running. Every generated item still
needs a 30-second human review pass (delete unanswerable/trivial ones) --
synthetic queries are cheap labels, not free labels.
"""

import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb

LANCEDB_DIR = Path(__file__).parent.parent / ".lancedb"
CACHE_DIR = Path(__file__).parent.parent / "rag" / ".md_cache"
OUT_FILE = Path(__file__).parent / "queries_synthetic.jsonl"

N_QUERIES = 60
SEED = 42

QUERY_TYPE_INSTRUCTIONS = {
    "conceptual": (
        "Ask a CONCEPTUAL question about a mechanism, effect, risk, or "
        "trade-off discussed in the section (why/how something works or "
        "what consequences it has). Paraphrase heavily."),
    "numerical": (
        "Ask a question whose answer is a SPECIFIC NUMBER, magnitude, "
        "projection, or quantitative comparison from the section."),
    "terminological": (
        "Ask about a SPECIFIC NAMED concept, indicator, institution, or "
        "technical term from the section, using its exact name in the "
        "question (the rest of the question paraphrased)."),
}

TYPE_CYCLE = ["conceptual", "conceptual", "numerical", "terminological"]

PROMPT = """You are building retrieval evaluation data from an economic policy document.

Read the section below and write ONE question that a policy analyst might
realistically ask, which this section answers.

Question type for THIS question:
{type_instruction}

Rules:
- The question must be answerable from this section alone.
- The question must be SELF-CONTAINED: an analyst who has never seen this
  document must understand exactly what is asked. Name the concrete subject
  ("commercial real estate risks", "the natural rate of interest") -- never
  "this mechanism", "these effects", "the process".
- Ask about ECONOMIC SUBSTANCE. NEVER ask about the document itself:
  methodology, data samples, definitions of conventions, certification,
  or which countries/sources are included in a calculation.
- PARAPHRASE: never copy more than 4 consecutive words from the section
  (except the named term in terminological questions).
- Also return the verbatim quote(s) from the section (1-3 sentences, copied
  EXACTLY) that contain the answer.

Return ONLY this JSON, nothing else:
{{"question": "...", "answer_quotes": ["..."]}}

Section (from {source}, {section_path}):
---
{content}
---"""


def call_llm(prompt: str) -> str:
    """Wire this to the project's LLM client (Instructor/whatever the agents
    use). Anthropic fallback included if a key is present."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    raise NotImplementedError("Set ANTHROPIC_API_KEY or wire call_llm() "
                              "to the project's LLM client.")


def _locate_quote(canonical: str, quote: str) -> tuple[int, int] | None:
    """Exact find first; whitespace-normalized regex find as fallback
    (extraction artifacts can perturb spacing)."""
    idx = canonical.find(quote)
    if idx != -1:
        return idx, idx + len(quote)
    pattern = re.escape(quote.strip())
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)
    m = re.search(pattern, canonical)
    return (m.start(), m.end()) if m else None


def _copied_run_too_long(question: str, content: str, max_words: int = 4) -> bool:
    """True if the question copies more than max_words consecutive words."""
    q_words = re.findall(r"\w+", question.lower())
    c_norm = " ".join(re.findall(r"\w+", content.lower()))
    for i in range(len(q_words) - max_words):
        window = " ".join(q_words[i:i + max_words + 1])
        if window in c_norm:
            return True
    return False


def generate() -> None:
    rng = random.Random(SEED)
    db = lancedb.connect(str(LANCEDB_DIR))
    parents = db.open_table("parent_chunks").to_pandas()
    
    # Filter out methodological/administrative parents: formally valid but
    # unrealistic questions come from appendices, assumptions, and samples.
    bad_path = parents["section_path"].str.contains(
        "annex|assumption|convention|country code|currency code|glossary|"
        "sample|data and|methodolog|archives|abbreviation",
        case=False, na=False)
    parents = parents[~bad_path & (parents["chunk_size"] > 600)]
    print(f"Sampling pool: {len(parents)} parents after relevance filter")

    # Stratify: proportional over sources, oversample non-prose so tables
    # and boxes appear in the label set at all.
    weights = parents["content_type"].map(
        {"prose": 1.0, "box": 2.5, "table": 2.5}).fillna(1.0)
    picked = parents.sample(n=min(N_QUERIES * 2, len(parents)),
                            weights=weights, random_state=SEED)

    canonicals = {p.stem: p.read_text(encoding="utf-8")
                  for p in CACHE_DIR.glob("*.md")}

    records, q_idx = [], 0
    for _, row in picked.iterrows():
        if len(records) >= N_QUERIES:
            break
        slug_candidates = [s for s in canonicals
                           if row["content"][:200] in canonicals[s]]
        canonical = canonicals.get(slug_candidates[0]) if slug_candidates else None
        source_slug = slug_candidates[0] if slug_candidates else None
        if canonical is None:
            continue

        qtype = TYPE_CYCLE[len(records) % len(TYPE_CYCLE)]
        try:
            raw = call_llm(PROMPT.format(
                type_instruction=QUERY_TYPE_INSTRUCTIONS[qtype],
                source=row["doc_title"],
                section_path=row["section_path"],
                content=row["content"]))
            data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
        except Exception as exc:
            print(f"  skip (generation failed: {exc})")
            continue

        question = data.get("question", "").strip()
        quotes = [q for q in data.get("answer_quotes", []) if q.strip()]
        if not question or not quotes:
            continue
        if _copied_run_too_long(question, row["content"]):
            print(f"  skip (question copies source): {question[:60]}")
            continue

        spans = []
        for quote in quotes:
            loc = _locate_quote(canonical, quote)
            if loc:
                spans.append({"source": source_slug, "char_start": loc[0],
                              "char_end": loc[1], "grade": 2})
        if not spans:
            print(f"  skip (quotes not found in canonical): {question[:60]}")
            continue

        q_idx += 1
        records.append({
            "query_id": f"s{q_idx:03d}",
            "query": question,
            "query_type": qtype,
            "origin": "synthetic",
            "tier2": False,
            "gold_spans": spans,
            "notes": f"generated from {row['chunk_id']}",
        })
        print(f"  ok [{q_idx}/{N_QUERIES}] {question[:70]}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} queries to {OUT_FILE}. "
          f"Now REVIEW them manually before merging into queries.jsonl.")


if __name__ == "__main__":
    generate()