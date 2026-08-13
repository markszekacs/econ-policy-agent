"""Build the Tier 2 query set: complex existing queries + perspective queries.

Tier 2 judges ANSWER quality (LLM-judge on Synthesizer outputs), not span
coverage, so gold spans are not required here. The set mixes:
  - the most complex queries from the Tier 1 set (multi-span first, then the
    longest conceptual ones): factual-synthesis ground
  - freshly generated PERSPECTIVE questions (distributional, feasibility,
    stakeholder trade-offs): the terrain where a multi-agent architecture
    should shine if it earns its keep at all. Testing agents only on
    single-fact retrieval questions would be rigged against them.

Usage:
    py -3.12 eval\\tier2_generate_queries.py
Writes eval/tier2_queries.jsonl
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb

QUERIES_FILE = Path(__file__).parent / "queries.jsonl"
OUT_FILE = Path(__file__).parent / "tier2_queries.jsonl"
LANCEDB_DIR = Path(__file__).parent.parent / ".lancedb"

N_FROM_TIER1 = 8
N_PERSPECTIVE = 8
GEN_MODEL = "claude-sonnet-4-6"

PERSPECTIVE_PROMPT = """You are building evaluation questions for an economic
policy analysis system. The system's corpus contains these documents:

{doc_titles}

Write {n} policy questions that REQUIRE weighing multiple perspectives --
macroeconomic effects, labor market impact, distributional consequences,
institutional feasibility, fiscal cost. Good questions sound like what a
minister's office would actually ask, e.g. "What would a significant minimum
wage increase mean for employment, inequality, and public finances, and how
feasible is it to implement?"

Rules:
- Each question must be answerable from documents like the ones listed
  (broad policy topics, not obscure specifics).
- Each must span at least three distinct perspectives.
- Self-contained, no references to "the documents".

Return ONLY a JSON array of {n} strings."""


def call_llm(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(model=GEN_MODEL, max_tokens=2000,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def main() -> None:
    tier1 = [json.loads(l) for l in open(QUERIES_FILE, encoding="utf-8")
             if l.strip()]

    # Complexity ordering: multi-span queries first, then longest conceptual
    tier1.sort(key=lambda q: (len(q["gold_spans"]), len(q["query"])),
               reverse=True)
    picked = tier1[:N_FROM_TIER1]

    db = lancedb.connect(str(LANCEDB_DIR))
    titles = sorted(set(db.open_table("parent_chunks")
                        .to_pandas()["doc_title"].tolist()))
    raw = call_llm(PERSPECTIVE_PROMPT.format(
        doc_titles="\n".join(f"- {t}" for t in titles), n=N_PERSPECTIVE))
    perspective = json.loads(raw.strip().removeprefix("```json")
                             .removesuffix("```"))

    records = []
    for q in picked:
        records.append({"query_id": f"t2_{q['query_id']}", "query": q["query"],
                        "query_type": q["query_type"], "origin": "tier1"})
    for i, q in enumerate(perspective, 1):
        records.append({"query_id": f"t2_persp{i:02d}", "query": q,
                        "query_type": "perspective", "origin": "generated"})

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} Tier 2 queries -> {OUT_FILE}")
    for r in records:
        print(f"  {r['query_id']} [{r['query_type']}] {r['query'][:70]}")
    print("\nReview the generated perspective questions before running the "
          "harness -- delete any unrealistic lines by hand.")


if __name__ == "__main__":
    main()