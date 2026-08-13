"""Tier 2 harness: agent-architecture ablation (arms A / B / C).

  A  full multi-agent: 5 persona retrievals + 5 persona analyses + Synthesizer
  B  single generalist agent, single raw-query retrieval
  C  single generalist agent, but the UNION of all 5 persona retrievals as
     context -- the decisive arm: (A - C) isolates thinking-diversity,
     (C - B) isolates retrieval-diversity

Design notes:
  - Retrieval goes through the REAL retriever module (hybrid, child-rerank),
    so the retrieval layer is the production path, not an approximation.
  - APPROXIMATION FLAG: agent prompts below approximate the production
    personas, and the Critic step is omitted from ALL arms (keeping the
    comparison internally fair). To measure the exact production system,
    replace AGENT_PROMPTS/SYNTH_PROMPT with the real ones from the graph.
  - C's context is merged by RANK ROUND-ROBIN across agents (rank-0 of each
    agent, then rank-1, ...), never by comparing rerank scores across agents
    -- those scores are query-dependent and not comparable.
  - Every answer is appended to eval/tier2/answers.jsonl IMMEDIATELY and
    existing (query, arm) pairs are skipped on rerun: the run is resumable
    and judge experiments never require re-generating answers.

Usage:
    py -3.12 eval\\tier2_harness.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

import retriever  # the real production retrieval pipeline

QUERIES_FILE = Path(__file__).parent / "tier2_queries.jsonl"
OUT_DIR = Path(__file__).parent / "tier2"
ANSWERS_FILE = OUT_DIR / "answers.jsonl"

COMPETITOR_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500
C_CONTEXT_CAP = 10          # max unique parents in arm C's merged context
SEARCH_MODE = "hybrid"

# Allow template-free retrieval for arms B and C's generalist
retriever.AGENT_QUERY_TEMPLATES["generalist"] = "{query}"

AGENT_PROMPTS = {
    # Approximations of the production personas -- replace with the real
    # prompts from the graph for an exact measurement.
    "macroeconomist": "You are a macroeconomist. Analyze the question through "
        "GDP, inflation, growth, and monetary-fiscal interactions.",
    "labor_economist": "You are a labor economist. Analyze the question through "
        "employment, wages, labor force participation, and automation effects.",
    "trade_unionist": "You are a workers' representative. Analyze the question "
        "through workers' rights, inequality, and social protection.",
    "institutional": "You are an institutional economist. Analyze implementation "
        "feasibility, regulation, governance, and political economy.",
    "fiscal_expert": "You are a fiscal policy expert. Analyze budget impact, "
        "taxation, public spending, and debt sustainability.",
}

ANALYSIS_TEMPLATE = """{persona}

Policy question: {query}

Source excerpts (cite as [S1], [S2], ...):
{sources}

Write a focused analysis (300-500 words) from your perspective. Ground every
substantive claim in the sources with [S#] citations. If the sources do not
support a claim, say so rather than asserting it."""

GENERALIST_PROMPT = ("You are a senior economic policy analyst producing a "
                     "final assessment for a decision maker.")

FINAL_ANSWER_TEMPLATE = """{persona}

Policy question: {query}

Source excerpts (cite as [S1], [S2], ...):
{sources}

Write the final policy assessment (500-800 words): key effects, risks,
trade-offs, and a recommendation. Ground claims in the sources with [S#]
citations; flag important considerations the sources do not cover."""

SYNTH_TEMPLATE = """You are the synthesizer of a multi-expert economic policy
analysis system.

Policy question: {query}

Expert analyses:
{analyses}

Write the final policy assessment (500-800 words) for a decision maker:
integrate the perspectives, surface agreements and tensions between experts,
and give a recommendation. Preserve the experts' [S#] citations for claims
you carry over."""


# ---------------------------------------------------------------------------

def call_llm(prompt: str, retries: int = 4) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=COMPETITOR_MODEL, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    (API error: {exc}; retrying in {wait}s)")
            time.sleep(wait)


def format_sources(docs: list[dict]) -> tuple[str, int]:
    parts = []
    total = 0
    for i, d in enumerate(docs, 1):
        head = (f"[S{i}] {d.get('doc_title', d['source'])}, "
                f"pp.{d.get('page', '?')}-{d.get('page_end', '?')}"
                f" ({d.get('section_path', '')})")
        parts.append(f"{head}\n{d['content']}")
        total += len(d["content"])
    return "\n\n".join(parts), total


def merge_round_robin(per_agent: dict[str, list[dict]], cap: int) -> list[dict]:
    """Interleave agents' ranked lists (rank 0 of each, then rank 1, ...),
    dedup by parent id. Avoids cross-agent score comparison."""
    merged, seen = [], set()
    max_len = max((len(v) for v in per_agent.values()), default=0)
    for rank in range(max_len):
        for agent, docs in per_agent.items():
            if rank >= len(docs):
                continue
            pid = docs[rank]["parent_chunk_id"]
            if pid in seen:
                continue
            seen.add(pid)
            merged.append(docs[rank])
            if len(merged) >= cap:
                return merged
    return merged


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run_arm_a(query: str) -> dict:
    """5 persona retrievals -> 5 parallel analyses -> synthesizer."""
    def analyze(agent: str) -> str:
        docs = retriever.retrieve_for_agent(query, agent,
                                            search_mode=SEARCH_MODE)
        sources, _ = format_sources(docs)
        return call_llm(ANALYSIS_TEMPLATE.format(
            persona=AGENT_PROMPTS[agent], query=query, sources=sources))

    with ThreadPoolExecutor(max_workers=5) as ex:
        analyses = list(ex.map(analyze, AGENT_PROMPTS.keys()))

    blocks = "\n\n".join(f"=== {name} ===\n{text}"
                         for name, text in zip(AGENT_PROMPTS, analyses))
    answer = call_llm(SYNTH_TEMPLATE.format(query=query, analyses=blocks))
    return {"answer": answer, "n_llm_calls": 6,
            "context_chars": sum(len(a) for a in analyses)}


def run_arm_b(query: str) -> dict:
    """Single generalist: raw-query retrieval -> one final answer."""
    docs = retriever.retrieve_for_agent(query, "generalist",
                                        search_mode=SEARCH_MODE)
    sources, ctx = format_sources(docs)
    answer = call_llm(FINAL_ANSWER_TEMPLATE.format(
        persona=GENERALIST_PROMPT, query=query, sources=sources))
    return {"answer": answer, "n_llm_calls": 1, "context_chars": ctx}


def run_arm_c(query: str) -> dict:
    """Single generalist, but the union of all 5 persona retrievals."""
    per_agent = {agent: retriever.retrieve_for_agent(query, agent,
                                                     search_mode=SEARCH_MODE)
                 for agent in AGENT_PROMPTS}
    merged = merge_round_robin(per_agent, C_CONTEXT_CAP)
    sources, ctx = format_sources(merged)
    answer = call_llm(FINAL_ANSWER_TEMPLATE.format(
        persona=GENERALIST_PROMPT, query=query, sources=sources))
    return {"answer": answer, "n_llm_calls": 1, "context_chars": ctx}


ARMS = {"A": run_arm_a, "B": run_arm_b, "C": run_arm_c}


# ---------------------------------------------------------------------------

def main() -> None:
    queries = [json.loads(l) for l in open(QUERIES_FILE, encoding="utf-8")
               if l.strip()]
    OUT_DIR.mkdir(exist_ok=True)

    done = set()
    if ANSWERS_FILE.exists():
        for line in open(ANSWERS_FILE, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["query_id"], r["arm"]))
        print(f"Resuming: {len(done)} (query, arm) pairs already answered")

    for i, q in enumerate(queries):
        for arm, fn in ARMS.items():
            if (q["query_id"], arm) in done:
                continue
            print(f"[{i + 1}/{len(queries)}] {q['query_id']} arm {arm} ...")
            t0 = time.perf_counter()
            result = fn(q["query"])
            record = {"query_id": q["query_id"], "query": q["query"],
                      "query_type": q.get("query_type", ""), "arm": arm,
                      "model": COMPETITOR_MODEL,
                      "elapsed_s": round(time.perf_counter() - t0, 1),
                      **result}
            with open(ANSWERS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"    done in {record['elapsed_s']}s "
                  f"({record['n_llm_calls']} calls)")

    print(f"\nAll arms answered -> {ANSWERS_FILE}")
    print("Next: py -3.12 eval\\tier2_judge.py")


if __name__ == "__main__":
    main()