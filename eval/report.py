"""Final evaluation report: one clean table per experiment question.

Reads eval/results.csv (produced by score_arms.py) plus the traces, and
writes eval/report.md. Design principle: every table answers ONE question,
varying ONE axis while all other parameters sit at the reference operating
point, so the numbers in a table are directly comparable:

    REFERENCE POINT: mode=hybrid, cand_n=20, ordering=rerank_child, top_k=5

Cells show "mean [95% CI]" from query-level bootstrap. The last section
reports PAIRED differences (same queries, difference resampled), which is
the statistically correct way to compare arms -- two overlapping marginal
CIs do NOT imply "no difference", the paired test decides.

Usage:
    python eval/score_arms.py          # produces results.csv
    python eval/report.py              # produces report.md (+ prints it)
"""

import csv
import json
import math
from pathlib import Path

from eval_common import load_queries, recall_at_k, mrr, paired_bootstrap_diff
from score_arms import final_parents

RESULTS_FILE = Path(__file__).parent / "results.csv"
QUERIES_FILE = Path(__file__).parent / "queries.jsonl"
TRACES_DIR = Path(__file__).parent / "traces"
REPORT_FILE = Path(__file__).parent / "report.md"

REF = {"mode": "hybrid", "cand_n": "20", "ordering": "rerank_child",
       "top_k": "5", "slice": "all"}
REF_STRATEGY = "section"

STRATEGY_LABELS = {
    "section": "section-aware + prefix",
    "section_bare": "section-aware, no prefix",
    "recursive": "recursive splitter",
    "fixed": "fixed windows + overlap",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(row: dict, metric: str) -> str:
    if not row or row.get(metric, "") == "":
        return "--"
    return f"{float(row[metric]):.3f} [{float(row[metric + '_lo']):.3f}, " \
           f"{float(row[metric + '_hi']):.3f}]"


def pick(rows: list[dict], **fixed) -> dict | None:
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in fixed.items()):
            return r
    return None


def fmt_chars(row: dict) -> str:
    if not row or row.get("context_chars", "") == "":
        return "--"
    return f"{float(row['context_chars']):,.0f}"


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p / 100 * len(vals)))]


def timing_stats() -> dict[str, tuple[float, float]]:
    """(p50, p95) in ms per timing key, over queries of the section trace.
    Percentiles, never means: latency distributions are right-skewed and a
    single slow API call would distort an average."""
    tf = TRACES_DIR / "trace__section.jsonl"
    if not tf.exists():
        return {}
    per_key: dict[str, list[float]] = {}
    for line in open(tf, encoding="utf-8"):
        t = json.loads(line)
        for k, v in t.get("timings", {}).items():
            per_key.setdefault(k, []).append(v)
    return {k: (_pct(v, 50), _pct(v, 95)) for k, v in per_key.items()}


def _lat(tm: dict, keys: list[str]) -> str:
    if not tm or any(k not in tm for k in keys):
        return "--"
    p50 = sum(tm[k][0] for k in keys)
    p95 = sum(tm[k][1] for k in keys)
    return f"{p50:.0f} / {p95:.0f}"


def table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def section(title: str, question: str, tbl: str, note: str = "") -> str:
    out = f"\n## {title}\n\n*Question: {question}*\n\n{tbl}\n"
    if note:
        out += f"\n> {note}\n"
    return out


# ---------------------------------------------------------------------------
# Report sections (one experiment question each)
# ---------------------------------------------------------------------------

def sec_chunking(rows) -> str:
    body = []
    for strat, label in STRATEGY_LABELS.items():
        r = pick(rows, strategy=strat, **REF)
        body.append([label, fmt(r, "cand_recall"), fmt(r, "recall"),
                     fmt(r, "mrr"), fmt(r, "ndcg")])
    return section(
        "1. Chunking strategy",
        "does section-aware chunking beat size-based baselines, and does the "
        "enrichment prefix contribute on its own?",
        table(["strategy", "cand recall@20", "recall@5", "MRR", "nDCG@5"], body),
        "Read (section vs section_bare) for the prefix effect and "
        "(section_bare vs recursive vs fixed) for the boundary effect -- the "
        "two treatments are deliberately separable. LIMITATION: labels are "
        "synthetic-only; questions were generated from section-strategy "
        "parents, so although gold spans are quote-grounded and boundary-"
        "neutral, the question DISTRIBUTION favors section-coherent asks. "
        "Treat the section-vs-baseline margin as an upper bound.")


def sec_search_mode(rows, tm) -> str:
    lat_keys = {"dense": ["embed_ms", "dense_ms"], "keyword": ["kw_ms"],
                "hybrid": ["embed_ms", "dense_ms", "kw_ms"]}
    body = []
    for mode in ["dense", "keyword", "hybrid"]:
        r = pick(rows, strategy=REF_STRATEGY, **{**REF, "mode": mode})
        body.append([mode, fmt(r, "cand_recall"), fmt(r, "recall"),
                     fmt(r, "mrr"), fmt(r, "ndcg"), _lat(tm, lat_keys[mode])])
    return section(
        "2. Search mode",
        "does the BM25 leg recover candidates the bi-encoder misses, and "
        "what does it cost in latency?",
        table(["mode", "cand recall@20", "recall@5", "MRR", "nDCG@5",
               "search ms (p50/p95)"], body),
        "Candidate recall is the primary column here: the legs differ in what "
        "they FIND; the shared reranker handles the ordering. Hybrid latency "
        "is the sequential sum of both legs (an upper bound -- they can run "
        "in parallel in production). See section 6 for the query_type "
        "breakdown where hybrid is expected to earn its keep.")


def sec_ordering(rows, tm) -> str:
    orderings = ["rerank_child", "rerank_parent", "no_rerank"]
    labels = {"rerank_child": "rerank on children (+prefix)",
              "rerank_parent": "rerank on parents",
              "no_rerank": "no rerank (RRF order)"}
    lat = {"rerank_child": _lat(tm, ["rerank_child_ms"]),
           "rerank_parent": _lat(tm, ["rerank_parent_ms"]),
           "no_rerank": "0 (skipped)"}
    body = []
    for o in orderings:
        r = pick(rows, strategy=REF_STRATEGY, **{**REF, "ordering": o})
        body.append([labels[o], fmt(r, "recall"), fmt(r, "mrr"),
                     fmt(r, "ndcg"), lat[o]])
    return section(
        "3. Reranking",
        "what does the local cross-encoder reranker buy over vector/RRF "
        "order, at which granularity, and at what latency cost?",
        table(["ordering", "recall@5", "MRR", "nDCG@5",
               "rerank ms (p50/p95)"], body),
        "MRR is the primary column: reranking exists to move the first "
        "relevant hit up. The latency column is the price of that MRR gain "
        "-- this pair of numbers IS the reranker ROI. Note the child-vs-"
        "parent latency gap too: children are shorter documents.")


def sec_cand_n(rows) -> str:
    body = []
    for n in ["10", "20", "35", "50"]:
        r = pick(rows, strategy=REF_STRATEGY, **{**REF, "cand_n": n})
        body.append([n, fmt(r, "cand_recall"), fmt(r, "recall"), fmt(r, "mrr")])
    return section(
        "4. Candidate budget",
        "where does candidate recall saturate -- how many candidates are "
        "worth paying for?",
        table(["candidates/leg", "cand recall@N", "recall@5", "MRR"], body),
        "Past the plateau, extra candidates only add rerank cost. The final "
        "recall@5 column shows whether a bigger pool ever hurts the top-5.")


def sec_top_k(rows) -> str:
    body = []
    for k in ["3", "5", "8", "10"]:
        r = pick(rows, strategy=REF_STRATEGY, **{**REF, "top_k": k})
        body.append([k, fmt(r, "recall"), fmt(r, "ndcg"), fmt_chars(r)])
    return section(
        "5. Context size (top_k)",
        "how much of the gold material fits into k parents, and what does "
        "each extra parent cost in context?",
        table(["top_k", "recall@k", "nDCG@k", "avg context chars"], body),
        "Recall is monotone in k BY CONSTRUCTION -- read the MARGINAL recall "
        "gain per row against the context-chars growth: that ratio is the "
        "real decision variable (token cost and context dilution scale with "
        "chars). The downstream half of this question (does a diluted "
        "context hurt the Synthesizer) is a Tier-2 end-to-end measurement.")


def sec_slices(rows_sliced) -> str:
    if not rows_sliced:
        return ("\n## 6. Breakdown by query type\n\n"
                "*Run `python eval/score_arms.py --slice query_type` "
                "to populate this section.*\n")
    types = sorted({r["slice"] for r in rows_sliced if r["slice"] != "all"})
    body = []
    for t in types:
        r_h = pick(rows_sliced, strategy=REF_STRATEGY,
                   **{**REF, "mode": "hybrid", "slice": t})
        r_d = pick(rows_sliced, strategy=REF_STRATEGY,
                   **{**REF, "mode": "dense", "slice": t})
        n = r_h["n_queries"] if r_h else "--"
        body.append([t, str(n), fmt(r_d, "recall"), fmt(r_h, "recall")])
    return section(
        "6. Breakdown by query type",
        "WHERE does hybrid help -- which query types need the BM25 leg?",
        table(["query type", "n", "dense recall@5", "hybrid recall@5"], body),
        "Expected pattern: terminological and numerical queries drive the "
        "hybrid gain; conceptual queries should be flat. Small n per cell -- "
        "read CIs, not point estimates.")


def sec_paired() -> str:
    """Paired differences at the reference point -- the statistically
    decisive comparisons for the session's standing questions."""
    queries = {q["query_id"]: q for q in load_queries(QUERIES_FILE)}
    tf = TRACES_DIR / "trace__section.jsonl"
    if not tf.exists():
        return "\n## 7. Paired comparisons\n\n*No section trace found.*\n"
    traces = [json.loads(l) for l in open(tf, encoding="utf-8")]
    has_parent = any("parent" in t.get("rerank", {}) for t in traces)

    def vec(mode, ordering, metric):
        out = []
        for t in traces:
            q = queries.get(t["query_id"])
            if q is None:
                continue
            finals = final_parents(t, mode, 20, ordering, 5)
            out.append(recall_at_k(finals, q["gold_spans"], 5)
                       if metric == "recall" else mrr(finals, q["gold_spans"]))
        return out

    comps = [
        ("hybrid - dense", "recall",
         vec("hybrid", "rerank_child", "recall"),
         vec("dense", "rerank_child", "recall")),
        ("rerank - no_rerank", "MRR",
         vec("hybrid", "rerank_child", "mrr"),
         vec("hybrid", "no_rerank", "mrr")),
    ]
    if has_parent:
        comps.append(("child - parent rerank", "MRR",
                      vec("hybrid", "rerank_child", "mrr"),
                      vec("hybrid", "rerank_parent", "mrr")))

    body = []
    for name, metric, a, b in comps:
        mean, lo, hi = paired_bootstrap_diff(a, b)
        verdict = ("**significant**" if (lo > 0 or hi < 0) else "not significant")
        body.append([name, metric, f"{mean:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]",
                     verdict])
    return section(
        "7. Paired comparisons (strategy=section, cand_n=20, top_k=5)",
        "which pipeline choices make a statistically defensible difference?",
        table(["comparison", "metric", "mean diff", "95% CI (paired)",
               "verdict"], body),
        "Paired over queries: between-query variance is removed, so these "
        "CIs are much tighter than the marginal CIs above. This table is "
        "the one to quote.")


# ---------------------------------------------------------------------------

def main() -> None:
    plain = [r for r in _read_csv(RESULTS_FILE) if r.get("slice") == "all"]
    sliced = [r for r in _read_csv(
        RESULTS_FILE.with_name("results__query_type.csv"))
        if r.get("slice") != "all"]

    n_q = plain[0]["n_queries"] if plain else "?"
    tm = timing_stats()
    report = (f"# Retrieval evaluation report\n\n"
              f"Queries: {n_q} | Reference point: mode=hybrid, cand_n=20, "
              f"ordering=rerank_child, top_k=5 | Cells: mean [95% bootstrap CI] "
              f"| Latencies: p50/p95 ms over queries\n")
    report += sec_chunking(plain)
    report += sec_search_mode(plain, tm)
    report += sec_ordering(plain, tm)
    report += sec_cand_n(plain)
    report += sec_top_k(plain)
    report += sec_slices(sliced)
    report += sec_paired()

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport written to {REPORT_FILE}")


if __name__ == "__main__":
    main()