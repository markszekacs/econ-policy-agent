"""Score every pipeline arm offline from the recorded traces.

An "arm" is a full pipeline configuration reconstructed from a trace:
  strategy   : section | section_bare | recursive | fixed   (per trace file)
  mode       : dense | keyword | hybrid       (filter by per-leg rank presence)
  cand_n     : 10 | 20 | 35 | 50              (prefix cut of each leg's ranks)
  ordering   : rerank_child | rerank_parent | no_rerank (RRF)
  top_k      : 3 | 5 | 8 | 10                 (prefix cut of the final list)

Derivation is exact, not approximate: restricting a cross-encoder ordering
to a candidate subset preserves relative order (documents are scored
independently), and RRF is computed from the stored per-leg ranks.

Two metric families, on purpose:
  - CANDIDATE metrics (recall@cand_n on child spans): how good is candidate
    generation -- scores the material the reranker gets to see.
  - FINAL metrics (recall/MRR/nDCG@top_k on PARENT spans, after parent
    dedup): how good is what the agent actually receives. Parent spans are
    used here because the parent content is what the LLM reads.

Usage:
    python eval/score_arms.py                 # full grid -> results.csv
    python eval/score_arms.py --slice query_type
"""

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

from eval_common import (load_queries, recall_at_k, mrr, ndcg_at_k,
                         bootstrap_ci, paired_bootstrap_diff)

QUERIES_FILE = Path(__file__).parent / "queries.jsonl"
TRACES_DIR = Path(__file__).parent / "traces"
RESULTS_FILE = Path(__file__).parent / "results.csv"

RRF_K = 60
CAND_NS = [10, 20, 35, 50]
TOP_KS = [3, 5, 8, 10]
MODES = ["dense", "keyword", "hybrid"]
ORDERINGS = ["rerank_child", "no_rerank"]  # rerank_parent added if traced


# ---------------------------------------------------------------------------
# Arm reconstruction from a single trace record
# ---------------------------------------------------------------------------

def _eligible(c: dict, mode: str, cand_n: int) -> bool:
    d = c.get("dense_rank")
    k = c.get("kw_rank")
    in_dense = d is not None and d < cand_n
    in_kw = k is not None and k < cand_n
    if mode == "dense":
        return in_dense
    if mode == "keyword":
        return in_kw
    return in_dense or in_kw


def _rrf(c: dict, cand_n: int, mode: str) -> float:
    score = 0.0
    d, k = c.get("dense_rank"), c.get("kw_rank")
    if mode in ("dense", "hybrid") and d is not None and d < cand_n:
        score += 1.0 / (RRF_K + d + 1)
    if mode in ("keyword", "hybrid") and k is not None and k < cand_n:
        score += 1.0 / (RRF_K + k + 1)
    return score


def final_parents(trace: dict, mode: str, cand_n: int,
                  ordering: str, top_k: int) -> list[dict]:
    """Reconstruct the top_k unique parents this arm would deliver."""
    cands = [c for c in trace["candidates"] if _eligible(c, mode, cand_n)]
    if not cands:
        return []
    idx_of = {c["chunk_id"]: i for i, c in enumerate(trace["candidates"])}

    if ordering.startswith("rerank"):
        key = ordering.removeprefix("rerank_")
        order = trace["rerank"].get(key)
        if order is None:
            return []  # this ordering was not traced
        eligible_ids = {c["chunk_id"] for c in cands}
        ranked = [trace["candidates"][r["index"]] for r in order
                  if trace["candidates"][r["index"]]["chunk_id"] in eligible_ids]
    else:
        ranked = sorted(cands, key=lambda c: (-_rrf(c, cand_n, mode),
                                              idx_of[c["chunk_id"]]))

    selected, seen = [], set()
    for c in ranked:
        pid = c["parent_chunk_id"]
        if pid in seen:
            continue
        seen.add(pid)
        selected.append({
            "source": c["source"],
            # Final metrics use PARENT spans: that content reaches the agent.
            "char_start": c["parent_char_start"],
            "char_end": c["parent_char_end"],
            "content_type": c["content_type"],
        })
        if len(selected) >= top_k:
            break
    return selected


def candidate_items(trace: dict, mode: str, cand_n: int) -> list[dict]:
    """Unordered candidate pool at CHILD spans (for recall@cand_n)."""
    return [{"source": c["source"], "char_start": c["char_start"],
             "char_end": c["char_end"]}
            for c in trace["candidates"] if _eligible(c, mode, cand_n)]


# ---------------------------------------------------------------------------
# Grid scoring
# ---------------------------------------------------------------------------

def score_all(slice_field: str | None) -> None:
    queries = {q["query_id"]: q for q in load_queries(QUERIES_FILE)}
    trace_files = sorted(TRACES_DIR.glob("trace__*.jsonl"))
    if not trace_files:
        raise SystemExit("No traces found -- run eval/run_harness.py first.")

    rows = []
    for tf in trace_files:
        strategy = tf.stem.removeprefix("trace__")
        traces = [json.loads(l) for l in open(tf, encoding="utf-8")]
        has_parent = any("parent" in t.get("rerank", {}) for t in traces)
        orderings = ORDERINGS + (["rerank_parent"] if has_parent else [])

        for mode, cand_n, ordering, top_k in product(
                MODES, CAND_NS, orderings, TOP_KS):
            per_query: dict[str, dict[str, float]] = {}
            for t in traces:
                q = queries.get(t["query_id"])
                if q is None:
                    continue
                gold = q["gold_spans"]
                finals = final_parents(t, mode, cand_n, ordering, top_k)
                cands = candidate_items(t, mode, cand_n)
                per_query[t["query_id"]] = {
                    "cand_recall": recall_at_k(cands, gold, len(cands)),
                    "recall": recall_at_k(finals, gold, top_k),
                    "mrr": mrr(finals, gold),
                    "ndcg": ndcg_at_k(finals, gold, top_k),
                    # Cost side of top_k: total parent chars handed to the
                    # agent (drives token cost and context dilution).
                    "context_chars": float(sum(
                        max(0, f["char_end"] - f["char_start"])
                        for f in finals)),
                    "_slice": q.get(slice_field, "all") if slice_field else "all",
                }

            slices = sorted({v["_slice"] for v in per_query.values()})
            for sl in slices:
                sub = [v for v in per_query.values() if v["_slice"] == sl]
                row = {"strategy": strategy, "mode": mode, "cand_n": cand_n,
                       "ordering": ordering, "top_k": top_k,
                       "slice": sl, "n_queries": len(sub)}
                for metric in ("cand_recall", "recall", "mrr", "ndcg",
                               "context_chars"):
                    mean, lo, hi = bootstrap_ci([v[metric] for v in sub])
                    row[metric] = round(mean, 4) if not math.isnan(mean) else ""
                    row[f"{metric}_lo"] = round(lo, 4) if not math.isnan(lo) else ""
                    row[f"{metric}_hi"] = round(hi, 4) if not math.isnan(hi) else ""
                rows.append(row)

    out_file = (RESULTS_FILE if slice_field is None
                else RESULTS_FILE.with_name(f"results__{slice_field}.csv"))
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} arm x slice rows -> {out_file}")

    # Headline paired comparisons at the reference operating point
    _headline(trace_files, queries)


def _headline(trace_files, queries) -> None:
    """Paired comparisons that answer the session's standing questions,
    at the reference point cand_n=20, top_k=5."""
    print("\n--- Headline paired comparisons (cand_n=20, top_k=5) ---")

    def metric_vector(traces, mode, ordering, metric="recall"):
        out = []
        for t in traces:
            q = queries.get(t["query_id"])
            if q is None:
                continue
            finals = final_parents(t, mode, 20, ordering, 5)
            val = {"recall": recall_at_k(finals, q["gold_spans"], 5),
                   "mrr": mrr(finals, q["gold_spans"])}[metric]
            out.append(val)
        return out

    for tf in trace_files:
        strategy = tf.stem.removeprefix("trace__")
        traces = [json.loads(l) for l in open(tf, encoding="utf-8")]

        pairs = [
            ("hybrid vs dense (recall)",
             metric_vector(traces, "hybrid", "rerank_child"),
             metric_vector(traces, "dense", "rerank_child")),
            ("rerank vs no-rerank (mrr)",
             metric_vector(traces, "hybrid", "rerank_child", "mrr"),
             metric_vector(traces, "hybrid", "no_rerank", "mrr")),
        ]
        if any("parent" in t.get("rerank", {}) for t in traces):
            pairs.append(
                ("child vs parent rerank (mrr)",
                 metric_vector(traces, "hybrid", "rerank_child", "mrr"),
                 metric_vector(traces, "hybrid", "rerank_parent", "mrr")))

        print(f"\n[{strategy}]")
        for name, a, b in pairs:
            mean, lo, hi = paired_bootstrap_diff(a, b)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {name:32s} diff={mean:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}] {sig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", dest="slice_field", default=None,
                    help="query field to slice by, e.g. query_type or origin")
    args = ap.parse_args()
    score_all(args.slice_field)