"""Does the parent layer earn its keep? Free analysis from existing traces.

The two-layer (small-to-big) design decouples two granularities:
  - RETRIEVAL granularity: small, focused children -- where bi-encoder
    embeddings and cross-encoder scoring work best
  - READING granularity: parents -- what the LLM needs to actually answer

This script quantifies whether that decoupling matters IN THIS CORPUS,
using only the recorded traces (no reruns, no API calls). For every query
whose gold span was found by some candidate child, it measures:

  child_coverage  = overlap(child, gold) / len(gold)
  parent_coverage = overlap(parent, gold) / len(gold)

and buckets the outcome:
  CHILD_ENOUGH   child alone covers >=95% of the gold span
                 (a flat, child-only pipeline would have sufficed here)
  PARENT_RESCUES child covers <95% but the parent covers >=95%
                 (the second layer is doing real work here)
  BOTH_PARTIAL   even the parent covers <95%
                 (answer spans a boundary -- chunking loses information)

Usage (project root):
    py -3.12 eval\\analyze_parent_child.py
"""

import json
import re
import statistics
import sys
from pathlib import Path

QUERIES_FILE = Path(__file__).parent / "queries.jsonl"
TRACE_FILE = Path(__file__).parent / "traces" / "trace__section.jsonl"
FULL_COVERAGE = 0.95


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def overlap(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def main() -> None:
    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(QUERIES_FILE, encoding="utf-8")
                if l.strip())}
    traces = [json.loads(l) for l in open(TRACE_FILE, encoding="utf-8")]

    buckets = {"CHILD_ENOUGH": [], "PARENT_RESCUES": [], "BOTH_PARTIAL": []}
    overhangs = []          # gold chars NOT covered by the best child
    child_sizes, parent_sizes = [], []
    not_found = 0

    for t in traces:
        q = queries.get(t["query_id"])
        if q is None:
            continue

        # Best gold coverage over all (candidate, gold span) pairs of the
        # same source. "Best" = the child that covers most of the gold.
        best = None  # (child_cov, parent_cov, overhang, c)
        for g in q["gold_spans"]:
            g_len = g["char_end"] - g["char_start"]
            if g_len <= 0:
                continue
            for c in t["candidates"]:
                if norm(c["source"]) != norm(g["source"]):
                    continue
                c_ov = overlap(c["char_start"], c["char_end"],
                               g["char_start"], g["char_end"])
                if c_ov == 0:
                    continue
                p_ov = overlap(c["parent_char_start"], c["parent_char_end"],
                               g["char_start"], g["char_end"])
                cand = (c_ov / g_len, p_ov / g_len, g_len - c_ov, c)
                if best is None or cand[0] > best[0]:
                    best = cand

        if best is None:
            not_found += 1
            continue

        child_cov, parent_cov, overhang, c = best
        overhangs.append(overhang)
        child_sizes.append(c["char_end"] - c["char_start"])
        parent_sizes.append(c["parent_char_end"] - c["parent_char_start"])

        if child_cov >= FULL_COVERAGE:
            buckets["CHILD_ENOUGH"].append((t["query_id"], child_cov, parent_cov))
        elif parent_cov >= FULL_COVERAGE:
            buckets["PARENT_RESCUES"].append((t["query_id"], child_cov, parent_cov))
        else:
            buckets["BOTH_PARTIAL"].append((t["query_id"], child_cov, parent_cov))

    n = sum(len(v) for v in buckets.values())
    print(f"Queries with a gold-overlapping candidate: {n} "
          f"(+{not_found} where no candidate touched the gold at all)\n")

    for name, items in buckets.items():
        share = len(items) / n if n else 0
        print(f"{name:15s} {len(items):3d}  ({share:.0%})")
    print()

    if overhangs:
        covered = [o for (_, c, _) in buckets["PARENT_RESCUES"] for o in [0]]
        print(f"Gold chars beyond the best child (overhang): "
              f"median {statistics.median(overhangs):.0f}, "
              f"max {max(overhangs)}")
    if child_sizes:
        print(f"Delivered context if child-only: median "
              f"{statistics.median(child_sizes):.0f} chars; "
              f"with parent expansion: median "
              f"{statistics.median(parent_sizes):.0f} chars "
              f"({statistics.median(parent_sizes) / max(1, statistics.median(child_sizes)):.1f}x)")

    print("\nInterpretation guide:")
    print("  - High CHILD_ENOUGH: the parent layer mostly pays a context-size")
    print("    cost for little coverage gain -- a flat child-only pipeline is")
    print("    a serious Tier-2 candidate (cheaper contexts, less dilution).")
    print("  - High PARENT_RESCUES: the second layer does real work -- flat")
    print("    retrieval would hand the LLM truncated answers.")
    print("  - Nontrivial BOTH_PARTIAL: answers cross section boundaries --")
    print("    an argument for neighbor-expansion or larger parents.")

    if buckets["BOTH_PARTIAL"]:
        print("\nBOTH_PARTIAL queries (answers crossing chunk boundaries):")
        for qid, c_cov, p_cov in buckets["BOTH_PARTIAL"]:
            print(f"  {qid}: child covers {c_cov:.0%}, parent covers {p_cov:.0%}"
                  f"  | {queries[qid]['query'][:60]}")


if __name__ == "__main__":
    main()