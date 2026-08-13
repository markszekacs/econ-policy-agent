"""Instrumented retrieval harness: run once wide, derive many arms offline.

Why this exists separately from retriever.py: the production retriever
returns only the selected top-k -- evaluation needs the FULL trace (every
candidate with its per-leg ranks and its position in every rerank ordering).
This harness mirrors the retriever's logic step by step (drift between the
two must be checked when either changes) but records everything.

The economics: with candidates=WIDE_N per leg and full-list reranking, one
traced run per (query x strategy) lets score_arms.py derive OFFLINE, with
zero extra API calls:
  - search modes dense / keyword / hybrid  (per-leg ranks are stored)
  - candidate budgets 10/20/35/50          (prefix cuts of the leg ranks)
  - rerank ON (child), rerank OFF (RRF)    (both orderings stored)
  - optionally rerank at parent granularity (one extra Cohere call)
  - top_k 3/5/8/10                          (prefix cuts of final lists)
This works because a cross-encoder scores each document independently:
restricting to a candidate subset preserves the relative rerank order.

Run AFTER `python rag/ingest.py all`:
    python eval/run_harness.py                 # child-granularity traces
    python eval/run_harness.py --parent-too    # + parent-granularity rerank
Traces land in eval/traces/, one JSONL per strategy.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from sentence_transformers import SentenceTransformer, CrossEncoder

from eval_common import load_queries

LANCEDB_DIR = Path(__file__).parent.parent / ".lancedb"
QUERIES_FILE = Path(__file__).parent / "queries.jsonl"
TRACES_DIR = Path(__file__).parent / "traces"

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WIDE_N = 50                      # per-leg candidate budget for the traced run
STRATEGIES = ["", "__section_bare", "__recursive", "__fixed"]

# Same anchor idea as retriever._section_prefix -- keep in sync.
def _section_prefix(row: dict) -> str:
    parts = []
    if row.get("chapter"):
        parts.append(row["chapter"][:80])
    if row.get("section_path") and row.get("section_path") != row.get("chapter"):
        parts.append(row["section_path"])
    return f"[{' > '.join(parts)}]\n" if parts else ""


def _rerank_order(reranker: CrossEncoder, query: str,
                  texts: list[str]) -> list[dict]:
    """Full rerank ordering as [{index, score}] over the given texts, using
    the local cross-encoder (deterministic, free, no rate limits)."""
    logits = reranker.predict([(query, t) for t in texts],
                              batch_size=32, show_progress_bar=False)
    order = sorted(range(len(texts)), key=lambda i: float(logits[i]),
                   reverse=True)
    return [{"index": i, "score": float(logits[i])} for i in order]


def trace_query(q: dict, child_table, parent_table, model, reranker,
                parent_too: bool) -> dict:
    query = q["query"]
    timings: dict[str, float] = {}

    # Dense leg (raw query; template variants are a Tier-1 arm run separately
    # if needed -- keeping the default trace agent-agnostic matches the
    # agent-agnostic ground truth)
    t0 = time.perf_counter()
    qvec = model.encode(query, normalize_embeddings=True).tolist()
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    dense = (child_table.search(qvec).metric("cosine")
             .limit(WIDE_N).to_list())
    timings["dense_ms"] = (time.perf_counter() - t0) * 1000

    # Keyword leg (raw query on the FTS index)
    t0 = time.perf_counter()
    try:
        kw = (child_table.search(query.replace('"', " "), query_type="fts")
              .limit(WIDE_N).to_list())
    except Exception:
        kw = []
    timings["kw_ms"] = (time.perf_counter() - t0) * 1000

    # Union with per-leg ranks
    cands: dict[str, dict] = {}
    for rank, row in enumerate(dense):
        cands[row["chunk_id"]] = {**row, "dense_rank": rank}
    for rank, row in enumerate(kw):
        if row["chunk_id"] in cands:
            cands[row["chunk_id"]]["kw_rank"] = rank
        else:
            cands[row["chunk_id"]] = {**row, "kw_rank": rank}
    cand_list = list(cands.values())
    if not cand_list:
        return {"query_id": q["query_id"], "candidates": [], "rerank": {},
                "timings": timings}

    # Parent metadata for spans/content
    parent_ids = sorted({c["parent_chunk_id"] for c in cand_list})
    pid_filter = "chunk_id IN ({})".format(", ".join(repr(p) for p in parent_ids))
    parent_rows = (parent_table.search().where(pid_filter, prefilter=True)
                   .limit(len(parent_ids)).to_list())
    pmap = {r["chunk_id"]: r for r in parent_rows}

    # Rerank orderings (child granularity; parent granularity optional)
    rerank: dict[str, list] = {}
    child_texts = [_section_prefix(c) + c["content"] for c in cand_list]
    t0 = time.perf_counter()
    rerank["child"] = _rerank_order(reranker, query, child_texts)
    timings["rerank_child_ms"] = (time.perf_counter() - t0) * 1000
    if parent_too:
        parent_texts = [pmap.get(c["parent_chunk_id"], c)["content"]
                        for c in cand_list]
        t0 = time.perf_counter()
        rerank["parent"] = _rerank_order(reranker, query, parent_texts)
        timings["rerank_parent_ms"] = (time.perf_counter() - t0) * 1000

    # Serialize only what scoring needs (not vectors, not full content)
    slim = []
    for c in cand_list:
        par = pmap.get(c["parent_chunk_id"], {})
        slim.append({
            "chunk_id": c["chunk_id"],
            "parent_chunk_id": c["parent_chunk_id"],
            "source": c["source"],
            "content_type": c.get("content_type", ""),
            "embed_truncated": bool(c.get("embed_truncated", False)),
            "char_start": c.get("char_start", -1),
            "char_end": c.get("char_end", -1),
            "parent_char_start": par.get("char_start", -1),
            "parent_char_end": par.get("char_end", -1),
            "dense_rank": c.get("dense_rank"),
            "kw_rank": c.get("kw_rank"),
        })
    return {"query_id": q["query_id"], "candidates": slim, "rerank": rerank,
            "timings": timings}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-too", action="store_true",
                    help="also record parent-granularity rerank orderings")
    args = ap.parse_args()

    queries = load_queries(QUERIES_FILE)
    print(f"{len(queries)} queries loaded")
    TRACES_DIR.mkdir(exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    db = lancedb.connect(str(LANCEDB_DIR))

    for suffix in STRATEGIES:
        child_name, parent_name = f"child_chunks{suffix}", f"parent_chunks{suffix}"
        if child_name not in set(db.table_names()):
            print(f"skip strategy '{suffix or 'section'}' (table missing -- "
                  f"run `python rag/ingest.py all`)")
            continue
        child_table = db.open_table(child_name)
        parent_table = db.open_table(parent_name)

        out_file = TRACES_DIR / f"trace{suffix or '__section'}.jsonl"
        print(f"Strategy '{suffix or 'section'}' -> {out_file.name}")
        with open(out_file, "w", encoding="utf-8") as f:
            for i, q in enumerate(queries):
                trace = trace_query(q, child_table, parent_table, model,
                                    reranker, args.parent_too)
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                print(f"  [{i + 1}/{len(queries)}] {q['query_id']}")

    print("Tracing complete.")


if __name__ == "__main__":
    main()