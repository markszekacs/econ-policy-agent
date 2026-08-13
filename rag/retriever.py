"""Query -> reranked chunks retrieval pipeline (LanceDB backend).

Reranking is done by a LOCAL cross-encoder (sentence-transformers,
cross-encoder/ms-marco-MiniLM-L-6-v2, ~80MB, runs on CPU): zero API cost,
no rate limits, no network latency, deterministic. Same architecture class
as hosted rerank APIs (query and document scored jointly), one quality
tier below the large hosted models -- an acceptable trade at zero cost,
and the rerank on/off arm in the eval quantifies what it contributes.

Search mode (search_mode argument):
  "dense" (default) | "keyword" (BM25 only) | "hybrid" (union of both legs).
  The cross-encoder scores the united pool on one scale, so the reranker IS
  the fusion mechanism; RRF is only the no-reranker fallback ordering.
  The dense leg uses the keyword-augmented agent template; the BM25 leg
  uses the RAW user query (template keyword bags would drown it).

Rerank granularity (rerank_granularity argument):
  "child" (default): all candidates reranked with a section-path prefix;
      dedup by parent AFTER reranking. The agent receives parent content.
  "parent": dedup by parent first, rerank full parent contents (eval arm).

Output: top_k UNIQUE parents with citation metadata and child/parent char
spans (for span-based evaluation ground truth).
"""

import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import RetrievalConfig, DEFAULT_CONFIG

LANCEDB_DIR = Path(__file__).parent.parent / ".lancedb"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_GRANULARITIES = ("child", "parent")
SEARCH_MODES = ("dense", "keyword", "hybrid")
RRF_K = 60  # standard reciprocal-rank-fusion constant

AGENT_QUERY_TEMPLATES = {
    "macroeconomist": "{query} macroeconomic effects GDP inflation growth monetary fiscal",
    "labor_economist": "{query} labor market employment wages workers automation",
    "trade_unionist": "{query} workers rights inequality redistribution social protection",
    "institutional": "{query} policy implementation regulation governance feasibility",
    "fiscal_expert": "{query} fiscal budget taxation government spending debt sustainability",
}

# ---------------------------------------------------------------------------
# Lazily initialized shared resources (thread-safe; retrieve_all_agents runs
# one thread per agent). The cross-encoder predict() is serialized under the
# lock: concurrent torch forward passes from multiple threads gain nothing
# on CPU and can misbehave.
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_db = None
_tables: dict[str, object] = {}
_fts_warned = False


def _get_model() -> SentenceTransformer:
    global _model
    with _lock:
        if _model is None:
            _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_reranker() -> CrossEncoder:
    global _reranker
    with _lock:
        if _reranker is None:
            _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _get_tables(lancedb_dir: Path):
    global _db, _tables
    with _lock:
        if _db is None:
            _db = lancedb.connect(str(lancedb_dir))
            _tables = {
                "child": _db.open_table("child_chunks"),
                "parent": _db.open_table("parent_chunks"),
            }
    return _tables["child"], _tables["parent"]


# ---------------------------------------------------------------------------
# Candidate generation (dense and keyword legs)
# ---------------------------------------------------------------------------

def _dense_candidates(child_table, agent_query: str, limit: int) -> list[dict]:
    model = _get_model()
    query_vector = model.encode(agent_query, normalize_embeddings=True).tolist()
    rows = (
        child_table.search(query_vector)
        .metric("cosine")  # LanceDB default is L2; wrong for our normalized vectors
        .limit(limit)
        .to_list()
    )
    for rank, row in enumerate(rows):
        row["_dense_rank"] = rank
        row["_cosine"] = 1.0 - row.get("_distance", 1.0)
    return rows


def _keyword_candidates(child_table, user_query: str, limit: int) -> list[dict]:
    """BM25 leg over the FTS index. Degrades gracefully (empty result) when
    the index does not exist, e.g. on a database built by an older ingest."""
    global _fts_warned
    fts_query = user_query.replace('"', " ").strip()
    if not fts_query:
        return []
    try:
        rows = (
            child_table.search(fts_query, query_type="fts")
            .limit(limit)
            .to_list()
        )
    except Exception as exc:
        with _lock:
            if not _fts_warned:
                print(f"Keyword search unavailable ({exc}); "
                      f"continuing with dense candidates only. "
                      f"Re-run ingestion to build the FTS index.")
                _fts_warned = True
        return []
    for rank, row in enumerate(rows):
        row["_kw_rank"] = rank
        row["_bm25"] = row.get("_score", 0.0)
    return rows


def _union_candidates(dense: list[dict], keyword: list[dict]) -> list[dict]:
    """Union by chunk_id, keeping rank/score info from both legs."""
    by_id: dict[str, dict] = {}
    for row in dense:
        by_id[row["chunk_id"]] = row
    for row in keyword:
        if row["chunk_id"] in by_id:
            by_id[row["chunk_id"]]["_kw_rank"] = row["_kw_rank"]
            by_id[row["chunk_id"]]["_bm25"] = row["_bm25"]
        else:
            by_id[row["chunk_id"]] = row
    return list(by_id.values())


def _rrf_score(row: dict) -> float:
    """Reciprocal rank fusion over whichever ranked lists this row appears in.
    Used for fallback ordering only; the main path fuses via the reranker."""
    score = 0.0
    if "_dense_rank" in row:
        score += 1.0 / (RRF_K + row["_dense_rank"] + 1)
    if "_kw_rank" in row:
        score += 1.0 / (RRF_K + row["_kw_rank"] + 1)
    return score


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _section_prefix(doc: dict) -> str:
    """Same idea as the ingestion enrichment prefix: anchor a context-dependent
    child ('This effect, however, ...') to its section for the cross-encoder."""
    parts = []
    if doc.get("chapter"):
        parts.append(doc["chapter"][:80])
    if doc.get("section_path") and doc.get("section_path") != doc.get("chapter"):
        parts.append(doc["section_path"])
    return f"[{' > '.join(parts)}]\n" if parts else ""


def retrieve_for_agent(
    query: str,
    agent_name: str,
    retrieval_config: RetrievalConfig = DEFAULT_CONFIG.retrieval,
    lancedb_dir: Path = LANCEDB_DIR,
    cohere_api_key: str | None = None,  # deprecated, ignored (local reranker)
    rerank_granularity: str = "child",
    search_mode: str = "dense",
) -> list[dict]:
    if agent_name not in AGENT_QUERY_TEMPLATES:
        raise ValueError(
            f"Unknown agent: {agent_name!r}. "
            f"Available: {', '.join(AGENT_QUERY_TEMPLATES)}"
        )
    if rerank_granularity not in RERANK_GRANULARITIES:
        raise ValueError(
            f"Unknown rerank_granularity: {rerank_granularity!r}. "
            f"Available: {', '.join(RERANK_GRANULARITIES)}"
        )
    if search_mode not in SEARCH_MODES:
        raise ValueError(
            f"Unknown search_mode: {search_mode!r}. "
            f"Available: {', '.join(SEARCH_MODES)}"
        )
    agent_query = AGENT_QUERY_TEMPLATES[agent_name].format(query=query)

    child_table, parent_table = _get_tables(lancedb_dir)
    n = retrieval_config.retrieval_candidates

    # Step 1: candidate generation. Dense leg uses the keyword-augmented agent
    # query; the BM25 leg uses the raw user query (see module docstring).
    dense_rows: list[dict] = []
    kw_rows: list[dict] = []
    if search_mode in ("dense", "hybrid"):
        dense_rows = _dense_candidates(child_table, agent_query, n)
    if search_mode in ("keyword", "hybrid"):
        kw_rows = _keyword_candidates(child_table, query, n)
    raw = _union_candidates(dense_rows, kw_rows)
    if not raw:
        return []

    # Step 2: fetch full parent rows (content + citation metadata)
    parent_ids = sorted({row["parent_chunk_id"] for row in raw})
    pid_filter = "chunk_id IN ({})".format(", ".join(repr(p) for p in parent_ids))
    parent_rows = (
        parent_table.search()
        .where(pid_filter, prefilter=True)
        .limit(len(parent_ids))
        .to_list()
    )
    parent_map = {row["chunk_id"]: row for row in parent_rows}

    docs: list[dict] = []
    for row in raw:
        pid = row["parent_chunk_id"]
        par = parent_map.get(pid)
        in_dense = "_dense_rank" in row
        in_kw = "_kw_rank" in row

        docs.append({
            "chunk_id": row["chunk_id"],
            "parent_chunk_id": pid,
            "content": par["content"] if par else row["content"],
            "child_content": row["content"],
            "source": row["source"],
            "doc_title": row.get("doc_title", row["source"]),
            "institution": row.get("institution", ""),
            "doc_year": row.get("doc_year", 0),
            "content_type": row.get("content_type", ""),
            "section_path": row.get("section_path", ""),
            "chapter": row.get("chapter", ""),
            # Page range of the parent -- that is the content the LLM sees.
            "page": par["page"] if par else row["page"],
            "page_end": (par.get("page_end", par["page"]) if par
                         else row.get("page_end", row["page"])),
            # Char spans for span-based evaluation: the matched child and the
            # parent context around it.
            "char_start": row.get("char_start", -1),
            "char_end": row.get("char_end", -1),
            "parent_char_start": par.get("char_start", -1) if par else -1,
            "parent_char_end": par.get("char_end", -1) if par else -1,
            # Leg provenance and scores (diagnostics + fallback ordering)
            "retrieval_source": ("both" if in_dense and in_kw
                                 else "dense" if in_dense else "keyword"),
            "cosine_score": row.get("_cosine"),
            "bm25_score": row.get("_bm25"),
            "score": _rrf_score(row),
        })

    # Step 3: rerank + dedup, order depending on granularity. The reranker
    # sees the united candidate pool and scores it on one scale -- this is
    # the fusion step; no explicit RRF is needed on the main path.
    if rerank_granularity == "parent":
        by_parent: dict[str, dict] = {}
        for doc in docs:
            pid = doc["parent_chunk_id"]
            if pid not in by_parent or doc["score"] > by_parent[pid]["score"]:
                by_parent[pid] = doc
        candidates = sorted(by_parent.values(), key=lambda d: d["score"], reverse=True)
        rerank_texts = [d["content"] for d in candidates]
        ordered = _local_rerank(agent_query, rerank_texts, candidates)
        selected = ordered[:retrieval_config.top_k]
    else:
        rerank_texts = [_section_prefix(d) + d["child_content"] for d in docs]
        ordered = _local_rerank(agent_query, rerank_texts, docs)
        selected = []
        seen_parents: set[str] = set()
        for doc in ordered:
            pid = doc["parent_chunk_id"]
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            selected.append(doc)
            if len(selected) >= retrieval_config.top_k:
                break

    for rank, doc in enumerate(selected):
        doc["agent"] = agent_name
        doc["rank"] = rank
        doc["rerank_granularity"] = rerank_granularity
        doc["search_mode"] = search_mode
    return selected


def _local_rerank(query: str, texts: list[str], docs: list[dict]) -> list[dict]:
    """Score (query, text) pairs with the local cross-encoder and return docs
    in descending score order over the FULL list (selection is the caller's
    job). rerank_score is sigmoid(logit), so it lives in (0, 1) like hosted
    rerank APIs' relevance scores. Falls back to RRF order on any failure."""
    if not docs:
        return []

    def fallback_order() -> list[dict]:
        ordered = sorted(docs, key=lambda d: d["score"], reverse=True)
        for doc in ordered:
            doc["rerank_score"] = doc["score"]
        return ordered

    try:
        reranker = _get_reranker()
        with _lock:  # serialize torch forward passes across agent threads
            logits = reranker.predict([(query, t) for t in texts],
                                      batch_size=32, show_progress_bar=False)
    except Exception as exc:
        print(f"Local rerank failed ({exc}); falling back to RRF order.")
        return fallback_order()

    scored = sorted(zip(docs, logits), key=lambda p: p[1], reverse=True)
    ordered = []
    for doc, logit in scored:
        d = doc.copy()
        d["rerank_score"] = 1.0 / (1.0 + math.exp(-float(logit)))
        ordered.append(d)
    return ordered


def retrieve_all_agents(
    query: str,
    retrieval_config: RetrievalConfig = DEFAULT_CONFIG.retrieval,
    lancedb_dir: Path = LANCEDB_DIR,
    cohere_api_key: str | None = None,  # deprecated, ignored (local reranker)
    agent_names: list[str] | None = None,
    rerank_granularity: str = "child",
    search_mode: str = "dense",
) -> list[dict]:
    if agent_names is None:
        agent_names = list(AGENT_QUERY_TEMPLATES.keys())

    all_docs: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(agent_names)) as ex:
        futures = {
            ex.submit(
                retrieve_for_agent,
                query, agent, retrieval_config,
                lancedb_dir, cohere_api_key,
                rerank_granularity, search_mode,
            ): agent
            for agent in agent_names
        }
        for future, agent in futures.items():
            try:
                all_docs.extend(future.result())
            except Exception as exc:
                # A single agent's failure degrades the answer but should not
                # abort the other agents' retrievals.
                print(f"Retrieval failed for agent '{agent}': {exc}")
    return all_docs