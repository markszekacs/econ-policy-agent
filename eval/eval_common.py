"""Shared evaluation core: query loading, span matching, metrics, statistics.

Design principles:
  - Ground truth lives at (query, source document, char span) level in the
    CANONICAL markdown. Spans are chunking-independent, so one label set
    scores every chunking strategy, search mode, and pipeline variant.
  - A retrieved item matches a gold span if their overlap covers at least
    MATCH_RATIO of the SHORTER of the two. Rationale: chunk sizes vary wildly
    across strategies (400-char children vs 3000-char parents); a symmetric
    IoU would punish a small gold span inside a large-but-correct parent,
    while "overlap >= 50% of the shorter" treats both directions fairly.
  - All comparisons between arms are PAIRED (same queries), so differences
    are tested with paired bootstrap -- much higher power than comparing
    two independent means at n~70.
"""

import json
import math
import random
from pathlib import Path
import re as _re

MATCH_RATIO = 0.5


def norm_source(name: str) -> str:
    """Canonical source id: matches ingest's _slug. Normalizing here makes
    labels comparable across raw-stem and slugged source fields."""
    return _re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Span matching
# ---------------------------------------------------------------------------

def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def is_match(item: dict, gold: dict) -> bool:
    """item: {source, char_start, char_end}; gold: same fields.
    Sources must agree; spans must overlap >= MATCH_RATIO of the shorter."""
    if norm_source(item["source"]) != norm_source(gold["source"]):
        return False
    if item["char_start"] < 0 or gold["char_start"] < 0:
        return False
    ov = overlap(item["char_start"], item["char_end"],
                 gold["char_start"], gold["char_end"])
    shorter = min(item["char_end"] - item["char_start"],
                  gold["char_end"] - gold["char_start"])
    return shorter > 0 and ov >= MATCH_RATIO * shorter


def grade_of(item: dict, gold_spans: list[dict]) -> int:
    """Highest grade among gold spans this item matches (0 if none)."""
    return max((g.get("grade", 1) for g in gold_spans if is_match(item, g)),
               default=0)


# ---------------------------------------------------------------------------
# Metrics over a RANKED list of items (each with source/char_start/char_end)
# ---------------------------------------------------------------------------

def recall_at_k(items: list[dict], gold_spans: list[dict], k: int) -> float:
    """Fraction of gold spans matched by at least one of the top-k items.
    This is the 'did the relevant material make it' metric -- the primary
    number for candidate-generation experiments (search modes, candidates-N),
    because the reranker can only promote what is present."""
    if not gold_spans:
        return float("nan")
    top = items[:k]
    hit = sum(1 for g in gold_spans if any(is_match(it, g) for it in top))
    return hit / len(gold_spans)


def mrr(items: list[dict], gold_spans: list[dict]) -> float:
    """1/rank of the first relevant item. The 'how high is the first good
    hit' metric -- sensitive exactly where reranking should help, so it is
    the primary number for the rerank on/off and rerank-variant arms."""
    for rank, it in enumerate(items, start=1):
        if grade_of(it, gold_spans) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(items: list[dict], gold_spans: list[dict], k: int) -> float:
    """Graded position-weighted quality of the whole top-k list. Uses the
    grade field (2 = directly answers, 1 = supporting context), so a list
    that puts direct answers above supporting context scores higher even
    when binary recall is identical."""
    gains = [grade_of(it, gold_spans) for it in items[:k]]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_grades = sorted((g.get("grade", 1) for g in gold_spans), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    return dcg / idcg if idcg > 0 else float("nan")


# ---------------------------------------------------------------------------
# Statistics: bootstrap CIs and paired comparisons over queries
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) over queries, NaNs dropped."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(vals) / len(vals), lo, hi


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float, float]:
    """Mean difference (a - b) with CI, resampling QUERY-LEVEL paired
    differences. Pairing removes between-query variance, which dominates
    at n~70 -- this is what makes small but consistent arm differences
    detectable at all."""
    diffs = [x - y for x, y in zip(a, b)
             if not math.isnan(x) and not math.isnan(y)]
    return bootstrap_ci(diffs, n_boot=n_boot, seed=seed)