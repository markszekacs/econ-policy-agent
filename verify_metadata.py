"""Read-only verification of ingested LanceDB tables and their metadata.

Checks per strategy table pair:
  1. Row counts and schema
  2. Sample child rows with their parent (linkage spot check)
  3. Metadata health: empty section_path / chapter rates, page sanity
  4. Invariants: content == canonical[char_start:char_end], child span
     inside parent span, every parent_chunk_id resolves, vector norms ~= 1
"""

import json
import sys
import random
from pathlib import Path

import lancedb
import numpy as np

LANCEDB_DIR = Path(__file__).parent / ".lancedb"
CACHE_DIR = Path(__file__).parent / "src" / ".md_cache"
# ^ adjust if ingest.py lives elsewhere; cache sits next to ingest.py

SUFFIXES = ["", "__section_bare", "__recursive", "__fixed"]


def load_canonical(source: str) -> str | None:
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    md_file = CACHE_DIR / f"{slug}.md"
    return md_file.read_text(encoding="utf-8") if md_file.exists() else None


def verify_pair(db, suffix: str) -> None:
    child_name = f"child_chunks{suffix}"
    parent_name = f"parent_chunks{suffix}"
    existing = set(db.table_names())
    if child_name not in existing:
        return

    print("=" * 70)
    print(f"STRATEGY TABLES: {child_name} / {parent_name}")
    print("=" * 70)

    children = db.open_table(child_name).to_pandas()
    parents = db.open_table(parent_name).to_pandas()
    print(f"Rows: {len(parents)} parents, {len(children)} children "
          f"(ratio {len(children) / max(len(parents), 1):.1f}x)")
    print(f"Child columns: {list(children.columns)}")

    # --- Metadata health ---
    NEW_SCHEMA_COLS = {"section_path", "chapter", "char_start", "char_end", "page_end"}
    missing = NEW_SCHEMA_COLS - set(children.columns)
    if missing:
        print(f"!! Missing columns {sorted(missing)} -- this table was built "
              f"by the OLD ingest.py. Re-run ingestion with the new pipeline.")
        print()
        return

    for col in ["section_path", "chapter"]:
        empty = (children[col].fillna("") == "").mean()
        print(f"Empty {col}: {empty:.0%} of children")

    # --- Invariant 1: every child's parent exists ---
    parent_ids = set(parents["chunk_id"])
    orphans = children[~children["parent_chunk_id"].isin(parent_ids)]
    print(f"Orphan children (parent_chunk_id not found): {len(orphans)}"
          + ("  <-- FAIL" if len(orphans) else "  OK"))

    # --- Invariant 2: child span inside parent span ---
    merged = children.merge(
        parents[["chunk_id", "char_start", "char_end"]],
        left_on="parent_chunk_id", right_on="chunk_id", suffixes=("", "_par"))
    outside = merged[(merged["char_start"] < merged["char_start_par"]) |
                     (merged["char_end"] > merged["char_end_par"])]
    print(f"Children outside parent span: {len(outside)}"
          + ("  <-- FAIL" if len(outside) else "  OK"))

    # --- Invariant 3: content == canonical[char_start:char_end] ---
    mismatches, checked = 0, 0
    for source in children["source"].unique():
        canonical = load_canonical(source)
        if canonical is None:
            print(f"!! No cache file for source '{source}' - skipping offset check")
            continue
        sample = children[children["source"] == source].sample(
            min(50, (children["source"] == source).sum()), random_state=0)
        for _, row in sample.iterrows():
            checked += 1
            if canonical[row["char_start"]:row["char_end"]] != row["content"]:
                mismatches += 1
    print(f"Offset mismatches: {mismatches}/{checked} sampled"
          + ("  <-- FAIL" if mismatches else "  OK"))

    # --- Invariant 4: normalized embeddings ---
    vecs = np.stack(children["vector"].sample(min(100, len(children)),
                                              random_state=0).to_numpy())
    norms = np.linalg.norm(vecs, axis=1)
    print(f"Vector norms: mean {norms.mean():.4f} (expect ~1.0)"
          + ("  <-- FAIL" if abs(norms.mean() - 1.0) > 0.01 else "  OK"))

    # --- Sample rows ---
    print("\n--- SAMPLE: 3 random children with parent context ---")
    for _, row in children.sample(min(3, len(children)), random_state=1).iterrows():
        par = parents[parents["chunk_id"] == row["parent_chunk_id"]].iloc[0]
        print(f"\n  child {row['chunk_id']} | p.{row['page']}"
              f" | span [{row['char_start']}:{row['char_end']}]")
        print(f"  chapter: {row['chapter'][:70] or '(empty)'}")
        print(f"  section_path: {row['section_path'][:70] or '(empty)'}")
        print(f"  content: {row['content'][:150]!r}...")
        print(f"  parent {par['chunk_id']} ({par['chunk_size']} chars): "
              f"{par['content'][:120]!r}...")
    print()


if __name__ == "__main__":
    db = lancedb.connect(str(LANCEDB_DIR))
    print(f"All tables in DB: {db.table_names()}\n")
    for suffix in SUFFIXES:
        verify_pair(db, suffix)