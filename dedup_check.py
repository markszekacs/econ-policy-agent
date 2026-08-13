"""Duplicate-document audit for the corpus.

Detects pairs of canonical texts (rag/.md_cache/*.md) that share substantial
content -- the symptom observed in the synthetic labels, where a chunk was
sampled from one source but its quote was located in another. Duplicated
content breaks span-based evaluation (gold source mismatch), distorts
retrieval (the same passage competes twice for candidate slots), and
weakens top-k diversity.

For each overlapping pair the script reports:
  - the estimated overlap direction (A in B / B in A / mutual)
  - how many gold spans in the eval labels point to each side
  - a KEEP / DELETE recommendation: prefer the side the labels point to;
    tiebreak on text length (the superset survives)

The script only REPORTS -- deleting PDFs is your call. After deleting,
clear the stale cache entries and re-run ingestion.

Usage (project root):
    py -3.12 dedup_check.py
"""

import json
import random
import re
from itertools import combinations
from pathlib import Path

CACHE_DIR = Path("rag") / ".md_cache"
DOCUMENTS_DIR = Path("rag") / "documents"
LABEL_FILES = [Path("eval") / "queries_synthetic.jsonl",
               Path("eval") / "queries.jsonl"]

N_SAMPLES = 20          # substring probes per direction
SAMPLE_LEN = 200        # probe length in chars
OVERLAP_THRESHOLD = 0.3  # fraction of probes found in the other text


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _probe(a_text: str, b_text: str, rng: random.Random) -> float:
    """Fraction of random substrings of A found verbatim in B."""
    if len(a_text) <= SAMPLE_LEN:
        return 1.0 if a_text in b_text else 0.0
    hits = 0
    for _ in range(N_SAMPLES):
        i = rng.randrange(len(a_text) - SAMPLE_LEN)
        if a_text[i:i + SAMPLE_LEN] in b_text:
            hits += 1
    return hits / N_SAMPLES


def _load_label_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in LABEL_FILES:
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            for span in json.loads(line).get("gold_spans", []):
                counts[span["source"]] = counts.get(span["source"], 0) + 1
    return counts


def _pdf_for_slug(slug: str) -> str:
    for pdf in DOCUMENTS_DIR.glob("*.pdf"):
        if _slug(pdf.stem) == slug:
            return pdf.name
    return "(no matching PDF found)"


def main() -> None:
    texts = {p.stem: p.read_text(encoding="utf-8")
             for p in sorted(CACHE_DIR.glob("*.md"))}
    if len(texts) < 2:
        print(f"Fewer than 2 canonical files in {CACHE_DIR} -- nothing to compare.")
        return
    label_counts = _load_label_counts()
    rng = random.Random(0)

    print(f"Comparing {len(texts)} canonical documents "
          f"({len(list(combinations(texts, 2)))} pairs)...\n")

    findings = []
    for a, b in combinations(texts, 2):
        a_in_b = _probe(texts[a], texts[b], rng)
        b_in_a = _probe(texts[b], texts[a], rng)
        if max(a_in_b, b_in_a) < OVERLAP_THRESHOLD:
            continue

        if a_in_b >= 0.8 and b_in_a >= 0.8:
            relation = "near-identical"
        elif a_in_b > b_in_a:
            relation = f"'{a}' largely contained in '{b}'"
        else:
            relation = f"'{b}' largely contained in '{a}'"

        la, lb = label_counts.get(a, 0), label_counts.get(b, 0)
        # Keep preference: labeled side first, then the longer text
        if la != lb:
            keep = a if la > lb else b
            reason = "eval labels point to it"
        else:
            keep = a if len(texts[a]) >= len(texts[b]) else b
            reason = "longer text (superset)"
        drop = b if keep == a else a

        findings.append((a, b, a_in_b, b_in_a, relation, keep, drop, reason,
                         la, lb))

    if not findings:
        print("No overlapping document pairs found -- corpus looks clean.")
        return

    for (a, b, ab, ba, relation, keep, drop, reason, la, lb) in findings:
        print("=" * 70)
        print(f"OVERLAP: {a}  <->  {b}")
        print(f"  probes: {ab:.0%} of '{a}' found in '{b}', "
              f"{ba:.0%} of '{b}' found in '{a}'  ({relation})")
        print(f"  gold spans referencing: '{a}': {la}, '{b}': {lb}")
        print(f"  sizes: '{a}': {len(texts[a]):,} chars, "
              f"'{b}': {len(texts[b]):,} chars")
        print(f"  -> KEEP   '{keep}'  ({reason})")
        print(f"  -> DELETE '{drop}'  (PDF: {DOCUMENTS_DIR / _pdf_for_slug(drop)})")
        print(f"            and its cache: {CACHE_DIR / (drop + '.md')} "
              f"+ {CACHE_DIR / (drop + '.meta.json')}")

    print("\n" + "=" * 70)
    print("After deleting the listed PDFs and their cache files, re-run:")
    print("  py -3.12 rag\\ingest.py all")
    print("Labels pointing to KEPT sources stay valid (their canonical files "
          "and offsets are untouched).")


if __name__ == "__main__":
    main()