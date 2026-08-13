"""Manual labeling helper for building gold spans in the canonical markdown.

Workflow (REPL / Jupyter friendly):
    >>> from label_helper import find, show, entry
    >>> find("bis_annual_economic_report_2024", "natural rate")
    ... prints every occurrence with char offsets and surrounding context ...
    >>> show("bis_annual_economic_report_2024", 118200, 119400)
    ... prints that exact span so you can tighten the boundaries ...
    >>> entry("q014", "What makes r-star a poor guide for policy?",
    ...       "terminological", "bis_annual_economic_report_2024", 118250, 119100,
    ...       grade=2, tier2=True)
    ... prints a JSON line ready to append to eval/queries.jsonl ...

Why spans are labeled in the canonical markdown and not in the PDF or in
chunks: the canonical text is the shared substrate every chunking strategy
cuts from, so a span labeled once is valid ground truth for every arm.
"""

import json
import re
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "rag" / ".md_cache"

_texts: dict[str, str] = {}


def _load(source_slug: str) -> str:
    if source_slug not in _texts:
        md_file = CACHE_DIR / f"{source_slug}.md"
        if not md_file.exists():
            available = sorted(p.stem for p in CACHE_DIR.glob("*.md"))
            raise FileNotFoundError(
                f"No canonical file for '{source_slug}'. Available: {available}")
        _texts[source_slug] = md_file.read_text(encoding="utf-8")
    return _texts[source_slug]


def sources() -> list[str]:
    return sorted(p.stem for p in CACHE_DIR.glob("*.md"))


def find(source_slug: str, pattern: str, context: int = 150,
         regex: bool = False) -> None:
    """Print all occurrences of a pattern with char offsets and context."""
    text = _load(source_slug)
    if regex:
        matches = [(m.start(), m.end()) for m in re.finditer(pattern, text,
                                                             re.IGNORECASE)]
    else:
        matches, pos = [], 0
        low_text, low_pat = text.lower(), pattern.lower()
        while (i := low_text.find(low_pat, pos)) != -1:
            matches.append((i, i + len(pattern)))
            pos = i + 1
    print(f"{len(matches)} occurrence(s) of {pattern!r} in {source_slug}")
    for start, end in matches:
        c0, c1 = max(0, start - context), min(len(text), end + context)
        snippet = text[c0:c1].replace("\n", " ")
        print(f"\n  [{start}:{end}]")
        print(f"  ...{snippet}...")


def show(source_slug: str, char_start: int, char_end: int) -> None:
    """Print the exact span, to verify and tighten boundaries."""
    text = _load(source_slug)
    print(f"--- {source_slug}[{char_start}:{char_end}] "
          f"({char_end - char_start} chars) ---")
    print(text[char_start:char_end])
    print("--- end of span ---")


def entry(query_id: str, query: str, query_type: str,
          source_slug: str, char_start: int, char_end: int,
          grade: int = 2, tier2: bool = False, notes: str = "",
          extra_spans: list[dict] | None = None) -> None:
    """Print a ready-to-append JSON line for eval/queries.jsonl.
    Validates the span against the canonical text before printing."""
    text = _load(source_slug)
    assert 0 <= char_start < char_end <= len(text), "span out of bounds"
    spans = [{"source": source_slug, "char_start": char_start,
              "char_end": char_end, "grade": grade}]
    if extra_spans:
        spans.extend(extra_spans)
    record = {
        "query_id": query_id,
        "query": query,
        "query_type": query_type,   # conceptual | numerical | terminological
                                    # | multi_doc | perspective
        "origin": "manual",
        "tier2": tier2,
        "gold_spans": spans,
        "notes": notes,
    }
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    # CLI shortcut: python label_helper.py <source_slug> <search terms...>
    if len(sys.argv) >= 3:
        find(sys.argv[1], " ".join(sys.argv[2:]))
    else:
        print("Sources:", ", ".join(sources()))
        print("Usage: python label_helper.py <source_slug> <search terms>")