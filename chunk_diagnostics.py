"""Diagnostics: PDF -> markdown -> section tree. Read-only analysis, writes nothing."""

import sys
from pathlib import Path
from collections import Counter

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

DOCUMENTS_DIR = Path(__file__).parent / "src" / "documents"
# ^ adjust to wherever the PDFs live (ingest.py uses: Path(__file__).parent / "documents")

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"), ("#####", "h5"), ("######", "h6")]

"""Diagnostics: PDF -> markdown -> section tree. Read-only analysis, writes nothing."""

import re
import sys
from pathlib import Path
from collections import Counter

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

DOCUMENTS_DIR = Path(__file__).parent / "src" / "documents"

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"),
           ("#####", "h5"), ("######", "h6")]

CHAPTER_RE = re.compile(r"^(?P<num>I{1,3}|IV|V)\.\s+\S")


def promote_chapter_headers(md_text: str) -> str:
    """Promote plain-text chapter title lines (e.g. 'II. Monetary policy...')
    to h5 headers so they match detected chapter headers like I. and III."""
    out_lines = []
    for line in md_text.split("\n"):
        stripped = line.strip()
        if CHAPTER_RE.match(stripped) and not stripped.startswith("#"):
            out_lines.append(f"##### {stripped}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)



def analyze_pdf(pdf_path: Path) -> None:
    print("=" * 70)
    print(f"DOCUMENT: {pdf_path.name}")
    print("=" * 70)

    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    # --- Preprocessing pipeline (order matters) ---
    md_text = re.sub(r"<!-- Start of picture text -->.*?<!-- End of picture text -->",
                     "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"~~(.+?)~~", r"\1", md_text)
    md_text = re.sub(r"<sup>.*?</sup>", "", md_text)
    promoted = [l.strip() for l in md_text.split("\n")
                if CHAPTER_RE.match(l.strip()) and not l.strip().startswith("#")]
    print(f"Promoted to chapter headers: {promoted}")
    md_text = promote_chapter_headers(md_text)

    # --- 1. Raw markdown output sample ---
    print("\n--- FIRST 1500 CHARACTERS (raw markdown) ---")
    print(md_text[:1500])
    print("--- END OF SAMPLE ---\n")

    # --- 2. Header statistics ---
    lines = md_text.split("\n")
    header_counts = Counter()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            header_counts[f"h{level}"] += 1
    print(f"Headers by level: {dict(sorted(header_counts.items()))}")

    # --- 3. Section splitting ---
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    sections = splitter.split_text(md_text)
    sizes = [len(s.page_content) for s in sections]

    print(f"Number of sections: {len(sections)}")
    if not sections:
        print("!! No sections found — pymupdf4llm detected no headers in this document.")
        return

    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    print(f"Size min/median/max: {sizes_sorted[0]} / {sizes_sorted[n // 2]} / {sizes_sorted[-1]} chars")
    print(f"  < 300 chars (merge candidate):  {sum(1 for s in sizes if s < 300)}")
    print(f"  300-3000 chars (good parent):   {sum(1 for s in sizes if 300 <= s <= 3000)}")
    print(f"  > 3000 chars (needs splitting): {sum(1 for s in sizes if s > 3000)}")

    # --- 4. Section tree with header paths ---
    print("\n--- SECTION TREE (deepest headers + size) ---")
    for s in sections:
        levels = [s.metadata.get(k, "") for k in ["h1", "h2", "h3", "h4", "h5", "h6"] if s.metadata.get(k)]
        tail = " > ".join(levels[-2:]) if levels else "(preamble text before first header)"
        flag = ""
        if len(s.page_content) < 300:
            flag = "  [SMALL]"
        elif len(s.page_content) > 3000:
            flag = "  [LARGE]"
        print(f"  {len(s.page_content):>6} chars | {tail[:100]}{flag}")

    # --- 5. Paragraph boundary integrity ---
    para_count = md_text.count("\n\n")
    print(f"\nParagraph boundaries (\\n\\n) count: {para_count}")
    if para_count < len(sections):
        print("!! Suspiciously few paragraph boundaries — extraction may have produced flat text.")

    # --- 6. Repeated lines (header/footer suspects) ---
    line_counts = Counter(l.strip() for l in lines if 5 < len(l.strip()) < 100)
    repeated = [(l, c) for l, c in line_counts.most_common(10) if c >= 5]
    if repeated:
        print("\n--- REPEATED LINES (header/footer suspects) ---")
        for line, count in repeated:
            print(f"  {count}x | {line[:70]}")
    
    # --- 7. Locate chapter II header in raw markdown ---
    for line in lines:
        if line.strip().lstrip("#").strip().startswith("II."):
            level = len(line.strip()) - len(line.strip().lstrip("#"))
            print(f"Chapter II header found at level h{level}: {line.strip()[:90]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdfs = [Path(sys.argv[1])]
    else:
        pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in: {DOCUMENTS_DIR}")
    for pdf in pdfs:
        analyze_pdf(pdf)
        print("\n")