"""PDF -> LanceDB ingestion, structured for chunking-strategy ablation.

PHASE 1 -- Canonicalization (shared by all strategies, cached to disk):
  extract per-page markdown -> remove figure-scrape noise -> fix artifacts ->
  drop repeated header/footer lines -> promote plain-text chapter titles ->
  drop excluded sections (TOC, references, endnotes, table-dense, boilerplate).
  Output: one canonical markdown text per document + a char-offset -> page map.

PHASE 2 -- Chunking strategies (all operate on the same canonical text):
  section      : section-aware parents, paragraph-first children, enrichment prefix
  section_bare : same boundaries, no enrichment prefix (isolates prefix effect)
  recursive    : RecursiveCharacterTextSplitter parents/children (separator-aware)
  fixed        : hard character windows with overlap (naive baseline)

Every chunk stores char_start/char_end into the canonical text, so one
evaluation set labelled at the (query, document, char-span) level can score
all strategies with the same ground truth.

Document-level metadata comes from documents/manifest.json (optional):
  { "<filename>.pdf": {"title": "...", "institution": "IMF", "year": 2024}, ... }
Missing entries fall back to the filename stem (with a year guess from the name).

Usage:
  python ingest.py              # default: section strategy -> child_chunks/parent_chunks
  python ingest.py section recursive fixed
  python ingest.py all
"""

import datetime
import json
import re
import sys
from bisect import bisect_right
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import lancedb
import pyarrow as pa
import pymupdf4llm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

INGEST_VERSION = "2.1"

DOCUMENTS_DIR = Path(__file__).parent / "documents"
LANCEDB_DIR = Path(__file__).parent.parent / ".lancedb"
CACHE_DIR = Path(__file__).parent / ".md_cache"
MANIFEST_FILE = DOCUMENTS_DIR / "manifest.json"
RUNS_FILE = LANCEDB_DIR / "ingest_runs.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Size thresholds (chars), validated on BIS AER 2024 / IMF WEO Oct 2024 diagnostics
PARENT_MIN = 300        # below this: merge into an adjacent section
PARENT_TARGET = 1800    # greedy packing target when splitting oversized sections
PARENT_MAX = 3000       # above this: paragraph-greedy split
CHILD_TARGET = 400      # child chunk target size

# Baseline strategy parameters (mirror the original fixed-window pipeline)
BASELINE_PARENT_SIZE = 1200
BASELINE_CHILD_SIZE = 400
BASELINE_CHILD_OVERLAP = 50

REPEATED_LINE_THRESHOLD = 5     # normalized line repeated this often = header/footer noise
TABLE_DENSITY_MAX = 0.40        # sections with a higher share of table rows are excluded
CHUNK_TABLE_DENSITY = 0.50      # chunks above this share of table rows are content_type=table

EXCLUDED_TITLES = {
    "contents", "table of contents", "references", "endnotes",
    "bibliography", "acknowledgements",
}
BOILERPLATE_RE = re.compile(
    r"ISSN|ISBN|All rights reserved|Cataloging-in-Publication", re.IGNORECASE
)
BOX_TITLE_RE = re.compile(r"^box\b|\bkey takeaways\b", re.IGNORECASE)

# Best-effort chapter title pattern; length guard avoids short list items
CHAPTER_RE = re.compile(r"^(I{1,3}|IV|V|VI{1,3}|IX|X)\.\s+\S")
CHAPTER_MIN_TITLE_LEN = 25

PAGE_MARKER = "<!-- p:{page} -->"
PAGE_MARKER_RE = re.compile(r"<!--\s*p:(\d+)\s*-->")
PICTURE_TEXT_RE = re.compile(
    r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
    re.DOTALL,
)
HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
YEAR_IN_NAME_RE = re.compile(r"(20\d{2})")

CHILD_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("parent_chunk_id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("source", pa.string()),
    pa.field("doc_title", pa.string()),
    pa.field("institution", pa.string()),
    pa.field("doc_year", pa.int16()),          # 0 = unknown
    pa.field("content_type", pa.string()),     # prose | table | box
    pa.field("section_path", pa.string()),
    pa.field("chapter", pa.string()),
    pa.field("page", pa.int32()),              # page at char_start; name kept for compat
    pa.field("page_end", pa.int32()),
    pa.field("char_start", pa.int32()),        # offsets into canonical text (eval ground truth)
    pa.field("char_end", pa.int32()),
    pa.field("chunk_size", pa.int32()),
    pa.field("embed_tokens", pa.int16()),      # token count of the embedded (enriched) text
    pa.field("embed_truncated", pa.bool_()),   # embedded text exceeded the model's max length
    pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
])

PARENT_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("source", pa.string()),
    pa.field("doc_title", pa.string()),
    pa.field("institution", pa.string()),
    pa.field("doc_year", pa.int16()),
    pa.field("content_type", pa.string()),
    pa.field("section_path", pa.string()),
    pa.field("chapter", pa.string()),
    pa.field("page", pa.int32()),
    pa.field("page_end", pa.int32()),
    pa.field("char_start", pa.int32()),
    pa.field("char_end", pa.int32()),
    pa.field("chunk_size", pa.int32()),
])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _clean_title(raw: str) -> str:
    t = re.sub(r"[*_]+", "", raw)
    return re.sub(r"\s{2,}", " ", t).strip()


def _normalize_line(line: str) -> str:
    t = line.strip().lstrip("#").strip()
    t = re.sub(r"[*_]+", "", t)
    return re.sub(r"\s+", " ", t).lower()


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# ---------------------------------------------------------------------------
# Document manifest (title / institution / year for citations and filtering)
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"!! manifest.json is invalid JSON ({exc}); using filename fallbacks")
    return {}


def doc_metadata(pdf_path: Path, manifest: dict) -> dict:
    entry = manifest.get(pdf_path.name, {})
    year = entry.get("year")
    if year is None:
        m = YEAR_IN_NAME_RE.search(pdf_path.stem)
        year = int(m.group(1)) if m else 0
    meta = {
        "doc_title": entry.get("title", pdf_path.stem),
        "institution": entry.get("institution", ""),
        "doc_year": int(year),
    }
    if not entry:
        print(f"  (no manifest entry for {pdf_path.name}; "
              f"using title='{meta['doc_title']}', year={meta['doc_year']})")
    return meta


# ---------------------------------------------------------------------------
# Content-type classification (per chunk)
# ---------------------------------------------------------------------------

def classify_content(content: str, section_path: str) -> str:
    """prose | table | box. Table wins over box; both win over prose."""
    lines = [l for l in content.splitlines() if l.strip()]
    if lines:
        table_share = sum(1 for l in lines if l.strip().startswith("|")) / len(lines)
        if table_share > CHUNK_TABLE_DENSITY:
            return "table"
    if BOX_TITLE_RE.search(section_path):
        return "box"
    return "prose"


# ---------------------------------------------------------------------------
# PHASE 1: canonicalization (shared preprocessing, cached)
# ---------------------------------------------------------------------------

def _extract_marked_text(pdf_path: Path) -> str:
    """Per-page markdown with page markers; figure-scrape blocks removed per page."""
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    parts: list[str] = []
    for i, chunk in enumerate(page_chunks):
        page_num = chunk.get("metadata", {}).get("page", i + 1)
        text = chunk.get("text", "")
        text = PICTURE_TEXT_RE.sub("", text)
        text = re.sub(r"<!--\s*(Start|End) of picture text\s*-->", "", text)
        parts.append(PAGE_MARKER.format(page=page_num) + "\n" + text)
    return "\n".join(parts)


def _clean_artifacts(md_text: str) -> str:
    md_text = re.sub(r"~~\s*(.+?)\s*~~", r"\1", md_text)
    md_text = re.sub(r"<sup>.*?</sup>", "", md_text)
    return re.sub(r"[ \t]{2,}", " ", md_text)


def _remove_repeated_lines(md_text: str) -> str:
    """Drop running headers/footers (normalized lines repeated >= threshold).
    Table rows and page markers are never dropped here."""
    lines = md_text.split("\n")
    counts = Counter()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("|") or PAGE_MARKER_RE.match(s):
            continue
        norm = _normalize_line(s)
        if 5 < len(norm) < 100:
            counts[norm] += 1
    noisy = {n for n, c in counts.items() if c >= REPEATED_LINE_THRESHOLD}

    kept = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("|") and not PAGE_MARKER_RE.match(s):
            if _normalize_line(s) in noisy:
                continue
        kept.append(line)
    return "\n".join(kept)


def _promote_chapter_headers(md_text: str) -> str:
    """Promote undetected plain-text chapter titles to h5 headers (best effort)."""
    out = []
    for line in md_text.split("\n"):
        s = line.strip()
        if (not s.startswith("#") and len(s) >= CHAPTER_MIN_TITLE_LEN
                and CHAPTER_RE.match(s)):
            out.append(f"##### {s}")
        else:
            out.append(line)
    return "\n".join(out)


def _strip_markers(md_text: str) -> tuple[str, list[tuple[int, int]]]:
    """Remove page markers; return (text, page_map) with page_map as
    ascending (char_offset, page_number) pairs over the marker-free text."""
    out_parts: list[str] = []
    page_map: list[tuple[int, int]] = [(0, 1)]
    out_len = 0
    pos = 0
    for m in PAGE_MARKER_RE.finditer(md_text):
        seg = md_text[pos:m.start()]
        out_parts.append(seg)
        out_len += len(seg)
        page_map.append((out_len, int(m.group(1))))
        pos = m.end()
    out_parts.append(md_text[pos:])
    return "".join(out_parts), page_map


def scan_sections(text: str) -> list[dict]:
    """Line-based section scan with exact char offsets.

    A section span starts at its header line and runs to the next header line
    at any level (font-size level detection is unreliable, so any header cuts).
    Chapter membership is forward-filled, best effort.
    """
    sections: list[dict] = []
    header_stack: dict[int, str] = {}
    current_chapter = ""
    span_start = 0
    offset = 0

    def close(end: int) -> None:
        s, e = _trim_span(text, span_start, end)
        if e <= s:
            return
        titles = [header_stack[k] for k in sorted(header_stack)]
        sections.append({
            "char_start": s,
            "char_end": e,
            "headers": titles,
            "path": " > ".join(titles[-2:]) if titles else "",
            "chapter": current_chapter,
        })

    for line in text.split("\n"):
        m = HEADER_LINE_RE.match(line.strip())
        if m:
            close(offset)
            span_start = offset
            level = len(m.group(1))
            title = _clean_title(m.group(2))
            header_stack = {k: v for k, v in header_stack.items() if k < level}
            header_stack[level] = title
            if len(title) >= CHAPTER_MIN_TITLE_LEN and CHAPTER_RE.match(title):
                current_chapter = title
        offset += len(line) + 1
    close(len(text))
    return sections


def _table_density(content: str) -> float:
    lines = [l for l in content.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if l.strip().startswith("|")) / len(lines)


def _is_excluded(section: dict, text: str) -> bool:
    for value in section["headers"]:
        if value.lower() in EXCLUDED_TITLES:
            return True
    content = text[section["char_start"]:section["char_end"]]
    if _table_density(content) > TABLE_DENSITY_MAX:
        return True
    if len(content) < 1200 and BOILERPLATE_RE.search(content):
        return True
    return False


def build_canonical(pdf_path: Path) -> tuple[str, dict]:
    """Full Phase 1: canonical text (exclusions removed) + page-lookup metadata."""
    md_text = _extract_marked_text(pdf_path)
    md_text = _clean_artifacts(md_text)
    md_text = _remove_repeated_lines(md_text)
    md_text = _promote_chapter_headers(md_text)
    full_text, page_map = _strip_markers(md_text)

    sections = scan_sections(full_text)
    kept = [s for s in sections if not _is_excluded(s, full_text)]

    # Concatenate kept spans; record (canonical_start, original_start, length)
    # segments so canonical offsets can be mapped back to pages.
    parts: list[str] = []
    segments: list[tuple[int, int, int]] = []
    canon_len = 0
    for s in kept:
        seg = full_text[s["char_start"]:s["char_end"]]
        segments.append((canon_len, s["char_start"], len(seg)))
        parts.append(seg)
        canon_len += len(seg) + 2  # account for the "\n\n" joiner

    canonical = "\n\n".join(parts)
    meta = {
        "ingest_version": INGEST_VERSION,
        "page_map": page_map,
        "segments": segments,
        "n_sections_raw": len(sections),
        "n_sections_kept": len(kept),
    }
    return canonical, meta


def load_or_build_canonical(pdf_path: Path) -> tuple[str, dict]:
    CACHE_DIR.mkdir(exist_ok=True)
    slug = _slug(pdf_path.stem)
    md_file = CACHE_DIR / f"{slug}.md"
    meta_file = CACHE_DIR / f"{slug}.meta.json"

    if (md_file.exists() and meta_file.exists()
            and md_file.stat().st_mtime >= pdf_path.stat().st_mtime):
        return md_file.read_text(encoding="utf-8"), json.loads(meta_file.read_text())

    canonical, meta = build_canonical(pdf_path)
    md_file.write_text(canonical, encoding="utf-8")
    meta_file.write_text(json.dumps(meta))
    return canonical, meta


def make_page_lookup(meta: dict):
    """Return page_at(canonical_offset) using segment and page maps."""
    segments = [tuple(s) for s in meta["segments"]]
    seg_starts = [s[0] for s in segments]
    page_map = [tuple(p) for p in meta["page_map"]]
    page_offsets = [p[0] for p in page_map]

    def page_at(canon_offset: int) -> int:
        if not segments:
            return 1
        i = max(0, bisect_right(seg_starts, canon_offset) - 1)
        canon_start, orig_start, length = segments[i]
        orig_offset = orig_start + min(max(canon_offset - canon_start, 0), length)
        j = max(0, bisect_right(page_offsets, orig_offset) - 1)
        return page_map[j][1]

    return page_at


# ---------------------------------------------------------------------------
# Span utilities (all chunks are exact substrings of the canonical text)
# ---------------------------------------------------------------------------

def _blocks_in_span(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraph blocks (blank-line separated) within [start, end), trimmed."""
    blocks: list[tuple[int, int]] = []
    pos = start
    for m in re.finditer(r"\n[ \t]*\n+", text[start:end]):
        b = _trim_span(text, pos, start + m.start())
        if b[1] > b[0]:
            blocks.append(b)
        pos = start + m.end()
    b = _trim_span(text, pos, end)
    if b[1] > b[0]:
        blocks.append(b)
    return blocks


def _is_table_span(text: str, start: int, end: int) -> bool:
    first = next((l for l in text[start:end].splitlines() if l.strip()), "")
    return first.strip().startswith("|")


def _window_span(text: str, start: int, end: int, target: int) -> list[tuple[int, int]]:
    """Last-resort character windows, cut at whitespace where possible."""
    spans: list[tuple[int, int]] = []
    cur = start
    while end - cur > target:
        cut = text.rfind(" ", cur + target // 2, cur + target)
        if cut <= cur:
            cut = cur + target
        spans.append((cur, cut))
        cur = cut
    spans.append((cur, end))
    return [_trim_span(text, s, e) for s, e in spans if e > s]


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    bounds = [start] + [start + m.end() for m in
                        SENTENCE_SPLIT_RE.finditer(text[start:end])] + [end]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    return [_trim_span(text, s, e) for s, e in spans if e > s]


def _pack_spans(spans: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    """Greedily union adjacent spans into chunks of roughly target length."""
    packed: list[tuple[int, int]] = []
    cur_start = cur_end = None
    for s, e in spans:
        if cur_start is None:
            cur_start, cur_end = s, e
        elif e - cur_start > target:
            packed.append((cur_start, cur_end))
            cur_start, cur_end = s, e
        else:
            cur_end = e
    if cur_start is not None:
        packed.append((cur_start, cur_end))
    return packed


# ---------------------------------------------------------------------------
# PHASE 2: chunking strategies
# Each returns a list of parent dicts, each with a "children" list of spans.
# All spans index into the canonical text.
# ---------------------------------------------------------------------------

def _child_spans_in_parent(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraph-first children with sentence and character fallbacks.
    Table blocks stay atomic even when oversized."""
    units: list[tuple[int, int]] = []
    for b_start, b_end in _blocks_in_span(text, start, end):
        if b_end - b_start <= CHILD_TARGET or _is_table_span(text, b_start, b_end):
            units.append((b_start, b_end))
            continue
        sent_units: list[tuple[int, int]] = []
        for s, e in _sentence_spans(text, b_start, b_end):
            if e - s > 2 * CHILD_TARGET:
                sent_units.extend(_window_span(text, s, e, CHILD_TARGET))
            else:
                sent_units.append((s, e))
        units.extend(_pack_spans(sent_units, CHILD_TARGET))
    return _pack_spans(units, CHILD_TARGET)


def chunk_section_aware(canonical: str) -> list[dict]:
    sections = scan_sections(canonical)

    # Merge small sections into an adjacent same-chapter neighbour (span union;
    # sections are contiguous in the canonical text by construction).
    merged: list[dict] = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        size = sec["char_end"] - sec["char_start"]
        if size >= PARENT_MIN:
            merged.append(dict(sec))
            i += 1
            continue
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        if nxt is not None and nxt["chapter"] == sec["chapter"]:
            nxt["char_start"] = sec["char_start"]   # small section folds into the next
            i += 1
        elif merged and merged[-1]["chapter"] == sec["chapter"]:
            merged[-1]["char_end"] = sec["char_end"]
            i += 1
        else:
            merged.append(dict(sec))                # standalone small parent is harmless
            i += 1

    # Split oversized sections on paragraph boundaries (tables atomic).
    parents: list[dict] = []
    for sec in merged:
        size = sec["char_end"] - sec["char_start"]
        if size <= PARENT_MAX:
            spans = [(sec["char_start"], sec["char_end"])]
        else:
            blocks: list[tuple[int, int]] = []
            for b_start, b_end in _blocks_in_span(canonical, sec["char_start"], sec["char_end"]):
                if b_end - b_start > PARENT_MAX and not _is_table_span(canonical, b_start, b_end):
                    blocks.extend(_window_span(canonical, b_start, b_end, PARENT_TARGET))
                else:
                    blocks.append((b_start, b_end))
            spans = _pack_spans(blocks, PARENT_TARGET)

        for s, e in spans:
            parents.append({
                "char_start": s,
                "char_end": e,
                "path": sec["path"],
                "chapter": sec["chapter"],
                "children": _child_spans_in_parent(canonical, s, e),
            })
    return parents


def chunk_recursive(canonical: str) -> list[dict]:
    """Separator-aware baseline: recursive splitter, fixed-size windows,
    no section knowledge. Small-to-big structure preserved for fairness."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASELINE_PARENT_SIZE, chunk_overlap=0, add_start_index=True)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASELINE_CHILD_SIZE, chunk_overlap=BASELINE_CHILD_OVERLAP,
        add_start_index=True)

    parents: list[dict] = []
    for p_doc in parent_splitter.create_documents([canonical]):
        p_start = p_doc.metadata["start_index"]
        p_end = p_start + len(p_doc.page_content)
        children = []
        for c_doc in child_splitter.create_documents([p_doc.page_content]):
            c_start = p_start + c_doc.metadata["start_index"]
            children.append((c_start, c_start + len(c_doc.page_content)))
        parents.append({
            "char_start": p_start, "char_end": p_end,
            "path": "", "chapter": "", "children": children,
        })
    return parents


def chunk_fixed(canonical: str) -> list[dict]:
    """Naive baseline: hard character windows with child overlap."""
    parents: list[dict] = []
    n = len(canonical)
    p_start = 0
    while p_start < n:
        p_end = min(p_start + BASELINE_PARENT_SIZE, n)
        children: list[tuple[int, int]] = []
        c_start = p_start
        step = BASELINE_CHILD_SIZE - BASELINE_CHILD_OVERLAP
        while c_start < p_end:
            children.append((c_start, min(c_start + BASELINE_CHILD_SIZE, p_end)))
            c_start += step
        parents.append({
            "char_start": p_start, "char_end": p_end,
            "path": "", "chapter": "", "children": children,
        })
        p_start = p_end
    return parents


# name -> (chunker, use_enrichment_prefix, table_suffix)
STRATEGIES = {
    "section": (chunk_section_aware, True, ""),
    "section_bare": (chunk_section_aware, False, "__section_bare"),
    "recursive": (chunk_recursive, False, "__recursive"),
    "fixed": (chunk_fixed, False, "__fixed"),
}


def enrichment_prefix(chapter: str, path: str) -> str:
    parts = []
    if chapter:
        parts.append(chapter[:80])
    if path and path != chapter:
        parts.append(path)
    return f"[{' > '.join(parts)}]\n" if parts else ""


# ---------------------------------------------------------------------------
# Run log (which pipeline version / parameters built the current tables)
# ---------------------------------------------------------------------------

def record_run(strategy_names: list[str], files: list[str]) -> None:
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "ingest_version": INGEST_VERSION,
        "strategies": strategy_names,
        "embed_model": EMBED_MODEL,
        "params": {
            "parent_min": PARENT_MIN,
            "parent_target": PARENT_TARGET,
            "parent_max": PARENT_MAX,
            "child_target": CHILD_TARGET,
            "baseline_parent_size": BASELINE_PARENT_SIZE,
            "baseline_child_size": BASELINE_CHILD_SIZE,
            "baseline_child_overlap": BASELINE_CHILD_OVERLAP,
            "repeated_line_threshold": REPEATED_LINE_THRESHOLD,
            "table_density_max": TABLE_DENSITY_MAX,
            "chunk_table_density": CHUNK_TABLE_DENSITY,
        },
        "files": files,
    }
    runs = []
    if RUNS_FILE.exists():
        try:
            runs = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runs = []
    runs.append(record)
    RUNS_FILE.parent.mkdir(exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_documents(strategy_names: list[str] | None = None) -> None:
    strategy_names = strategy_names or ["section"]
    for name in strategy_names:
        if name not in STRATEGIES:
            print(f"Unknown strategy: {name}. Available: {', '.join(STRATEGIES)}")
            return

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCUMENTS_DIR}")
        return

    manifest = load_manifest()
    if not manifest:
        print(f"(no {MANIFEST_FILE.name} found; document titles fall back to filenames)")

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)
    max_embed_tokens = int(model.max_seq_length)
    db = lancedb.connect(str(LANCEDB_DIR))

    for name in strategy_names:
        chunker, use_prefix, suffix = STRATEGIES[name]
        child_table_name = f"child_chunks{suffix}"
        parent_table_name = f"parent_chunks{suffix}"

        existing = set(db.table_names())
        for t in (child_table_name, parent_table_name):
            if t in existing:
                db.drop_table(t)
        child_table = db.create_table(child_table_name, schema=CHILD_SCHEMA)
        parent_table = db.create_table(parent_table_name, schema=PARENT_SCHEMA)

        print(f"\n=== Strategy: {name} -> {child_table_name} / {parent_table_name} ===")

        for pdf_path in pdf_files:
            source_slug = _slug(pdf_path.stem)
            print(f"Ingesting {pdf_path.name} ...")
            dmeta = doc_metadata(pdf_path, manifest)

            canonical, meta = load_or_build_canonical(pdf_path)
            page_at = make_page_lookup(meta)
            parents = chunker(canonical)

            parent_rows: list[dict] = []
            child_rows: list[dict] = []
            enriched_texts: list[str] = []

            for p_idx, par in enumerate(parents):
                parent_id = f"{source_slug}_p{p_idx:04d}"
                p_content = canonical[par["char_start"]:par["char_end"]]
                parent_rows.append({
                    "chunk_id": parent_id,
                    "content": p_content,
                    "source": source_slug,
                    **dmeta,
                    "content_type": classify_content(p_content, par["path"]),
                    "section_path": par["path"],
                    "chapter": par["chapter"],
                    "page": page_at(par["char_start"]),
                    "page_end": page_at(par["char_end"]),
                    "char_start": par["char_start"],
                    "char_end": par["char_end"],
                    "chunk_size": len(p_content),
                })

                prefix = enrichment_prefix(par["chapter"], par["path"]) if use_prefix else ""
                for c_idx, (c_start, c_end) in enumerate(par["children"]):
                    c_content = canonical[c_start:c_end]
                    child_rows.append({
                        "chunk_id": f"{parent_id}_c{c_idx:02d}",
                        "parent_chunk_id": parent_id,
                        "content": c_content,
                        "source": source_slug,
                        **dmeta,
                        "content_type": classify_content(c_content, par["path"]),
                        "section_path": par["path"],
                        "chapter": par["chapter"],
                        "page": page_at(c_start),
                        "page_end": page_at(c_end),
                        "char_start": c_start,
                        "char_end": c_end,
                        "chunk_size": len(c_content),
                    })
                    enriched_texts.append(prefix + c_content)

            print(f"  Sections: {meta['n_sections_raw']} raw -> "
                  f"{meta['n_sections_kept']} kept -> "
                  f"{len(parent_rows)} parents, {len(child_rows)} children")

            # Token counts of the texts actually embedded, to flag truncation
            # (MiniLM cuts at max_seq_length tokens; oversized atomic tables
            # are the usual suspects). Counted without truncation.
            token_ids = model.tokenizer(
                enriched_texts, add_special_tokens=True, truncation=False,
            )["input_ids"]
            n_truncated = 0
            for row, ids in zip(child_rows, token_ids):
                row["embed_tokens"] = min(len(ids), 32767)
                row["embed_truncated"] = len(ids) > max_embed_tokens
                n_truncated += row["embed_truncated"]
            if n_truncated:
                print(f"  Truncated embeddings: {n_truncated}/{len(child_rows)} "
                      f"children exceed {max_embed_tokens} tokens")

            print(f"  Embedding {len(enriched_texts)} child chunks...")
            vectors = model.encode(
                enriched_texts,
                batch_size=64,
                normalize_embeddings=True,  # retriever must use cosine / normalized queries
                show_progress_bar=False,
            )
            for row, vec in zip(child_rows, vectors):
                row["vector"] = vec.tolist()

            if parent_rows:
                parent_table.add(parent_rows)
            if child_rows:
                child_table.add(child_rows)

        # Full-text index for the BM25 leg of hybrid search. Native Lance FTS
        # (use_tantivy=False) -- the tantivy path has unreliable Windows wheels.
        print(f"  Building full-text index on {child_table_name}...")
        child_table.create_fts_index("content", use_tantivy=False)

    record_run(strategy_names, [p.name for p in pdf_files])
    print(f"\nIngestion complete. Run recorded in {RUNS_FILE.name} "
          f"(pipeline v{INGEST_VERSION}).")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["all"]:
        args = list(STRATEGIES)
    ingest_documents(args or ["section"])