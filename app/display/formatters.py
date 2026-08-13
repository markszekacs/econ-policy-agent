import re


def confidence_label(score: float) -> str:
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Moderate"
    return "Low"


def strip_prior_knowledge_tags(text: str) -> str:
    """Remove [PRIOR KNOWLEDGE — ...] markers from agent response text."""
    return re.sub(r"\[PRIOR KNOWLEDGE[^\]]*\]", "", text).strip()


def grounding_pct(prior_knowledge_rate: float | None) -> float:
    if prior_knowledge_rate is None:
        return 1.0
    return round(1.0 - prior_knowledge_rate, 3)


def strip_filler_opener(text: str) -> str:
    """Remove filler label at the very start of text, e.g. 'Economic Analysis: ' or 'Key Insight — '."""
    return re.sub(r"^[A-Z][A-Za-z/ ]{2,50}[:—]\s+", "", text).strip()


def strip_bullets(text: str) -> str:
    """Convert bullet/list/header/markdown formatting to clean plain prose."""
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
    # Remove bold/italic section labels acting as headers ("**Key point:**")
    text = re.sub(r"\*{1,2}[^*\n]{1,60}:\*{1,2}\s*", "", text)
    # Strip remaining inline bold/italic asterisks (**word** → word, *word* → word)
    text = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)
    # Replace em-dashes with commas for clean prose
    text = re.sub(r"\s*—\s*", ", ", text)
    # Strip leading bullet/list markers from each line
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Ensure each non-empty line ends with sentence punctuation before joining
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    normalized = []
    for line in lines:
        if line and line[-1] not in ".!?":
            line += "."
        normalized.append(line)
    text = " ".join(normalized)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def strip_chunk_refs(text: str) -> str:
    """Remove all chunk/document citation artefacts the model may hallucinate."""
    # word_chunk_0042, imf-weo_chunk_047, any_prefix_chunk_N
    text = re.sub(r"\b[\w.-]+_chunk_\d+\b", "", text, flags=re.IGNORECASE)
    # standalone chunk_047 or chunk 047 or chunk #47
    text = re.sub(r"\bchunk[_ #]*\d+\b", "", text, flags=re.IGNORECASE)
    # bare "chunk_id:" or "chunk_id :" labels left after ID removal
    text = re.sub(r"\bchunk_id\s*:\s*", "", text, flags=re.IGNORECASE)
    # [chunk_id: xxx] or (chunk_id: xxx)
    text = re.sub(r"[\[(]chunk_id\s*:[^\])\n]+[\])]", "", text, flags=re.IGNORECASE)
    # [DOC | ...] or [DOC: ...] blocks
    text = re.sub(r"\[DOC[^\]]*\]", "", text, flags=re.IGNORECASE)
    # (Source: ...) or [Source: ...] inline citations
    text = re.sub(r"[\[(]Source[^\])\n]{0,120}[\])]", "", text, flags=re.IGNORECASE)
    # (Doc N) or [Doc N] or (Document N)
    text = re.sub(r"[\[(]Docs?\.?\s*\d+[^\])\n]{0,60}[\])]", "", text, flags=re.IGNORECASE)
    # Parenthetical page references: (p. 12), (pp. 3-5), (page 12)
    text = re.sub(r"\(\s*pp?\.\s*\d[\d\s,–-]*\)", "", text)
    # Leftover empty brackets/parens
    text = re.sub(r"[\[(]\s*[\])]", "", text)
    text = re.sub(r"\s*\(\s*\)", "", text)
    return re.sub(r" {2,}", " ", text).strip()


def truncate_to_sentences(text: str, max_sentences: int = 4) -> str:
    """Hard-limit to max_sentences sentences.

    Splits on sentence-ending punctuation followed by a capital letter,
    which avoids false splits on 'e.g.', '3.5%', 'U.S.', etc.
    """
    # Split at ". ", "! ", "? " where next token starts with uppercase
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= max_sentences:
        return text.strip()
    truncated = " ".join(parts[:max_sentences])
    # Ensure ends with a period
    if not truncated[-1] in ".!?":
        truncated += "."
    return truncated


def extract_core_analysis(response: str) -> str:
    """Return only the Step 3 analysis section, stripping document summaries,
    chunk citations, and confidence statements that belong to internal reasoning.

    If the response has no step markers, returns it unchanged.
    """
    # Strip bold markdown around "Step N" labels for uniform matching
    text = re.sub(r"\*{1,2}(Step\s+\d+[^:\n]*:?)\*{1,2}", r"\1", response)

    if not re.search(r"Step\s+3\b", text, re.IGNORECASE):
        return response.strip()

    match = re.search(
        r"Step\s+3[^:\n]*:?\s*\n?(.*?)(?=\n\s*Step\s+[4-9]|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return response.strip()
