from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


CHAPTER_PATTERN = re.compile(r"^\s*Chapter\s+([IVXLCDM]+)\.?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
# Accept headings like (title can be on the same line or the next line):
#  - "304A. Causing death by negligence"
#  - "304-A. Causing death by negligence"
#  - "Section 304A. Causing death by negligence"
#  - "Sec. 302 — Punishment for murder"
#  - "302." followed by title on the next line
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:(?:Section|Sec\.|S\.)\s*)?(?P<num>\d+)\s*-?\s*(?P<letter>[A-Z]?)\s*(?:[\.:—–-]\s*(?P<title>.*))?$",
    re.IGNORECASE | re.MULTILINE,
)

# Additional pattern to catch section numbers in brackets like "8[14." or "1[17"
BRACKETED_SECTION_PATTERN = re.compile(
    r"^\s*(?P<prefix>\d+)\s*\[\s*(?P<num>\d+)\s*\.\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Pattern to catch standalone section numbers that might be missed
STANDALONE_SECTION_PATTERN = re.compile(
    r"^\s*(?P<num>\d+)\s*\.\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Pattern to catch simple section headers like "13. [Omitted]." or "14. Title"
SIMPLE_SECTION_PATTERN = re.compile(
    r"^\s*(?P<num>\d+)\s*\.\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Pattern to catch sections that start with just a number and period
NUMBER_ONLY_SECTION_PATTERN = re.compile(
    r"^\s*(?P<num>\d+)\s*\.\s*$",
    re.MULTILINE,
)

# Pattern to catch sections with various separators
FLEXIBLE_SECTION_PATTERN = re.compile(
    r"^\s*(?P<num>\d+)\s*[\.:—–-]\s*(?P<title>.*)$",
    re.MULTILINE,
)

# Pattern to catch footnote-style section references like "8[14." where the actual section number is in brackets
FOOTNOTE_SECTION_PATTERN = re.compile(
    r"^\s*(?P<prefix>\d+)\s*\[\s*(?P<num>\d+)\s*\.\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class SectionContext:
    chapter_number: Optional[str]
    chapter_title: Optional[str]
    section_number: str
    section_title: str
    section_text: str


def _load_pdf_text(pdf_path: Path) -> str:
    loader = PyMuPDFLoader(str(pdf_path))
    pages = loader.load()
    full_text = "\n\n".join(p.page_content for p in pages)
    return full_text


def _remove_arrangement_of_sections(text: str) -> str:
    """Remove the 'Arrangement of Sections' table of contents block if present."""
    pattern = re.compile(
        r"\bARRANGEMENT\s+OF\s+SECTIONS\b[\s\S]*?(?=^\s*Chapter\s+[IVXLCDM]+\b|^\s*CHAPTER\s+[IVXLCDM]+\b)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return re.sub(pattern, "", text)


def _remove_standalone_headers(text: str) -> str:
    """Drop known page header lines like 'SECTIONS' that pollute parsing."""
    headers = {"SECTIONS", "SECTION", "ARRANGEMENT OF SECTIONS"}
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if s.upper() in headers:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


def _remove_footnotes_and_annotations(text: str) -> str:
    """Remove footnote-like content that appears at the bottom of pages."""
    # Remove lines that are clearly footnotes (start with numbers, contain amendment info)
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        s = ln.strip()
        # Skip footnote patterns
        if re.match(r"^\d+\.\s*(The\s+words|Inserted|Ins\.|Substituted|Omitted|Amended|Repealed|Extended|Application)", s, re.IGNORECASE):
            continue
        # Skip lines that are just amendment references
        if re.match(r"^\[.*\]\s*$", s):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


def _clean_text(text: str) -> str:
    text = _remove_arrangement_of_sections(text)
    text = _remove_standalone_headers(text)
    text = _remove_footnotes_and_annotations(text)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _iter_chapter_boundaries(text: str) -> List[tuple[int, dict]]:
    boundaries: List[tuple[int, dict]] = []
    for m in CHAPTER_PATTERN.finditer(text):
        chap_num = m.group(1).strip() if m.group(1) else None
        chap_title = m.group(2).strip() if m.group(2) else None
        boundaries.append((m.start(), {"chapter_number": chap_num, "chapter_title": chap_title}))
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def _extract_section_spans(text: str) -> list[dict]:
    # Find all section matches using all patterns
    matches = list(SECTION_HEADING_PATTERN.finditer(text))
    
    # Also find bracketed section numbers like "8[14." or "1[17"
    bracket_matches = list(BRACKETED_SECTION_PATTERN.finditer(text))
    
    # Find standalone section numbers like "14. Title"
    standalone_matches = list(STANDALONE_SECTION_PATTERN.finditer(text))
    
    # Find simple section headers like "13. [Omitted]." or "14. Title"
    simple_matches = list(SIMPLE_SECTION_PATTERN.finditer(text))
    
    # Find sections that start with just a number and period
    number_only_matches = list(NUMBER_ONLY_SECTION_PATTERN.finditer(text))
    
    # Find sections with various separators
    flexible_matches = list(FLEXIBLE_SECTION_PATTERN.finditer(text))
    
    # Find footnote-style section references like "8[14." where the actual section number is in brackets
    footnote_matches = list(FOOTNOTE_SECTION_PATTERN.finditer(text))
    
    # Combine and sort all matches by position
    all_matches = matches + bracket_matches + standalone_matches + simple_matches + number_only_matches + flexible_matches + footnote_matches
    all_matches.sort(key=lambda x: x.start())
    
    # Deduplicate matches by position (keep the first match at each position)
    unique_matches = []
    seen_positions = set()
    
    for m in all_matches:
        # Use a small range around the position to avoid duplicates
        pos_range = (max(0, m.start() - 5), m.end() + 5)
        if not any(pos_range[0] <= seen_pos <= pos_range[1] for seen_pos in seen_positions):
            unique_matches.append(m)
            seen_positions.add(m.start())
    
    # Further deduplicate by section number (keep the most complete match)
    section_map = {}
    for m in unique_matches:
        if m.groupdict().get('prefix'):  # This is a bracketed or footnote match
            base = m.group("num").strip()
        elif m.groupdict().get('num') and not m.groupdict().get('letter'):  # This is a standalone or simple match
            base = m.group("num").strip()
        else:  # This is a regular section match
            base = m.group("num").strip()
        
        # Keep the match with the most content (longest title + body)
        if base not in section_map or len(m.group(0)) > len(section_map[base].group(0)):
            section_map[base] = m
    
    # Use the deduplicated matches
    unique_matches = list(section_map.values())
    unique_matches.sort(key=lambda x: x.start())
    
    spans: list[dict] = []
    for idx, m in enumerate(unique_matches):
        # Handle different match types
        if m.groupdict().get('prefix'):  # This is a bracketed or footnote match
            base = m.group("num").strip()
            letter = ""
            same_line_title = (m.group("title") or "").strip()
        elif m.groupdict().get('num') and not m.groupdict().get('letter'):  # This is a standalone, simple, or flexible match
            base = m.group("num").strip()
            letter = ""
            same_line_title = (m.group("title") or "").strip()
        else:  # This is a regular section match
            base = m.group("num").strip()
            letter = (m.group("letter") or "").strip().upper()
            same_line_title = (m.group("title") or "").strip()

        start = m.end()
        end = unique_matches[idx + 1].start() if idx + 1 < len(unique_matches) else len(text)
        raw_span = text[start:end]

        # Determine title: use same-line title if present; otherwise, peek first non-empty line
        title = same_line_title
        if not title:
            for ln in raw_span.splitlines():
                s = ln.strip()
                if s:
                    title = s
                    break

        body = raw_span.strip()
        
        # For sections with no body content, use the title as the body
        if not body and title:
            body = title
        
        # For sections that start with just a number and period, look for content on next lines
        if not body and not title:
            # This might be a section like "9." followed by content on next lines
            for ln in raw_span.splitlines():
                s = ln.strip()
                if s and not s.isdigit() and not s.endswith('.'):
                    title = s
                    body = s
                    break

        # Skip if this looks like a TOC entry or footnote
        if not title or title.upper() in {"SECTIONS", "ARRANGEMENT OF SECTIONS"}:
            continue
        
        # Skip amendment footnotes and other non-legal content
        if re.match(r"^(The\s+words|Inserted|Ins\.|Substituted|Omitted|Amended|Repealed|Subs\.|Act|w\.e\.f\.|Vide)", title, flags=re.IGNORECASE):
            continue
        
        # Skip if title is too short (likely not a real section)
        if len(title) < 3:
            continue
        
        # Skip if body is too short or contains mostly non-legal content
        if len(body) < 10:  # Reduced from 20 to catch very short definition sections
            continue
            
        # Check for legal content indicators - be more lenient for definition sections
        legal_indicators = ["shall", "punishable", "imprisonment", "fine", "death", "life", "years", "months", "denotes", "means", "includes", "word", "words", "person", "government", "servant", "property", "offence", "act", "law", "code", "chapter", "section"]
        has_legal_content = any(indicator in body.lower() for indicator in legal_indicators)
        if not has_legal_content and len(body) < 30:  # Reduced from 50 to catch short definition sections
            continue
        
        spans.append({"base": base, "letter": letter, "title": title, "body": body})
    return spans


def parse_ipc_sections(pdf_path: Path) -> List[Document]:
    """
    Parse the IPC PDF into parent Documents, each representing a full Section.

    Parents metadata include: section_number, section_title, chapter_number, chapter_title.
    """
    text = _clean_text(_load_pdf_text(pdf_path))

    chapter_bounds = _iter_chapter_boundaries(text)
    parents: List[Document] = []

    # If no chapters found, treat the entire document as one chapter span
    def build_sections(span_text: str, chap_meta: dict):
        spans = _extract_section_spans(span_text)
        
        # Sort spans by their position in the text to maintain order
        spans.sort(key=lambda x: span_text.find(x["body"]))
        
        for sp in spans:
            # Compose one parent per section (e.g., 153, 153A, 153B, 153AA)
            full_num = f"{sp['base']}{sp['letter']}" if sp["letter"] else sp["base"]
            title = sp["title"].strip()
            
            # Additional filtering for edge cases
            if not title or len(title) < 3:
                continue
                
            body = sp["body"].strip()
            
            # Skip if body is too short or contains mostly non-legal content
            if len(body) < 20:  # Reduced from 50 to catch very short definition sections
                continue
                
            # Check for legal content indicators - be more lenient for definition sections
            legal_indicators = ["shall", "punishable", "imprisonment", "fine", "death", "life", "years", "months", "denotes", "means", "includes", "word", "words", "person", "government", "servant"]
            has_legal_content = any(indicator in body.lower() for indicator in legal_indicators)
            if not has_legal_content and len(body) < 50:  # Reduced from 100 to catch short definition sections
                continue

            content = f"Section {full_num}: {title}\n{body}".strip()
            meta = {
                "section_number": full_num,
                "section_title": title,
                "chapter_number": chap_meta.get("chapter_number"),
                "chapter_title": chap_meta.get("chapter_title"),
            }
            parents.append(Document(page_content=content, metadata=meta))
            
    if not chapter_bounds:
        # No explicit chapter markers; process entire text
        build_sections(text, {"chapter_number": None, "chapter_title": None})
    else:
        # Iterate chapter spans
        for i, (start_idx, chap_meta) in enumerate(chapter_bounds):
            end_idx = chapter_bounds[i + 1][0] if i + 1 < len(chapter_bounds) else len(text)
            span_text = text[start_idx:end_idx]
            build_sections(span_text, chap_meta)

    # Final deduplication: ensure each section number appears only once
    seen_sections = {}
    unique_parents = []
    
    for parent in parents:
        sec_num = parent.metadata.get("section_number")
        if sec_num not in seen_sections:
            seen_sections[sec_num] = parent
            unique_parents.append(parent)
        else:
            # Keep the one with more content
            existing = seen_sections[sec_num]
            if len(parent.page_content) > len(existing.page_content):
                seen_sections[sec_num] = parent
                # Replace the existing one
                unique_parents[unique_parents.index(existing)] = parent

    return unique_parents


def generate_child_chunks(
    parent: Document,
    enable_llm: bool = False,
    llm: Optional[object] = None,
) -> List[Document]:
    """
    Produce child Documents for a given parent Section:
    - One-sentence summary
    - Clause/exception/explanation-based chunks
    - Optional LLM-generated QA pair
    """
    children: List[Document] = []
    meta = {
        "section_number": parent.metadata.get("section_number"),
        "section_title": parent.metadata.get("section_title"),
        "chapter_number": parent.metadata.get("chapter_number"),
        "chapter_title": parent.metadata.get("chapter_title"),
    }

    # Summary child
    summary_text = _summarize_text(parent.page_content, enable_llm=enable_llm, llm=llm)
    children.append(Document(page_content=f"Summary: {summary_text}", metadata={**meta, "child_type": "summary"}))

    # Heading child to improve keyword/BM25 matching
    sec_num = meta.get("section_number") or ""
    sec_title = meta.get("section_title") or ""
    heading = f"Section {sec_num}: {sec_title}".strip()
    if sec_num:
        children.append(Document(page_content=heading, metadata={**meta, "child_type": "heading"}))

    # Alias child for section-number searches (e.g., "IPC 304A", "304-A")
    if sec_num:
        norm = _normalize_section_number(sec_num)
        aliases = _aliases_for_section(norm)
        children.append(Document(page_content=" ".join(sorted(aliases)), metadata={**meta, "child_type": "aliases", "section_number_norm": norm}))

    # Children per sub-section if available; else split entire parent
    sub_sections = parent.metadata.get("sub_sections") or []
    if sub_sections:
        for sub in sub_sections:
            sub_num = str(sub.get("number"))
            sub_title = str(sub.get("title", ""))
            sub_text = str(sub.get("text", ""))
            if not sub_text.strip():
                continue
            sub_meta = {**meta, "child_type": "sub_section", "sub_section_number": sub_num, "sub_section_title": sub_title}
            # Summary per sub-section
            sub_summary = _summarize_text(sub_text, enable_llm=enable_llm, llm=llm)
            children.append(Document(page_content=f"Summary ({sub_num}): {sub_summary}", metadata=sub_meta))
            # Clause chunks
            for chunk in _split_into_clauses(sub_text):
                children.append(Document(page_content=chunk, metadata=sub_meta))
            # Heading and aliases for sub-section
            heading = f"Section {sub_num}: {sub_title}".strip()
            children.append(Document(page_content=heading, metadata={**sub_meta, "child_type": "heading"}))
            aliases = _aliases_for_section(sub_num)
            children.append(Document(page_content=" ".join(sorted(aliases)), metadata={**sub_meta, "child_type": "aliases", "section_number_norm": _normalize_section_number(sub_num)}))
    else:
        for chunk in _split_into_clauses(parent.page_content):
            children.append(Document(page_content=chunk, metadata={**meta, "child_type": "clause"}))

    # Optional QA child via LLM
    if enable_llm and llm is not None:
        qa = _make_qa(parent.page_content, llm)
        if qa:
            children.append(Document(page_content=qa, metadata={**meta, "child_type": "qa"}))

    return children


def _summarize_text(text: str, enable_llm: bool, llm: Optional[object]) -> str:
    first_sentence = _first_sentence(text)
    if not enable_llm or llm is None:
        return first_sentence or text[:240].strip()
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the legal section in one concise sentence."),
            ("human", "{text}"),
        ])
        chain = prompt | llm | (lambda x: x.content if hasattr(x, "content") else str(x))
        return chain.invoke({"text": text})
    except Exception:
        return first_sentence or text[:240].strip()


def _first_sentence(text: str) -> str:
    m = re.search(r"(.{10,}?[\.!?])\s", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip().split("\n")[0][:240]


def _split_into_clauses(text: str) -> List[str]:
    """Deterministic, rule-based chunking for legal sections.

    Strategy:
    - Detect clause boundaries by keywords and enumeration markers.
    - Merge adjacent short lines; keep chunks near a target size.
    - Add small overlap between chunks to preserve context.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    clause_units: List[str] = []
    buffer: List[str] = []

    keyword_starts = (
        "Explanation", "Exception", "Illustration",
        "Provided that", "Provided further",
    )

    enum_pattern = re.compile(r"^\s*(\(?[a-zA-Z0-9ivx]+\)|[a-zA-Z0-9ivx]+[\)\.:-])\s+")

    def flush():
        if buffer:
            combined = " ".join(s.strip() for s in buffer if s.strip()).strip()
            if combined:
                clause_units.append(combined)
            buffer.clear()

    for raw in lines:
        ln = raw.strip()
        if not ln:
            flush()
            continue
        if ln.startswith(keyword_starts) or enum_pattern.match(ln):
            flush()
            clause_units.append(ln)
        else:
            buffer.append(ln)
    flush()

    # Now pack clause_units into chunks with soft limits and overlap
    max_chunk = 600  # characters
    min_chunk = 200
    overlap = 80  # characters of overlap between consecutive chunks

    chunks: List[str] = []
    current = ""
    for unit in clause_units:
        if not current:
            current = unit
            continue
        if len(current) + 1 + len(unit) <= max_chunk:
            current = f"{current} {unit}"
        else:
            # emit current
            chunks.append(current)
            # start next with overlap tail of current
            tail = current[-overlap:] if len(current) > overlap else current
            current = f"{tail} {unit}" if tail else unit

    if current:
        chunks.append(current)

    # Ensure minimum size by merging tiny trailing chunks
    if len(chunks) >= 2 and len(chunks[-1]) < min_chunk:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    if not chunks:
        chunks = _split_by_punctuation(text, target_size=400)
    return chunks


def _split_by_punctuation(text: str, target_size: int = 300) -> List[str]:
    parts = re.split(r"(?<=[\.!?])\s+", text)
    chunks: List[str] = []
    cur = ""
    for s in parts:
        if len(cur) + 1 + len(s) <= target_size:
            cur = f"{cur} {s}".strip()
    else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def _normalize_section_number(sec_num: str) -> str:
    return sec_num.replace(" ", "").replace("-", "").upper()


def _aliases_for_section(sec_num: str) -> set[str]:
    norm = _normalize_section_number(sec_num)
    hyphen = f"{norm[:-1]}-{norm[-1]}" if norm[-1:].isalpha() else norm
    spaced = f"{norm[:-1]} {norm[-1]}" if norm[-1:].isalpha() else norm
    return {
        f"IPC {norm}", f"IPC {hyphen}", f"IPC {spaced}",
        f"Section {norm}", f"Section {hyphen}", f"Section {spaced}",
        f"Sec {norm}", f"Sec {hyphen}", f"Sec {spaced}",
        f"S. {norm}", f"S. {hyphen}", f"S. {spaced}",
        norm, hyphen, spaced,
    }


def _make_qa(text: str, llm: object) -> Optional[str]:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a single focused Q&A pair about the legal section. Keep it concise."),
            ("human", "Section text:\n{text}\n\nReturn as 'Q: ...\nA: ...'")
        ])
        chain = prompt | llm | (lambda x: x.content if hasattr(x, "content") else str(x))
        out = chain.invoke({"text": text})
        if isinstance(out, str) and out.strip():
            return out.strip()
    except Exception:
        return None
    return None



