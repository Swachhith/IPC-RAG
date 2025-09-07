#!/usr/bin/env python3
"""
Index parents and children from a manually extracted sections JSONL file.
Input: extracted_sections_final.jsonl
Output: storage/parents.jsonl, storage/children.jsonl and Chroma vector store.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.indexer import persist_corpora, build_vector_store
from src.parser import generate_child_chunks


def _normalize_section_number(value) -> str:
    s = str(value).strip()
    return s


def _extract_chapter_number(chapter_field: str) -> str | None:
    if not chapter_field:
        return None
    m = re.search(r"CHAPTER\s+([IVXLCDM]+)", chapter_field, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _clean_prefix(line: str) -> str:
    # Some lines begin with a non-JSON prefix like [cite_start]
    return line.lstrip().removeprefix("[cite_start]").lstrip()


def load_parents_from_jsonl(path: Path) -> List[Document]:
    parents: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            cleaned = _clean_prefix(raw)
            try:
                rec = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try again if trailing markers exist
                try:
                    cleaned2 = cleaned.split("[cite:")[0].rstrip()
                    rec = json.loads(cleaned2)
                except Exception:
                    continue

            chapter_field = rec.get("chapter") or ""
            chapter_number = _extract_chapter_number(chapter_field)

            sec_num = _normalize_section_number(rec.get("section_number", ""))
            title = str(rec.get("title", "")).strip()
            content = str(rec.get("content", "")).strip()

            # Optionally append illustrations to content (they are part of law text)
            illustrations = rec.get("illustrations") or []
            if illustrations:
                content += "\n\nIllustrations:\n" + "\n".join(str(x) for x in illustrations)

            # Skip amendments in content; they are not core legal text

            page_content = f"Section {sec_num}: {title}\n{content}".strip()
            meta = {
                "section_number": sec_num,
                "section_title": title,
                "chapter_number": chapter_number,
                "chapter_title": chapter_field,
            }
            parents.append(Document(page_content=page_content, metadata=meta))
    return parents


def main():
    parser = argparse.ArgumentParser(description="Index IPC parents/children from manual JSONL")
    parser.add_argument("--file", default="extracted_sections_final.jsonl", help="Path to JSONL file")
    args = parser.parse_args()

    src_path = Path(args.file)
    if not src_path.exists():
        print(f"File not found: {src_path}")
        return

    print(f"Loading parents from: {src_path}")
    parents = load_parents_from_jsonl(src_path)
    print(f"Loaded {len(parents)} parent sections")

    print("Generating child chunks (rule-based)...")
    all_children: List[Document] = []
    for i, p in enumerate(parents, 1):
        children = generate_child_chunks(p, enable_llm=False, llm=None)
        all_children.extend(children)
        if i % 25 == 0:
            print(f"  Processed {i}/{len(parents)} parents; children so far: {len(all_children)}")

    print(f"Total children generated: {len(all_children)}")

    print("Persisting parents and children to storage...")
    persist_corpora(parents, all_children)
    print("Saved storage/parents.jsonl and storage/children.jsonl")

    print("Building vector store (Chroma)...")
    build_vector_store(all_children)
    print("Vector store built and persisted.")

    print("Done.")


if __name__ == "__main__":
    main()


