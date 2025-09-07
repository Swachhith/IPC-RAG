#!/usr/bin/env python3
"""
Query the indexed IPC corpus built from extracted_sections_final.jsonl.
Requires that index_from_manual.py has been run to build storage and the vector store.
"""
import argparse
from typing import List

from langchain_core.documents import Document

from src.indexer import load_corpora
from src.retrieval import build_hybrid_retriever, build_qa_chain, format_sources, parent_expander


def main():
    parser = argparse.ArgumentParser(description="Query IPC RAG (manual sections)")
    parser.add_argument("question", help="Your question")
    parser.add_argument("--k", type=int, default=10, help="Top-k documents")
    args = parser.parse_args()

    parents, children = load_corpora()
    if not parents or not children:
        print("No indexed data found. Run: python index_from_manual.py --file extracted_sections_final.jsonl")
        return

    retriever = build_hybrid_retriever(children, k=args.k)
    retrieved_children: List[Document] = retriever.invoke(args.question)
    print(f"Retrieved {len(retrieved_children)} child docs")

    # Expand to parents to ensure full section text is available
    expander = parent_expander(children_docs=children, parents=parents)
    retrieved_parents: List[Document] = expander.invoke(retrieved_children)
    print(f"Expanded to {len(retrieved_parents)} parent sections")

    # Build QA chain using env LLM with parent content
    chain = build_qa_chain()
    context_parts = [p.page_content for p in retrieved_parents]
    # Fallback: if no parents (edge case), use children content
    if not context_parts:
        context_parts = [c.page_content for c in retrieved_children]
    context = "\n\n".join(context_parts)
    answer = chain.invoke({"question": args.question, "context": context})

    print("\nAnswer:\n" + answer)
    print("\nSources:\n" + format_sources(retrieved_children))


if __name__ == "__main__":
    main()


