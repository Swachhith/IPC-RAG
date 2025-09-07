from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from .config import CHROMA_DIR, CHILDREN_JSONL, PARENTS_JSONL, HF_EMBEDDING_MODEL


def _save_jsonl(path: Path, docs: Iterable[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {"page_content": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["page_content"], metadata=rec.get("metadata", {})))
    return docs


def build_vector_store(children: List[Document]) -> Chroma:
    # Prefer local HuggingFace embeddings; fallback to OpenAI embeddings if unavailable
    try:
        embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    except Exception:
        embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_documents(
        documents=children,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.persist()
    return vector_store


def build_bm25(children: List[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(children)
    retriever.k = 8
    return retriever


def persist_corpora(parents: List[Document], children: List[Document]) -> None:
    _save_jsonl(PARENTS_JSONL, parents)
    _save_jsonl(CHILDREN_JSONL, children)


def load_corpora() -> Tuple[List[Document], List[Document]]:
    parents = _load_jsonl(PARENTS_JSONL) if PARENTS_JSONL.exists() else []
    children = _load_jsonl(CHILDREN_JSONL) if CHILDREN_JSONL.exists() else []
    return parents, children


