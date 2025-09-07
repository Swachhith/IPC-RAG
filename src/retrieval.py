from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import CHROMA_DIR, OPENAI_MODEL, GOOGLE_MODEL
import os

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def load_vector_retriever(k: int = 8):
    # Lazy import embedding consistent with indexer
    from langchain_huggingface import HuggingFaceEmbeddings
    import os

    hf_model = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=hf_model, model_kwargs={"device": "cpu"})
    except Exception:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

    vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    return vs.as_retriever(search_kwargs={"k": k})


def _expand_query_terms(query: str) -> str:
    """Add domain synonyms and IPC section hints to the query to improve recall."""
    q = query.strip().lower()
    additions: list[str] = []
    if ("group" in q or "gang" in q) and "rape" in q:
        additions += ["gang rape", "Section 376D", "376D"]
        if "twelve" in q or "12" in q:
            additions += ["376DB", "under twelve years"]
        if "sixteen" in q or "16" in q:
            additions += ["376DA", "under sixteen years"]
    if "repeat" in q and ("offender" in q or "offenders" in q):
        additions += ["Section 376E", "376E", "repeat offenders"]
    if "rape" in q and not any(x in q for x in ["376", "376a", "376b", "376c", "376d", "376da", "376db", "376e"]):
        additions += ["Section 376", "376"]
    expanded = query
    if additions:
        expanded = f"{query} | " + " ".join(additions)
    return expanded


def build_hybrid_retriever(children: List[Document], k: int = 8):
    """Hybrid retrieval with query expansion and heuristic reranking.

    - Dense: Chroma vectors
    - Sparse: BM25
    - Query expansion: domain synonyms/section hints
    - Reranking: boosts for exact phrases, section-number hits, and headings/aliases
    """
    inner_k = max(k * 3, 24)
    vector_ret = load_vector_retriever(k=inner_k)
    bm25_ret = BM25Retriever.from_documents(children)
    bm25_ret.k = inner_k

    def score_boost(doc: Document, expanded_q_lower: str) -> float:
        text = (doc.page_content or "").lower()
        meta = doc.metadata or {}
        boost = 0.0
        # Exact phrase boosts
        if "gang rape" in text:
            boost += 0.8
        if "section 376d" in text or meta.get("section_number", "").upper() == "376D":
            boost += 1.0
        if "376db" in text or meta.get("section_number", "").upper() == "376DB":
            boost += 0.4
        if "376da" in text or meta.get("section_number", "").upper() == "376DA":
            boost += 0.4
        if "376e" in text or meta.get("section_number", "").upper() == "376E":
            boost += 0.2
        # Keyword overlap boosts
        for kw in ("rape", "woman", "group", "gang"):
            if kw in text and kw in expanded_q_lower:
                boost += 0.1
        # Prefer headings/aliases child types
        ctype = meta.get("child_type")
        if ctype in ("heading", "aliases"):
            boost += 0.15
        return boost

    def combine(query: str) -> List[Document]:
        expanded_query = _expand_query_terms(query)
        expanded_lower = expanded_query.lower()
        vect_docs = vector_ret.invoke(expanded_query)
        bm_docs = bm25_ret.invoke(expanded_query)
        # Rank fusion + heuristic boosts
        seen = {}
        score = {}
        for rank, d in enumerate(vect_docs):
            key = (d.page_content, tuple(sorted(d.metadata.items())))
            seen[key] = d
            score[key] = score.get(key, 0.0) + 1.0 / (1 + rank)
        for rank, d in enumerate(bm_docs):
            key = (d.page_content, tuple(sorted(d.metadata.items())))
            if key not in seen:
                seen[key] = d
            score[key] = score.get(key, 0.0) + 1.0 / (1 + rank)
        # Apply boosts
        for key, d in seen.items():
            score[key] = score.get(key, 0.0) + score_boost(d, expanded_lower)
        ordered = sorted(seen.keys(), key=lambda k_: score[k_], reverse=True)
        return [seen[k_] for k_ in ordered][:k]

    return RunnableLambda(lambda q: combine(q))


def parent_expander(children_docs: List[Document], parents: List[Document]):
    index = {}
    for p in parents:
        key = (p.metadata.get("section_number"), p.metadata.get("chapter_number"))
        index[key] = p

    def expand(docs: List[Document]) -> List[Document]:
        expanded: List[Document] = []
        added = set()
        for d in docs:
            key = (d.metadata.get("section_number"), d.metadata.get("chapter_number"))
            parent = index.get(key)
            if parent is not None:
                sig = (parent.page_content, tuple(sorted(parent.metadata.items())))
                if sig not in added:
                    expanded.append(parent)
                    added.add(sig)
        return expanded

    return RunnableLambda(expand)


def build_env_llm():
    # Attempt to read keys; if missing, try loading .env again
    gkey = os.environ.get("GOOGLE_API_KEY")
    okey = os.environ.get("OPENAI_API_KEY")
    if not (gkey or okey):
        try:
            from dotenv import load_dotenv as _ld
            _ld()
            gkey = os.environ.get("GOOGLE_API_KEY")
            okey = os.environ.get("OPENAI_API_KEY")
        except Exception:
            pass

    if gkey:
        return ChatGoogleGenerativeAI(model=GOOGLE_MODEL, temperature=0)
    if okey:
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    return None


def build_qa_chain(system_prompt: Optional[str] = None, llm: Optional[object] = None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or (
            "You are a legal assistant for the Indian Penal Code. Answer strictly using the provided context.\n"
            "If the answer is not contained in the context, say you do not know.\n"
            "Cite sections exactly as metadata in the Sources section."
        )),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    if llm is None:
        llm = build_env_llm()
    if llm is None:
        raise RuntimeError("No LLM configured. Set GOOGLE_API_KEY or OPENAI_API_KEY in .env.")
    chain = prompt | llm | StrOutputParser()
    return chain


def format_sources(docs: List[Document]) -> str:
    entries = []
    for d in docs:
        sec = d.metadata.get("section_number")
        sec_title = d.metadata.get("section_title")
        ch = d.metadata.get("chapter_number")
        ch_title = d.metadata.get("chapter_title")
        entries.append(f"Section {sec}: {sec_title}; Chapter {ch}: {ch_title}")
    # Deduplicate while preserving order
    seen = set()
    unique_entries = []
    for e in entries:
        if e not in seen:
            unique_entries.append(e)
            seen.add(e)
    return "\n".join(unique_entries)


