#!/usr/bin/env python3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import re
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from pathlib import Path
from src.indexer import load_corpora, persist_corpora, build_vector_store
from src.retrieval import build_hybrid_retriever, build_qa_chain, parent_expander, format_sources
from src.config import DATA_DIR, PARENTS_JSONL, CHILDREN_JSONL, CHROMA_DIR
from src.parser import generate_child_chunks
from index_from_manual import load_parents_from_jsonl


def ensure_env():
    # Prefer Streamlit secrets; else .env
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    try:
        load_dotenv()
    except Exception:
        pass


def highlight_terms(text: str, query: str) -> str:
    terms = [t for t in re.split(r"\W+", query) if len(t) > 2]
    highlighted = text
    for t in sorted(set(terms), key=len, reverse=True):
        try:
            highlighted = re.sub(rf"(?i)({re.escape(t)})", r"<mark>\1</mark>", highlighted)
        except re.error:
            continue
    return highlighted


def render_minimal_console():
    st.set_page_config(page_title="IPC RAG", page_icon="🤖", layout="wide")
    st.markdown(
        """
        <style>
        html, body, [class^="css"]  {
            background-color: #0b0f14;
            color: #e6edf3;
        }
        .header { display:flex; flex-direction:column; gap:6px; margin-bottom: 8px; }
        .brand { font-size: 22px; font-weight: 800; letter-spacing: .2px; }
        .subtle { color:#9fb0c0; font-size: 13px; }
        .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border:1px solid #1f2a35; border-radius:999px; background:#0f141a; color:#9fb0c0; font-size:12px; }
        .stTextInput>div>div>input {
            background: #0f141a; color: #e6edf3; border: 1px solid #1f2a35; border-radius: 8px;
        }
        .stButton>button {
            background: linear-gradient(90deg,#0ea5ea,#0bd1d1);
            color: #0b0f14; border: 0; border-radius: 8px; font-weight: 700;
        }
        .console-card { background: #0f141a; border: 1px solid #1f2a35; border-radius: 12px; padding: 16px; }
        .badge { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background:#13202c; color:#68d6ff; border:1px solid #1f2a35; }
        .muted { color:#9fb0c0; }
        .title { font-weight:700; font-size: 18px; }
        mark { background: #204b61; color:#e6edf3; padding: 0 3px; border-radius:3px; }
        details summary { cursor: pointer; }
        .answer-card { background: linear-gradient(180deg,#0f141a,#0b0f14); border:1px solid #1f2a35; border-radius:12px; padding:16px; box-shadow: 0 0 0 1px rgba(14,165,234,.12) inset; }
        .gradline { height:1px; background: linear-gradient(90deg, rgba(14,165,234,.0), rgba(14,165,234,.6), rgba(14,165,234,.0)); margin: 8px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='header'>
          <div class='brand'>IPC Knowledge Console</div>
          <div class='subtle'>Indian Penal Code · Retrieval‑Augmented QA</div>
          <div class='gradline'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ensure_env()

    # Auto-build index on first run if missing
    def ensure_index_ready():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        have_parents = PARENTS_JSONL.exists()
        have_children = CHILDREN_JSONL.exists()
        have_chroma = (CHROMA_DIR / "chroma.sqlite3").exists()
        if have_parents and have_children and have_chroma:
            return
        with st.spinner("Building index (first run)..."):
            src_path = Path("extracted_sections_final.jsonl")
            parents_local = load_parents_from_jsonl(src_path)
            all_children = []
            for p in parents_local:
                all_children.extend(generate_child_chunks(p, enable_llm=False, llm=None))
            persist_corpora(parents_local, all_children)
            build_vector_store(all_children)

    ensure_index_ready()
    parents, children = load_corpora()
    if not parents or not children:
        st.error("No index found. Run: python index_from_manual.py --file extracted_sections_final.jsonl")
        return

    # Sidebar controls
    with st.sidebar:
        st.markdown("**Settings**")
        k = st.number_input("Top-k (children)", min_value=3, max_value=30, value=12, step=1)
        highlight = st.toggle("Highlight matches", value=True)
        st.markdown("**Quick examples**")
        ex_cols = st.columns(1)
        with ex_cols[0]:
            if st.button("Definition of public servant (21)"):
                st.session_state["query"] = "What is the definition of public servant?"
            if st.button("Punishment for theft (379)"):
                st.session_state["query"] = "What is the punishment for theft?"

    # Search form
    with st.form("search_form", clear_on_submit=False):
        q = st.text_input("Type your query", key="query", placeholder="e.g., What is Section 376D about?", label_visibility="collapsed")
        submitted = st.form_submit_button("Ask")

    if not submitted or not q.strip():
        st.info("Enter a question to get started.")
        return

    # Retrieval + QA with spinner animations
    with st.spinner("Retrieving relevant sections..."):
        retriever = build_hybrid_retriever(children, k=k)
        children_hits: List = retriever.invoke(q)
        expander = parent_expander(children_docs=children, parents=parents)
        parent_hits: List = expander.invoke(children_hits)

    with st.spinner("Synthesizing answer..."):
        chain = build_qa_chain()
        context = "\n\n".join(p.page_content for p in parent_hits)
        answer = chain.invoke({"question": q, "context": context})

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Children retrieved", len(children_hits))
    m2.metric("Parents expanded", len(parent_hits))
    m3.metric("k", k)

    # Layer 1: concise answer card
    st.markdown("<div class='answer-card'><div class='badge'>Answer</div><div class='gradline'></div>" + (answer or "No answer.") + "</div>", unsafe_allow_html=True)

    # Layer 2: interactive sources list + detail viewer
    st.markdown("\n**Top Sections**")
    options: List[Tuple[str, str]] = [
        (f"Section {p.metadata.get('section_number')}: {p.metadata.get('section_title')}", str(i)) for i, p in enumerate(parent_hits)
    ]
    if options:
        sel = st.radio("Select a section to view details", options=[o[1] for o in options], format_func=lambda v: options[int(v)][0], horizontal=False, label_visibility="collapsed")
        idx = int(sel)
        sel_parent = parent_hits[idx]
        sec = sel_parent.metadata.get("section_number")
        title = sel_parent.metadata.get("section_title")
        ch = sel_parent.metadata.get("chapter_title") or ""
        st.markdown(f"<div class='console-card'><span class='badge'>Section {sec}</span> <span class='title'>{title}</span><br><span class='muted'>{ch}</span></div>", unsafe_allow_html=True)
        tabs = st.tabs(["Full text", "Metadata"])
        with tabs[0]:
            text = sel_parent.page_content
            st.markdown(
                f"<div class='console-card' style='margin-top:8px'>{highlight_terms(text, q) if highlight else text}</div>",
                unsafe_allow_html=True,
            )
        with tabs[1]:
            st.json(sel_parent.metadata)


if __name__ == "__main__":
    render_minimal_console()



