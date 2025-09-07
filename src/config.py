import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Default PDF file name present in the workspace root
DEFAULT_PDF_FILENAME = "repealedfileopen.pdf"

DATA_DIR = REPO_ROOT / "storage"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Persisted stores/indices
CHROMA_DIR = DATA_DIR / "chroma_ipc"
PARENTS_JSONL = DATA_DIR / "parents.jsonl"
CHILDREN_JSONL = DATA_DIR / "children.jsonl"
BM25_JSONL = DATA_DIR / "bm25_children.jsonl"  # to rebuild BM25 retriever quickly

# Embeddings
HF_EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Model for optional LLM summarization/QA
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-pro")


def resolve_pdf_path(cli_path: str | None = None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if p.is_file():
            return p
    # Fallback to default at repository root
    default_path = REPO_ROOT / DEFAULT_PDF_FILENAME
    return default_path


