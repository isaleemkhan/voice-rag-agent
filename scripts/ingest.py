"""Ingest documents from the docs/ directory into Pinecone."""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

# Allow running as `python scripts/ingest.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.embeddings import ensure_index, ingest_text  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


def read_file(path: Path) -> str:
    """Read text from a .txt, .md, or .pdf file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    ensure_index()
    if not DOCS_DIR.exists():
        print(f"No docs directory found at {DOCS_DIR}")
        return

    files = [
        p
        for p in DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
    ]
    if not files:
        print(f"No ingestible files in {DOCS_DIR} (.txt, .md, .pdf).")
        return

    total = 0
    for path in files:
        text = read_file(path).strip()
        if not text:
            print(f"Skipped (empty): {path.name}")
            continue
        chunks = ingest_text(text, source=path.name)
        total += chunks
        print(f"Ingested {chunks} chunks from {path.name}")

    print(f"Done. {total} chunks ingested into Pinecone.")


if __name__ == "__main__":
    main()
