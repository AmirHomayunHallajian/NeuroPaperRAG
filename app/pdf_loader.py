"""PDF loading helpers."""
from pathlib import Path
from typing import Any
from pypdf import PdfReader


def load_pdf(path: Path) -> list[dict[str, Any]]:
    """Load a single PDF into page-level documents."""
    docs: list[dict[str, Any]] = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        print(f"Warning: could not read PDF {path}: {exc}")
        return docs

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "path": str(path.resolve()),
                    "page": i,
                },
            }
        )
    return docs


def load_pdf_folder(pdf_dir: Path) -> list[dict[str, Any]]:
    """Load all PDFs from a folder."""
    all_docs: list[dict[str, Any]] = []
    if not pdf_dir.exists():
        print(f"Warning: PDF directory not found: {pdf_dir}")
        return all_docs

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        all_docs.extend(load_pdf(pdf_path))
    return all_docs
