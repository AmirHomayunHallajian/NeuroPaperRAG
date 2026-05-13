"""Document chunking utilities."""
from typing import Any


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    if not text.strip():
        return chunks

    step = max(1, chunk_size - chunk_overlap)
    start = 0
    length = len(text)

    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += step
    return chunks


def chunk_documents(
    documents: list[dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[dict[str, Any]]:
    """Split page-level documents into overlapping chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunked_docs: list[dict[str, Any]] = []

    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        metadata = doc.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page", 0)

        text_chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(text_chunks):
            chunk_id = f"{source}_p{page}_c{idx}"
            chunked_docs.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "source": source,
                        "path": metadata.get("path", ""),
                        "page": page,
                        "chunk_index": idx,
                    },
                }
            )

    return chunked_docs
