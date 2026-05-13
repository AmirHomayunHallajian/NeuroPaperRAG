"""Ingest local PDFs into ChromaDB."""
from app.config import CHROMA_DIR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, PDF_DIR
from app.pdf_loader import load_pdf_folder
from app.text_splitter import chunk_documents
from app.vector_store import add_chunks_to_vector_store, reset_collection


def run_ingestion() -> dict[str, int]:
    pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    docs = load_pdf_folder(PDF_DIR)
    chunks = chunk_documents(docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)

    reset_collection()
    add_chunks_to_vector_store(chunks)

    print("Ingestion complete.")
    print(f"PDFs found: {pdf_count}")
    print(f"Pages extracted: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Vector DB path: {CHROMA_DIR}")

    return {"pdf_count": pdf_count, "pages": len(docs), "chunks": len(chunks)}


if __name__ == "__main__":
    run_ingestion()
