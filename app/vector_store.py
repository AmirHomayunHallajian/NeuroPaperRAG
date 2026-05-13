"""ChromaDB vector store utilities."""
from functools import lru_cache
from typing import Any
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection() -> Any:
    client = get_chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def reset_collection() -> Any:
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_chunks_to_vector_store(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    collection = get_or_create_collection()
    embedder = get_embedding_model()

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)


def query_vector_store(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    collection = get_or_create_collection()
    embedder = get_embedding_model()
    query_embedding = embedder.encode([question], show_progress_bar=False).tolist()[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    return [
        {"text": doc, "metadata": meta or {}, "distance": dist}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
