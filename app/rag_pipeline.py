"""RAG pipeline for querying source-grounded answers."""
from typing import Any
from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.vector_store import query_vector_store


def format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        md = chunk.get("metadata", {})
        source = md.get("source", "unknown")
        page = md.get("page", "?")
        lines.append(f"[{i}] {source}, p. {page}\n{chunk.get('text', '')}")
    return "\n\n".join(lines)


def build_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context = format_context(retrieved_chunks)
    return (
        "You are a scientific assistant. Answer ONLY using the provided context. "
        "If context is insufficient, explicitly say so. Do not invent citations. "
        "Cite sources in [source, p. page] format.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )


def generate_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "No relevant passages were retrieved. Please ingest PDFs first or ask a different question."

    if not OPENAI_API_KEY:
        snippets = []
        for chunk in retrieved_chunks[:3]:
            md = chunk.get("metadata", {})
            snippets.append(f"[{md.get('source', 'unknown')}, p. {md.get('page', '?')}] {chunk.get('text', '')}")
        return (
            "No OpenAI API key found. Below are the most relevant retrieved passages and a retrieval-based synthesis:\n\n"
            + "\n\n".join(snippets)
        )

    prompt = build_prompt(question, retrieved_chunks)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You answer with source-grounded scientific rigor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "No answer returned."


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    retrieved_chunks = query_vector_store(question, top_k=top_k)
    answer = generate_answer(question, retrieved_chunks)
    sources = [
        {
            "source": c.get("metadata", {}).get("source", "unknown"),
            "page": c.get("metadata", {}).get("page", "?"),
            "distance": c.get("distance"),
        }
        for c in retrieved_chunks
    ]
    return {"answer": answer, "sources": sources, "retrieved_chunks": retrieved_chunks}
