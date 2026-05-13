"""Streamlit UI for NeuroPaperRAG."""
import os
import pandas as pd
import streamlit as st

from app.config import PDF_DIR, TOP_K
from app.ingest import run_ingestion
from app.rag_pipeline import answer_question

st.set_page_config(page_title="NeuroPaperRAG", layout="wide")

st.title("NeuroPaperRAG: Source-Grounded Assistant for Scientific PDFs")
st.caption("Query psychology and neuroscience papers using retrieval-augmented generation.")

with st.sidebar:
    st.header("Project")
    st.write("A lightweight RAG assistant for local scientific PDFs.")
    st.markdown("**Instructions**\n1. Add PDFs to `data/sample_papers/`\n2. Run ingestion\n3. Ask questions")
    top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=TOP_K)
    st.write(f"OpenAI key detected: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    if st.button("Run Ingestion"):
        stats = run_ingestion()
        st.success(f"Ingested {stats['pdf_count']} PDFs, {stats['pages']} pages, {stats['chunks']} chunks.")

examples = [
    "What is representational alignment in communication?",
    "How do researchers define shared meaning?",
    "What methods are used to study dyadic interaction?",
    "What does the literature say about autism and communication?",
    "What are the limitations of current communication models?",
]

st.markdown("**Example questions**")
for ex in examples:
    st.write(f"- {ex}")

question = st.text_area("Ask a question", height=120)

if st.button("Ask"):
    pdfs = list(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    if not pdfs:
        st.warning("No PDFs found in data/sample_papers/. Please add PDFs and run ingestion first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        result = answer_question(question, top_k=top_k)
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        if result["sources"]:
            st.dataframe(pd.DataFrame(result["sources"]), use_container_width=True)
        else:
            st.info("No sources returned.")

        st.subheader("Retrieved passages")
        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            md = chunk.get("metadata", {})
            title = f"{i}. {md.get('source', 'unknown')} - p.{md.get('page', '?')} (distance={chunk.get('distance')})"
            with st.expander(title):
                st.write(chunk.get("text", ""))
