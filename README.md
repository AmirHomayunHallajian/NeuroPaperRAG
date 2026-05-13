# NeuroPaperRAG: Source-Grounded RAG Assistant for Psychology and Neuroscience Papers

## Overview
NeuroPaperRAG is a lightweight Retrieval-Augmented Generation (RAG) application for querying local psychology and neuroscience PDFs with source-grounded outputs.

## Problem
Researchers, analysts, and knowledge workers need a fast way to query dense scientific documents. Keyword search is limited and generic LLMs may hallucinate.

## Solution
This app retrieves relevant passages from local PDFs before generating source-grounded answers.

## Why This Project Matters
This project demonstrates end-to-end AI product development:
- document ingestion
- preprocessing
- embeddings
- vector search
- LLM integration
- source-grounded generation
- user-facing Streamlit app
- Dockerized reproducibility

## Business Relevance
Although this demo uses neuroscience papers, the same architecture applies to:
- enterprise knowledge bases
- internal documentation
- legal/regulatory documents
- technical reports
- customer support knowledge bases
- healthcare/pharma R&D literature
- consulting knowledge management

## Features
- PDF ingestion
- page-level metadata
- text chunking
- local embeddings
- ChromaDB vector database
- semantic search
- source-grounded answers
- Streamlit app
- Docker support
- optional OpenAI answer generation
- fallback retrieval-only mode without API key

## Tech Stack
Python, Streamlit, pypdf, ChromaDB, SentenceTransformers, OpenAI API, Docker.

## Architecture
```text
PDF Folder
  → Text Extraction
  → Chunking
  → Embeddings
  → ChromaDB Vector Store
  → Semantic Retrieval
  → LLM / Retrieval-Based Answer
  → Streamlit UI with Sources
```

## Repository Structure
```text
NeuroPaperRAG/
├── app/
├── data/
├── notebooks/
├── tests/
├── screenshots/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── LICENSE
```

## Quickstart
### 1. Clone the repository
```bash
git clone https://github.com/AmirHomayunHallajian/NeuroPaperRAG.git
cd NeuroPaperRAG
```

### 2. Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows activation:
```powershell
.venv\Scripts\activate
```

### 3. Add PDFs
Place PDFs into:
`data/sample_papers/`

### 4. Optional: configure OpenAI
```bash
cp .env.example .env
```
Then add `OPENAI_API_KEY`.

### 5. Build vector database
```bash
python app/ingest.py
```

### 6. Run app
```bash
streamlit run app/streamlit_app.py
```

## Docker Usage
```bash
docker build -t neuropaper-rag .
docker run -p 8501:8501 -v $(pwd)/data:/app/data neuropaper-rag
```
Using compose:
```bash
docker compose up --build
```

## Example Questions
- What is representational alignment in communication?
- How do researchers define shared meaning?
- What methods are used to study dyadic interaction?
- What does the literature say about autism and communication?
- Which papers discuss prediction, interaction, and common ground?
- What are the limitations of current communication models?

## Evaluation Plan
- Are retrieved passages relevant?
- Is the answer faithful to the retrieved context?
- Are sources correctly cited?
- Does the model avoid unsupported claims?

## Future Improvements
- RAG evaluation dataset
- hallucination checks
- metadata filters by author/year/topic
- cloud deployment
- local LLM support with Ollama
- user-uploaded PDFs through UI
- chunking strategy comparison
- CI tests with GitHub Actions


