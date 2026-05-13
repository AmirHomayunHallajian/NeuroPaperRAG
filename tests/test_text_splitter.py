from app.text_splitter import chunk_documents


def test_chunk_documents_returns_chunks():
    docs = [{"text": "a" * 2200, "metadata": {"source": "a.pdf", "path": "/tmp/a.pdf", "page": 1}}]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) >= 2


def test_chunk_metadata_and_ids():
    docs = [{"text": "hello world " * 100, "metadata": {"source": "paper.pdf", "path": "/x/paper.pdf", "page": 2}}]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert chunks[0]["metadata"]["source"] == "paper.pdf"
    assert chunks[0]["metadata"]["page"] == 2
    assert chunks[0]["id"].startswith("paper.pdf_p2_c")


def test_empty_documents_ignored():
    docs = [{"text": "   ", "metadata": {"source": "x.pdf", "path": "/x/x.pdf", "page": 1}}]
    chunks = chunk_documents(docs)
    assert chunks == []
