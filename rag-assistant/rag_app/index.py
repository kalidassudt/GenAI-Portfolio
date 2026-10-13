"""Chunk documents, build local/Ollama embeddings, and index them in FAISS."""

from __future__ import annotations

from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import HashingVectorizer

from rag_app.config import Settings


class HashEmbeddings(Embeddings):
    """Deterministic offline embeddings using a stateless hashing vectorizer."""

    def __init__(self, dimensions: int = 2048) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=dimensions,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def split_documents(
    documents: Sequence[Document], settings: Settings | None = None
) -> list[Document]:
    selected = settings or Settings.from_env()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=selected.chunk_size,
        chunk_overlap=selected.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


def build_vector_store(
    documents: Sequence[Document], settings: Settings | None = None
) -> tuple[FAISS, int, str]:
    """Build a FAISS index using Ollama embeddings or local hashing."""
    selected = settings or Settings.from_env()
    chunks = split_documents(documents, selected)
    if not chunks:
        raise ValueError("No document chunks are available to index")

    if selected.embedding_model:
        from langchain_ollama import OllamaEmbeddings

        embeddings: Embeddings = OllamaEmbeddings(
            base_url=selected.ollama_base_url,
            model=selected.embedding_model,
        )
        provider = "ollama"
    else:
        embeddings = HashEmbeddings()
        provider = "local-hashing"
    return FAISS.from_documents(chunks, embeddings), len(chunks), provider