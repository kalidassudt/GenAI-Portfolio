"""Environment-backed settings for embeddings and answer generation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "llama3.2:latest"
    embedding_model: str | None = None
    ollama_enabled: bool = False
    chunk_size: int = 700
    chunk_overlap: int = 120
    retrieval_k: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "700"))
        chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
        retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "4"))
        if chunk_size < 100:
            raise ValueError("RAG_CHUNK_SIZE must be at least 100")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be less than chunk size")
        if not 1 <= retrieval_k <= 10:
            raise ValueError("RAG_RETRIEVAL_K must be between 1 and 10")
        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            chat_model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL") or None,
            ollama_enabled=os.getenv("OLLAMA_ENABLED", "true").lower()
            in {"1", "true", "yes"},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_k=retrieval_k,
        )