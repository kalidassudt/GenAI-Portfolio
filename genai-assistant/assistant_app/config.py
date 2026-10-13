"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_enabled: bool = False
    temperature: float = 0.2
    max_tokens: int = 600
    request_timeout: float = 30.0

    @property
    def ollama_configured(self) -> bool:
        return bool(self.ollama_base_url and self.ollama_model)

    @classmethod
    def from_env(cls) -> Settings:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "600"))
        timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
        if not 0 <= temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("LLM_MAX_TOKENS must be between 1 and 4096")
        if timeout <= 0:
            raise ValueError("LLM_REQUEST_TIMEOUT must be positive")

        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            ollama_enabled=os.getenv("OLLAMA_ENABLED", "true").lower()
            in {"1", "true", "yes"},
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout,
        )