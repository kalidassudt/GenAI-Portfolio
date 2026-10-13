"""Provider selection and chat generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from assistant_app.config import Settings
from assistant_app.prompts import build_messages

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    provider: str


class ChatService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def answer(
        self,
        question: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        selected_temperature = (
            self.settings.temperature if temperature is None else temperature
        )
        selected_max_tokens = (
            self.settings.max_tokens if max_tokens is None else max_tokens
        )
        if not self.settings.ollama_enabled or not self.settings.ollama_configured:
            return self._mock_response(question)

        try:
            model = ChatOllama(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=selected_temperature,
                num_predict=selected_max_tokens,
                timeout=self.settings.request_timeout,
            )
            response = model.invoke(build_messages(question))
            answer_text = str(response.content).strip()
            if not answer_text:
                raise ValueError("Ollama returned an empty response")
            return ChatResult(answer=answer_text, provider="ollama")
        except Exception:
            logger.exception("Ollama request failed; using the mock response")
            return self._mock_response(question)

    @staticmethod
    def _mock_response(question: str) -> ChatResult:
        return ChatResult(
            answer=f"Mock response (Ollama unavailable): {question.strip()}",
            provider="mock",
        )