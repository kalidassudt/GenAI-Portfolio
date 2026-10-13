from __future__ import annotations

import pytest

from assistant_app.config import Settings
from assistant_app.prompts import build_messages
from assistant_app.service import ChatService


def test_disabled_ollama_uses_mock_response() -> None:
    result = ChatService(Settings(ollama_enabled=False)).answer(
        "What is RAG?", temperature=0.8
    )
    assert result.provider == "mock"
    assert result.answer == "Mock response (Ollama unavailable): What is RAG?"


def test_prompt_includes_system_instructions_and_question() -> None:
    messages = build_messages("Explain embeddings")
    assert len(messages) == 2
    assert "precise and helpful" in messages[0].content
    assert messages[1].content == "Explain embeddings"


def test_ollama_failures_fall_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(ollama_enabled=True)

    def fail_model(**kwargs: object) -> None:
        raise OSError("upstream detail")

    monkeypatch.setattr("assistant_app.service.ChatOllama", fail_model)
    result = ChatService(settings).answer("hello")
    assert result.provider == "mock"
    assert result.answer == "Mock response (Ollama unavailable): hello"


def test_ollama_response_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(ollama_enabled=True, ollama_model="test-model")
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def invoke(self, messages: list[object]) -> object:
            return type("Response", (), {"content": "Local answer"})()

    monkeypatch.setattr("assistant_app.service.ChatOllama", FakeModel)
    result = ChatService(settings).answer("hello", temperature=0.7)

    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.7
    assert result.provider == "ollama"
    assert result.answer == "Local answer"