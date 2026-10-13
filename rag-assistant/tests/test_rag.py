from __future__ import annotations

import pytest
from types import SimpleNamespace

from rag_app.assistant import KnowledgeAssistant
from rag_app.config import Settings
from rag_app.documents import load_documents
from rag_app.evaluation import RetrievalCase, evaluate_retrieval
from rag_app.index import build_vector_store, split_documents


def test_text_ingestion_chunking_and_citations() -> None:
    documents = load_documents(
        [("handbook.txt", b"Parental leave provides twelve weeks of paid leave.")]
    )
    chunks = split_documents(documents, Settings(chunk_size=120, chunk_overlap=20))
    vector_store, count, provider = build_vector_store(documents)
    assistant = KnowledgeAssistant(vector_store, Settings(ollama_enabled=False))

    answer = assistant.ask("How many weeks of parental leave are provided?")
    assert chunks[0].metadata["source"] == "handbook.txt"
    assert count >= 1
    assert provider == "local-hashing"
    assert "twelve weeks" in answer.text
    assert "handbook.txt, p. 1" in answer.text
    assert "travel receipts" not in answer.text.lower()


def test_retrieval_evaluation_reports_recall_at_k() -> None:
    documents = load_documents(
        [
            (
                "security.txt",
                b"Report suspected security incidents to the security team within 24 hours.",
            ),
            ("travel.txt", b"Submit travel receipts within thirty days of a trip."),
        ]
    )
    vector_store, _, _ = build_vector_store(documents)
    assistant = KnowledgeAssistant(vector_store, Settings(ollama_enabled=False))
    report = evaluate_retrieval(
        assistant,
        [RetrievalCase("When must a security incident be reported?", {"security.txt"})],
        k=1,
    )
    answer = assistant.ask("When must a security incident be reported?", k=2)
    assert report.evaluated_queries == 1
    assert report.recall_at_k == 1.0
    assert "security incident" in answer.text.lower()
    assert "travel receipts" not in answer.text.lower()


def test_ollama_grounded_answer_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_app.assistant as assistant_module

    vector_store, _, _ = build_vector_store(
        load_documents([("handbook.txt", b"Employees receive twelve weeks of leave.")])
    )
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def invoke(self, messages: list[object]) -> object:
            return type(
                "Response",
                (),
                {"content": "Employees receive twelve weeks of leave [handbook.txt, p. 1]."},
            )()

    monkeypatch.setattr(assistant_module, "ChatOllama", FakeModel)
    settings = Settings(ollama_enabled=True, chat_model="test-model")
    answer = KnowledgeAssistant(vector_store, settings).ask("How much leave?")

    assert captured["model"] == "test-model"
    assert answer.provider == "ollama"
    assert "handbook.txt, p. 1" in answer.text


def test_ingestion_rejects_unsupported_and_empty_files() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_documents([("archive.csv", b"a,b")])
    with pytest.raises(ValueError, match="No readable text"):
        load_documents([("empty.txt", b"   ")])


def test_pdf_ingestion_preserves_one_based_page_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pypdf

    pages = [
        SimpleNamespace(extract_text=lambda: "First page policy"),
        SimpleNamespace(extract_text=lambda: "Second page policy"),
    ]
    monkeypatch.setattr(pypdf, "PdfReader", lambda content: SimpleNamespace(pages=pages))

    documents = load_documents([("policy.pdf", b"fake pdf bytes")])

    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert all(document.metadata["source"] == "policy.pdf" for document in documents)


def test_question_and_evaluation_inputs_are_validated() -> None:
    vector_store, _, _ = build_vector_store(
        load_documents([("notes.txt", b"FAISS stores vectors for semantic retrieval.")])
    )
    assistant = KnowledgeAssistant(vector_store, Settings(ollama_enabled=False))
    with pytest.raises(ValueError, match="cannot be empty"):
        assistant.retrieve("  ")
    with pytest.raises(ValueError, match="At least one"):
        evaluate_retrieval(assistant, [])