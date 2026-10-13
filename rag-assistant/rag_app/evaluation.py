"""Small, inspectable retrieval evaluation for labeled query/source pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rag_app.assistant import KnowledgeAssistant


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    relevant_sources: set[str]


@dataclass(frozen=True)
class RetrievalReport:
    recall_at_k: float
    evaluated_queries: int
    k: int


def evaluate_retrieval(
    assistant: KnowledgeAssistant,
    cases: Sequence[RetrievalCase],
    k: int = 4,
) -> RetrievalReport:
    """Compute macro recall@k over expected source filenames."""
    if not cases:
        raise ValueError("At least one labeled retrieval case is required")
    recalls = []
    for case in cases:
        if not case.relevant_sources:
            raise ValueError("Each retrieval case needs a relevant source")
        retrieved = {
            source.source for source in assistant.retrieve(case.question, k=k)
        }
        recalls.append(
            len(retrieved & case.relevant_sources) / len(case.relevant_sources)
        )
    return RetrievalReport(
        recall_at_k=sum(recalls) / len(recalls),
        evaluated_queries=len(cases),
        k=k,
    )