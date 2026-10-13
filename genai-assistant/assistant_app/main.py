"""FastAPI application exposing the assistant service."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from assistant_app.config import Settings
from assistant_app.service import ChatService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Portfolio GenAI Assistant", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)


class ChatResponse(BaseModel):
    answer: str
    provider: str


@lru_cache(maxsize=1)
def get_service() -> ChatService:
    return ChatService(Settings.from_env())


@app.get("/health")
def health() -> dict[str, str]:
    settings = Settings.from_env()
    provider = "ollama" if settings.ollama_enabled else "mock"
    return {"status": "ok", "provider": provider}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest, service: ChatService = Depends(get_service)
) -> ChatResponse:
    try:
        result = service.answer(
            request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=500, detail="Invalid LLM configuration"
        ) from error
    return ChatResponse(answer=result.answer, provider=result.provider)