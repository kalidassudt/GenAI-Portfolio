from __future__ import annotations

from fastapi.testclient import TestClient

from assistant_app.main import app, get_service
from assistant_app.service import ChatResult

client = TestClient(app)


class StubService:
    def answer(self, question: str, **kwargs: object) -> ChatResult:
        return ChatResult(answer=f"Echo: {question}", provider="test")


def test_chat_endpoint_returns_service_result() -> None:
    app.dependency_overrides[get_service] = lambda: StubService()
    try:
        response = client.post(
            "/chat", json={"question": "Hello", "temperature": 0.4}
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"answer": "Echo: Hello", "provider": "test"}


def test_chat_endpoint_validates_request() -> None:
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422