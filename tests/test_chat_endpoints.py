from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.memory.in_memory import InMemoryStore
from app.services.chat_service import ChatService
import app.api.routes as routes


def _client_with_store() -> tuple[TestClient, InMemoryStore]:
    store = InMemoryStore(ttl_seconds=0)
    routes.chat_service = ChatService(store)
    return TestClient(app), store


def test_list_chats_returns_conversations() -> None:
    client, store = _client_with_store()
    store.append_messages(
        "chat-1",
        [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Buenas"},
        ],
    )
    store.append_messages(
        "chat-2",
        [
            {"role": "user", "content": "Otra pregunta"},
            {"role": "assistant", "content": "Otra respuesta"},
        ],
    )

    response = client.get("/v1/chats?limit=10&offset=0")
    assert response.status_code == 200
    payload = response.json()
    ids = {item["id"] for item in payload["data"]}

    assert "chat-1" in ids
    assert "chat-2" in ids


def test_get_chat_returns_full_history() -> None:
    client, store = _client_with_store()
    store.append_messages(
        "chat-1",
        [
            {"role": "user", "content": "Pregunta"},
            {"role": "assistant", "content": "Respuesta"},
        ],
    )

    response = client.get("/v1/chats/chat-1")
    assert response.status_code == 200
    payload = response.json()

    assert payload["id"] == "chat-1"
    assert payload["message_count"] == 2
    assert payload["messages"] == [
        {"role": "user", "content": "Pregunta"},
        {"role": "assistant", "content": "Respuesta"},
    ]


def test_get_chat_not_found() -> None:
    client, _store = _client_with_store()
    response = client.get("/v1/chats/missing-chat")
    assert response.status_code == 404
