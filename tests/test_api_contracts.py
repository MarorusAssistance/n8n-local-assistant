from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from app.config import settings
from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    EntryIntent,
    MultiAgentGraphResult,
)
from app.main import app
from app.memory.in_memory import InMemoryStore
from app.services.chat_service import ChatService
import app.api.routes as routes


def _client_with_store() -> tuple[TestClient, InMemoryStore]:
    store = InMemoryStore(ttl_seconds=0)
    routes.chat_service = ChatService(store)
    return TestClient(app), store


def _fake_reasoning_result() -> MultiAgentGraphResult:
    return MultiAgentGraphResult(
        user_query="Crea un flujo",
        entry_intent=EntryIntent.workflow_build_request,
        target_stage=AgentStage.product_manager_agent,
        confidence=0.84,
        routing_signals=["build_signals_detected"],
        current_stage="product_manager_agent",
        missing_user_inputs=[],
        qa_enabled=True,
        needs_replan=False,
        status="stub_routed",
    )


def test_health_contract_shape(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr("app.services.chat_service.check_db", lambda: (True, None))
    monkeypatch.setattr(
        "app.services.chat_service.list_models",
        lambda: {"object": "list", "data": [{"id": "local-model"}]},
    )

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) >= {"status", "db_ok", "lmstudio_ok"}


def test_models_contract_shape(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(
        "app.services.chat_service.list_models",
        lambda: {
            "object": "list",
            "data": [{"id": "local-model", "object": "model", "owned_by": "local"}],
        },
    )

    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload.get("data"), list)
    assert "id" in payload["data"][0]


def test_chat_completion_contract_non_stream(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(
        routes.chat_service._graph_runtime,  # noqa: SLF001
        "run_reasoning",
        lambda **kwargs: _fake_reasoning_result(),
    )
    monkeypatch.setattr("app.services.chat_service.resolve_model", lambda *_: "local-model")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Crea un flujo"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert isinstance(payload["choices"], list) and payload["choices"]
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert parsed["entry_intent"] == "workflow_build_request"
    assert parsed["target_stage"] == "product_manager_agent"


def test_chat_completion_contract_stream(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(
        routes.chat_service._graph_runtime,  # noqa: SLF001
        "run_reasoning",
        lambda **kwargs: _fake_reasoning_result(),
    )
    monkeypatch.setattr("app.services.chat_service.resolve_model", lambda *_: "local-model")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "stream": True,
            "messages": [{"role": "user", "content": "Crea un flujo"}],
        },
    )
    assert response.status_code == 200
    body = response.text
    assert "data: " in body
    assert "[DONE]" in body


def test_chat_completion_preserves_conversation_header(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(
        routes.chat_service._graph_runtime,  # noqa: SLF001
        "run_reasoning",
        lambda **kwargs: _fake_reasoning_result(),
    )
    monkeypatch.setattr("app.services.chat_service.resolve_model", lambda *_: "local-model")

    response = client.post(
        "/v1/chat/completions",
        headers={"x-conversation-id": "conv-contract-1"},
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Crea un flujo"}],
        },
    )
    assert response.status_code == 200
    assert response.headers.get("x-conversation-id") == "conv-contract-1"
