from __future__ import annotations

import json

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


def _client_with_store() -> tuple[TestClient, ChatService]:
    store = InMemoryStore(ttl_seconds=0)
    service = ChatService(store)
    routes.chat_service = service
    return TestClient(app), service


def _fake_reasoning_result() -> MultiAgentGraphResult:
    return MultiAgentGraphResult(
        user_query="Crea un workflow",
        entry_intent=EntryIntent.workflow_build_request,
        target_stage=AgentStage.product_manager_agent,
        confidence=0.83,
        routing_signals=["build_signals_detected"],
        current_stage="product_manager_agent",
        missing_user_inputs=[],
        qa_enabled=True,
        needs_replan=False,
        status="stub_routed",
    )


def test_docs_only_uses_langgraph_reasoning_runtime_when_enabled(monkeypatch) -> None:
    client, service = _client_with_store()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)

    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: _fake_reasoning_result(),
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy retrieval should not run")),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Crea un workflow"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert parsed["entry_intent"] == "workflow_build_request"
    assert parsed["target_stage"] == "product_manager_agent"


def test_workflow_uses_langgraph_runtime_when_enabled(monkeypatch) -> None:
    client, service = _client_with_store()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_WORKFLOW_ENABLED", True, raising=False)

    monkeypatch.setattr(
        service._graph_runtime,
        "run_workflow",
        lambda **kwargs: {
            "clarification_text": "Necesito el nodo exacto para continuar.",
            "fallback_to_docs_only": False,
        },
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "/wf wf_123 revisa este flujo"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    assert "Necesito el nodo exacto" in content
