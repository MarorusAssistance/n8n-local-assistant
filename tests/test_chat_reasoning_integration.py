from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.memory.in_memory import InMemoryStore
from app.reasoning.types import (
    CheckerResult,
    ContextDocChunk,
    ContextPack,
    ContextPackBudget,
    PlanSpec,
    PlanStep,
    ReasoningPipelineResult,
    RouterConstraints,
    RouterOutput,
)
from app.services.chat_service import ChatService
import app.api.routes as routes


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def model_dump(self):
        return {
            "id": "cmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": self._content},
                    "finish_reason": "stop",
                }
            ],
        }


def _client_with_store() -> tuple[TestClient, InMemoryStore]:
    store = InMemoryStore(ttl_seconds=0)
    routes.chat_service = ChatService(store)
    return TestClient(app), store


def test_docs_only_legacy_path_when_reasoning_flag_disabled(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "legacy", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", False, raising=False)

    monkeypatch.setattr(
        "app.services.chat_service.retrieve_context",
        lambda *args, **kwargs: [
            {"doc_id": "d1", "text": "doc", "url": "", "title": "Doc", "section": ""}
        ],
    )
    monkeypatch.setattr("app.services.chat_service.resolve_model", lambda *_: "test-model")
    monkeypatch.setattr(
        "app.services.chat_service.chat_completion",
        lambda *args, **kwargs: _FakeResponse("legacy response"),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Crea un workflow simple"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"].startswith("legacy response")


def test_docs_only_reasoning_path_when_flag_enabled(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "legacy", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", False, raising=False)
    monkeypatch.setattr("app.services.chat_service.resolve_model", lambda *_: "test-model")

    fake_result = ReasoningPipelineResult(
        plan=PlanSpec(
            summary="Workflow plan",
            steps=[
                PlanStep(
                    id="s1",
                    nodeType="n8n-nodes-base.webhook",
                    purpose="Trigger workflow",
                    inputs=[],
                    outputs=["json"],
                )
            ],
            dataFlowNotes=["Webhook payload flows to next node."],
            questionsForUser=[],
        ),
        checker=CheckerResult(ok=True, issues=[]),
        router=RouterOutput(
            intent="create",
            goal="create workflow",
            constraints=RouterConstraints(),
            missing_info=[],
            complexity_score=1,
        ),
        context_pack=ContextPack(
            nodeCards=[],
            docChunks=[ContextDocChunk(id="d1", text="doc", source="docs")],
            budget=ContextPackBudget(
                maxNodeCards=10,
                maxDocChunks=6,
                maxContextTokens=2500,
                estimatedTokens=140,
            ),
        ),
        second_iteration_used=False,
        attempts=1,
    )

    monkeypatch.setattr(
        "app.services.chat_service.run_reasoning_pipeline",
        lambda *args, **kwargs: fake_result,
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy retrieval should not run")),
    )
    monkeypatch.setattr(
        "app.services.chat_service.chat_completion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy llm should not run")),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Crea un workflow simple"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    assert parsed["summary"] == "Workflow plan"
