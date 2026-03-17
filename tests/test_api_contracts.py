from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from app.config import settings
from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    ConsultantQueryAnalysis,
    ConsultantResponse,
    ConsultantSource,
    ImplementationStatus,
    BusinessContextSummary,
    EntryIntent,
    MultiAgentGraphResult,
    NodeRequirement,
    UseCase,
    WorkflowContext,
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
        entry_intent=EntryIntent.business_discovery_conversation,
        target_stage=None,
        confidence=0.84,
        routing_signals=["entered_commercial_agent", "handoff_ready_product_manager", "handoff_ready_architect"],
        current_stage="product_manager_agent",
        missing_user_inputs=[],
        business_context_summary=BusinessContextSummary(
            process_scope="Finance operations",
            pain_points=["manual approvals"],
            desired_outcomes=["faster approvals"],
            constraints=[],
        ),
        discovered_use_cases=[
            UseCase(
                id="uc_1",
                title="Invoice approval reminders",
                business_problem="Invoice approvals are delayed due to manual follow-up.",
                desired_outcome="Automate reminders and escalation for pending approvals.",
                expected_value="Reduce delays and improve payment throughput.",
                feasibility="medium",
                priority_score=85.0,
                why_selected="Selected due to high business value.",
            )
        ],
        selected_use_case=UseCase(
            id="uc_1",
            title="Invoice approval reminders",
            business_problem="Invoice approvals are delayed due to manual follow-up.",
            desired_outcome="Automate reminders and escalation for pending approvals.",
            expected_value="Reduce delays and improve payment throughput.",
            feasibility="medium",
            priority_score=85.0,
            why_selected="Selected due to high business value.",
        ),
        alternative_use_cases=[],
        selection_reason="Top business-value opportunity with clear desired outcome.",
        architecture_plan=ArchitecturePlan(
            use_case_id="uc_1",
            title="Invoice approval reminders",
            business_objective="Reduce delays in invoice approvals.",
            desired_outcome="Automate reminders and escalation for pending approvals.",
            workflow_summary="Three-stage architecture with intake, decisioning and delivery.",
            stages=[
                ArchitectureStage(
                    id="stage_intake",
                    name="Intake",
                    purpose="Capture approval events",
                    required_capabilities=["Capture events"],
                expected_inputs=["Approval event"],
                expected_outputs=["Normalized payload"],
                dependencies=[],
                success_criteria=["The approval event is captured once and normalized."],
            )
        ],
            data_flow=[
                ArchitectureDataFlowItem(
                    source_stage_id="stage_intake",
                    target_stage_id="stage_intake",
                    data_items=["payload"],
                )
            ],
            assumptions=["Approval source emits stable events."],
            missing_information=[],
            implementation_notes_for_engineer=["Architect agent should preserve the abstract stage order."],
            required_nodes=[],
        ),
        workflow_context=WorkflowContext(
            use_case_id="uc_1",
            planning_ready=True,
            handoff_target=AgentStage.architect_agent,
            required_node_types=[],
            unresolved_inputs=[],
            notes=["abstract_plan_only", "handoff_target=architect_agent"],
        ),
        planning_summary="Abstract architecture plan prepared for architect handoff.",
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
    assert parsed["entry_intent"] == "business_discovery_conversation"
    assert parsed["target_stage"] is None
    assert parsed["selected_use_case"]["id"] == "uc_1"
    assert parsed["workflow_context"]["handoff_target"] == "architect_agent"
    assert "required_nodes" not in parsed["architecture_plan"]
    assert "required_node_types" not in parsed["workflow_context"]


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


def test_chat_completion_contract_accepts_additive_engineer_fields(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)

    result = _fake_reasoning_result().model_copy(
        update={
            "current_stage": "engineer_agent",
            "target_stage": AgentStage.qa_agent,
            "implementation_status": ImplementationStatus.blocked_waiting_user,
            "missing_user_input_details": [],
            "workflow_versions": [],
            "active_workflow_id": "wf_contract_1",
            "active_workflow_name": "Contract Flow",
            "active_workflow_url": "http://localhost:5678/workflow/wf_contract_1",
            "workflow_persisted": True,
            "workflow_persist_action": "updated",
            "workflow_api_sync_result": {"ok": True, "action": "updated", "id": "wf_contract_1"},
        }
    )
    monkeypatch.setattr(
        routes.chat_service._graph_runtime,  # noqa: SLF001
        "run_reasoning",
        lambda **kwargs: result,
    )
    monkeypatch.setattr("app.services.chat_service.resolve_model", lambda *_: "local-model")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Continue workflow implementation"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    parsed = json.loads(payload["choices"][0]["message"]["content"])
    assert parsed["entry_intent"] == "business_discovery_conversation"
    assert parsed["current_stage"] == "engineer_agent"
    assert parsed["implementation_status"] == "blocked_waiting_user"
    assert "workflow_versions" in parsed
    assert parsed["workflow_reference"]["id"] == "wf_contract_1"
    assert parsed["workflow_persist_action"] == "updated"
    assert "final_workflow_json" not in parsed


def test_chat_completion_contract_consultant_content_is_text(monkeypatch) -> None:
    client, _store = _client_with_store()
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)

    consultant_result = MultiAgentGraphResult(
        user_query="Explain webhook vs schedule",
        entry_intent=EntryIntent.information_request,
        target_stage=AgentStage.consultant_agent,
        confidence=0.91,
        routing_signals=["entered_consultant_agent"],
        current_stage="consultant_agent",
        missing_user_inputs=[],
        consultant_query_analysis=ConsultantQueryAnalysis(
            request_type="comparison",
            key_topics=["webhook", "schedule"],
            needs_active_workflow_context=False,
            retrieval_needed=False,
            conversation_history_sufficient=True,
            source_limited=False,
            selected_sources=[ConsultantSource.conversation_history],
            analysis_notes=[],
        ),
        consultant_selected_sources=[ConsultantSource.conversation_history],
        consultant_tools_used=[],
        consultant_used_retrieval=False,
        consultant_retrieval_results=[],
        consultant_response=ConsultantResponse(
            text="Webhook is event-driven. Schedule Trigger is time-driven.",
            directly_supported=[],
            inferred_guidance=[],
            uncertainties=[],
        ),
        consultant_notes=[],
        qa_enabled=False,
        needs_replan=False,
        status="stub_routed",
    )
    monkeypatch.setattr(
        routes.chat_service._graph_runtime,  # noqa: SLF001
        "run_reasoning",
        lambda **kwargs: consultant_result,
    )
    monkeypatch.setattr("app.services.chat_service.resolve_model", lambda *_: "local-model")

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Explain webhook vs schedule"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    content = payload["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert "Webhook is event-driven" in content
