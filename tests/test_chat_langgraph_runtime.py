from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import settings
from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    ImplementationQueueItem,
    ImplementationStatus,
    ImplementedNode,
    MissingUserInput,
    ProposedNode,
    RequiredCredential,
    BusinessContextSummary,
    EntryIntent,
    MultiAgentGraphResult,
    NodeRequirement,
    UseCase,
    VariableDefinition,
    WorkflowDraft,
    WorkflowDraftConnection,
    WorkflowDraftNode,
    WorkflowVersion,
    WorkflowContext,
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
        entry_intent=EntryIntent.business_discovery_conversation,
        target_stage=AgentStage.engineer_agent,
        confidence=0.83,
        routing_signals=["entered_commercial_agent", "handoff_ready_product_manager", "handoff_ready_engineer"],
        current_stage="product_manager_agent",
        missing_user_inputs=[],
        business_context_summary=BusinessContextSummary(
            process_scope="Customer support operations",
            pain_points=["manual escalations"],
            desired_outcomes=["faster response times"],
            constraints=[],
        ),
        discovered_use_cases=[
            UseCase(
                id="uc_1",
                title="Support escalation automation",
                business_problem="Critical support tickets are escalated too late.",
                desired_outcome="Escalate priority tickets automatically before SLA breach.",
                expected_value="Reduce churn and avoid SLA penalties.",
                feasibility="medium",
                priority_score=88.0,
                why_selected="Selected due to highest business value.",
            )
        ],
        selected_use_case=UseCase(
            id="uc_1",
            title="Support escalation automation",
            business_problem="Critical support tickets are escalated too late.",
            desired_outcome="Escalate priority tickets automatically before SLA breach.",
            expected_value="Reduce churn and avoid SLA penalties.",
            feasibility="medium",
            priority_score=88.0,
            why_selected="Selected due to highest business value.",
        ),
        alternative_use_cases=[],
        selection_reason="Highest expected business value with clear and actionable scope.",
        architecture_plan=ArchitecturePlan(
            use_case_id="uc_1",
            title="Support escalation automation",
            business_objective="Prevent late escalations for high-priority support tickets.",
            desired_outcome="Escalate priority tickets before SLA breach and track delivery outcomes.",
            workflow_summary="Three-stage plan from event intake to escalation delivery and logging.",
            stages=[
                ArchitectureStage(
                    id="stage_intake",
                    name="Intake",
                    purpose="Capture and normalize support ticket events.",
                    required_capabilities=["Capture support events", "Normalize payload"],
                    expected_inputs=["Incoming support event"],
                    expected_outputs=["Normalized ticket payload"],
                    dependencies=[],
                ),
                ArchitectureStage(
                    id="stage_escalation",
                    name="Escalation Logic",
                    purpose="Apply business escalation rules to classify urgency and route actions.",
                    required_capabilities=["Evaluate SLA thresholds", "Route escalation path"],
                    expected_inputs=["Normalized ticket payload"],
                    expected_outputs=["Escalation command"],
                    dependencies=["stage_intake"],
                ),
            ],
            data_flow=[
                ArchitectureDataFlowItem(
                    source_stage_id="stage_intake",
                    target_stage_id="stage_escalation",
                    data_items=["normalized payload", "priority metadata"],
                )
            ],
            assumptions=["Ticket priority metadata is reliable."],
            missing_information=[],
            implementation_notes_for_engineer=["Configure node parameters and credentials in engineering phase."],
            required_nodes=[
                NodeRequirement(
                    node_type="n8n-nodes-base.webhook",
                    why_required="Evidence-backed trigger node from API docs retrieval.",
                    evidence_chunk_ids=["linked:node:n8n-nodes-base.webhook"],
                    evidence_refs=["Node: Webhook"],
                    evidence_confidence=0.87,
                    rerank_confidence=0.74,
                    blended_confidence=0.84,
                )
            ],
        ),
        workflow_context=WorkflowContext(
            use_case_id="uc_1",
            planning_ready=True,
            handoff_target=AgentStage.engineer_agent,
            required_node_types=["n8n-nodes-base.webhook"],
            unresolved_inputs=[],
            notes=["required_nodes=1"],
        ),
        planning_summary="Architecture plan is ready for engineer handoff with evidence-backed required nodes.",
        qa_enabled=True,
        needs_replan=False,
        status="stub_routed",
    )


def _fake_engineer_reasoning_result() -> MultiAgentGraphResult:
    draft = WorkflowDraft(
        name="Engineer Draft",
        use_case_id="uc_1",
        summary="Draft under construction",
        nodes=[
            WorkflowDraftNode(
                node_id="pn_1",
                name="webhook_1",
                node_type="n8n-nodes-base.webhook",
                purpose="Receive inbound payload",
                parameters_known={},
                parameters_inferred={"path": "incoming-event"},
                expected_outputs=["payload"],
                dependencies=[],
                position=[0, 0],
            )
        ],
        connections=[],
        metadata={},
    )
    return MultiAgentGraphResult(
        user_query="Build this workflow",
        entry_intent=EntryIntent.workflow_build_request,
        target_stage=AgentStage.qa_agent,
        confidence=0.81,
        routing_signals=["handoff_ready_engineer", "entered_engineer_agent", "handoff_ready_qa"],
        current_stage="engineer_agent",
        missing_user_inputs=[],
        architecture_plan=ArchitecturePlan(
            use_case_id="uc_1",
            title="Support escalation automation",
            business_objective="Escalate high-priority tickets",
            desired_outcome="Escalate before SLA breach",
            workflow_summary="Trigger + escalation branch",
            stages=[
                ArchitectureStage(
                    id="stage_1",
                    name="Trigger",
                    purpose="Receive event",
                    required_capabilities=["Receive event"],
                    expected_inputs=["event"],
                    expected_outputs=["payload"],
                    dependencies=[],
                )
            ],
            data_flow=[],
            assumptions=[],
            missing_information=[],
            implementation_notes_for_engineer=[],
            required_nodes=[
                NodeRequirement(
                    node_type="n8n-nodes-base.webhook",
                    why_required="Evidence-backed trigger node",
                    evidence_chunk_ids=["chunk-1"],
                    evidence_refs=["ref-1"],
                    evidence_confidence=0.86,
                )
            ],
        ),
        workflow_context=WorkflowContext(
            use_case_id="uc_1",
            planning_ready=True,
            handoff_target=AgentStage.qa_agent,
            required_node_types=["n8n-nodes-base.webhook"],
            unresolved_inputs=[],
            notes=[],
        ),
        proposed_nodes=[
            ProposedNode(
                node_id="pn_1",
                node_type="n8n-nodes-base.webhook",
                purpose="Receive inbound payload",
            )
        ],
        required_credentials=[
            RequiredCredential(
                credential_key="cred_1",
                node_type="n8n-nodes-base.googleSheets",
                credential_name="googleSheetsOAuth2Api",
                required_for="pn_2",
            )
        ],
        workflow_draft=draft,
        workflow_versions=[
            WorkflowVersion(
                version=1,
                reason="initialized engineer workflow draft",
                workflow_draft=draft,
                implemented_node_count=1,
            )
        ],
        node_implementation_queue=[
            ImplementationQueueItem(
                queue_id="pn_1",
                node_type="n8n-nodes-base.webhook",
                status="implemented",
            )
        ],
        implemented_nodes=[
            ImplementedNode(
                queue_id="pn_1",
                node_id="pn_1",
                node_type="n8n-nodes-base.webhook",
                version=1,
            )
        ],
        blocked_nodes=[],
        variable_registry=[
            VariableDefinition(
                name="pn_1.payload",
                origin_node_id="pn_1",
                destination_node_ids=[],
                semantic_meaning="Incoming webhook payload",
            )
        ],
        missing_user_input_details=[
            MissingUserInput(
                input_id="credential:pn_2:googleSheetsOAuth2Api",
                input_key="credential:pn_2:googleSheetsOAuth2Api",
                missing_item="googleSheetsOAuth2Api",
                reason="Credential needed for destination",
                blocking_node_id="pn_2",
                category="credential",
                question="Provide credential reference",
            )
        ],
        implementation_status=ImplementationStatus.in_progress,
        engineer_notes=["Engineer started iterative construction."],
        final_workflow_json={
            "name": "Engineer Draft",
            "nodes": [{"id": "pn_1", "type": "n8n-nodes-base.webhook"}],
            "connections": {},
        },
        active_workflow_id="wf_200",
        active_workflow_name="Engineer Draft",
        active_workflow_url="http://localhost:5678/workflow/wf_200",
        workflow_persisted=True,
        workflow_persist_action="created",
        workflow_api_sync_result={"ok": True, "action": "created", "id": "wf_200"},
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
    assert parsed["entry_intent"] == "business_discovery_conversation"
    assert parsed["target_stage"] == "engineer_agent"
    assert parsed["current_stage"] == "product_manager_agent"
    assert parsed["selected_use_case"]["id"] == "uc_1"
    assert parsed["alternative_use_cases"] == []
    assert parsed["architecture_plan"]["required_nodes"][0]["node_type"] == "n8n-nodes-base.webhook"
    assert parsed["workflow_context"]["planning_ready"] is True
    assert "selection_reason" in parsed


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


def test_reasoning_payload_sanitizes_malformed_reference_spillover(monkeypatch) -> None:
    client, service = _client_with_store()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)

    bad = _fake_reasoning_result()
    bad.architecture_plan.required_nodes[0].evidence_refs = [
        (
            "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.sendemail/"
            '&quot;], "confidence": 0.66}, {"node_type":"n8n-nodes-base.code"}'
        )
    ]
    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: bad,
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
    ref = parsed["architecture_plan"]["required_nodes"][0]["evidence_refs"][0]
    assert ref == "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.sendemail/"


def test_reasoning_payload_includes_engineer_fields_when_engineer_stage_runs(monkeypatch) -> None:
    client, service = _client_with_store()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)

    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: _fake_engineer_reasoning_result(),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Build and continue workflow"}],
        },
    )
    assert response.status_code == 200
    parsed = json.loads(response.json()["choices"][0]["message"]["content"])
    assert parsed["current_stage"] == "engineer_agent"
    assert parsed["workflow_draft"]["nodes"]
    assert parsed["workflow_versions"]
    assert parsed["node_implementation_queue"]
    assert parsed["implemented_nodes"]
    assert parsed["variable_registry"]
    assert parsed["implementation_status"] == "in_progress"
    assert parsed["active_workflow_id"] == "wf_200"
    assert parsed["workflow_persisted"] is True
    assert parsed["workflow_reference"]["id"] == "wf_200"
    assert "final_workflow_json" not in parsed
