from __future__ import annotations

import json

import pytest

from fastapi.testclient import TestClient

from app.config import settings
from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectClarificationState,
    ArchitectStageSearchState,
    ArchitectStageSelection,
    ArchitectStatus,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    ConsultantQueryAnalysis,
    ConsultantResponse,
    ConsultantRetrievalResult,
    ConsultantSource,
    ConsultantToolUsage,
    ImplementationQueueItem,
    ImplementationStatus,
    ImplementedNode,
    MissingUserInput,
    PMClarificationState,
    PMNodeCandidate,
    PMProgressState,
    PMStagePlan,
    PMStageSelection,
    PMStatus,
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
        routing_signals=[
            "entered_commercial_agent",
            "handoff_ready_product_manager",
            "handoff_ready_architect",
            "entered_architect_agent",
            "handoff_ready_engineer",
        ],
        current_stage="architect_agent",
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
                success_criteria=["The support event is normalized for downstream stages."],
            ),
            ArchitectureStage(
                id="stage_escalation",
                name="Escalation Logic",
                purpose="Apply business escalation rules to classify urgency and route actions.",
                required_capabilities=["Evaluate SLA thresholds", "Route escalation path"],
                expected_inputs=["Normalized ticket payload"],
                expected_outputs=["Escalation command"],
                dependencies=["stage_intake"],
                success_criteria=["A single escalation outcome is produced."],
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
            implementation_notes_for_engineer=[
                "Architect agent should preserve the abstract stage order and business intent."
            ],
            required_nodes=[],
        ),
        workflow_context=WorkflowContext(
            use_case_id="uc_1",
            planning_ready=True,
            handoff_target=AgentStage.engineer_agent,
            required_node_types=["n8n-nodes-base.webhook", "n8n-nodes-base.code"],
            unresolved_inputs=[],
            notes=["abstract_plan_only", "architect_complete", "handoff_target=engineer_agent"],
        ),
        planning_summary="Abstract architecture plan is ready for architect handoff.",
        pm_status=PMStatus.pm_completed,
        pm_stage_plan=[
            PMStagePlan(
                id="stage_intake",
                name="Intake",
                objective="Capture support event",
                expected_inputs=["Support event"],
                expected_outputs=["Normalized payload"],
                success_criteria=["Event captured"],
                dependencies=[],
            )
        ],
        pm_stage_selections=[],
        pm_stage_progress=PMProgressState(
            total_stages=1,
            current_stage_id=None,
            completed_stage_ids=["stage_intake"],
            blocked_stage_ids=[],
            passes_by_stage={},
        ),
        pm_clarification_state=PMClarificationState(
            attempts_used=0,
            max_attempts=2,
            pending_questions=[],
            turns=[],
        ),
        architect_status=ArchitectStatus.architect_completed,
        architect_stage_search_history=[
            ArchitectStageSearchState(
                stage_id="stage_intake",
                pass_index=1,
                query="support escalation intake node",
                doc_chunk_ids=["doc-1"],
                candidate_node_types=["n8n-nodes-base.webhook"],
                result_count=1,
                top_rerank_confidence=0.81,
                notes=[],
            )
        ],
        architect_stage_selections=[
            ArchitectStageSelection(
                stage_id="stage_intake",
                selected_node_types=["n8n-nodes-base.webhook"],
                selected_nodes=[],
                rationale="Webhook best matches inbound ticket capture.",
                passes_used=1,
                blocked=False,
                missing_information=[],
            )
        ],
        architect_clarification_state=ArchitectClarificationState(
            attempts_used=0,
            max_attempts=3,
            pending_questions=[],
            turns=[],
        ),
        architect_notes=["Architect grounded the abstract plan into standard workflow nodes."],
        proposed_nodes=[
            ProposedNode(
                node_id="an_1",
                node_type="n8n-nodes-base.webhook",
                stage_id="stage_intake",
                purpose="Capture inbound ticket event",
                expected_inputs=["ticket event"],
                expected_outputs=["normalized payload"],
            ),
            ProposedNode(
                node_id="an_2",
                node_type="n8n-nodes-base.code",
                stage_id="stage_escalation",
                purpose="Classify urgency for escalation",
                depends_on=["an_1"],
                expected_inputs=["normalized payload"],
                expected_outputs=["escalation command"],
            ),
        ],
        workflow_draft=WorkflowDraft(
            name="Support escalation automation",
            use_case_id="uc_1",
            summary="Architect draft",
            nodes=[
                WorkflowDraftNode(
                    node_id="an_1",
                    name="webhook_1",
                    node_type="n8n-nodes-base.webhook",
                    type_version=1,
                    purpose="Capture inbound ticket event",
                    stage_id="stage_intake",
                    expected_inputs=["ticket event"],
                    expected_outputs=["normalized payload"],
                    position=[260, 300],
                ),
                WorkflowDraftNode(
                    node_id="an_2",
                    name="code_2",
                    node_type="n8n-nodes-base.code",
                    type_version=2,
                    purpose="Classify urgency",
                    stage_id="stage_escalation",
                    expected_inputs=["normalized payload"],
                    expected_outputs=["escalation command"],
                    dependencies=["an_1"],
                    position=[540, 300],
                ),
            ],
            connections=[
                WorkflowDraftConnection(
                    source_node_id="an_1",
                    target_node_id="an_2",
                )
            ],
            metadata={},
        ),
        workflow_versions=[
            WorkflowVersion(
                version=1,
                reason="initialized architect workflow draft",
                workflow_draft=WorkflowDraft(
                    name="Support escalation automation",
                    use_case_id="uc_1",
                    summary="Architect draft",
                    nodes=[],
                    connections=[],
                    metadata={},
                ),
                implemented_node_count=0,
            )
        ],
        final_workflow_json={
            "name": "Support escalation automation",
            "nodes": [{"id": "an_1"}, {"id": "an_2"}],
            "connections": {"webhook_1": {"main": [[{"node": "code_2", "type": "main", "index": 0}]]}},
        },
        active_workflow_id="wf_architect_1",
        active_workflow_name="Support escalation automation",
        active_workflow_url="http://localhost:5678/workflow/wf_architect_1",
        workflow_persisted=True,
        workflow_persist_action="created",
        workflow_api_sync_result={"ok": True, "action": "created", "id": "wf_architect_1"},
        qa_enabled=True,
        needs_replan=False,
        status="stub_routed",
    )


def _fake_pm_reasoning_result() -> MultiAgentGraphResult:
    return MultiAgentGraphResult(
        user_query="Crea un workflow",
        entry_intent=EntryIntent.business_discovery_conversation,
        target_stage=None,
        confidence=0.83,
        routing_signals=["entered_commercial_agent", "handoff_ready_product_manager", "handoff_ready_architect"],
        current_stage="product_manager_agent",
        missing_user_inputs=[],
        business_context_summary=BusinessContextSummary(
            process_scope="Customer support operations",
            pain_points=["manual escalations"],
            desired_outcomes=["faster response times"],
            constraints=[],
        ),
        discovered_use_cases=[],
        architecture_plan=ArchitecturePlan(
            use_case_id="uc_1",
            title="Support escalation automation",
            business_objective="Prevent late escalations for high-priority support tickets.",
            desired_outcome="Escalate priority tickets before SLA breach and track delivery outcomes.",
            workflow_summary="Three-stage plan from event intake to escalation delivery and logging.",
            stages=[],
            data_flow=[],
            assumptions=[],
            missing_information=[],
            implementation_notes_for_engineer=[],
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
        pm_status=PMStatus.pm_completed,
        pm_stage_plan=[],
        pm_stage_selections=[],
        pm_stage_progress=PMProgressState(
            total_stages=1,
            current_stage_id=None,
            completed_stage_ids=[],
            blocked_stage_ids=[],
            passes_by_stage={},
        ),
        pm_clarification_state=PMClarificationState(
            attempts_used=0,
            max_attempts=2,
            pending_questions=[],
            turns=[],
        ),
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


def _fake_consultant_reasoning_result() -> MultiAgentGraphResult:
    return MultiAgentGraphResult(
        user_query="What is the difference between webhook and schedule?",
        entry_intent=EntryIntent.information_request,
        target_stage=AgentStage.consultant_agent,
        confidence=0.9,
        routing_signals=["entered_consultant_agent"],
        current_stage="consultant_agent",
        missing_user_inputs=[],
        consultant_query_analysis=ConsultantQueryAnalysis(
            request_type="comparison",
            key_topics=["webhook", "schedule trigger"],
            needs_active_workflow_context=False,
            retrieval_needed=True,
            conversation_history_sufficient=False,
            source_limited=False,
            selected_sources=[ConsultantSource.nodes_index, ConsultantSource.api_docs_index],
            analysis_notes=["Need node/docs evidence."],
        ),
        consultant_selected_sources=[ConsultantSource.nodes_index, ConsultantSource.api_docs_index],
        consultant_tools_used=[
            ConsultantToolUsage(
                tool_name="nodes_index_tool",
                source=ConsultantSource.nodes_index,
                call_order=1,
                query="webhook schedule trigger",
                result_count=3,
            ),
            ConsultantToolUsage(
                tool_name="api_docs_index_tool",
                source=ConsultantSource.api_docs_index,
                call_order=2,
                query="webhook schedule trigger docs",
                result_count=2,
            ),
        ],
        consultant_used_retrieval=True,
        consultant_retrieval_results=[
            ConsultantRetrievalResult(
                source=ConsultantSource.nodes_index,
                query="webhook schedule trigger",
                result_count=3,
                chunk_ids=["node-1"],
                references=["Node: Webhook"],
                snippets=["Webhook starts from incoming HTTP requests."],
            )
        ],
        consultant_response=ConsultantResponse(
            text="Webhook reacts to external events, while Schedule Trigger runs on a time schedule.",
            directly_supported=["nodes_index: Node: Webhook"],
            inferred_guidance=[],
            uncertainties=[],
        ),
        consultant_notes=["consultant completed"],
        qa_enabled=False,
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
    assert parsed["current_stage"] == "architect_agent"
    assert parsed["selected_use_case"]["id"] == "uc_1"
    assert parsed["alternative_use_cases"] == []
    assert parsed["workflow_context"]["handoff_target"] == "engineer_agent"
    assert parsed["workflow_context"]["planning_ready"] is True
    assert parsed["pm_status"] == "pm_completed"
    assert parsed["architect_status"] == "architect_completed"
    assert parsed["proposed_nodes"][0]["node_type"] == "n8n-nodes-base.webhook"
    assert parsed["workflow_draft"]["nodes"][0]["node_type"] == "n8n-nodes-base.webhook"
    assert parsed["workflow_reference"]["id"] == "wf_architect_1"
    assert "architect_stage_search_history" not in parsed
    assert "workflow_versions" not in parsed
    assert "final_workflow_json" not in parsed
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


def test_reasoning_payload_hides_pm_legacy_node_fields(monkeypatch) -> None:
    client, service = _client_with_store()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)

    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: _fake_pm_reasoning_result(),
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
    assert "required_nodes" not in parsed["architecture_plan"]
    assert "pm_stage_selections" not in parsed
    assert "proposed_nodes" not in parsed
    assert "required_credentials" not in parsed


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


def test_reasoning_consultant_returns_plain_text_content(monkeypatch) -> None:
    client, service = _client_with_store()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)
    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: _fake_consultant_reasoning_result(),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "Explain webhook vs schedule."}],
        },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert content.startswith("Webhook reacts to external events")
    with pytest.raises(json.JSONDecodeError):
        json.loads(content)


def test_reasoning_consultant_streams_text_in_multiple_chunks(monkeypatch) -> None:
    client, service = _client_with_store()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)
    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: _fake_consultant_reasoning_result(),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "stream": True,
            "messages": [{"role": "user", "content": "Explain webhook vs schedule."}],
        },
    )
    assert response.status_code == 200
    body = response.text
    assert "data: " in body
    assert body.count("chat.completion.chunk") > 2
    assert "[DONE]" in body
