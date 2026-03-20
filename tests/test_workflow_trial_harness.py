from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import settings
from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitecturePlan,
    ArchitectureStage,
    DecisionSlot,
    DecisionSlotAnswerStatus,
    EntryIntent,
    ImplementationStatus,
    MultiAgentGraphResult,
    StageKind,
    WorkflowContext,
    WorkflowDraft,
    WorkflowDraftConnection,
    WorkflowDraftNode,
)
from app.main import app
from app.memory.in_memory import InMemoryStore
from app.services.chat_service import ChatService
from app.services.workflow_trial_harness import (
    DEFAULT_EMAIL_URGENCY_REQUEST,
    run_email_urgency_trial,
    validate_email_urgency_workflow,
)
import app.api.routes as routes


def _client_with_service() -> tuple[TestClient, ChatService]:
    store = InMemoryStore(ttl_seconds=0)
    service = ChatService(store)
    routes.chat_service = service
    return TestClient(app), service


def _question_result() -> MultiAgentGraphResult:
    slot = DecisionSlot(
        slot_key="result_application_mode",
        owner_agent=AgentStage.product_manager_agent,
        stage_id="stage_apply",
        question_text="Que quieres hacer con el resultado de la clasificacion?",
        question_intent="result_application_mode",
        answer_status=DecisionSlotAnswerStatus.pending,
    )
    return MultiAgentGraphResult(
        user_query=DEFAULT_EMAIL_URGENCY_REQUEST,
        request_context_query=DEFAULT_EMAIL_URGENCY_REQUEST,
        entry_intent=EntryIntent.workflow_build_request,
        confidence=0.95,
        current_stage="product_manager_agent",
        missing_user_inputs=["Que quieres hacer con el resultado de la clasificacion?"],
        pending_decision_slots=[slot],
        clarification_owner=AgentStage.product_manager_agent,
        clarification_reason="planning_gap",
        last_block_cause="planning_gap",
        workflow_context=WorkflowContext(
            use_case_id="uc_email_urgency",
            planning_ready=False,
            clarification_owner=AgentStage.product_manager_agent,
            clarification_reason="missing_result_application_mode",
            last_block_cause="planning_gap",
            pending_decision_slots=[slot],
            unresolved_inputs=["result_application_mode"],
        ),
        status="unknown_terminal",
    )


def _successful_workflow_result() -> MultiAgentGraphResult:
    plan = ArchitecturePlan(
        use_case_id="uc_email_urgency",
        title="Email urgency classifier",
        business_objective="Classify incoming Gmail messages by urgency.",
        desired_outcome="Apply urgency labels in Gmail.",
        workflow_summary="Trigger -> LLM classify -> Gmail add label",
        stages=[
            ArchitectureStage(
                id="stage_trigger",
                name="Receive Email",
                purpose="Trigger on each Gmail email received.",
                stage_kind=StageKind.trigger_intake,
                business_effect="Capture the incoming email payload.",
                target_entity="gmail_message",
                user_visible_goal="Start automatically on incoming email.",
                required_capabilities=["Trigger on incoming Gmail email"],
                expected_inputs=["Gmail mailbox"],
                expected_outputs=["Email subject and body"],
                dependencies=[],
                success_criteria=["Each incoming email starts the workflow."],
            ),
            ArchitectureStage(
                id="stage_classify",
                name="Classify Urgency",
                purpose="Use an LLM to classify urgency.",
                stage_kind=StageKind.classify_decision,
                business_effect="Produce urgency level.",
                target_entity="gmail_message",
                user_visible_goal="Infer urgency semantically from subject and body.",
                required_capabilities=["LLM classification"],
                expected_inputs=["Email subject and body"],
                expected_outputs=["Urgency level"],
                dependencies=["stage_trigger"],
                success_criteria=["Email is classified as low/medium/high/critical."],
            ),
            ArchitectureStage(
                id="stage_apply",
                name="Apply Gmail Label",
                purpose="Apply urgency label back to Gmail message.",
                stage_kind=StageKind.apply_update_source,
                business_effect="Label the message in Gmail.",
                target_entity="gmail_message",
                user_visible_goal="Filter emails later from Gmail UI.",
                required_capabilities=["Add label to Gmail message"],
                expected_inputs=["Gmail message id", "Urgency label"],
                expected_outputs=["Labeled Gmail message"],
                dependencies=["stage_classify"],
                success_criteria=["Urgency label is visible in Gmail UI."],
            ),
        ],
    )
    draft = WorkflowDraft(
        name="Email urgency classifier",
        use_case_id="uc_email_urgency",
        summary="Trigger -> LLM classify -> Gmail add label",
        nodes=[
            WorkflowDraftNode(
                node_id="gmail_trigger",
                name="Gmail Trigger",
                node_type="n8n-nodes-base.gmailTrigger",
                stage_id="stage_trigger",
                purpose="Trigger on each new email in Gmail.",
                position=[260, 300],
            ),
            WorkflowDraftNode(
                node_id="openai_classify",
                name="OpenAI",
                node_type="n8n-nodes-base.openAi",
                stage_id="stage_classify",
                purpose="Classify urgency semantically from subject and body.",
                parameters_known={"model": "gpt-4o-mini"},
                position=[520, 300],
            ),
            WorkflowDraftNode(
                node_id="gmail_add_label",
                name="Gmail",
                node_type="n8n-nodes-base.gmail",
                stage_id="stage_apply",
                purpose="Add urgency label to the Gmail message.",
                parameters_known={
                    "resource": "message",
                    "operation": "addLabel",
                    "labelId": "Review",
                },
                implementation_hints={"action": "apply_label"},
                position=[780, 300],
            ),
        ],
        connections=[
            WorkflowDraftConnection(source_node_id="gmail_trigger", target_node_id="openai_classify"),
            WorkflowDraftConnection(source_node_id="openai_classify", target_node_id="gmail_add_label"),
        ],
    )
    return MultiAgentGraphResult(
        user_query=DEFAULT_EMAIL_URGENCY_REQUEST,
        request_context_query=DEFAULT_EMAIL_URGENCY_REQUEST,
        entry_intent=EntryIntent.workflow_build_request,
        confidence=0.95,
        current_stage="engineer_agent",
        architecture_plan=plan,
        workflow_context=WorkflowContext(
            use_case_id="uc_email_urgency",
            planning_ready=True,
            handoff_target=AgentStage.engineer_agent,
            stage_bundle_map={
                "stage_trigger": ["gmail_trigger"],
                "stage_classify": ["openai_classify"],
                "stage_apply": ["gmail_add_label"],
            },
        ),
        workflow_draft=draft,
        stage_bundle_map={
            "stage_trigger": ["gmail_trigger"],
            "stage_classify": ["openai_classify"],
            "stage_apply": ["gmail_add_label"],
        },
        implementation_status=ImplementationStatus.completed,
        workflow_persisted=True,
        workflow_persist_action="created",
        active_workflow_id="wf_email_urgency",
        active_workflow_url="http://localhost:5678/workflow/wf_email_urgency",
        status="unknown_terminal",
    )


def _incomplete_workflow_result() -> MultiAgentGraphResult:
    plan = ArchitecturePlan(
        use_case_id="uc_email_urgency",
        title="Email urgency classifier",
        business_objective="Classify incoming Gmail messages by urgency.",
        desired_outcome="Apply urgency labels in Gmail.",
        workflow_summary="Trigger -> LLM classify",
        stages=[
            ArchitectureStage(
                id="stage_trigger",
                name="Receive Email",
                purpose="Trigger on each Gmail email received.",
                stage_kind=StageKind.trigger_intake,
                business_effect="Capture the incoming email payload.",
                target_entity="gmail_message",
                user_visible_goal="Start automatically on incoming email.",
                required_capabilities=["Trigger on incoming Gmail email"],
                expected_inputs=["Gmail mailbox"],
                expected_outputs=["Email subject and body"],
                dependencies=[],
                success_criteria=["Each incoming email starts the workflow."],
            ),
            ArchitectureStage(
                id="stage_classify",
                name="Classify Urgency",
                purpose="Use an LLM to classify urgency.",
                stage_kind=StageKind.classify_decision,
                business_effect="Produce urgency level.",
                target_entity="gmail_message",
                user_visible_goal="Infer urgency semantically from subject and body.",
                required_capabilities=["LLM classification"],
                expected_inputs=["Email subject and body"],
                expected_outputs=["Urgency level"],
                dependencies=["stage_trigger"],
                success_criteria=["Email is classified as low/medium/high/critical."],
            ),
        ],
    )
    draft = WorkflowDraft(
        name="Email urgency classifier",
        use_case_id="uc_email_urgency",
        summary="Trigger -> LLM classify",
        nodes=[
            WorkflowDraftNode(
                node_id="gmail_trigger",
                name="Gmail Trigger",
                node_type="n8n-nodes-base.gmailTrigger",
                stage_id="stage_trigger",
                purpose="Trigger on each new email in Gmail.",
                position=[260, 300],
            ),
            WorkflowDraftNode(
                node_id="openai_classify",
                name="OpenAI",
                node_type="n8n-nodes-base.openAi",
                stage_id="stage_classify",
                purpose="Classify urgency semantically from subject and body.",
                parameters_known={"model": "gpt-4o-mini"},
                position=[520, 300],
            ),
        ],
        connections=[
            WorkflowDraftConnection(source_node_id="gmail_trigger", target_node_id="openai_classify"),
        ],
    )
    return MultiAgentGraphResult(
        user_query=DEFAULT_EMAIL_URGENCY_REQUEST,
        request_context_query=DEFAULT_EMAIL_URGENCY_REQUEST,
        entry_intent=EntryIntent.workflow_build_request,
        confidence=0.95,
        current_stage="engineer_agent",
        architecture_plan=plan,
        workflow_context=WorkflowContext(
            use_case_id="uc_email_urgency",
            planning_ready=True,
            handoff_target=AgentStage.engineer_agent,
            stage_bundle_map={
                "stage_trigger": ["gmail_trigger"],
                "stage_classify": ["openai_classify"],
            },
        ),
        workflow_draft=draft,
        stage_bundle_map={
            "stage_trigger": ["gmail_trigger"],
            "stage_classify": ["openai_classify"],
        },
        implementation_status=ImplementationStatus.completed,
        workflow_persisted=True,
        workflow_persist_action="created",
        active_workflow_id="wf_bad_email_urgency",
        active_workflow_url="http://localhost:5678/workflow/wf_bad_email_urgency",
        status="unknown_terminal",
    )


def test_email_urgency_trial_handles_semantic_clarification_and_succeeds(monkeypatch) -> None:
    client, service = _client_with_service()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)

    calls: list[str] = []
    results = [_question_result(), _successful_workflow_result()]

    def _fake_run_reasoning(**kwargs):
        calls.append(str(kwargs.get("user_prompt") or ""))
        return results.pop(0)

    monkeypatch.setattr(service._graph_runtime, "run_reasoning", _fake_run_reasoning)

    result = run_email_urgency_trial(client=client, service=service, max_turns=4)

    assert result.success is True
    assert result.structural_success is True
    assert result.persistence_success is True
    assert result.workflow_url == "http://localhost:5678/workflow/wf_email_urgency"
    assert result.structural_issues == []
    assert len(result.transcript) == 2
    assert calls[0] == DEFAULT_EMAIL_URGENCY_REQUEST
    assert "Gmail" in calls[1]
    assert "etiqueta visible en Gmail" in calls[1]
    assert "clasificar semanticamente" in calls[1]


def test_email_urgency_trial_flags_incomplete_created_workflow(monkeypatch) -> None:
    client, service = _client_with_service()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)
    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: _incomplete_workflow_result(),
    )

    result = run_email_urgency_trial(client=client, service=service, max_turns=2)

    assert result.success is False
    assert result.structural_success is False
    assert result.workflow_url == "http://localhost:5678/workflow/wf_bad_email_urgency"
    assert "missing_label_apply_node" in result.structural_issues


def test_harness_flags_semantically_wrong_nodes_and_blocked_followup() -> None:
    payload = _successful_workflow_result().model_dump(mode="json")
    nodes = payload["workflow_draft"]["nodes"]
    nodes[1]["node_id"] = "postmark_trigger"
    nodes[1]["name"] = "Postmark Trigger"
    nodes[1]["node_type"] = "n8n-nodes-base.postmarkTrigger"
    nodes[1]["purpose"] = "Trigger on inbound Postmark events."
    nodes[1]["parameters_known"] = {}
    nodes[1]["parameters_inferred"] = {}
    nodes[2]["node_id"] = "email_trigger_apply"
    nodes[2]["name"] = "Email Trigger (IMAP)"
    nodes[2]["node_type"] = "n8n-nodes-base.emailReadImap"
    nodes[2]["purpose"] = "Trigger when a new email is received."
    nodes[2]["parameters_known"] = {}
    nodes[2]["parameters_inferred"] = {}
    payload["implementation_status"] = "blocked_waiting_user"
    payload["missing_user_inputs"] = ["Configure this node to apply a label or tag."]
    payload["workflow_persisted"] = False
    payload["workflow_api_sync_result"] = {"ok": False, "error": "Failed to contact n8n"}

    issues = validate_email_urgency_workflow(payload)

    assert "missing_llm_classification_node" in issues
    assert "missing_label_apply_node" in issues
    assert "workflow_blocked_waiting_user" in issues
    assert "workflow_requires_user_followup" in issues
    assert "workflow_not_persisted" in issues


def test_reasoning_payload_exposes_pending_slots_for_harness(monkeypatch) -> None:
    client, service = _client_with_service()
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "langgraph", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)
    monkeypatch.setattr(
        service._graph_runtime,
        "run_reasoning",
        lambda **kwargs: _question_result(),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": DEFAULT_EMAIL_URGENCY_REQUEST}],
        },
    )

    assert response.status_code == 200
    parsed = json.loads(service._temporal_result_path.read_text(encoding="utf-8"))  # noqa: SLF001
    assert parsed["pending_decision_slots"][0]["slot_key"] == "result_application_mode"
    assert parsed["clarification_owner"] == "product_manager_agent"
    assert parsed["last_block_cause"] == "planning_gap"
