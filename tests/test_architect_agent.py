from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.doc_links import derive_doc_page_key
from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectClarificationState,
    ArchitectNodeCandidate,
    ArchitectStageSelection,
    ArchitectStatus,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    EntryIntent,
    StageKind,
    WorkflowContext,
)
from app.graphs.nodes import architect_agent as architect_mod
from app.graphs.nodes.architect_agent import (
    _StageSelectionOutput,
    _augment_candidates_with_definition_fallbacks,
    _build_workflow_blueprint_with_structured_output,
    _WorkflowConnectionBlueprint,
    _WorkflowBlueprintOutput,
    _WorkflowNodeBlueprint,
    _candidate_rejection_reasons,
    _is_trigger_candidate,
    _build_candidates,
    _normalize_confidence,
    _plan_terminal_outcome_gap_question,
    _refine_terminal_outcome_plan_from_slots,
    _select_linking_page_keys,
    _select_stage_nodes_with_structured_output,
    architect_agent_node,
)


def _plan() -> ArchitecturePlan:
    return ArchitecturePlan(
        use_case_id="uc_architect",
        title="Email urgency classification",
        business_objective="Classify incoming emails by urgency and store the result.",
        desired_outcome="Persist a first workflow draft that captures, classifies, and stores email urgency.",
        workflow_summary="Three-stage workflow from ingestion to classification to persistence.",
        stages=[
            ArchitectureStage(
                id="stage_1",
                name="Email Intake",
                purpose="Receive new incoming emails from the mail provider.",
                required_capabilities=["Start from inbound email events"],
                expected_inputs=["incoming email"],
                expected_outputs=["normalized email payload"],
                dependencies=[],
                success_criteria=["The workflow starts reliably for each incoming email."],
            ),
            ArchitectureStage(
                id="stage_2",
                name="Urgency Classification",
                purpose="Classify each email into urgency levels.",
                required_capabilities=["Apply urgency logic"],
                expected_inputs=["normalized email payload"],
                expected_outputs=["urgency label"],
                dependencies=["stage_1"],
                success_criteria=["A single urgency label is produced."],
            ),
            ArchitectureStage(
                id="stage_3",
                name="Store Result",
                purpose="Persist the resulting urgency label for later use.",
                required_capabilities=["Store or append the classification result"],
                expected_inputs=["urgency label"],
                expected_outputs=["stored classification"],
                dependencies=["stage_2"],
                success_criteria=["The classified email is stored."],
            ),
        ],
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_1",
                target_stage_id="stage_2",
                data_items=["normalized email payload"],
            ),
            ArchitectureDataFlowItem(
                source_stage_id="stage_2",
                target_stage_id="stage_3",
                data_items=["urgency label"],
            ),
        ],
        assumptions=[],
        missing_information=[],
        implementation_notes_for_engineer=[],
        required_nodes=[],
    )


def _state(*, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "user_query": "Crea un workflow para leer emails y clasificarlos por urgencia.",
        "entry_intent": EntryIntent.workflow_build_request,
        "current_stage": "product_manager_agent",
        "target_stage": None,
        "routing_signals": ["handoff_ready_architect"],
        "architecture_plan": _plan(),
        "workflow_context": WorkflowContext(
            use_case_id="uc_architect",
            planning_ready=True,
            handoff_target=AgentStage.architect_agent,
            required_node_types=[],
            unresolved_inputs=[],
            notes=["abstract_plan_only"],
        ),
        "architect_status": None,
        "architect_stage_search_history": [],
        "architect_stage_selections": [],
        "architect_clarification_state": None,
        "architect_notes": [],
        "proposed_nodes": [],
        "workflow_draft": None,
        "workflow_versions": [],
        "final_workflow_json": {},
        "missing_user_inputs": [],
        "missing_user_input_details": [],
        "runtime_context": {"model": None, "request_id": "req-architect", "persist_to_n8n": True},
        "resume_requested": False,
    }
    if extra:
        state.update(extra)
    return state


def _candidate(
    node_type: str,
    *,
    stage_id: str,
    usage_mode: str = "action_only",
    has_main_input: bool | None = True,
    usable_as_tool: bool | None = False,
    capability_summary: str = "",
    type_version: int = 1,
    input_connection_types: List[str] | None = None,
    output_connection_types: List[str] | None = None,
) -> ArchitectNodeCandidate:
    return ArchitectNodeCandidate(
        node_type=node_type,
        display_name=node_type.split(".")[-1],
        stage_id=stage_id,
        capability_summary=capability_summary or f"Capabilities for {node_type}",
        limitations=[],
        rationale=f"Use {node_type} for {stage_id}",
        usage_mode=usage_mode,  # type: ignore[arg-type]
        usable_as_tool=usable_as_tool,
        has_main_input=has_main_input,
        input_connection_types=(
            input_connection_types
            if input_connection_types is not None
            else (["main"] if has_main_input else [])
        ),
        output_connection_types=output_connection_types if output_connection_types is not None else ["main"],
        evidence_chunk_ids=[f"chunk:{stage_id}:{node_type}"],
        evidence_refs=[f"ref:{node_type}"],
        rerank_confidence=0.8,
        link_confidence=0.7,
        type_version=type_version,
    )


def test_architect_normalizes_raw_rerank_scores_above_one() -> None:
    page_key, _, _ = derive_doc_page_key(
        {},
        row_url="https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/",
    )
    docs_chunks = [
        {
            "doc_id": "doc-1",
            "url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/",
            "metadata": {},
            "rerank_score": 2.020512819290161,
            "text": "Webhook node docs",
        }
    ]
    linked_chunks = [
            {
                "doc_id": "linked-1",
                "linked_def_type": "node",
                "linked_entity_id": "n8n-nodes-base.webhook",
                "link_doc_page_key": page_key,
                "link_confidence": 0.82,
                "metadata": {"displayName": "Webhook", "version": 2},
            "text": (
                "Display Name: Webhook\n"
                "Description: Receive inbound HTTP events.\n"
                "Inputs: \n"
                "Outputs: main\n"
                "Usable As Tool: false\n"
                "Version: 2\n"
            ),
            "url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/",
        }
    ]

    candidates = _build_candidates(
        stage_id="stage_1",
        docs_chunks=docs_chunks,
        linked_chunks=linked_chunks,
    )

    assert _normalize_confidence(2.020512819290161) is not None
    assert len(candidates) == 1
    assert candidates[0].rerank_confidence is not None
    assert 0.0 <= candidates[0].rerank_confidence <= 1.0


def test_architect_stage_selection_returns_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._invoke_structured_output",
        lambda **_kwargs: _StageSelectionOutput(
            selected_node_types=["n8n-nodes-base.webhook"],
            rationale="Webhook best matches the intake stage.",
            needs_clarification=False,
            clarification_questions=[],
        ),
    )

    output = _select_stage_nodes_with_structured_output(
        plan=_plan(),
        stage=_plan().stages[0],
        candidates=[_candidate("n8n-nodes-base.webhook", stage_id="stage_1", has_main_input=False)],
        previous_selections=[],
        stage_requires_trigger=True,
        model="fake-model",
        request_id="req-architect-selector",
    )

    assert output is not None
    assert output.selected_node_types == ["n8n-nodes-base.webhook"]


def test_architect_rejects_trigger_nodes_for_classification_stage() -> None:
    plan = _plan()
    stage = plan.stages[1]
    candidate = _candidate(
        "n8n-nodes-base.postmarkTrigger",
        stage_id=stage.id,
        has_main_input=False,
        capability_summary="Trigger the workflow when a new Postmark email event is received.",
    )

    reasons = _candidate_rejection_reasons(
        candidate=candidate,
        stage=stage,
        plan=plan,
        stage_requires_trigger=False,
    )

    assert "trigger_invalid_for_classification" in reasons
    assert "not_a_classification_node" in reasons


def test_architect_rejects_trigger_nodes_for_apply_label_stage() -> None:
    plan = ArchitecturePlan(
        use_case_id="uc_apply_label",
        title="Apply Gmail labels",
        business_objective="Apply urgency labels back to Gmail.",
        desired_outcome="Label the original Gmail message.",
        workflow_summary="Receive, classify, and apply a visible label in Gmail.",
        stages=[
            ArchitectureStage(
                id="stage_apply",
                name="Apply Urgency Label in Gmail",
                purpose="Apply the resulting urgency level back onto the same Gmail message as a visible label.",
                stage_kind=StageKind.apply_update_source,
                business_effect="The source email is updated with the workflow result.",
                target_entity="gmail_message",
                user_visible_goal="The urgency label is visible in Gmail.",
                required_capabilities=["Add a visible label to the Gmail message"],
                expected_inputs=["message id", "urgency label"],
                expected_outputs=["labeled message"],
                dependencies=[],
                success_criteria=["The Gmail message is updated with the correct label."],
            )
        ],
        data_flow=[],
        assumptions=[],
        missing_information=[],
        implementation_notes_for_engineer=[],
        required_nodes=[],
    )
    stage = plan.stages[0]
    candidate = _candidate(
        "n8n-nodes-base.emailReadImap",
        stage_id=stage.id,
        has_main_input=False,
        capability_summary="Triggers the workflow when a new email is received. Can mark messages as read.",
    )

    reasons = _candidate_rejection_reasons(
        candidate=candidate,
        stage=stage,
        plan=plan,
        stage_requires_trigger=False,
    )

    assert "trigger_invalid_for_apply_update" in reasons


def test_source_update_stage_rejects_generic_wait_even_with_contaminated_summary() -> None:
    plan = ArchitecturePlan(
        use_case_id="uc_apply_label",
        title="Apply Gmail labels",
        business_objective="Apply urgency labels back to Gmail.",
        desired_outcome="Label Gmail messages by urgency.",
        workflow_summary="Receive, classify, and apply a visible label in Gmail.",
        stages=[
            ArchitectureStage(
                id="stage_apply",
                name="Apply Urgency Label in Gmail",
                purpose="Apply the resulting urgency level back onto the same Gmail message as a visible label.",
                stage_kind=StageKind.apply_update_source,
                business_effect="Update the Gmail message with the urgency result.",
                target_entity="gmail_message",
                user_visible_goal="The urgency label is visible in Gmail.",
                required_capabilities=["Add a visible label to the Gmail message"],
                expected_inputs=["message id", "urgency label"],
                expected_outputs=["labeled message"],
                dependencies=[],
                success_criteria=["The Gmail message is updated with the correct label."],
            )
        ],
    )
    candidate = _candidate(
        "n8n-nodes-base.wait",
        stage_id="stage_apply",
        capability_summary=(
            "Wait before continue with execution. Gmail node Message Operations / Add Label to a message."
        ),
    )

    reasons = _candidate_rejection_reasons(
        candidate=candidate,
        stage=plan.stages[0],
        plan=plan,
        stage_requires_trigger=False,
    )

    assert "not_a_source_update_node" in reasons


def test_definition_fallback_replaces_invalid_same_type_candidate_for_gmail_apply_stage() -> None:
    plan = ArchitecturePlan(
        use_case_id="uc_apply_label",
        title="Apply Gmail labels",
        business_objective="Apply urgency labels back to Gmail.",
        desired_outcome="Label Gmail messages by urgency.",
        workflow_summary="Receive, classify, and apply a visible label in Gmail.",
        stages=[
            ArchitectureStage(
                id="stage_apply",
                name="Apply Urgency Label in Gmail",
                purpose="Apply the resulting urgency level back onto the same Gmail message as a visible label.",
                stage_kind=StageKind.apply_update_source,
                business_effect="Update the Gmail message with the urgency result.",
                target_entity="gmail_message",
                user_visible_goal="The urgency label is visible in Gmail.",
                required_capabilities=["Add a visible label to the Gmail message"],
                expected_inputs=["message id", "urgency label"],
                expected_outputs=["labeled message"],
                dependencies=[],
                success_criteria=["The Gmail message is updated with the correct label."],
            )
        ],
    )
    invalid_gmail = _candidate(
        "n8n-nodes-base.gmail",
        stage_id="stage_apply",
        capability_summary="Generic Gmail node summary without a concrete update action.",
        output_connection_types=[],
    )

    augmented = _augment_candidates_with_definition_fallbacks(
        stage=plan.stages[0],
        plan=plan,
        stage_requires_trigger=False,
        candidates=[invalid_gmail],
    )

    gmail_candidates = [item for item in augmented if item.node_type == "n8n-nodes-base.gmail"]
    assert len(gmail_candidates) == 1
    reasons = _candidate_rejection_reasons(
        candidate=gmail_candidates[0],
        stage=plan.stages[0],
        plan=plan,
        stage_requires_trigger=False,
    )
    assert reasons == []


def test_architect_refines_terminal_outcome_plan_from_resolved_slot() -> None:
    plan = ArchitecturePlan(
        use_case_id="uc_architect_refine",
        title="Email urgency classification",
        business_objective="Classify incoming emails by urgency.",
        desired_outcome="Classify incoming emails by urgency.",
        workflow_summary="Receive each email, classify it, and then handle the result.",
        stages=[
            ArchitectureStage(
                id="stage_1",
                name="Email Intake",
                purpose="Receive new incoming emails.",
                stage_kind=StageKind.trigger_intake,
                required_capabilities=["Receive inbound email"],
                expected_inputs=["incoming email"],
                expected_outputs=["normalized email payload"],
                dependencies=[],
                success_criteria=["Each email starts the workflow."],
            ),
            ArchitectureStage(
                id="stage_2",
                name="Urgency Classification",
                purpose="Classify the email by urgency.",
                stage_kind=StageKind.classify_decision,
                required_capabilities=["Assign urgency"],
                expected_inputs=["normalized email payload"],
                expected_outputs=["urgency label"],
                dependencies=["stage_1"],
                success_criteria=["A single urgency label is produced."],
            ),
            ArchitectureStage(
                id="stage_3",
                name="Outcome Handling",
                purpose="Persist, notify, or otherwise record the final outcome of the workflow.",
                stage_kind=StageKind.transform_process,
                required_capabilities=["Execute the final outcome step"],
                expected_inputs=["urgency label"],
                expected_outputs=["stored outcome or outbound action result"],
                dependencies=["stage_2"],
                success_criteria=["The final outcome is executed and traceable."],
            ),
        ],
        data_flow=[
            ArchitectureDataFlowItem(source_stage_id="stage_1", target_stage_id="stage_2", data_items=["normalized email payload"]),
            ArchitectureDataFlowItem(source_stage_id="stage_2", target_stage_id="stage_3", data_items=["urgency label"]),
        ],
        assumptions=[],
        missing_information=[],
        implementation_notes_for_engineer=[],
        required_nodes=[],
    )
    clarification_state = ArchitectClarificationState(
        resolved_slots=[
            {
                "slot_key": "result_application_mode",
                "owner_agent": "architect_agent",
                "stage_id": "plan_terminal_outcome",
                "question_text": "What should happen with that result next?",
                "question_intent": "result_application_mode",
                "answer_status": "resolved",
                "answer": "Aplicalo al mismo correo de Gmail como una etiqueta visible para poder filtrarlo despues.",
            }
        ]
    )

    refined = _refine_terminal_outcome_plan_from_slots(
        plan=plan,
        clarification_state=clarification_state,
    )

    assert refined.stages[-1].stage_kind == StageKind.apply_update_source
    assert refined.stages[-1].target_entity == "gmail_message"
    assert "label" in (refined.stages[-1].purpose or "").lower()
    assert _plan_terminal_outcome_gap_question(refined) is None


def test_architect_stage_selection_prompt_includes_recent_upstream_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _StageSelectionOutput(
            selected_node_types=["n8n-nodes-base.code"],
            rationale="Use code after the upstream trigger.",
            needs_clarification=False,
            clarification_questions=[],
        )

    monkeypatch.setattr("app.graphs.nodes.architect_agent._invoke_structured_output", _capture)

    previous_selection = ArchitectStageSelection(
        stage_id="stage_1",
        selected_node_types=["n8n-nodes-base.gmailTrigger"],
        selected_nodes=[
            _candidate(
                "n8n-nodes-base.gmailTrigger",
                stage_id="stage_1",
                has_main_input=False,
                capability_summary="Receive inbound emails from Gmail.",
            )
        ],
        rationale="Inbound email trigger",
    )

    _select_stage_nodes_with_structured_output(
        plan=_plan(),
        stage=_plan().stages[1],
        candidates=[_candidate("n8n-nodes-base.code", stage_id="stage_2")],
        previous_selections=[previous_selection],
        stage_requires_trigger=False,
        model="fake-model",
        request_id="req-architect-upstream-context",
    )

    prompt = str(captured["user_prompt"])
    assert "Recent upstream selected nodes and connectors" in prompt
    assert "n8n-nodes-base.gmailTrigger" in prompt
    assert "inputs=['-']" in prompt or "inputs=[]" in prompt


def test_architect_prompts_emphasize_structural_role_and_id_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: List[Dict[str, Any]] = []

    def _capture(**kwargs):
        captured.append(kwargs)
        output_model = kwargs["output_model"]
        if output_model is _StageSelectionOutput:
            return _StageSelectionOutput(
                selected_node_types=["n8n-nodes-base.gmailTrigger"],
                rationale="Selected by trigger role.",
                needs_clarification=False,
                clarification_questions=[],
            )
        return _WorkflowBlueprintOutput(
            workflow_name="Email urgency classification",
            summary="Architect draft",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="gmail_trigger_1",
                    node_type="n8n-nodes-base.gmailTrigger",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive emails",
                    depends_on=[],
                )
            ],
            connections=[],
        )

    monkeypatch.setattr("app.graphs.nodes.architect_agent._invoke_structured_output", _capture)

    _select_stage_nodes_with_structured_output(
        plan=_plan(),
        stage=_plan().stages[0],
        candidates=[_candidate("n8n-nodes-base.gmailTrigger", stage_id="stage_1", has_main_input=False)],
        previous_selections=[],
        stage_requires_trigger=True,
        model="fake-model",
        request_id="req-architect-prompt-selection",
    )
    _build_workflow_blueprint_with_structured_output(
        plan=_plan(),
        stage_selections=[
            ArchitectStageSelection(
                stage_id="stage_1",
                selected_node_types=["n8n-nodes-base.gmailTrigger"],
                selected_nodes=[_candidate("n8n-nodes-base.gmailTrigger", stage_id="stage_1", has_main_input=False)],
                rationale="Selected by trigger role.",
            )
        ],
        model="fake-model",
        request_id="req-architect-prompt-blueprint",
    )

    selection_prompt = str(captured[0]["user_prompt"])
    blueprint_prompt = str(captured[1]["user_prompt"])
    assert "Decide by functional role, not just semantic similarity." in selection_prompt
    assert "Reject provider-only or infrastructure-only nodes when the stage needs a complete business operation node." in selection_prompt
    assert "Never reference a source_node_id or target_node_id that is not present in the returned nodes list." in blueprint_prompt
    assert "Allowed blueprint nodes (use these exact ids and names in the output):" in blueprint_prompt
    assert "Never use placeholder ids such as '-', '', null, none, output, end, terminal, or similar." in blueprint_prompt
    assert "perform a private self-check" in blueprint_prompt


def test_architect_normalizes_blueprint_ids_and_strips_invalid_placeholder_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture(**kwargs):
        output_model = kwargs["output_model"]
        if output_model is not _WorkflowBlueprintOutput:
            raise AssertionError("unexpected output model")
        return _WorkflowBlueprintOutput(
            workflow_name="Email urgency classification",
            summary="Architect draft",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="cronTrigger",
                    name="Cron Trigger",
                    node_type="n8n-nodes-base.gmailTrigger",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive emails",
                    depends_on=["-"],
                ),
                _WorkflowNodeBlueprint(
                    node_id="classify",
                    name="Classifier",
                    node_type="n8n-nodes-base.code",
                    type_version=2,
                    stage_id="stage_2",
                    purpose="Classify urgency",
                    depends_on=["cronTrigger"],
                ),
            ],
            connections=[
                _WorkflowConnectionBlueprint(source_node_id="cronTrigger", target_node_id="classify"),
                _WorkflowConnectionBlueprint(source_node_id="classify", target_node_id="-"),
            ],
        )

    monkeypatch.setattr("app.graphs.nodes.architect_agent._invoke_structured_output", _capture)

    output = _build_workflow_blueprint_with_structured_output(
        plan=_plan(),
        stage_selections=[
            ArchitectStageSelection(
                stage_id="stage_1",
                selected_node_types=["n8n-nodes-base.gmailTrigger"],
                selected_nodes=[_candidate("n8n-nodes-base.gmailTrigger", stage_id="stage_1", has_main_input=False)],
                rationale="Receive emails",
            ),
            ArchitectStageSelection(
                stage_id="stage_2",
                selected_node_types=["n8n-nodes-base.code"],
                selected_nodes=[_candidate("n8n-nodes-base.code", stage_id="stage_2", type_version=2)],
                rationale="Classify urgency",
            ),
        ],
        model="fake-model",
        request_id="req-architect-blueprint-normalize",
    )

    assert [node.node_id for node in output.nodes] == ["stage_1_gmailtrigger", "stage_2_code"]
    assert [node.name for node in output.nodes] == ["gmailTrigger", "code"]
    assert output.nodes[0].depends_on == []
    assert output.nodes[1].depends_on == ["stage_1_gmailtrigger"]
    assert len(output.connections) == 1
    assert output.connections[0].source_node_id == "stage_1_gmailtrigger"
    assert output.connections[0].target_node_id == "stage_2_code"


def test_architect_parses_malformed_connector_fragments_into_clean_connector_types() -> None:
    page_key, _, _ = derive_doc_page_key(
        {},
        row_url="https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.textclassifier/",
    )
    candidates = _build_candidates(
        stage_id="stage_2",
        docs_chunks=[],
        linked_chunks=[
            {
                "doc_id": "linked-connector-1",
                "linked_def_type": "node",
                "linked_entity_id": "@n8n/n8n-nodes-langchain.textClassifier",
                "link_doc_page_key": page_key,
                "metadata": {"displayName": "Text Classifier", "version": 1},
                "text": (
                    "Display Name: Text Classifier\n"
                    "Description: Classify text using an AI model.\n"
                    "Inputs: \"{'displayName': ''\", \"'type': 'main'}\", \"'type': 'ai_languageModel'\"\n"
                    "Outputs: \"'type': 'main'\"\n"
                    "Usable As Tool: false\n"
                ),
                "url": "https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.textclassifier/",
            }
        ],
    )

    assert candidates[0].input_connection_types == ["main", "ai_languagemodel"]
    assert candidates[0].output_connection_types == ["main"]


def test_architect_rejects_outbound_send_node_as_trigger_candidate() -> None:
    candidate = _candidate(
        "n8n-nodes-base.emailSend",
        stage_id="stage_1",
        has_main_input=False,
        capability_summary="Send outgoing emails to recipients.",
    )

    assert _is_trigger_candidate(candidate) is False


def test_architect_rejects_non_main_ai_connectors_for_v1_rule_based_stage() -> None:
    plan = _plan().model_copy(
        update={
            "workflow_summary": "Receive incoming emails and classify them with heuristics only.",
            "business_objective": "Apply rule-based urgency heuristics to incoming emails.",
        }
    )
    stage = plan.stages[1].model_copy(
        update={
            "purpose": "Classify urgency using heuristic and rule-based logic.",
            "required_capabilities": ["Apply heuristic rules"],
        }
    )
    candidate = _candidate(
        "@n8n/n8n-nodes-langchain.textClassifier",
        stage_id=stage.id,
        has_main_input=True,
        capability_summary="AI text classifier for categorization.",
        input_connection_types=["main", "ai_languageModel"],
    )

    reasons = _candidate_rejection_reasons(
        candidate=candidate,
        stage=stage,
        plan=plan,
        stage_requires_trigger=False,
    )

    assert "requires_non_main_input_connectors" in reasons
    assert "rule_based_stage_rejects_ai_candidate" in reasons


def test_architect_linking_page_selection_preserves_specific_node_docs() -> None:
    stage = _plan().stages[0]
    docs_chunks = [
        {
            "doc_id": "doc-imap",
            "title": "Email Trigger (IMAP)",
            "url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.emailreadimap/",
            "metadata": {},
            "rerank_score": 0.98,
            "text": "Email Trigger (IMAP) receives incoming emails from IMAP inboxes.",
        },
        {
            "doc_id": "doc-privacy",
            "title": "Privacy",
            "url": "https://docs.n8n.io/privacy/",
            "metadata": {},
            "rerank_score": 0.97,
            "text": "Privacy policy and generic documentation.",
        },
        {
            "doc_id": "doc-tutorial",
            "title": "AI Workflow Builder",
            "url": "https://docs.n8n.io/advanced-ai/ai-workflow-builder/",
            "metadata": {},
            "rerank_score": 0.96,
            "text": "Generic AI tutorial page unrelated to inbound email triggers.",
        },
    ]

    selected_page_keys, selected_pages, _discarded = _select_linking_page_keys(
        docs_chunks=docs_chunks,
        user_query="Consume incoming Gmail emails as they arrive.",
        plan=_plan(),
        stage=stage,
        previous_selections=[],
        stage_requires_trigger=True,
    )

    assert selected_page_keys
    assert "emailreadimap" in selected_pages[0]["page_key"]


def test_architect_rejects_generic_scheduler_for_source_specific_trigger_stage() -> None:
    plan = _plan()
    stage = plan.stages[0]
    candidate = _candidate(
        "n8n-nodes-base.cron",
        stage_id=stage.id,
        has_main_input=False,
        capability_summary="Schedule workflow execution on time-based intervals.",
    )

    reasons = _candidate_rejection_reasons(
        candidate=candidate,
        stage=stage,
        plan=plan,
        stage_requires_trigger=True,
    )

    assert "generic_scheduler_rejected_for_source_specific_trigger" in reasons


def test_architect_blocks_when_plan_leaves_terminal_business_outcome_undefined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan().model_copy(
        update={
            "business_objective": "Classify incoming emails by urgency.",
            "desired_outcome": "Classify incoming emails by urgency.",
            "workflow_summary": "Two-stage workflow that receives emails and classifies urgency.",
            "stages": _plan().stages[:2],
            "data_flow": _plan().data_flow[:1],
        }
    )

    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not retrieve")),
    )

    updates = architect_agent_node(
        _state(
            extra={
                "architecture_plan": plan,
                "user_query": "Crea un workflow que reciba emails y los clasifique por urgencia",
            }
        )
    )

    assert updates["architect_status"] == ArchitectStatus.architect_blocked_waiting_user
    assert updates["missing_user_inputs"]
    assert "what should happen with that result" in updates["missing_user_inputs"][0].lower()
    assert updates["missing_user_input_details"][0].missing_item == "plan_terminal_outcome"


def test_architect_blocks_when_apply_update_stage_still_lacks_concrete_action(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan().model_copy(
        update={
            "stages": [
                _plan().stages[0],
                ArchitectureStage(
                    id="stage_2",
                    name="Filter Urgent Emails in Gmail",
                    purpose="Make urgent emails easy to filter in Gmail later.",
                    stage_kind=StageKind.apply_update_source,
                    business_effect="Keep urgent emails visible for later filtering in Gmail.",
                    target_entity="gmail",
                    user_visible_goal="See urgent emails in Gmail filters.",
                    required_capabilities=["Apply the result back into Gmail"],
                    expected_inputs=["normalized email payload", "urgency label"],
                    expected_outputs=["updated Gmail message"],
                    dependencies=["stage_1"],
                    success_criteria=["Urgent emails are easy to find in Gmail."],
                ),
            ],
            "data_flow": [
                ArchitectureDataFlowItem(
                    source_stage_id="stage_1",
                    target_stage_id="stage_2",
                    data_items=["normalized email payload", "urgency label"],
                )
            ],
        }
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/gmail"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])

    def _build_candidates(*, stage_id: str, **_kwargs):
        if stage_id == "stage_1":
            return [_candidate("n8n-nodes-base.gmailTrigger", stage_id=stage_id, has_main_input=False)]
        return [
            _candidate(
                "n8n-nodes-base.gmail",
                stage_id=stage_id,
                capability_summary="Update Gmail messages by adding labels or changing message status.",
            )
        ]

    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", _build_candidates)
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._select_stage_nodes_with_structured_output",
        lambda *, candidates, **_kwargs: _StageSelectionOutput(
            selected_node_types=[candidates[0].node_type],
            rationale="selected",
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_workflow_blueprint_with_structured_output",
        lambda **_kwargs: _WorkflowBlueprintOutput(
            workflow_name="Ambiguous Gmail Update",
            summary="Architect draft",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="gmail_trigger_1",
                    node_type="n8n-nodes-base.gmailTrigger",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive emails",
                    depends_on=[],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_2",
                    name="gmail_2",
                    node_type="n8n-nodes-base.gmail",
                    type_version=1,
                    stage_id="stage_2",
                    purpose="Apply result in Gmail",
                    depends_on=["an_1"],
                ),
            ],
            connections=[_WorkflowConnectionBlueprint(source_node_id="an_1", target_node_id="an_2")],
        ),
    )

    updates = architect_agent_node(
        _state(
            extra={
                "architecture_plan": plan,
                "runtime_context": {"model": None, "request_id": "req-architect", "persist_to_n8n": False},
            }
        )
    )

    assert updates["architect_status"] == ArchitectStatus.architect_blocked_waiting_user
    assert updates["missing_user_inputs"]
    assert "etiquetar" in updates["missing_user_inputs"][0].lower() or "archivar" in updates["missing_user_inputs"][0].lower()


def test_architect_attaches_operation_hints_for_generic_source_update_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan().model_copy(
        update={
            "stages": [
                _plan().stages[0],
                ArchitectureStage(
                    id="stage_2",
                    name="Apply Urgency Label",
                    purpose="Apply an urgency label to the Gmail message so users can filter it later.",
                    stage_kind=StageKind.apply_update_source,
                    business_effect="Add the urgency label to the Gmail message.",
                    target_entity="gmail",
                    user_visible_goal="Users can filter Gmail by urgency label.",
                    required_capabilities=["Update the source Gmail message"],
                    expected_inputs=["normalized email payload", "urgency label"],
                    expected_outputs=["Gmail message updated with urgency label"],
                    dependencies=["stage_1"],
                    success_criteria=["The Gmail message receives the urgency label."],
                ),
            ],
            "data_flow": [
                ArchitectureDataFlowItem(
                    source_stage_id="stage_1",
                    target_stage_id="stage_2",
                    data_items=["normalized email payload", "urgency label"],
                )
            ],
        }
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/gmail"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])

    def _build_candidates(*, stage_id: str, **_kwargs):
        if stage_id == "stage_1":
            return [_candidate("n8n-nodes-base.gmailTrigger", stage_id=stage_id, has_main_input=False)]
        return [
            _candidate(
                "n8n-nodes-base.gmail",
                stage_id=stage_id,
                capability_summary="Update Gmail messages by adding labels or changing message status.",
            )
        ]

    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", _build_candidates)
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._select_stage_nodes_with_structured_output",
        lambda *, candidates, **_kwargs: _StageSelectionOutput(
            selected_node_types=[candidates[0].node_type],
            rationale="selected",
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_workflow_blueprint_with_structured_output",
        lambda **_kwargs: _WorkflowBlueprintOutput(
            workflow_name="Gmail Label Flow",
            summary="Architect draft",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="gmail_trigger_1",
                    node_type="n8n-nodes-base.gmailTrigger",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive emails",
                    depends_on=[],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_2",
                    name="gmail_2",
                    node_type="n8n-nodes-base.gmail",
                    type_version=1,
                    stage_id="stage_2",
                    purpose="Apply urgency label",
                    depends_on=["an_1"],
                ),
            ],
            connections=[_WorkflowConnectionBlueprint(source_node_id="an_1", target_node_id="an_2")],
        ),
    )

    updates = architect_agent_node(
        _state(
            extra={
                "architecture_plan": plan,
                "runtime_context": {"model": None, "request_id": "req-architect", "persist_to_n8n": False},
            }
        )
    )

    assert updates["architect_status"] == ArchitectStatus.architect_completed
    assert updates["proposed_nodes"][1].implementation_hints["semantic_action"] == "apply_label"
    assert updates["proposed_nodes"][1].implementation_hints["require_action_selection"] is True
    assert updates["workflow_draft"].nodes[1].implementation_hints["preferred_resource"] == "message"


def test_architect_builds_and_persists_new_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _retrieve(*_args, **_kwargs):
        return [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/gmail"}]

    monkeypatch.setattr("app.graphs.nodes.architect_agent.retrieve_context", _retrieve)
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])

    def _build_candidates(*, stage_id: str, **_kwargs):
        if stage_id == "stage_1":
            return [_candidate("n8n-nodes-base.gmailTrigger", stage_id=stage_id, has_main_input=False)]
        if stage_id == "stage_2":
            return [
                _candidate(
                    "n8n-nodes-base.code",
                    stage_id=stage_id,
                    type_version=2,
                    capability_summary="Use AI or code-based logic to classify urgency.",
                )
            ]
        return [
            _candidate(
                "n8n-nodes-base.googleSheets",
                stage_id=stage_id,
                capability_summary="Store workflow results in a spreadsheet table.",
            )
        ]

    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", _build_candidates)
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._select_stage_nodes_with_structured_output",
        lambda *, candidates, **_kwargs: _StageSelectionOutput(
            selected_node_types=[candidates[0].node_type],
            rationale=f"Selected {candidates[0].node_type}",
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_workflow_blueprint_with_structured_output",
        lambda **_kwargs: _WorkflowBlueprintOutput(
            workflow_name="Email urgency classification",
            summary="Architect draft",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="gmail_trigger_1",
                    node_type="n8n-nodes-base.gmailTrigger",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive emails",
                    depends_on=[],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_2",
                    name="code_2",
                    node_type="n8n-nodes-base.code",
                    type_version=2,
                    stage_id="stage_2",
                    purpose="Classify urgency",
                    depends_on=["an_1"],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_3",
                    name="google_sheets_3",
                    node_type="n8n-nodes-base.googleSheets",
                    type_version=1,
                    stage_id="stage_3",
                    purpose="Store result",
                    depends_on=["an_2"],
                ),
            ],
            connections=[
                _WorkflowConnectionBlueprint(source_node_id="an_1", target_node_id="an_2"),
                _WorkflowConnectionBlueprint(source_node_id="an_2", target_node_id="an_3"),
            ],
        ),
    )

    def _create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert payload["connections"]["gmail_trigger_1"]["main"][0][0]["node"] == "code_2"
        return {
            "id": "wf_arch_1",
            "name": payload.get("name"),
            "url": "http://localhost:5678/workflow/wf_arch_1",
        }

    monkeypatch.setattr("app.graphs.nodes.engineer_agent.N8NClient.create_workflow", _create)

    updates = architect_agent_node(_state())

    assert updates["current_stage"] == "architect_agent"
    assert updates["architect_status"] == ArchitectStatus.architect_completed
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert updates["workflow_persisted"] is True
    assert updates["workflow_persist_action"] == "created"
    assert updates["active_workflow_id"] == "wf_arch_1"
    assert updates["final_workflow_json"]["nodes"]
    assert updates["workflow_draft"].nodes[1].type_version == 2
    assert len(updates["proposed_nodes"]) == 3
    assert updates["workflow_context"].handoff_target == AgentStage.engineer_agent


def test_architect_prefers_explicit_model_node_for_ai_classification() -> None:
    plan = _plan()
    stage = plan.stages[1].model_copy(
        update={
            "stage_kind": StageKind.classify_decision,
            "purpose": "Use an AI model to classify urgency semantically from subject and body.",
        }
    )
    openai_candidate = _candidate(
        "n8n-nodes-base.openAi",
        stage_id=stage.id,
        capability_summary="OpenAI model call for text analysis and classification.",
    )
    ai_transform_candidate = _candidate(
        "n8n-nodes-base.aiTransform",
        stage_id=stage.id,
        capability_summary="Generic AI data transform node.",
    )

    openai_score = architect_mod._fallback_candidate_score(
        candidate=openai_candidate,
        stage=stage,
        plan=plan,
        stage_requires_trigger=False,
    )
    ai_transform_score = architect_mod._fallback_candidate_score(
        candidate=ai_transform_candidate,
        stage=stage,
        plan=plan,
        stage_requires_trigger=False,
    )

    assert openai_score > ai_transform_score


def test_architect_operation_hints_capture_label_contract() -> None:
    plan = _plan().model_copy(
        update={
            "workflow_summary": "Incoming Gmail urgency classification with labels bajo, medio, alto, critico and fallback Review.",
            "implementation_notes_for_engineer": [
                "Applied labels must stay visible in Gmail UI.",
                "Use Review when confidence is low.",
            ],
        }
    )
    stage = ArchitectureStage(
        id="stage_apply",
        name="Apply Urgency Label in Gmail",
        purpose="Apply the urgency label back to the same Gmail message.",
        stage_kind=StageKind.apply_update_source,
        target_entity="gmail_message",
        user_visible_goal="Visible in Gmail UI for filtering later.",
        required_capabilities=["Apply label"],
        expected_inputs=["Urgency classification result", "Source email identifiers"],
        expected_outputs=["Updated Gmail message"],
        dependencies=["stage_2"],
        success_criteria=["The message shows the chosen urgency label in Gmail."],
        notes="Use labels bajo, medio, alto, critico. If uncertain use Review.",
    )
    candidate = _candidate(
        "n8n-nodes-base.gmail",
        stage_id=stage.id,
        capability_summary="Update Gmail messages and labels.",
    )

    hints = architect_mod._derive_operation_hints(stage=stage, candidate=candidate, plan=plan)

    assert hints["semantic_action"] == "apply_label"
    assert hints["preferred_resource"] == "message"
    assert hints["allowed_label_values"] == ["bajo", "medio", "alto", "critico"]
    assert hints["fallback_label_name"] == "Review"
    assert "messageId" in hints["parameter_focus"]


def test_architect_operation_hints_capture_gmail_trigger_polling_contract() -> None:
    plan = _plan().model_copy(
        update={
            "workflow_summary": "Capture every new Gmail email and classify it by urgency.",
            "implementation_notes_for_engineer": ["Use near-real-time intake for each new Gmail email."],
        }
    )
    stage = ArchitectureStage(
        id="stage_trigger",
        name="Receive Gmail Emails",
        purpose="Capture each new Gmail email as soon as the native trigger allows.",
        stage_kind=StageKind.trigger_intake,
        business_effect="Capture incoming Gmail messages for downstream processing.",
        target_entity="gmail_message",
        user_visible_goal="Run automatically for each new email.",
        required_capabilities=["Trigger on each Gmail email received."],
        expected_inputs=["Gmail inbox"],
        expected_outputs=["Email subject and body"],
        dependencies=[],
        success_criteria=["Each new email starts the workflow promptly."],
    )
    candidate = _candidate("n8n-nodes-base.gmailTrigger", stage_id=stage.id, has_main_input=False)

    hints = architect_mod._derive_operation_hints(stage=stage, candidate=candidate, plan=plan)

    assert hints["semantic_action"] == "receive_incoming_item"
    assert hints["native_trigger_mechanism"] == "polling"
    assert hints["preferred_poll_mode"] == "everyMinute"
    assert "pollTimes.item.mode" in hints["parameter_focus"]
    assert "pollTimes.item.mode" in hints["allow_inferred_parameter_keys"]


def test_architect_operation_hints_capture_openai_classifier_contract() -> None:
    plan = _plan().model_copy(
        update={
            "workflow_summary": "Use AI to classify Gmail email urgency as low, medium, high, or critical.",
            "implementation_notes_for_engineer": [
                "Urgency levels must remain exactly: low, medium, high, critical.",
                "If the workflow cannot determine a confident urgency level, use fallback label 'Review'.",
            ],
        }
    )
    stage = ArchitectureStage(
        id="stage_classify",
        name="Classify Email Urgency",
        purpose="Use an AI model to classify email urgency from subject and body.",
        stage_kind=StageKind.classify_decision,
        business_effect="Produce one urgency level for each email.",
        target_entity="email",
        user_visible_goal="Every email has one urgency label.",
        required_capabilities=["Use AI to classify email urgency semantically."],
        expected_inputs=["Email subject and body"],
        expected_outputs=["Urgency label"],
        dependencies=["stage_trigger"],
        success_criteria=["Exactly one urgency label is returned for each email."],
    )
    candidate = _candidate("n8n-nodes-base.openAi", stage_id=stage.id)

    hints = architect_mod._derive_operation_hints(stage=stage, candidate=candidate, plan=plan)

    assert hints["semantic_action"] == "classify_payload"
    assert hints["classification_output_key"] == "urgency_level"
    assert hints["fallback_label_name"] == "Review"
    assert "model" in hints["parameter_focus"]
    assert "prompt" in hints["parameter_focus"]
    assert "model" in hints["allow_inferred_parameter_keys"]


def test_architect_blocks_when_only_tool_candidates_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/agent"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_candidates",
        lambda *, stage_id, **_kwargs: [
            _candidate(
                "n8n-nodes-base.emailReadImapTool",
                stage_id=stage_id,
                usage_mode="tool_only",
                has_main_input=False,
                usable_as_tool=True,
            )
        ],
    )

    updates = architect_agent_node(_state())

    assert updates["architect_status"] == ArchitectStatus.architect_blocked_waiting_user
    assert updates["target_stage"] is None
    assert updates["missing_user_inputs"]
    assert "iniciar" in updates["missing_user_inputs"][0].lower() or "evento" in updates["missing_user_inputs"][0].lower()


def test_architect_resume_after_user_clarification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/webhook"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])

    phase = {"resume": False}

    def _build_candidates(*, stage_id: str, **_kwargs):
        if stage_id == "stage_1":
            if not phase["resume"]:
                return []
            return [_candidate("n8n-nodes-base.webhook", stage_id=stage_id, has_main_input=False)]
        if stage_id == "stage_2":
            return [
                _candidate(
                    "n8n-nodes-base.code",
                    stage_id=stage_id,
                    type_version=2,
                    capability_summary="Use AI or code-based logic to classify urgency.",
                )
            ]
        return [
            _candidate(
                "n8n-nodes-base.googleSheets",
                stage_id=stage_id,
                capability_summary="Store workflow results in a spreadsheet table.",
            )
        ]

    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", _build_candidates)
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._select_stage_nodes_with_structured_output",
        lambda *, stage, candidates, **_kwargs: _StageSelectionOutput(
            selected_node_types=[candidates[0].node_type] if candidates else [],
            rationale=f"Selected stage {stage.id}",
            needs_clarification=not candidates,
            clarification_questions=[f"Need more detail for {stage.id}"] if not candidates else [],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_workflow_blueprint_with_structured_output",
        lambda **_kwargs: _WorkflowBlueprintOutput(
            workflow_name="Resume Architect",
            summary="Architect draft",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="webhook_1",
                    node_type="n8n-nodes-base.webhook",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive input",
                    depends_on=[],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_2",
                    name="code_2",
                    node_type="n8n-nodes-base.code",
                    type_version=2,
                    stage_id="stage_2",
                    purpose="Classify",
                    depends_on=["an_1"],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_3",
                    name="google_sheets_3",
                    node_type="n8n-nodes-base.googleSheets",
                    type_version=1,
                    stage_id="stage_3",
                    purpose="Store",
                    depends_on=["an_2"],
                ),
            ],
            connections=[
                _WorkflowConnectionBlueprint(source_node_id="an_1", target_node_id="an_2"),
                _WorkflowConnectionBlueprint(source_node_id="an_2", target_node_id="an_3"),
            ],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.engineer_agent.N8NClient.create_workflow",
        lambda self, payload: {"id": "wf_resume", "name": payload["name"], "url": "http://localhost:5678/workflow/wf_resume"},
    )

    first = architect_agent_node(_state())
    assert first["architect_status"] == ArchitectStatus.architect_blocked_waiting_user

    phase["resume"] = True
    resumed_state = _state(
        extra={
            **first,
            "user_query": "Empieza cuando llegue un correo nuevo por webhook o similar",
            "resume_requested": True,
            "runtime_context": {"model": None, "request_id": "req-architect-resume", "persist_to_n8n": True},
        }
    )
    second = architect_agent_node(resumed_state)

    assert second["architect_status"] == ArchitectStatus.architect_completed
    assert second["active_workflow_id"] == "wf_resume"
    assert second["architect_clarification_state"].turns[0].answer is not None


def test_architect_uses_request_context_query_for_stage_search(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_queries: List[str] = []

    def _retrieve_context(query: str, *args, **kwargs):
        seen_queries.append(query)
        return [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/webhook"}]

    monkeypatch.setattr("app.graphs.nodes.architect_agent.retrieve_context", _retrieve_context)
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", lambda **_kwargs: [])

    updates = architect_mod.architect_agent_node(
        _state(
            extra={
                "user_query": "nothing else",
                "request_context_query": "crea un workflow para coger emails y clasificarlos por urgencia",
            }
        )
    )

    assert updates["architect_status"] == ArchitectStatus.architect_blocked_waiting_user
    assert seen_queries
    assert any("coger emails" in query.lower() for query in seen_queries)
    assert all("nothing else" not in query.lower() for query in seen_queries)


def test_architect_rephrases_pending_question_in_spanish_without_consuming_answer() -> None:
    state = _state(
        extra={
            "architect_status": ArchitectStatus.architect_blocked_waiting_user,
            "architect_clarification_state": ArchitectClarificationState(
                attempts_used=1,
                max_attempts=3,
                pending_questions=["How should the result of stage 'Urgency Classification' be applied or stored?"],
                pending_slots=[
                    {
                        "slot_key": "result_application_mode",
                        "owner_agent": "architect_agent",
                        "stage_id": "stage_2",
                        "question_text": "How should the result of stage 'Urgency Classification' be applied or stored?",
                        "question_intent": "result_application_mode",
                        "answer_status": "pending",
                    }
                ],
                turns=[
                    {
                        "stage_id": "stage_2",
                        "slot_key": "result_application_mode",
                        "question": "How should the result of stage 'Urgency Classification' be applied or stored?",
                        "answer": None,
                    }
                ],
            ),
            "user_query": "Me lo puedes preguntar en espanol?",
            "resume_requested": True,
        }
    )

    updates = architect_agent_node(state)

    assert updates["architect_status"] == ArchitectStatus.architect_blocked_waiting_user
    assert "urgency classification" in updates["missing_user_inputs"][0].lower()
    assert updates["architect_clarification_state"].turns[0].answer is None
    assert "architect_question_rephrased" in updates["routing_signals"]


def test_architect_single_answer_can_clear_multiple_pending_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/webhook"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_candidates",
        lambda *, stage_id, **_kwargs: (
            [_candidate("n8n-nodes-base.webhook", stage_id=stage_id, has_main_input=False)]
            if stage_id == "stage_1"
            else [
                _candidate(
                    "n8n-nodes-base.code",
                    stage_id=stage_id,
                    type_version=2,
                    capability_summary="Use AI or code-based logic to classify urgency.",
                )
            ]
            if stage_id == "stage_2"
            else [
                _candidate(
                    "n8n-nodes-base.googleSheets",
                    stage_id=stage_id,
                    capability_summary="Store workflow results in a spreadsheet table.",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._select_stage_nodes_with_structured_output",
        lambda *, candidates, **_kwargs: _StageSelectionOutput(
            selected_node_types=[candidates[0].node_type],
            rationale="selected",
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_workflow_blueprint_with_structured_output",
        lambda **_kwargs: _WorkflowBlueprintOutput(
            workflow_name="Clarified architect workflow",
            summary="Workflow",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="webhook_1",
                    node_type="n8n-nodes-base.webhook",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive",
                    depends_on=[],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_2",
                    name="code_2",
                    node_type="n8n-nodes-base.code",
                    type_version=2,
                    stage_id="stage_2",
                    purpose="Classify",
                    depends_on=["an_1"],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_3",
                    name="google_sheets_3",
                    node_type="n8n-nodes-base.googleSheets",
                    type_version=1,
                    stage_id="stage_3",
                    purpose="Store",
                    depends_on=["an_2"],
                ),
            ],
            connections=[
                _WorkflowConnectionBlueprint(source_node_id="an_1", target_node_id="an_2"),
                _WorkflowConnectionBlueprint(source_node_id="an_2", target_node_id="an_3"),
            ],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.engineer_agent.N8NClient.create_workflow",
        lambda self, payload: {"id": "wf_multi_answer", "name": payload["name"], "url": "http://localhost:5678/workflow/wf_multi_answer"},
    )

    state = _state(
        extra={
            "architect_status": ArchitectStatus.architect_blocked_waiting_user,
            "architect_clarification_state": ArchitectClarificationState(
                attempts_used=1,
                max_attempts=3,
                pending_questions=[
                    "How should the result of stage 'Urgency Classification' be applied or stored?",
                    "Where should the result of stage 'Store Result' be stored?",
                ],
                pending_slots=[
                    {
                        "slot_key": "result_application_mode",
                        "owner_agent": "architect_agent",
                        "stage_id": "stage_2",
                        "question_text": "How should the result of stage 'Urgency Classification' be applied or stored?",
                        "question_intent": "result_application_mode",
                        "answer_status": "pending",
                    },
                    {
                        "slot_key": "storage_destination",
                        "owner_agent": "architect_agent",
                        "stage_id": "stage_3",
                        "question_text": "Where should the result of stage 'Store Result' be stored?",
                        "question_intent": "storage_destination",
                        "answer_status": "pending",
                    },
                ],
                turns=[
                    {"stage_id": "stage_2", "slot_key": "result_application_mode", "question": "How should the result of stage 'Urgency Classification' be applied or stored?", "answer": None},
                    {"stage_id": "stage_3", "slot_key": "storage_destination", "question": "Where should the result of stage 'Store Result' be stored?", "answer": None},
                ],
            ),
            "user_query": "Usa un LLM para clasificar y luego guarda el resultado en Google Sheets.",
            "resume_requested": True,
            "runtime_context": {"model": None, "request_id": "req-architect-multi", "persist_to_n8n": True},
        }
    )

    updates = architect_agent_node(state)

    assert updates["architect_status"] == ArchitectStatus.architect_completed
    assert updates["architect_clarification_state"].pending_questions == []
    assert all(turn.answer is not None for turn in updates["architect_clarification_state"].turns)


def test_architect_fails_after_max_clarifications(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/unknown"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", lambda **_kwargs: [])

    state = _state(
        extra={
            "architect_status": ArchitectStatus.architect_blocked_waiting_user,
            "architect_clarification_state": ArchitectClarificationState(
                attempts_used=3,
                max_attempts=3,
                pending_questions=["Need exact trigger"],
                turns=[{"stage_id": "stage_1", "question": "Need exact trigger", "answer": None}],
            ),
            "user_query": "No se, usa algo",
            "resume_requested": True,
        }
    )

    updates = architect_agent_node(state)

    assert updates["architect_status"] == ArchitectStatus.architect_failed_no_solution
    assert updates["target_stage"] is None
    assert "clarification attempts" in updates["missing_user_inputs"][0].lower()


def test_architect_updates_existing_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/gmail"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])

    def _build_candidates(*, stage_id: str, **_kwargs):
        if stage_id == "stage_1":
            return [_candidate("n8n-nodes-base.gmailTrigger", stage_id=stage_id, has_main_input=False)]
        if stage_id == "stage_2":
            return [
                _candidate(
                    "n8n-nodes-base.code",
                    stage_id=stage_id,
                    type_version=2,
                    capability_summary="Use AI or code-based logic to classify urgency.",
                )
            ]
        return [
            _candidate(
                "n8n-nodes-base.googleSheets",
                stage_id=stage_id,
                capability_summary="Store workflow results in a spreadsheet table.",
            )
        ]

    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", _build_candidates)
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._select_stage_nodes_with_structured_output",
        lambda *, candidates, **_kwargs: _StageSelectionOutput(
            selected_node_types=[candidates[0].node_type],
            rationale="selected",
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_workflow_blueprint_with_structured_output",
        lambda **_kwargs: _WorkflowBlueprintOutput(
            workflow_name="Architect Update",
            summary="Architect update draft",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="gmail_trigger_1",
                    node_type="n8n-nodes-base.gmailTrigger",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive emails",
                    depends_on=[],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_2",
                    name="code_2",
                    node_type="n8n-nodes-base.code",
                    type_version=2,
                    stage_id="stage_2",
                    purpose="Classify",
                    depends_on=["an_1"],
                ),
                _WorkflowNodeBlueprint(
                    node_id="an_3",
                    name="google_sheets_3",
                    node_type="n8n-nodes-base.googleSheets",
                    type_version=1,
                    stage_id="stage_3",
                    purpose="Store",
                    depends_on=["an_2"],
                ),
            ],
            connections=[
                _WorkflowConnectionBlueprint(source_node_id="an_1", target_node_id="an_2"),
                _WorkflowConnectionBlueprint(source_node_id="an_2", target_node_id="an_3"),
            ],
        ),
    )

    def _update(self, workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert workflow_id == "wf_existing"
        return {
            "id": workflow_id,
            "name": payload["name"],
            "url": f"http://localhost:5678/workflow/{workflow_id}",
        }

    monkeypatch.setattr("app.graphs.nodes.engineer_agent.N8NClient.update_workflow", _update)

    updates = architect_agent_node(
        _state(extra={"active_workflow_id": "wf_existing", "active_workflow_name": "Old Flow"})
    )

    assert updates["architect_status"] == ArchitectStatus.architect_completed
    assert updates["workflow_persist_action"] == "updated"
    assert updates["active_workflow_id"] == "wf_existing"


def test_architect_blocks_when_blueprint_does_not_cover_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent.retrieve_context",
        lambda *_args, **_kwargs: [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/gmail"}],
    )
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])

    def _build_candidates(*, stage_id: str, **_kwargs):
        if stage_id == "stage_1":
            return [_candidate("n8n-nodes-base.gmailTrigger", stage_id=stage_id, has_main_input=False)]
        if stage_id == "stage_2":
            return [
                _candidate(
                    "n8n-nodes-base.code",
                    stage_id=stage_id,
                    type_version=2,
                    capability_summary="Use AI or code-based logic to classify urgency.",
                )
            ]
        return [
            _candidate(
                "n8n-nodes-base.googleSheets",
                stage_id=stage_id,
                capability_summary="Store workflow results in a spreadsheet table.",
            )
        ]

    monkeypatch.setattr("app.graphs.nodes.architect_agent._build_candidates", _build_candidates)
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._select_stage_nodes_with_structured_output",
        lambda *, candidates, **_kwargs: _StageSelectionOutput(
            selected_node_types=[candidates[0].node_type],
            rationale="selected",
            needs_clarification=False,
            clarification_questions=[],
        ),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.architect_agent._build_workflow_blueprint_with_structured_output",
        lambda **_kwargs: _WorkflowBlueprintOutput(
            workflow_name="Broken blueprint",
            summary="Missing downstream stages",
            nodes=[
                _WorkflowNodeBlueprint(
                    node_id="an_1",
                    name="gmail_trigger_1",
                    node_type="n8n-nodes-base.gmailTrigger",
                    type_version=1,
                    stage_id="stage_1",
                    purpose="Receive emails",
                    depends_on=[],
                )
            ],
            connections=[],
        ),
    )

    updates = architect_agent_node(_state())

    assert updates["architect_status"] == ArchitectStatus.architect_blocked_waiting_user
    assert updates["missing_user_inputs"]
