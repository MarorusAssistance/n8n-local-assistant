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
    WorkflowContext,
)
from app.graphs.nodes.architect_agent import (
    _StageSelectionOutput,
    _WorkflowConnectionBlueprint,
    _WorkflowBlueprintOutput,
    _WorkflowNodeBlueprint,
    _candidate_rejection_reasons,
    _is_trigger_candidate,
    _build_candidates,
    _normalize_confidence,
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
    assert "perform a private self-check" in blueprint_prompt


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


def test_architect_builds_and_persists_new_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _retrieve(*_args, **_kwargs):
        return [{"doc_id": "doc-1", "text": "docs", "url": "https://docs.n8n.io/gmail"}]

    monkeypatch.setattr("app.graphs.nodes.architect_agent.retrieve_context", _retrieve)
    monkeypatch.setattr("app.graphs.nodes.architect_agent.query_related_definition_chunks", lambda *_args, **_kwargs: [])

    def _build_candidates(*, stage_id: str, **_kwargs):
        if stage_id == "stage_1":
            return [_candidate("n8n-nodes-base.gmailTrigger", stage_id=stage_id, has_main_input=False)]
        if stage_id == "stage_2":
            return [_candidate("n8n-nodes-base.code", stage_id=stage_id, type_version=2)]
        return [_candidate("n8n-nodes-base.googleSheets", stage_id=stage_id)]

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
    assert "trigger" in updates["missing_user_inputs"][0].lower()


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
            return [_candidate("n8n-nodes-base.code", stage_id=stage_id, type_version=2)]
        return [_candidate("n8n-nodes-base.googleSheets", stage_id=stage_id)]

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
            return [_candidate("n8n-nodes-base.code", stage_id=stage_id, type_version=2)]
        return [_candidate("n8n-nodes-base.googleSheets", stage_id=stage_id)]

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
