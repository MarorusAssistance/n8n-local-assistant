from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitectureStage,
    DecisionSlot,
    DecisionSlotAnswerStatus,
    EntryIntent,
    PMClarificationState,
    PMStatus,
    StageKind,
    UseCase,
)
from app.graphs.nodes import product_manager_agent as pm
from app.graphs.nodes.question_utils import infer_decision_slot_key


def _use_case(
    *,
    use_case_id: str = "uc_1",
    title: str = "Urgent Email Classification",
    business_problem: str = "The team manually reviews incoming emails and decides urgency.",
    desired_outcome: str = "Classify each incoming email by urgency and persist the result.",
    expected_value: str = "Reduce manual triage time and improve response speed.",
) -> UseCase:
    return UseCase(
        id=use_case_id,
        title=title,
        business_problem=business_problem,
        desired_outcome=desired_outcome,
        expected_value=expected_value,
        feasibility="medium",
        priority_score=82.0,
        why_selected="Selected in commercial stage.",
    )


def _state(
    *,
    selected_use_case: UseCase | Dict[str, Any] | None,
    entry_intent: EntryIntent = EntryIntent.business_discovery_conversation,
    user_query: str = "Build a workflow that classifies urgency",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "entry_intent": entry_intent,
        "selected_use_case": selected_use_case,
        "user_query": user_query,
        "routing_signals": ["entered_commercial_agent", "handoff_ready_product_manager"],
        "missing_user_inputs": [],
        "workflow_context": {"model": None, "request_id": "req-pm-test"},
    }
    if extra:
        payload.update(extra)
    return payload


def _abstract_plan(
    *,
    planning_ready: bool = True,
    missing_information: List[str] | None = None,
) -> pm._AbstractPlanningOutput:
    return pm._AbstractPlanningOutput(
        workflow_summary="Abstract workflow plan with intake, classification, and persistence.",
        stages=[
            ArchitectureStage(
                id="stage_intake",
                name="Intake",
                purpose="Receive each incoming email and normalize the working payload.",
                required_capabilities=["Capture inbound item", "Normalize payload"],
                expected_inputs=["Incoming email"],
                expected_outputs=["Normalized email payload"],
                dependencies=[],
                success_criteria=["Each relevant email enters the workflow once."],
            ),
            ArchitectureStage(
                id="stage_classification",
                name="Classification",
                purpose="Determine the urgency level from the subject and content.",
                required_capabilities=["Interpret content", "Apply urgency logic"],
                expected_inputs=["Normalized email payload"],
                expected_outputs=["Urgency decision"],
                dependencies=["stage_intake"],
                success_criteria=["A single urgency outcome is produced for each email."],
            ),
            ArchitectureStage(
                id="stage_persistence",
                name="Persistence",
                purpose="Store or communicate the assigned urgency result.",
                required_capabilities=["Persist result", "Make outcome available downstream"],
                expected_inputs=["Urgency decision"],
                expected_outputs=["Stored urgency record"],
                dependencies=["stage_classification"],
                success_criteria=["The urgency result is stored or forwarded."],
            ),
        ],
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_classification",
                data_items=["normalized email payload"],
            ),
            ArchitectureDataFlowItem(
                source_stage_id="stage_classification",
                target_stage_id="stage_persistence",
                data_items=["urgency decision"],
            ),
        ],
        assumptions=["The triggering source of the emails will be grounded later by architect_agent."],
        missing_information=missing_information or [],
        handoff_notes=[
            "Architect agent must preserve this three-stage flow and choose nodes that fit the full workflow."
        ],
        planning_ready=planning_ready,
    )


def _invalid_multi_operation_plan() -> pm._AbstractPlanningOutput:
    return pm._AbstractPlanningOutput(
        workflow_summary="Receive incoming emails and classify them by urgency before storing the result.",
        stages=[
            ArchitectureStage(
                id="stage_intake",
                name="Intake",
                purpose="Receive incoming emails.",
                required_capabilities=["Capture inbound item"],
                expected_inputs=["Incoming email"],
                expected_outputs=["Normalized email payload"],
                dependencies=[],
                success_criteria=["The workflow receives each email once."],
            )
        ],
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_classification",
                data_items=["normalized email payload"],
            )
        ],
        assumptions=[],
        missing_information=[],
        handoff_notes=["Architect should classify and store the result."],
        explicit_user_operations=["capture", "classify", "store"],
        covered_operations=["capture"],
        planning_ready=True,
    )


def _ambiguous_outcome_plan() -> pm._AbstractPlanningOutput:
    return pm._AbstractPlanningOutput(
        workflow_summary="Receive emails and classify them by urgency.",
        stages=[
            ArchitectureStage(
                id="stage_intake",
                name="Intake",
                purpose="Receive incoming emails.",
                required_capabilities=["Capture inbound item"],
                expected_inputs=["Incoming email"],
                expected_outputs=["Normalized email payload"],
                dependencies=[],
                success_criteria=["The workflow receives each email once."],
            ),
            ArchitectureStage(
                id="stage_classification",
                name="Classification",
                purpose="Classify each email by urgency.",
                required_capabilities=["Interpret content", "Assign urgency"],
                expected_inputs=["Normalized email payload"],
                expected_outputs=["Urgency label"],
                dependencies=["stage_intake"],
                success_criteria=["A single urgency label is produced."],
            ),
        ],
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_classification",
                data_items=["normalized email payload"],
            )
        ],
        assumptions=[],
        missing_information=[],
        handoff_notes=["Architect should ground intake and classification."],
        explicit_user_operations=["capture", "classify"],
        covered_operations=["capture", "classify"],
        planning_ready=True,
    )


def _mixed_stage_plan() -> pm._AbstractPlanningOutput:
    return pm._AbstractPlanningOutput(
        workflow_summary="Receive incoming emails, classify them, and apply the result in Gmail.",
        stages=[
            ArchitectureStage(
                id="stage_intake",
                name="Receive and Tag Emails",
                purpose="Receive incoming emails and apply an urgency label in Gmail.",
                stage_kind=StageKind.trigger_intake,
                required_capabilities=["Capture inbound email", "Apply urgency label"],
                expected_inputs=["Incoming email"],
                expected_outputs=["Email captured with urgency label applied"],
                dependencies=[],
                success_criteria=["Each email is received and labeled in Gmail."],
            ),
            ArchitectureStage(
                id="stage_notify",
                name="Notify Result",
                purpose="Send a summary notification for critical emails.",
                required_capabilities=["Notify downstream users"],
                expected_inputs=["Labeled email"],
                expected_outputs=["Notification sent"],
                dependencies=["stage_intake"],
                success_criteria=["Critical emails trigger a notification."],
            ),
        ],
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_notify",
                data_items=["Labeled email"],
            )
        ],
        assumptions=[],
        missing_information=[],
        handoff_notes=[],
        explicit_user_operations=["capture", "classify", "sync", "notify"],
        covered_operations=["capture", "notify"],
        planning_ready=True,
    )


def test_pm_completes_abstract_plan_for_architect_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_pm_ready")
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _abstract_plan(),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["target_stage"] is None
    assert updates["workflow_context"].planning_ready is True
    assert updates["workflow_context"].handoff_target == AgentStage.architect_agent
    assert updates["architecture_plan"] is not None
    assert updates["architecture_plan"].required_nodes == []
    assert updates["architecture_plan"].stages[0].stage_kind is not None
    assert updates["proposed_nodes"] == []
    assert updates["required_credentials"] == []
    assert updates["pm_stage_plan"][0].id == "stage_intake"
    assert updates["pm_stage_selections"] == []
    assert "handoff_ready_architect" in updates["routing_signals"]


def test_build_architecture_plan_propagates_user_constraints() -> None:
    selected = _use_case(use_case_id="uc_pm_constraints")
    clarification_state = PMClarificationState(
        resolved_slots=[
            DecisionSlot(
                slot_key="result_application_mode",
                owner_agent=AgentStage.product_manager_agent,
                stage_id="stage_persistence",
                answer_status=DecisionSlotAnswerStatus.resolved,
                answer=(
                    "Los niveles de urgencia son bajo, medio, alto y critico. "
                    "Aplica el resultado al mismo correo de Gmail para que quede visible en Gmail UI y pueda filtrarlo luego. "
                    "Si no hay suficiente confianza, usa Review."
                ),
            )
        ],
        turns=[],
    )
    plan = pm._build_architecture_plan(
        use_case=selected,
        plan_output=_abstract_plan(),
        request_context_query="Clasifica los correos con IA segun asunto y cuerpo.",
        clarification_state=clarification_state,
    )

    classification_stage = next(stage for stage in plan.stages if stage.id == "stage_classification")
    persistence_stage = next(stage for stage in plan.stages if stage.id == "stage_persistence")

    assert "bajo, medio, alto, critico" in (classification_stage.notes or "")
    assert "Review" in (classification_stage.notes or "")
    assert "Gmail UI" in (persistence_stage.notes or "")
    assert any("Gmail UI" in note for note in plan.implementation_notes_for_engineer)


def test_pm_accepts_direct_build_request_without_selected_use_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _abstract_plan(),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=None,
            entry_intent=EntryIntent.workflow_build_request,
            user_query="Create a workflow that triages incoming emails by urgency.",
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["target_stage"] is None
    assert "pm_use_case_derived_from_direct_build_request" in updates["routing_signals"]
    assert updates["workflow_context"].handoff_target == AgentStage.architect_agent
    assert updates["selected_use_case"] is not None
    assert "triages incoming emails by urgency" in updates["request_context_query"].lower()


def test_pm_resume_keeps_original_request_context_instead_of_latest_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(
        use_case_id="uc_pm_resume_context",
        title="Gmail urgency workflow",
        business_problem="User requested a new workflow for: crea un workflow para coger emails y clasificarlos por urgencia",
        desired_outcome="Deliver an abstract workflow plan that satisfies: crea un workflow para coger emails y clasificarlos por urgencia",
    )
    captured: Dict[str, Any] = {}

    def _fake_plan(**kwargs):
        captured.update(kwargs)
        return _abstract_plan()

    monkeypatch.setattr(pm, "_plan_abstract_workflow_with_structured_output", _fake_plan)

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            user_query="nothing else",
            extra={
                "request_context_query": "crea un workflow para coger emails y clasificarlos por urgencia",
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_clarification_state": PMClarificationState(
                    attempts_used=1,
                    max_attempts=2,
                    pending_questions=["Que quieres hacer con el resultado de la clasificacion?"],
                    pending_slots=[
                        {
                            "slot_key": "result_application_mode",
                            "owner_agent": "product_manager_agent",
                            "stage_id": "stage_classification",
                            "question_text": "Que quieres hacer con el resultado de la clasificacion?",
                            "question_intent": "result_application_mode",
                            "answer_status": "pending",
                        }
                    ],
                    turns=[
                        {
                            "stage_id": "stage_classification",
                            "slot_key": "result_application_mode",
                            "question": "Que quieres hacer con el resultado de la clasificacion?",
                            "answer": None,
                        }
                    ],
                ),
            },
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["selected_use_case"].title == selected.title
    assert captured["request_context_query"] == "crea un workflow para coger emails y clasificarlos por urgencia"


def test_terminal_outcome_question_maps_to_result_application_mode() -> None:
    question = (
        "The workflow currently ends at stage 'Outcome Handling' without a clear final business outcome. "
        "What should happen with that result next: apply it to the source item, save it somewhere, "
        "notify someone, or route it to a concrete downstream action?"
    )

    assert infer_decision_slot_key(question, stage_name="plan_terminal_outcome") == "result_application_mode"


def test_downstream_actions_question_maps_to_result_application_mode() -> None:
    question = (
        "Are there additional downstream actions (e.g., routing to a different mailbox, forwarding) "
        "that should occur after classification?"
    )

    assert infer_decision_slot_key(question, stage_name="stage_3") == "result_application_mode"


def test_pm_ignores_none_style_missing_information_when_plan_is_otherwise_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(use_case_id="uc_pm_none_missing_info")
    noisy_plan = _abstract_plan(
        planning_ready=False,
        missing_information=[
            "None. The workflow now covers all explicit operations and no clarification is required."
        ],
    )
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: noisy_plan,
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["workflow_context"].planning_ready is True
    assert updates["pending_decision_slots"] == []
    assert updates["missing_user_inputs"] == []


def test_pm_heuristic_email_plan_blocks_until_outcome_is_defined() -> None:
    selected = _use_case(
        use_case_id="uc_pm_heuristic_email",
        title="Email urgency workflow",
        business_problem="User requested a new workflow for: crea un workflow para coger los emails que vaya recibiendo y que los clasifique por nivel de urgencia con IA",
        desired_outcome="Deliver an abstract workflow plan that satisfies: crea un workflow para coger los emails que vaya recibiendo y que los clasifique por nivel de urgencia con IA",
    )

    plan = pm._heuristic_abstract_plan(
        selected,
        request_context_query="crea un workflow para coger los emails que vaya recibiendo y que los clasifique por nivel de urgencia con IA",
        clarification_state=PMClarificationState(),
    )

    assert plan.planning_ready is False
    assert [stage.stage_kind for stage in plan.stages] == [StageKind.trigger_intake, StageKind.classify_decision]
    assert any("classification next" in issue.lower() or "what should happen" in issue.lower() for issue in plan.missing_information)


def test_pm_heuristic_email_plan_uses_resolved_result_application_mode() -> None:
    selected = _use_case(
        use_case_id="uc_pm_heuristic_email_apply",
        title="Email urgency workflow",
        business_problem="User requested a new workflow for: crea un workflow para coger los emails que vaya recibiendo y que los clasifique por nivel de urgencia con IA",
        desired_outcome="Deliver an abstract workflow plan that satisfies: crea un workflow para coger los emails que vaya recibiendo y que los clasifique por nivel de urgencia con IA",
    )
    clarification_state = PMClarificationState(
        resolved_slots=[
            {
                "slot_key": "source_system",
                "owner_agent": "product_manager_agent",
                "question_text": "Que sistema debe usarse?",
                "question_intent": "source_system",
                "answer_status": "resolved",
                "answer": "Gmail",
            },
            {
                "slot_key": "result_application_mode",
                "owner_agent": "product_manager_agent",
                "stage_id": "stage_processing",
                "question_text": "Que quieres hacer con el resultado?",
                "question_intent": "result_application_mode",
                "answer_status": "resolved",
                "answer": "Aplicalo al mismo correo de Gmail como una etiqueta visible para poder filtrarlo despues.",
            },
            {
                "slot_key": "classification_method",
                "owner_agent": "product_manager_agent",
                "question_text": "Que metodo usar?",
                "question_intent": "classification_method",
                "answer_status": "resolved",
                "answer": "Usa IA semantica.",
            },
        ]
    )

    plan = pm._heuristic_abstract_plan(
        selected,
        request_context_query="crea un workflow para coger los emails que vaya recibiendo y que los clasifique por nivel de urgencia con IA",
        clarification_state=clarification_state,
    )

    assert plan.planning_ready is True
    assert len(plan.stages) == 3
    assert [stage.stage_kind for stage in plan.stages] == [
        StageKind.trigger_intake,
        StageKind.classify_decision,
        StageKind.apply_update_source,
    ]
    assert "gmail" in (plan.stages[-1].purpose or "").lower()


def test_pm_blocks_when_abstract_plan_needs_more_information(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_pm_blocked")
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _abstract_plan(
            planning_ready=False,
            missing_information=[
                "What source should provide the incoming emails?",
                "Where should the urgency classification be stored?",
            ],
        ),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["pm_status"] == PMStatus.pm_blocked_waiting_user
    assert updates["target_stage"] is None
    assert updates["workflow_context"].planning_ready is False
    assert updates["workflow_context"].handoff_target is None
    assert updates["pm_clarification_state"].attempts_used == 1
    assert updates["pm_clarification_state"].pending_questions
    assert updates["workflow_context"].pending_decision_slots
    assert "pm_blocked_waiting_user" in updates["routing_signals"]


def test_pm_blocks_when_plan_ends_in_classification_without_business_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(
        use_case_id="uc_pm_outcome_gap",
        business_problem="The team wants a workflow that receives emails and classifies urgency.",
        desired_outcome="Receive emails and classify them by urgency.",
    )
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _ambiguous_outcome_plan(),
    )
    monkeypatch.setattr(
        pm,
        "_repair_abstract_plan_with_structured_output",
        lambda **_: _ambiguous_outcome_plan(),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            user_query="Crea un workflow que reciba emails y los clasifique por urgencia",
        )
    )

    assert updates["pm_status"] == PMStatus.pm_blocked_waiting_user
    assert updates["workflow_context"].planning_ready is False
    assert updates["missing_user_inputs"]
    assert "clasificacion" in updates["missing_user_inputs"][0].lower() or "classification" in updates["missing_user_inputs"][0].lower()


def test_pm_structural_plan_issue_maps_to_workflow_goal_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_pm_structural_issue")
    monkeypatch.setattr(pm, "_plan_abstract_workflow_with_structured_output", lambda **_: _invalid_multi_operation_plan())
    monkeypatch.setattr(pm, "_repair_abstract_plan_with_structured_output", lambda **_: _invalid_multi_operation_plan())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["pm_status"] == PMStatus.pm_blocked_waiting_user
    assert updates["workflow_context"].pending_decision_slots
    assert updates["workflow_context"].pending_decision_slots[0].slot_key == "workflow_goal"


def test_pm_allows_analysis_only_request_without_extra_terminal_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(
        use_case_id="uc_pm_analysis_only",
        business_problem="The team wants to inspect email urgency, nothing else.",
        desired_outcome="Only classify each incoming email by urgency and show the result.",
    )
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _ambiguous_outcome_plan(),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            user_query="Solo quiero clasificar los emails por urgencia y mostrar el resultado",
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["workflow_context"].planning_ready is True
    assert updates["target_stage"] is None
    assert updates["workflow_context"].handoff_target == AgentStage.architect_agent


def test_pm_fails_after_max_clarification_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_pm_failed")
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _abstract_plan(
            planning_ready=False,
            missing_information=["Which business rule defines urgency?"],
        ),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            extra={
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_clarification_state": PMClarificationState(
                    attempts_used=2,
                    max_attempts=2,
                    pending_questions=["Which business rule defines urgency?"],
                    turns=[],
                ),
            },
        )
    )

    assert updates["pm_status"] == PMStatus.pm_failed_no_solution
    assert updates["target_stage"] is None
    assert updates["workflow_context"].planning_ready is False
    assert "pm_failed_no_solution" in updates["routing_signals"]


def test_pm_resume_consumes_user_answer_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_pm_resume")
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _abstract_plan(),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            user_query="Use the email inbox as source and persist urgency in our ticket record.",
            extra={
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_clarification_state": PMClarificationState(
                    attempts_used=1,
                    max_attempts=2,
                    pending_questions=["What source should provide the incoming emails?"],
                    turns=[
                        {
                            "stage_id": None,
                            "question": "What source should provide the incoming emails?",
                            "answer": None,
                        }
                    ],
                ),
            },
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["pm_clarification_state"].pending_questions == []
    assert updates["pm_clarification_state"].turns[-1].answer is not None
    assert "pm_clarification_answer_received" in updates["routing_signals"]
    assert updates["workflow_context"].handoff_target == AgentStage.architect_agent


def test_pm_rephrases_pending_question_in_spanish_without_consuming_answer() -> None:
    selected = _use_case(use_case_id="uc_pm_rephrase")

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            user_query="No entiendo la pregunta, me la puedes hacer en espanol?",
            extra={
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_clarification_state": PMClarificationState(
                    attempts_used=1,
                    max_attempts=2,
                    pending_questions=["Which business rule defines urgency?"],
                    turns=[
                        {
                            "stage_id": None,
                            "question": "Which business rule defines urgency?",
                            "answer": None,
                        }
                    ],
                ),
            },
        )
    )

    assert updates["pm_status"] == PMStatus.pm_blocked_waiting_user
    assert updates["missing_user_inputs"][0] == "Que regla de negocio define la urgencia?"
    assert updates["pm_clarification_state"].turns[0].answer is None
    assert "pm_question_rephrased" in updates["routing_signals"]


def test_pm_single_answer_can_clear_multiple_pending_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_pm_multi_answer")
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _abstract_plan(),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            user_query="Usa Gmail como origen y guarda el resultado en una etiqueta.",
            extra={
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_clarification_state": PMClarificationState(
                    attempts_used=1,
                    max_attempts=2,
                    pending_questions=[
                        "What source should provide the incoming emails?",
                        "Which business rule defines urgency?",
                    ],
                    turns=[
                        {"stage_id": None, "question": "What source should provide the incoming emails?", "answer": None},
                        {"stage_id": None, "question": "Which business rule defines urgency?", "answer": None},
                    ],
                ),
            },
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["pm_clarification_state"].pending_questions == []
    assert all(turn.answer is not None for turn in updates["pm_clarification_state"].turns)


def test_pm_does_not_call_retrieval_or_select_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_no_retrieval")
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _abstract_plan(),
    )
    monkeypatch.setattr(
        "app.graphs.nodes.product_manager_agent.get_langchain_chat_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("not expected in this test")),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            extra={"workflow_context": {"model": None, "request_id": "req-no-model"}},
        )
    )

    assert updates["architecture_plan"].required_nodes == []
    assert updates["pm_stage_selections"] == []
    assert updates["proposed_nodes"] == []
    assert updates["required_credentials"] == []


def test_pm_prompt_preserves_exact_trigger_processing_and_outcome_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return _abstract_plan()

    monkeypatch.setattr(pm, "_invoke_structured_output", _capture)

    pm._plan_abstract_workflow_with_structured_output(
        use_case=_use_case(),
        request_context_query=(
            "Clasifica correos entrantes por urgencia con heuristicas y guarda el resultado; "
            "no envies emails ni agregues revision manual."
        ),
        clarification_state=PMClarificationState(),
        model="fake-model",
        request_id="req-pm-prompt",
    )

    prompt = str(captured["user_prompt"])
    assert "Preserve the exact source system, trigger style, processing mode, and final outcome named by the user." in prompt
    assert "Distinguish receiving email from sending email; they are not interchangeable." in prompt
    assert "Do not invent validation, feedback, manual review, refinement, or approval stages unless the user explicitly asked for them." in prompt
    assert "Each stage must have one dominant operable action" in prompt
    assert "Do not mix source intake with downstream classification" in prompt


def test_pm_repairs_semantically_incomplete_multi_operation_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _invalid_multi_operation_plan(),
    )
    monkeypatch.setattr(
        pm,
        "_repair_abstract_plan_with_structured_output",
        lambda **_: _abstract_plan(),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=_use_case(),
            entry_intent=EntryIntent.workflow_build_request,
            user_query="Recibe correos, clasificalos por urgencia y guarda el resultado.",
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["workflow_context"].planning_ready is True
    assert len(updates["architecture_plan"].stages) == 3


def test_pm_validation_rejects_mixed_stage_roles() -> None:
    plan_output, issues = pm._validate_abstract_plan_output(
        use_case=_use_case(),
        user_query="Recibe correos, clasificalos por urgencia y aplica la etiqueta en Gmail.",
        plan_output=_mixed_stage_plan(),
    )

    assert plan_output.stages[0].stage_kind == StageKind.trigger_intake
    assert any("mixes intake/fetch work" in issue for issue in issues)


def test_pm_blocks_when_plan_still_incoherent_after_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pm,
        "_plan_abstract_workflow_with_structured_output",
        lambda **_: _invalid_multi_operation_plan(),
    )
    monkeypatch.setattr(
        pm,
        "_repair_abstract_plan_with_structured_output",
        lambda **_: _invalid_multi_operation_plan(),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=_use_case(),
            entry_intent=EntryIntent.workflow_build_request,
            user_query="Recibe correos, clasificalos por urgencia y guarda el resultado.",
        )
    )

    assert updates["pm_status"] == PMStatus.pm_blocked_waiting_user
    assert updates["workflow_context"].planning_ready is False
    assert updates["missing_user_inputs"]
