from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitectureStage,
    EntryIntent,
    PMClarificationState,
    PMStatus,
    UseCase,
)
from app.graphs.nodes import product_manager_agent as pm


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
    assert updates["proposed_nodes"] == []
    assert updates["required_credentials"] == []
    assert updates["pm_stage_plan"][0].id == "stage_intake"
    assert updates["pm_stage_selections"] == []
    assert "handoff_ready_architect" in updates["routing_signals"]


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
    assert "pm_blocked_waiting_user" in updates["routing_signals"]


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
        user_query=(
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
