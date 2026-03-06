from __future__ import annotations

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    BusinessContextSummary,
    EntryIntent,
    EntryRouterDecision,
    UseCase,
)
from app.graphs.reasoning_graph import ReasoningGraphRuntime


def _decision(intent: EntryIntent, stage: AgentStage | None) -> EntryRouterDecision:
    return EntryRouterDecision(
        entry_intent=intent,
        target_stage=stage,
        confidence=0.8 if stage is not None else 0.3,
        routing_signals=["test_route"],
        missing_user_inputs=[],
    )


@pytest.mark.parametrize(
    ("intent", "stage"),
    [
        (EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
        (EntryIntent.workflow_edit_request, AgentStage.engineer_agent),
        (EntryIntent.workflow_fix_request, AgentStage.qa_agent),
        (EntryIntent.information_request, AgentStage.consultant_agent),
    ],
)
def test_graph_routes_start_to_expected_stub(
    monkeypatch: pytest.MonkeyPatch,
    intent: EntryIntent,
    stage: AgentStage,
) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr("app.graphs.reasoning_graph.route_entry_intent", lambda **kwargs: _decision(intent, stage))

    result = runtime.run(
        user_prompt="route me",
        model=None,
        request_id="req-1",
        existing_workflow=None,
    )
    assert result.entry_intent == intent
    assert result.target_stage == stage
    assert result.current_stage == stage.value
    assert result.status == "stub_routed"


def test_graph_unknown_route_ends_without_forcing_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.unknown, None),
    )

    result = runtime.run(
        user_prompt="ambiguous",
        model=None,
        request_id="req-2",
        existing_workflow=None,
    )
    assert result.entry_intent == EntryIntent.unknown
    assert result.target_stage is None
    assert result.current_stage is None
    assert result.status == "unknown_terminal"


def test_graph_compiles_or_uses_sequential_fallback() -> None:
    runtime = ReasoningGraphRuntime()
    assert runtime._graph is None or hasattr(runtime._graph, "invoke")


def test_graph_business_discovery_sets_handoff_ready_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.commercial_agent_node",
        lambda state: {
            "current_stage": "commercial_agent",
            "business_context_summary": BusinessContextSummary(
                process_scope="Finance operations",
                pain_points=["manual approvals"],
                desired_outcomes=["faster approvals"],
                constraints=[],
            ),
            "discovered_use_cases": [
                UseCase(
                    id="uc_1",
                    title="Invoice approval reminders",
                    business_problem="Invoice approvals are delayed due to manual follow-up.",
                    desired_outcome="Automate reminders and escalation for pending approvals.",
                    expected_value="Reduce payment delays and improve throughput.",
                    feasibility="medium",
                    priority_score=86.0,
                    why_selected="Selected for high business value.",
                )
            ],
            "selected_use_case": UseCase(
                id="uc_1",
                title="Invoice approval reminders",
                business_problem="Invoice approvals are delayed due to manual follow-up.",
                desired_outcome="Automate reminders and escalation for pending approvals.",
                expected_value="Reduce payment delays and improve throughput.",
                feasibility="medium",
                priority_score=86.0,
                why_selected="Selected for high business value.",
            ),
            "alternative_use_cases": [],
            "selection_reason": "Highest value and actionable business outcome.",
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_product_manager"],
            "target_stage": AgentStage.product_manager_agent,
            "missing_user_inputs": [],
        },
    )
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
    )

    result = runtime.run(
        user_prompt="We have manual approval bottlenecks in finance.",
        model=None,
        request_id="req-commercial-1",
        existing_workflow=None,
    )
    assert result.current_stage == "commercial_agent"
    assert result.selected_use_case is not None
    assert result.target_stage == AgentStage.product_manager_agent
    assert "handoff_ready_product_manager" in result.routing_signals
    assert result.status == "stub_routed"


def test_graph_business_discovery_no_actionable_case_ends_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.commercial_agent_node",
        lambda state: {
            "current_stage": "commercial_agent",
            "business_context_summary": BusinessContextSummary(
                process_scope="General operations",
                pain_points=[],
                desired_outcomes=[],
                constraints=[],
            ),
            "discovered_use_cases": [],
            "selected_use_case": None,
            "alternative_use_cases": [],
            "selection_reason": "No actionable use case was extracted from the current input.",
            "routing_signals": list(state.get("routing_signals") or []) + ["no_actionable_use_case"],
            "target_stage": None,
            "missing_user_inputs": ["Please provide one concrete process and desired outcome."],
        },
    )
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
    )

    result = runtime.run(
        user_prompt="We should automate something eventually.",
        model=None,
        request_id="req-commercial-2",
        existing_workflow=None,
    )
    assert result.current_stage == "commercial_agent"
    assert result.selected_use_case is None
    assert result.alternative_use_cases == []
    assert "no actionable use case" in (result.selection_reason or "").lower()
    assert result.status == "unknown_terminal"
