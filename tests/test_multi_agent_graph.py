from __future__ import annotations

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    EntryIntent,
    EntryRouterDecision,
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
        (EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
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

