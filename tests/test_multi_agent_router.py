from __future__ import annotations

import pytest

from app.features.reasoning.multi_agent_contracts import AgentStage, EntryIntent
from app.graphs.nodes import multi_agent_router


def _route_with_heuristics(monkeypatch: pytest.MonkeyPatch, query: str):
    monkeypatch.setattr(
        multi_agent_router,
        "_classify_with_structured_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("llm offline")),
    )
    return multi_agent_router.route_entry_intent(query)


@pytest.mark.parametrize(
    ("query", "intent", "stage"),
    [
        ("Let's run a discovery session about automation opportunities in our finance process.", EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
        ("As consultant and client, we need to identify which business processes should be automated first.", EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
        ("Help me map business process bottlenecks and propose where automation gives highest value.", EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
        ("Create a new automation to capture webhook orders and store them in a database.", EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
        ("Build a new workflow from scratch for invoice approvals.", EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
        ("Set up a workflow that reads emails and writes rows to Google Sheets.", EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
        ("Modify this workflow to add Slack notifications after payment is confirmed.", EntryIntent.workflow_edit_request, AgentStage.engineer_agent),
        ("Update the existing workflow node mapping to include customer_id.", EntryIntent.workflow_edit_request, AgentStage.engineer_agent),
        ("Change current workflow behavior so it retries only once.", EntryIntent.workflow_edit_request, AgentStage.engineer_agent),
        ("Fix this workflow, it fails at HTTP Request with timeout errors.", EntryIntent.workflow_fix_request, AgentStage.qa_agent),
        ("Debug and repair the current workflow because node execution throws exception.", EntryIntent.workflow_fix_request, AgentStage.qa_agent),
        ("Workflow is broken and not working after credential rotation, please fix it.", EntryIntent.workflow_fix_request, AgentStage.qa_agent),
        ("Explain the difference between Webhook and Schedule Trigger in n8n.", EntryIntent.information_request, AgentStage.consultant_agent),
        ("Recommend best practices for idempotency in automation workflows.", EntryIntent.information_request, AgentStage.consultant_agent),
        ("Compare IF node versus Switch node usage.", EntryIntent.information_request, AgentStage.consultant_agent),
        ("Help", EntryIntent.unknown, None),
        ("Can you assist?", EntryIntent.unknown, None),
        ("I need something but not sure what.", EntryIntent.unknown, None),
    ],
)
def test_router_classifies_clear_examples(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    intent: EntryIntent,
    stage: AgentStage | None,
) -> None:
    result = _route_with_heuristics(monkeypatch, query)
    assert result.entry_intent == intent
    assert result.target_stage == stage


@pytest.mark.parametrize(
    "query",
    [
        "Maybe we should do something with automation, any thoughts?",
        "I have a workflow and also general questions, not sure where to start.",
        "Can you help me decide what to do next with our setup?",
    ],
)
def test_router_handles_ambiguous_examples_as_unknown(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    result = _route_with_heuristics(monkeypatch, query)
    assert result.entry_intent == EntryIntent.unknown
    assert result.target_stage is None


@pytest.mark.parametrize(
    "query",
    [
        "Edit this existing workflow and fix the HTTP node error that keeps failing.",
        "Please modify current workflow, but first debug and repair timeout failures.",
    ],
)
def test_router_fix_beats_edit_when_both_signals_present(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    result = _route_with_heuristics(monkeypatch, query)
    assert result.entry_intent == EntryIntent.workflow_fix_request
    assert result.target_stage == AgentStage.qa_agent


@pytest.mark.parametrize(
    "query",
    [
        (
            "User: We changed a node.\n"
            "Assistant: Which node fails?\n"
            "User: The workflow now throws timeout on HTTP Request."
        ),
        (
            "Client: Existing workflow breaks on credentials.\n"
            "Consultant: Is it an auth node issue?\n"
            "Client: Yes, debug what failed."
        ),
    ],
)
def test_multiturn_technical_conversations_do_not_route_to_commercial(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    result = _route_with_heuristics(monkeypatch, query)
    assert result.target_stage != AgentStage.commercial_agent
    assert result.entry_intent != EntryIntent.business_discovery_conversation


@pytest.mark.parametrize(
    "query",
    [
        "Create a new workflow to process support tickets automatically.",
        "Build a new automation from scratch for onboarding emails.",
    ],
)
def test_direct_build_requests_route_to_product_manager(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    result = _route_with_heuristics(monkeypatch, query)
    assert result.entry_intent == EntryIntent.workflow_build_request
    assert result.target_stage == AgentStage.product_manager_agent

