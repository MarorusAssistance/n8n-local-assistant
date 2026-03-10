from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    EntryIntent,
    PMClarificationState,
    PMNodeCandidate,
    PMStagePlan,
    PMStageSelection,
    PMStatus,
    UseCase,
)
from app.graphs.nodes import product_manager_agent as pm


def _use_case(
    *,
    use_case_id: str = "uc_1",
    title: str = "Webhook to Sheets",
    business_problem: str = "Operations team manually copies incoming submissions into a spreadsheet.",
    desired_outcome: str = "Automatically capture submissions and store them in Google Sheets.",
    expected_value: str = "Reduce manual work and data-entry errors.",
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
    user_query: str = "Build a workflow for inbound forms",
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


def _stage_plan() -> List[PMStagePlan]:
    return [
        PMStagePlan(
            id="stage_intake",
            name="Intake",
            objective="Capture incoming payload.",
            expected_inputs=["incoming event"],
            expected_outputs=["normalized payload"],
            success_criteria=["event captured"],
            dependencies=[],
        ),
        PMStagePlan(
            id="stage_delivery",
            name="Delivery",
            objective="Persist normalized payload to destination.",
            expected_inputs=["normalized payload"],
            expected_outputs=["delivery result"],
            success_criteria=["payload persisted"],
            dependencies=["stage_intake"],
        ),
    ]


def _selection(stage_id: str, node_type: str, *, gate_passed: bool = True) -> PMStageSelection:
    return PMStageSelection(
        stage_id=stage_id,
        selected_node_types=[node_type] if gate_passed else [],
        selected_nodes=(
            [
                PMNodeCandidate(
                    node_type=node_type,
                    capability_summary=f"Use {node_type} for {stage_id}.",
                    limitations=[],
                    usage_mode="action_only",
                    evidence_chunk_ids=[f"chunk-{stage_id}"],
                    evidence_refs=[f"ref-{stage_id}"],
                    rerank_confidence=0.72,
                    pm_fit_score=0.84,
                )
            ]
            if gate_passed
            else []
        ),
        rationale=f"Selected node for {stage_id}." if gate_passed else "Insufficient evidence.",
        pm_fit_score=0.84 if gate_passed else 0.42,
        rerank_confidence=0.72 if gate_passed else 0.30,
        top_margin=0.12 if gate_passed else 0.02,
        gate_passed=gate_passed,
        passes_used=1,
        missing_information=[] if gate_passed else ["Need more integration details."],
        search_history=[],
    )


def test_pm_completes_stage_first_and_builds_legacy_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_stage_bridge")
    monkeypatch.setattr(pm, "_plan_stages_with_structured_output", lambda **_: _stage_plan())
    monkeypatch.setattr(
        pm,
        "_run_stage_selection_passes",
        lambda **kwargs: _selection(kwargs["stage"].id, "n8n-nodes-base.webhook")
        if kwargs["stage"].id == "stage_intake"
        else _selection(kwargs["stage"].id, "n8n-nodes-base.googleSheets"),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert updates["architecture_plan"] is not None
    assert len(updates["pm_stage_plan"]) == 2
    assert len(updates["pm_stage_selections"]) == 2
    assert updates["workflow_context"].planning_ready is True
    assert updates["architecture_plan"].required_nodes[0].node_type == "n8n-nodes-base.webhook"
    assert updates["proposed_nodes"][0].purpose


def test_pm_accepts_direct_build_request_without_selected_use_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pm, "_plan_stages_with_structured_output", lambda **_: _stage_plan())
    monkeypatch.setattr(
        pm,
        "_run_stage_selection_passes",
        lambda **kwargs: _selection(kwargs["stage"].id, "n8n-nodes-base.webhook"),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=None,
            entry_intent=EntryIntent.workflow_build_request,
            user_query="Create a webhook workflow that writes data to Google Sheets.",
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert "pm_use_case_derived_from_direct_build_request" in updates["routing_signals"]


def test_pm_blocks_and_requests_clarification_when_stage_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_blocked")
    monkeypatch.setattr(pm, "_plan_stages_with_structured_output", lambda **_: _stage_plan())
    monkeypatch.setattr(
        pm,
        "_run_stage_selection_passes",
        lambda **kwargs: _selection(kwargs["stage"].id, "n8n-nodes-base.webhook", gate_passed=False),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["pm_status"] == PMStatus.pm_blocked_waiting_user
    assert updates["target_stage"] is None
    assert updates["workflow_context"].planning_ready is False
    assert updates["pm_clarification_state"].attempts_used == 1
    assert updates["pm_clarification_state"].pending_questions
    assert "pm_blocked_waiting_user" in updates["routing_signals"]


def test_pm_fails_after_max_clarification_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_failed")
    monkeypatch.setattr(pm, "_plan_stages_with_structured_output", lambda **_: _stage_plan())
    monkeypatch.setattr(
        pm,
        "_run_stage_selection_passes",
        lambda **kwargs: _selection(kwargs["stage"].id, "n8n-nodes-base.webhook", gate_passed=False),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            extra={
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_stage_plan": _stage_plan(),
                "pm_clarification_state": PMClarificationState(
                    attempts_used=2,
                    max_attempts=2,
                    pending_questions=["Need integration details."],
                    turns=[],
                ),
            },
        )
    )

    assert updates["pm_status"] == PMStatus.pm_failed_no_solution
    assert updates["target_stage"] is None
    assert updates["workflow_context"].planning_ready is False
    assert "pm_failed_no_solution" in updates["routing_signals"]


def test_pm_resume_consumes_user_answer_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_resume")
    monkeypatch.setattr(pm, "_plan_stages_with_structured_output", lambda **_: _stage_plan())
    monkeypatch.setattr(
        pm,
        "_run_stage_selection_passes",
        lambda **kwargs: _selection(kwargs["stage"].id, "n8n-nodes-base.webhook"),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=selected,
            user_query="Use Gmail inbound trigger and persist rows to Sheets.",
            extra={
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_stage_plan": _stage_plan(),
                "pm_clarification_state": PMClarificationState(
                    attempts_used=1,
                    max_attempts=2,
                    pending_questions=["Need trigger and destination details."],
                    turns=[
                        {"stage_id": "stage_intake", "question": "Need trigger and destination details.", "answer": None}
                    ],
                ),
            },
        )
    )

    assert updates["pm_status"] == PMStatus.pm_completed
    assert updates["pm_clarification_state"].pending_questions == []
    assert updates["pm_clarification_state"].turns[-1].answer is not None
    assert "pm_clarification_answer_received" in updates["routing_signals"]


def test_extract_required_nodes_uses_strict_explicit_evidence_only() -> None:
    docs = [
        {
            "doc_id": "doc-with-node-line-only",
            "title": "Guide",
            "url": "https://docs.n8n.io/guides",
            "text": "Node Type: n8n-nodes-base.webhook",
            "metadata": {"kind": "DOCS_PAGE"},
        },
        {
            "doc_id": "linked:node:n8n-nodes-base.webhook",
            "context_kind": "linked_def",
            "linked_def_type": "node",
            "linked_entity_id": "n8n-nodes-base.webhook",
            "metadata": {"kind": "NODE_OVERVIEW", "nodeType": "n8n-nodes-base.webhook"},
            "text": "Kind: NODE_OVERVIEW\nNode Type: n8n-nodes-base.webhook\nSupports webhook trigger.",
            "link_confidence": 0.91,
        },
    ]
    required = pm.extract_required_nodes_from_docs(docs)
    assert [item.node_type for item in required] == ["n8n-nodes-base.webhook"]


def test_rerank_confidence_comes_from_docs_rerank_not_link_confidence() -> None:
    docs = [
        {
            "doc_id": "linked:node:webhook",
            "context_kind": "linked_def",
            "linked_def_type": "node",
            "linked_entity_id": "n8n-nodes-base.webhook",
            "link_doc_page_key": "page-webhook",
            "metadata": {"kind": "NODE_OVERVIEW", "nodeType": "n8n-nodes-base.webhook"},
            "text": "Kind: NODE_OVERVIEW\nNode Type: n8n-nodes-base.webhook\nCan receive events.",
            "link_confidence": 0.95,
        },
        {
            "doc_id": "docs:webhook",
            "url": "https://docs.n8n.io/webhook",
            "metadata": {"kind": "DOCS_PAGE", "page_id": "page-webhook"},
            "text": "Webhook node docs page.",
            "rerank_score": 0.81,
        },
        {
            "doc_id": "linked:node:code",
            "context_kind": "linked_def",
            "linked_def_type": "node",
            "linked_entity_id": "n8n-nodes-base.code",
            "metadata": {"kind": "NODE_OVERVIEW", "nodeType": "n8n-nodes-base.code"},
            "text": "Kind: NODE_OVERVIEW\nNode Type: n8n-nodes-base.code",
            "link_confidence": 0.93,
        },
    ]
    required = pm.extract_required_nodes_from_docs(docs)
    by_type = {item.node_type: item for item in required}
    assert by_type["n8n-nodes-base.webhook"].rerank_confidence is not None
    assert by_type["n8n-nodes-base.code"].rerank_confidence is None


def test_filter_tool_only_nodes_depends_on_use_case_mode() -> None:
    non_agentic = _use_case(use_case_id="uc_non_agentic", title="Standard sync workflow")
    agentic = _use_case(use_case_id="uc_agentic", title="AI agent tool calling workflow")

    tool_only = pm.NodeRequirement(
        node_type="n8n-nodes-base.emailReadImapTool",
        why_required="Tool node",
        evidence_chunk_ids=[],
        evidence_refs=[],
        evidence_confidence=0.7,
        usage_mode="tool_only",
    )
    action_node = pm.NodeRequirement(
        node_type="n8n-nodes-base.emailReadImap",
        why_required="Action node",
        evidence_chunk_ids=[],
        evidence_refs=[],
        evidence_confidence=0.7,
        usage_mode="action_only",
    )

    kept_non_agentic, dropped_non_agentic = pm._filter_required_nodes_for_usage(  # noqa: SLF001
        use_case=non_agentic,
        required_nodes=[tool_only, action_node],
    )
    kept_agentic, dropped_agentic = pm._filter_required_nodes_for_usage(  # noqa: SLF001
        use_case=agentic,
        required_nodes=[tool_only, action_node],
    )

    assert [item.node_type for item in kept_non_agentic] == ["n8n-nodes-base.emailReadImap"]
    assert [item.node_type for item in dropped_non_agentic] == ["n8n-nodes-base.emailReadImapTool"]
    assert len(kept_agentic) == 2
    assert dropped_agentic == []


def test_stage_selection_respects_max_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    use_case = _use_case(use_case_id="uc_passes")
    stage = _stage_plan()[0]
    calls = {"retrieve": 0}

    def _retrieve_docs(_query: str, request_id: str | None = None) -> List[Dict[str, Any]]:
        calls["retrieve"] += 1
        return []

    monkeypatch.setattr(pm.settings, "PM_MAX_STAGE_RETRIEVAL_PASSES", 3, raising=False)
    monkeypatch.setattr(pm, "retrieve_docs", _retrieve_docs)

    selection = pm._run_stage_selection_passes(  # noqa: SLF001
        use_case=use_case,
        stage=stage,
        model=None,
        request_id="req-pass-limit",
        clarification_state=PMClarificationState(attempts_used=0, max_attempts=2, pending_questions=[], turns=[]),
        trace_events=[],
    )

    assert calls["retrieve"] == 3
    assert selection.passes_used == 3
    assert selection.gate_passed is False


def test_hybrid_gate_policy_thresholds() -> None:
    assert pm._stage_gate_pass(pm_fit_score=0.72, rerank_confidence=0.56, top_margin=0.01) is True  # noqa: SLF001
    assert pm._stage_gate_pass(pm_fit_score=0.72, rerank_confidence=None, top_margin=0.11) is True  # noqa: SLF001
    assert pm._stage_gate_pass(pm_fit_score=0.69, rerank_confidence=0.90, top_margin=0.50) is False  # noqa: SLF001
    assert pm._stage_gate_pass(pm_fit_score=0.72, rerank_confidence=0.40, top_margin=0.05) is False  # noqa: SLF001


def test_extract_required_nodes_sanitizes_reference_spillover() -> None:
    docs = [
        {
            "doc_id": "linked:node:n8n-nodes-base.emailSend",
            "context_kind": "linked_def",
            "linked_def_type": "node",
            "linked_entity_id": "n8n-nodes-base.emailSend",
            "metadata": {"kind": "NODE_OVERVIEW", "nodeType": "n8n-nodes-base.emailSend"},
            "url": (
                "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.sendemail/"
                '&quot;], "confidence": 0.66}, {"node_type":"n8n-nodes-base.code"}'
            ),
            "text": "Kind: NODE_OVERVIEW\nNode Type: n8n-nodes-base.emailSend",
        }
    ]
    required = pm.extract_required_nodes_from_docs(docs)
    assert required[0].evidence_refs == [
        "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.sendemail/"
    ]
