from __future__ import annotations

import json

from bench.checks import run_checks
from bench.models import CaseLimits, CaseSpec, RequirementSpec
from bench.trace import parse_trace_text


def _m3_payload() -> dict:
    return {
        "entry_intent": "workflow_build_request",
        "target_stage": "engineer_agent",
        "confidence": 0.86,
        "routing_signals": ["build_signals_detected", "handoff_ready_engineer"],
        "current_stage": "product_manager_agent",
        "missing_user_inputs": [],
        "status": "stub_routed",
        "selection_reason": "Top business value candidate was selected.",
        "alternative_use_cases": [],
        "architecture_plan": {
            "use_case_id": "uc_1",
            "required_nodes": [
                {
                    "node_type": "n8n-nodes-base.webhook",
                    "why_required": "explicit evidence",
                    "evidence_chunk_ids": ["linked:node:n8n-nodes-base.webhook"],
                    "evidence_refs": ["Node: Webhook"],
                    "confidence": 0.88,
                }
            ],
        },
        "workflow_context": {
            "use_case_id": "uc_1",
            "planning_ready": True,
            "handoff_target": "engineer_agent",
            "required_node_types": ["n8n-nodes-base.webhook"],
            "unresolved_inputs": [],
            "notes": ["required_nodes=1"],
        },
    }


def _m3_trace_entries():
    trace_text = """2026-03-07 10:00:00,000 INFO n8n-assistant.trace: TRACE EVENT {"event":"retrieval_final","request_id":"req-1","stage":"rag.retrieve_context","pre_rerank_chunks":[{"content":"doc pool","rrf_score":0.111}],"post_rerank_chunks":[{"content":"doc top","rerank_score":0.812}],"final_chunks":[{"content":"doc top","rerank_score":0.812},{"content":"node linked","rerank_score":0.801,"linked_def_type":"node"},{"content":"credential linked","rerank_score":0.755,"linked_def_type":"credential"}]}
"""
    return parse_trace_text(trace_text)


def test_run_checks_accepts_valid_multi_agent_payload() -> None:
    case = CaseSpec(
        id="product_manager_case_ok",
        stage="product_manager",
        user_message="x",
        requirements=[
            RequirementSpec(type="must_have_required_nodes_min", value=1),
            RequirementSpec(type="must_have_required_nodes_with_evidence", value=True),
            RequirementSpec(type="must_have_planning_ready", value=True),
            RequirementSpec(type="must_have_handoff_target", value="engineer_agent"),
            RequirementSpec(
                type="must_not_include_workflow_json_keys",
                value=["nodes", "connections", "final_workflow_json"],
            ),
        ],
        limits=CaseLimits(max_missing_user_inputs=1),
    )

    result = run_checks(
        json.dumps(_m3_payload()),
        case=case,
        trace_entries=_m3_trace_entries(),
    )

    assert result["parse_ok"] is True
    assert result["compliance_score"] == 100.0
    assert result["failures"] == []


def test_run_checks_detects_required_nodes_min_failure() -> None:
    payload = _m3_payload()
    payload["architecture_plan"]["required_nodes"] = []
    payload["workflow_context"]["planning_ready"] = False
    payload["workflow_context"]["handoff_target"] = None
    payload["target_stage"] = None

    case = CaseSpec(
        id="product_manager_case_fail_nodes",
        stage="product_manager",
        user_message="x",
        requirements=[RequirementSpec(type="must_have_required_nodes_min", value=1)],
    )

    result = run_checks(
        json.dumps(payload),
        case=case,
        trace_entries=_m3_trace_entries(),
    )

    assert result["parse_ok"] is True
    assert result["compliance_score"] < 100.0
    failures = result["trace"]["product_manager_planning"]["failures"]
    assert any("required_nodes count too low" in message for message in failures)


def test_run_checks_detects_incomplete_retrieval_trace_for_product_manager() -> None:
    case = CaseSpec(
        id="product_manager_case_trace_fail",
        stage="product_manager",
        user_message="x",
    )
    trace_text = """2026-03-07 10:00:00,000 INFO n8n-assistant.trace: TRACE EVENT {"event":"retrieval_final","request_id":"req-1","stage":"rag.retrieve_context","pre_rerank_chunks":[],"post_rerank_chunks":[],"final_chunks":[]}
"""

    result = run_checks(
        json.dumps(_m3_payload()),
        case=case,
        trace_entries=parse_trace_text(trace_text),
    )

    assert result["parse_ok"] is True
    assert result["compliance_score"] < 100.0
    trace_failures = result["trace"]["retrieval_trace_checks"]["failures"]
    assert any("pre-rerank" in message for message in trace_failures)
