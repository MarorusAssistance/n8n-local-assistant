from __future__ import annotations

from bench.scoring import compute_compliance_score


def test_compute_compliance_score_is_stable() -> None:
    score, breakdown = compute_compliance_score(
        parse_ok=True,
        section_ratios={
            "router_routing_graph": 0.5,
            "commercial_selection": 1.0,
            "product_manager_planning": 0.0,
            "planning_safety_guardrails": 0.5,
            "retrieval_trace_checks": 1.0,
        },
        section_failures={},
    )

    assert score == 65.0
    assert breakdown["json_parse_ok"]["score"] == 15.0
    assert breakdown["router_routing_graph"]["score"] == 10.0
    assert breakdown["product_manager_planning"]["score"] == 0.0


def test_compute_compliance_score_short_circuits_when_parse_fails() -> None:
    score, breakdown = compute_compliance_score(
        parse_ok=False,
        section_ratios={},
        section_failures={"json_parse_ok": ["no json"]},
    )

    assert score == 0.0
    assert breakdown["json_parse_ok"]["passed"] is False
    assert breakdown["router_routing_graph"]["skipped_due_parse_failure"] is True
