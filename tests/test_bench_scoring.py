from __future__ import annotations

from bench.scoring import compute_compliance_score


def test_compute_compliance_score_is_stable() -> None:
    score, breakdown = compute_compliance_score(
        parse_ok=True,
        section_ratios={
            "workflow_min_schema": 0.5,
            "node_types_exist": 1.0,
            "credentials_shape_and_existence": 1.0,
            "credential_compatibility": 0.0,
            "requirements_and_limits": 0.5,
        },
        section_failures={},
    )

    assert score == 72.5
    assert breakdown["json_parse_ok"]["score"] == 20.0
    assert breakdown["workflow_min_schema"]["score"] == 10.0
    assert breakdown["credential_compatibility"]["score"] == 0.0


def test_compute_compliance_score_short_circuits_when_parse_fails() -> None:
    score, breakdown = compute_compliance_score(
        parse_ok=False,
        section_ratios={},
        section_failures={"json_parse_ok": ["no json"]},
    )

    assert score == 0.0
    assert breakdown["json_parse_ok"]["passed"] is False
    assert breakdown["workflow_min_schema"]["skipped_due_parse_failure"] is True
