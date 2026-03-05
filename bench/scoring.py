from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple


SECTION_WEIGHTS: Dict[str, float] = {
    "json_parse_ok": 20.0,
    "workflow_min_schema": 20.0,
    "node_types_exist": 20.0,
    "credentials_shape_and_existence": 15.0,
    "credential_compatibility": 10.0,
    "requirements_and_limits": 15.0,
}


def compute_compliance_score(
    *,
    parse_ok: bool,
    section_ratios: Mapping[str, float],
    section_failures: Mapping[str, Sequence[str]],
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    breakdown: Dict[str, Dict[str, Any]] = {}

    if not parse_ok:
        breakdown["json_parse_ok"] = {
            "ratio": 0.0,
            "weight": SECTION_WEIGHTS["json_parse_ok"],
            "score": 0.0,
            "passed": False,
            "failures": list(section_failures.get("json_parse_ok", [])),
            "skipped_due_parse_failure": False,
        }
        for section, weight in SECTION_WEIGHTS.items():
            if section == "json_parse_ok":
                continue
            breakdown[section] = {
                "ratio": 0.0,
                "weight": weight,
                "score": 0.0,
                "passed": False,
                "failures": list(section_failures.get(section, [])),
                "skipped_due_parse_failure": True,
            }
        return 0.0, breakdown

    total_score = 0.0
    for section, weight in SECTION_WEIGHTS.items():
        if section == "json_parse_ok":
            ratio = 1.0
        else:
            ratio = float(section_ratios.get(section, 0.0))
            ratio = max(0.0, min(1.0, ratio))
        score = round(weight * ratio, 4)
        total_score += score
        failures = list(section_failures.get(section, []))
        breakdown[section] = {
            "ratio": ratio,
            "weight": weight,
            "score": score,
            "passed": ratio >= 1.0 and not failures,
            "failures": failures,
            "skipped_due_parse_failure": False,
        }

    return round(total_score, 4), breakdown
