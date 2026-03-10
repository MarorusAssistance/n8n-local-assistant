from __future__ import annotations

from pathlib import Path

import pytest

from bench.io import load_cases, load_experiments
from bench.models import CasesFile


def test_load_cases_and_experiments_yaml() -> None:
    cases = load_cases(Path("bench/cases.yaml"))
    experiments = load_experiments(Path("bench/experiments.yaml"))

    assert len(cases) >= 6
    assert any(case.id == "product_manager_build_webhook_sheets" for case in cases)
    assert {"router", "commercial", "product_manager"} <= {
        case.stage for case in cases
    }
    assert all(case.user_message.strip() for case in cases)

    assert len(experiments) >= 3
    assert experiments[0].name == "baseline_langgraph"


def test_case_validation_fails_for_unknown_requirement_type() -> None:
    with pytest.raises(Exception):
        CasesFile.from_raw(
            [
                {
                    "id": "bad_case",
                    "user_message": "Hola",
                    "requirements": [
                        {"type": "unsupported_requirement", "value": True}
                    ],
                }
            ]
        )
