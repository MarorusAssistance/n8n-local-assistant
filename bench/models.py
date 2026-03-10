from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


RequirementType = Literal[
    "must_equal_field",
    "must_not_equal_field",
    "must_be_null_field",
    "must_not_be_null_field",
    "must_include_routing_signal",
    "must_not_include_routing_signal",
    "must_have_required_nodes_min",
    "must_have_required_nodes_with_evidence",
    "must_not_include_workflow_json_keys",
    "must_have_planning_ready",
    "must_have_handoff_target",
]

AdapterType = Literal["direct", "http"]
StageType = Literal["router", "commercial", "product_manager"]

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_STAGE_CANONICAL_MAP = {
    "router": "router",
    "commercial": "commercial",
    "product_manager": "product_manager",
}


class RequirementSpec(BaseModel):
    type: RequirementType
    value: Any = None

    @model_validator(mode="after")
    def _validate_value(self) -> "RequirementSpec":
        if self.type in {
            "must_equal_field",
            "must_not_equal_field",
        }:
            if not isinstance(self.value, dict):
                raise ValueError(f"{self.type} requires an object value")
            path = self.value.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{self.type} requires value.path")
            required_key = "equals" if self.type == "must_equal_field" else "not_equals"
            if required_key not in self.value:
                raise ValueError(f"{self.type} requires value.{required_key}")
            return self

        if self.type in {"must_be_null_field", "must_not_be_null_field"}:
            if not isinstance(self.value, dict):
                raise ValueError(f"{self.type} requires an object value")
            path = self.value.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{self.type} requires value.path")
            return self

        if self.type in {"must_include_routing_signal", "must_not_include_routing_signal"}:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError(f"{self.type} requires a non-empty string value")
            return self

        if self.type == "must_have_required_nodes_min":
            if not isinstance(self.value, int) or self.value < 0:
                raise ValueError("must_have_required_nodes_min requires integer >= 0")
            return self

        if self.type == "must_have_required_nodes_with_evidence":
            if not isinstance(self.value, bool):
                raise ValueError("must_have_required_nodes_with_evidence requires boolean value")
            return self

        if self.type == "must_not_include_workflow_json_keys":
            if not isinstance(self.value, list) or not self.value:
                raise ValueError("must_not_include_workflow_json_keys requires non-empty list")
            if not all(isinstance(item, str) and item.strip() for item in self.value):
                raise ValueError(
                    "must_not_include_workflow_json_keys requires non-empty string keys"
                )
            return self

        if self.type == "must_have_planning_ready":
            if not isinstance(self.value, bool):
                raise ValueError("must_have_planning_ready requires boolean value")
            return self

        if self.type == "must_have_handoff_target":
            if self.value is not None and not isinstance(self.value, str):
                raise ValueError("must_have_handoff_target requires string or null value")
            return self

        return self


class CaseLimits(BaseModel):
    max_missing_user_inputs: Optional[int] = Field(default=None, ge=0)


class CaseSpec(BaseModel):
    id: str
    user_message: str
    stage: StageType = "product_manager"
    json_only: bool = False
    requirements: List[RequirementSpec] = Field(default_factory=list)
    limits: Optional[CaseLimits] = None

    @model_validator(mode="after")
    def _validate_case(self) -> "CaseSpec":
        if not self.id.strip():
            raise ValueError("case id cannot be empty")
        if not _CASE_ID_RE.match(self.id):
            raise ValueError(
                "case id can only use letters, numbers, '.', '_' and '-'"
            )
        if not self.user_message.strip():
            raise ValueError("user_message cannot be empty")
        normalized = _STAGE_CANONICAL_MAP.get(str(self.stage))
        if not normalized:
            raise ValueError(
                "stage must be one of: router, commercial, product_manager"
            )
        self.stage = normalized  # type: ignore[assignment]
        return self


class ExperimentSpec(BaseModel):
    name: str
    env_overrides: Dict[str, str] = Field(default_factory=dict)
    warmup: int = Field(default=0, ge=0)
    repetitions: int = Field(default=1, ge=1)
    adapter: AdapterType = "direct"
    json_only: bool = False

    @model_validator(mode="after")
    def _validate_experiment(self) -> "ExperimentSpec":
        if not self.name.strip():
            raise ValueError("experiment name cannot be empty")
        self.env_overrides = {
            str(key): str(value)
            for key, value in self.env_overrides.items()
        }
        return self


class CasesFile(BaseModel):
    cases: List[CaseSpec]

    @classmethod
    def from_raw(cls, raw: Any) -> "CasesFile":
        if isinstance(raw, list):
            return cls(cases=raw)
        if isinstance(raw, dict) and "cases" in raw:
            return cls(cases=raw["cases"])
        raise ValueError("cases file must be a list or an object with 'cases'")


class ExperimentsFile(BaseModel):
    experiments: List[ExperimentSpec]

    @classmethod
    def from_raw(cls, raw: Any) -> "ExperimentsFile":
        if isinstance(raw, list):
            return cls(experiments=raw)
        if isinstance(raw, dict) and "experiments" in raw:
            return cls(experiments=raw["experiments"])
        raise ValueError("experiments file must be a list or an object with 'experiments'")


class CheckSection(BaseModel):
    ratio: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    score: float = Field(ge=0.0)
    passed: bool
    failures: List[str] = Field(default_factory=list)
    skipped_due_parse_failure: bool = False


class RunArtifacts(BaseModel):
    response: str
    checks: str
    workflow: Optional[str] = None
    diff: Optional[str] = None
    logs_html: Optional[str] = None
    trace_events: Optional[str] = None
    trace_slice: Optional[str] = None
    prompts: Optional[str] = None
    prompt_diff: Optional[str] = None
    logs_note: Optional[str] = None


class RunRecord(BaseModel):
    experiment: str
    case_id: str
    stage: StageType
    rep: int = Field(ge=1)
    is_warmup: bool
    conversation_id: Optional[str] = None
    trace_request_id: Optional[str] = None
    started_at: str
    ended_at: str
    latency_ms: float = Field(ge=0.0)
    success: bool
    error: Optional[str] = None
    http_status: Optional[int] = None
    parse_ok: bool
    compliance_score: float = Field(ge=0.0, le=100.0)
    breakdown: Dict[str, CheckSection]
    artifacts: RunArtifacts
