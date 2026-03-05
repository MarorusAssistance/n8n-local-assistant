from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


RequirementType = Literal[
    "must_include_node_type",
    "must_include_keyword_in_node_params",
    "must_have_schedule_daily_at",
    "must_have_connection",
]

AdapterType = Literal["direct", "http"]

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class RequirementSpec(BaseModel):
    type: RequirementType
    value: Any = None

    @model_validator(mode="after")
    def _validate_value(self) -> "RequirementSpec":
        if self.type == "must_include_node_type":
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("must_include_node_type requires a non-empty string value")
            return self

        if self.type == "must_include_keyword_in_node_params":
            if not isinstance(self.value, dict):
                raise ValueError("must_include_keyword_in_node_params requires an object value")
            key = self.value.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("must_include_keyword_in_node_params requires value.key")
            if "equals" not in self.value and "contains" not in self.value:
                raise ValueError("must_include_keyword_in_node_params requires equals or contains")
            return self

        if self.type == "must_have_schedule_daily_at":
            if not isinstance(self.value, str) or not _TIME_RE.match(self.value.strip()):
                raise ValueError("must_have_schedule_daily_at requires HH:MM value")
            return self

        if self.type == "must_have_connection":
            if not isinstance(self.value, dict):
                raise ValueError("must_have_connection requires an object value")
            from_name = self.value.get("from")
            to_name = self.value.get("to")
            from_type = self.value.get("from_type")
            to_type = self.value.get("to_type")
            if not any((from_name and to_name, from_type and to_type)):
                raise ValueError(
                    "must_have_connection requires (from+to) or (from_type+to_type)"
                )
            return self

        return self


class CaseLimits(BaseModel):
    max_nodes: Optional[int] = Field(default=None, ge=1)


class CaseSpec(BaseModel):
    id: str
    user_message: str
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
        self.env_overrides = {str(key): str(value) for key, value in self.env_overrides.items()}
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
