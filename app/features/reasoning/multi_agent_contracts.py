from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EntryIntent(str, Enum):
    business_discovery_conversation = "business_discovery_conversation"
    workflow_build_request = "workflow_build_request"
    workflow_edit_request = "workflow_edit_request"
    workflow_fix_request = "workflow_fix_request"
    information_request = "information_request"
    unknown = "unknown"


class AgentStage(str, Enum):
    commercial_agent = "commercial_agent"
    consultant_agent = "consultant_agent"
    product_manager_agent = "product_manager_agent"
    engineer_agent = "engineer_agent"
    qa_agent = "qa_agent"


class UseCase(BaseModel):
    id: str
    title: str
    business_problem: str
    desired_outcome: str
    expected_value: str
    feasibility: str
    priority_score: float
    why_selected: str


class BusinessContextSummary(BaseModel):
    process_scope: str = ""
    pain_points: List[str] = Field(default_factory=list)
    desired_outcomes: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class TemplateCandidate(BaseModel):
    template_id: str
    title: str
    source: Optional[str] = None
    summary: str
    fit_score: float = Field(ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)


class EntryRouterDecision(BaseModel):
    entry_intent: EntryIntent
    target_stage: Optional[AgentStage] = None
    confidence: float = Field(ge=0.0, le=1.0)
    routing_signals: List[str] = Field(default_factory=list)
    missing_user_inputs: List[str] = Field(default_factory=list)


class MultiAgentGraphResult(BaseModel):
    user_query: str
    entry_intent: EntryIntent
    target_stage: Optional[AgentStage] = None
    confidence: float = Field(ge=0.0, le=1.0)
    routing_signals: List[str] = Field(default_factory=list)
    current_stage: Optional[str] = None
    missing_user_inputs: List[str] = Field(default_factory=list)
    business_context_summary: Optional[BusinessContextSummary] = None
    discovered_use_cases: List[UseCase] = Field(default_factory=list)
    selected_use_case: Optional[UseCase] = None
    alternative_use_cases: List[UseCase] = Field(default_factory=list)
    selection_reason: Optional[str] = None
    qa_enabled: bool = False
    needs_replan: bool = False
    status: Literal["stub_routed", "unknown_terminal"]
