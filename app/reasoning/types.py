from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RouterConstraints(BaseModel):
    services: List[str] = Field(default_factory=list)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    nonFunctional: List[str] = Field(default_factory=list)


class RouterOutput(BaseModel):
    intent: Literal["create", "fix", "extend"]
    goal: str
    constraints: RouterConstraints = Field(default_factory=RouterConstraints)
    missing_info: List[str] = Field(default_factory=list)
    complexity_score: Literal[1, 2, 3] = 1


class NodeCard(BaseModel):
    type: str
    displayName: Optional[str] = None
    purpose: Optional[str] = None
    requiredCredentials: List[str] = Field(default_factory=list)
    keyParams: List[str] = Field(default_factory=list)
    gotchas: List[str] = Field(default_factory=list)


class ContextDocChunk(BaseModel):
    id: str
    title: Optional[str] = None
    text: str
    source: Literal["docs", "nodes", "credentials"]


class ContextPackBudget(BaseModel):
    maxNodeCards: int
    maxDocChunks: int
    maxContextTokens: int
    estimatedTokens: int


class ContextPack(BaseModel):
    nodeCards: List[NodeCard] = Field(default_factory=list)
    docChunks: List[ContextDocChunk] = Field(default_factory=list)
    budget: ContextPackBudget


class PlanStep(BaseModel):
    id: str
    nodeType: str
    purpose: str
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    credentialsNeeded: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class PlanSpec(BaseModel):
    summary: str
    steps: List[PlanStep] = Field(default_factory=list)
    dataFlowNotes: List[str] = Field(default_factory=list)
    questionsForUser: List[str] = Field(default_factory=list)


class CheckerIssue(BaseModel):
    severity: Literal["error", "warn"]
    code: str
    msg: str
    stepId: Optional[str] = None


class CheckerResult(BaseModel):
    ok: bool
    issues: List[CheckerIssue] = Field(default_factory=list)


class ReasoningPipelineResult(BaseModel):
    plan: PlanSpec
    checker: CheckerResult
    router: RouterOutput
    context_pack: ContextPack
    second_iteration_used: bool = False
    attempts: int = 1
    metadata: Dict[str, int] = Field(default_factory=dict)
