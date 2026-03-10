from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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


class NodeRequirement(BaseModel):
    node_type: str
    display_name: Optional[str] = None
    why_required: str
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    evidence_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rerank_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    blended_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    usage_mode: Literal["action_only", "tool_only", "both", "unknown"] = "unknown"
    usable_as_tool: Optional[bool] = None
    has_main_input: Optional[bool] = None
    input_connection_types: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_confidence(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if "evidence_confidence" in values:
            return values
        legacy = values.get("confidence")
        if isinstance(legacy, (int, float)):
            migrated = dict(values)
            migrated["evidence_confidence"] = float(legacy)
            return migrated
        return values


class ArchitectureStage(BaseModel):
    id: str
    name: str
    purpose: str
    required_capabilities: List[str] = Field(default_factory=list)
    expected_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ArchitectureDataFlowItem(BaseModel):
    source_stage_id: str
    target_stage_id: str
    data_items: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ArchitecturePlan(BaseModel):
    use_case_id: str
    title: str
    business_objective: str
    desired_outcome: str
    workflow_summary: str
    stages: List[ArchitectureStage] = Field(default_factory=list)
    data_flow: List[ArchitectureDataFlowItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    implementation_notes_for_engineer: List[str] = Field(default_factory=list)
    required_nodes: List[NodeRequirement] = Field(default_factory=list)


class WorkflowContext(BaseModel):
    use_case_id: str
    planning_ready: bool = False
    handoff_target: Optional[AgentStage] = None
    required_node_types: List[str] = Field(default_factory=list)
    unresolved_inputs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ImplementationStatus(str, Enum):
    ready = "ready"
    in_progress = "in_progress"
    blocked_waiting_user = "blocked_waiting_user"
    completed = "completed"
    failed = "failed"


class ProposedNode(BaseModel):
    node_id: str
    node_type: str
    stage_id: Optional[str] = None
    purpose: str = ""
    depends_on: List[str] = Field(default_factory=list)
    expected_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    usage_mode: Literal["action_only", "tool_only", "both", "unknown"] = "unknown"
    usable_as_tool: Optional[bool] = None
    has_main_input: Optional[bool] = None
    input_connection_types: List[str] = Field(default_factory=list)


class RequiredCredential(BaseModel):
    credential_key: str
    node_type: str
    credential_name: str
    required_for: Optional[str] = None
    source: str = "derived"


class WorkflowDraftNode(BaseModel):
    node_id: str
    name: str
    node_type: str
    purpose: str = ""
    stage_id: Optional[str] = None
    parameters_known: Dict[str, Any] = Field(default_factory=dict)
    parameters_inferred: Dict[str, Any] = Field(default_factory=dict)
    parameters_unresolved: List[str] = Field(default_factory=list)
    credential_refs: Dict[str, str] = Field(default_factory=dict)
    expected_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    position: List[int] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class WorkflowDraftConnection(BaseModel):
    source_node_id: str
    target_node_id: str
    source_output: str = "main"
    target_input: str = "main"


class WorkflowDraft(BaseModel):
    name: str
    use_case_id: Optional[str] = None
    summary: Optional[str] = None
    nodes: List[WorkflowDraftNode] = Field(default_factory=list)
    connections: List[WorkflowDraftConnection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowVersion(BaseModel):
    version: int = Field(ge=1)
    reason: str
    workflow_draft: WorkflowDraft
    implemented_node_count: int = Field(default=0, ge=0)


class ImplementationQueueItem(BaseModel):
    queue_id: str
    node_type: str
    stage_id: Optional[str] = None
    purpose: str = ""
    dependencies: List[str] = Field(default_factory=list)
    expected_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    status: Literal["pending", "implemented", "blocked"] = "pending"


class ImplementedNode(BaseModel):
    queue_id: str
    node_id: str
    node_type: str
    purpose: str = ""
    version: int = Field(default=1, ge=1)


class BlockedNode(BaseModel):
    queue_id: str
    node_type: str
    reason: str
    missing_input_ids: List[str] = Field(default_factory=list)


class VariableDefinition(BaseModel):
    name: str
    origin_node_id: str
    destination_node_ids: List[str] = Field(default_factory=list)
    semantic_meaning: str
    expected_format: Optional[str] = None
    mapping_notes: Optional[str] = None


class MissingUserInput(BaseModel):
    input_id: str
    input_key: str
    missing_item: str
    reason: str
    blocking_node_id: Optional[str] = None
    category: Literal[
        "credential",
        "parameter",
        "mapping",
        "business_rule",
        "handoff",
        "dependency",
    ] = "parameter"
    question: str


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
    architecture_plan: Optional[ArchitecturePlan] = None
    workflow_context: Optional[WorkflowContext] = None
    planning_summary: Optional[str] = None
    proposed_nodes: List[ProposedNode] = Field(default_factory=list)
    required_credentials: List[RequiredCredential] = Field(default_factory=list)
    workflow_draft: Optional[WorkflowDraft] = None
    workflow_versions: List[WorkflowVersion] = Field(default_factory=list)
    node_implementation_queue: List[ImplementationQueueItem] = Field(default_factory=list)
    implemented_nodes: List[ImplementedNode] = Field(default_factory=list)
    blocked_nodes: List[BlockedNode] = Field(default_factory=list)
    variable_registry: List[VariableDefinition] = Field(default_factory=list)
    missing_user_input_details: List[MissingUserInput] = Field(default_factory=list)
    implementation_status: Optional[ImplementationStatus] = None
    engineer_notes: List[str] = Field(default_factory=list)
    final_workflow_json: Dict[str, Any] = Field(default_factory=dict)
    active_workflow_id: Optional[str] = None
    active_workflow_name: Optional[str] = None
    active_workflow_url: Optional[str] = None
    workflow_persisted: bool = False
    workflow_persist_action: Optional[str] = None
    workflow_api_sync_result: Dict[str, Any] = Field(default_factory=dict)
    qa_enabled: bool = False
    needs_replan: bool = False
    status: Literal["stub_routed", "unknown_terminal"]
