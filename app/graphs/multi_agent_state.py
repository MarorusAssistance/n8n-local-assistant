from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from ..features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitecturePlan,
    BlockedNode,
    BusinessContextSummary,
    EntryIntent,
    ImplementationQueueItem,
    ImplementationStatus,
    ImplementedNode,
    MissingUserInput,
    ProposedNode,
    RequiredCredential,
    UseCase,
    VariableDefinition,
    WorkflowDraft,
    WorkflowVersion,
    WorkflowContext,
)


class MultiAgentGraphState(TypedDict, total=False):
    user_query: str
    entry_intent: EntryIntent
    target_stage: Optional[AgentStage]
    confidence: float
    routing_signals: List[str]
    current_stage: Optional[str]
    business_context_summary: Optional[BusinessContextSummary]
    discovered_use_cases: List[UseCase]
    selected_use_case: Optional[UseCase]
    alternative_use_cases: List[UseCase]
    selection_reason: Optional[str]
    workflow_context: Optional[WorkflowContext | Dict[str, Any]]
    architecture_plan: Optional[ArchitecturePlan | Dict[str, Any]]
    planning_summary: Optional[str]
    proposed_nodes: List[ProposedNode | Dict[str, Any]]
    required_credentials: List[RequiredCredential | Dict[str, Any]]
    workflow_draft: Optional[WorkflowDraft | Dict[str, Any]]
    workflow_versions: List[WorkflowVersion | Dict[str, Any]]
    node_implementation_queue: List[ImplementationQueueItem | Dict[str, Any]]
    implemented_nodes: List[ImplementedNode | Dict[str, Any]]
    blocked_nodes: List[BlockedNode | Dict[str, Any]]
    variable_registry: List[VariableDefinition | Dict[str, Any]]
    missing_user_inputs: List[str]
    missing_user_input_details: List[MissingUserInput | Dict[str, Any]]
    implementation_status: Optional[ImplementationStatus | str]
    engineer_notes: List[str]
    qa_enabled: bool
    qa_result: Dict[str, Any]
    needs_replan: bool
    final_workflow_json: Dict[str, Any]
    active_workflow_id: Optional[str]
    active_workflow_name: Optional[str]
    active_workflow_url: Optional[str]
    workflow_persisted: bool
    workflow_persist_action: Optional[str]
    workflow_api_sync_result: Dict[str, Any]
    runtime_context: Dict[str, Any]
    resume_requested: bool
