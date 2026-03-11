from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from ..features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitecturePlan,
    BlockedNode,
    BusinessContextSummary,
    ConsultantQueryAnalysis,
    ConsultantResponse,
    ConsultantRetrievalResult,
    ConsultantSource,
    ConsultantToolUsage,
    EntryIntent,
    ImplementationQueueItem,
    ImplementationStatus,
    ImplementedNode,
    PMClarificationState,
    PMProgressState,
    PMStagePlan,
    PMStageSearchState,
    PMStageSelection,
    PMStatus,
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
    consultant_query_analysis: Optional[ConsultantQueryAnalysis | Dict[str, Any]]
    consultant_selected_sources: List[ConsultantSource | str]
    consultant_tools_used: List[ConsultantToolUsage | Dict[str, Any]]
    consultant_used_retrieval: bool
    consultant_retrieval_results: List[ConsultantRetrievalResult | Dict[str, Any]]
    consultant_response: Optional[ConsultantResponse | Dict[str, Any]]
    consultant_notes: List[str]
    business_context_summary: Optional[BusinessContextSummary]
    discovered_use_cases: List[UseCase]
    selected_use_case: Optional[UseCase]
    alternative_use_cases: List[UseCase]
    selection_reason: Optional[str]
    workflow_context: Optional[WorkflowContext | Dict[str, Any]]
    architecture_plan: Optional[ArchitecturePlan | Dict[str, Any]]
    planning_summary: Optional[str]
    pm_status: Optional[PMStatus | str]
    pm_stage_plan: List[PMStagePlan | Dict[str, Any]]
    pm_stage_selections: List[PMStageSelection | Dict[str, Any]]
    pm_stage_progress: Optional[PMProgressState | Dict[str, Any]]
    pm_clarification_state: Optional[PMClarificationState | Dict[str, Any]]
    pm_stage_search_history: List[PMStageSearchState | Dict[str, Any]]
    pm_reasoning_trace_full: List[Dict[str, Any]]
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
