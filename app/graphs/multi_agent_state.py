from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from ..features.reasoning.multi_agent_contracts import (
    AgentStage,
    BusinessContextSummary,
    EntryIntent,
    UseCase,
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
    workflow_context: Dict[str, Any]
    architecture_plan: Dict[str, Any]
    missing_user_inputs: List[str]
    qa_enabled: bool
    qa_result: Dict[str, Any]
    needs_replan: bool
    final_workflow_json: Dict[str, Any]
