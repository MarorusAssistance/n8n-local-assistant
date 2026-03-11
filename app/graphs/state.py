from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, Literal, TypedDict

if False:  # pragma: no cover
    from ..reasoning.types import CheckerResult, ContextPack, PlanSpec, ReasoningPipelineResult, RouterOutput
    from ..token_budget import PromptBudgetResult
    from ..workflow.node_analyzer import NodeFinding
    from ..workflow.planner import Plan
    from ..workflow.subgraph_builder import SubgraphSelection
    from ..workflow.workflow_summary import WorkflowSummary


class ReasoningGraphState(TypedDict, total=False):
    user_prompt: str
    model: Optional[str]
    request_id: Optional[str]
    existing_workflow: Any
    conversation_context: List[Dict[str, str]]
    active_workflow_context: Dict[str, Any]
    router_output: Any
    context_pack: Any
    plan: Any
    checker: Any
    second_iteration_used: bool
    attempts: int
    metadata: Dict[str, int]
    debug_events: Annotated[List[str], operator.add]


class WorkflowGraphState(TypedDict, total=False):
    question: str
    workflow_id: str
    request_model: Optional[str]
    request_id: Optional[str]
    chat_context: str
    summary: Any
    plan: Any
    subgraph: Any
    docs_chunks: List[Dict[str, Any]]
    findings: List[Any]
    llm_messages: List[Dict[str, str]]
    prompt_budget: Any
    refs: List[Dict[str, str]]
    clarification_text: str
    fallback_to_docs_only: bool
    debug_events: Annotated[List[str], operator.add]


class MasterGraphState(TypedDict, total=False):
    mode: Literal["reasoning", "workflow"]
    run_config: Dict[str, Any]
    reasoning_input: ReasoningGraphState
    workflow_input: WorkflowGraphState
    reasoning_result: Any
    workflow_result: Any
