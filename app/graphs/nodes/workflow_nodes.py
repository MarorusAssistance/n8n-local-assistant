from __future__ import annotations

from typing import Any, Dict, List

from ...config import settings
from ...rag import collect_references
from ...token_budget import trim_messages_to_budget
from ...workflow.planner import PlanScope, plan_scope
from ...workflow.subgraph_builder import build_subgraph
from ...workflow.synthesizer import build_synthesis_messages
from ...workflow.workflow_summary import build_workflow_summary
from ..state import WorkflowGraphState


def _apply_prompt_budget_with_docs_fallback(
    docs_chunks: List[Dict[str, Any]],
    message_builder,
) -> tuple[List[Dict[str, str]], List[Dict[str, Any]], Any]:
    limit = max(settings.CONVERSATION_MAX_TOKENS, 256)
    current_chunks = list(docs_chunks)

    while True:
        llm_messages = message_builder(current_chunks)
        result = trim_messages_to_budget(llm_messages, max_tokens=limit)
        if result.estimated_tokens_after <= limit or not current_chunks:
            return result.messages, current_chunks, result
        current_chunks = current_chunks[:-1]


class WorkflowNodes:
    def __init__(self, workflow_service: Any) -> None:
        self._workflow = workflow_service

    def fetch_summary_node(self, state: WorkflowGraphState) -> Dict[str, Any]:
        workflow = self._workflow.fetch_workflow(state["workflow_id"])
        summary = build_workflow_summary(workflow, workflow_id=state["workflow_id"])
        return {"summary": summary, "debug_events": ["fetch_summary"]}

    def scope_plan_node(self, state: WorkflowGraphState) -> Dict[str, Any]:
        plan = plan_scope(state["question"], state["summary"])
        return {"plan": plan, "debug_events": ["scope_plan"]}

    def no_workflow_node(self, _state: WorkflowGraphState) -> Dict[str, Any]:
        return {"fallback_to_docs_only": True, "debug_events": ["fallback_docs_only"]}

    def clarify_node(self, state: WorkflowGraphState) -> Dict[str, Any]:
        text = self._workflow.clarification_text(state["plan"], state["summary"])
        return {"clarification_text": text, "debug_events": ["clarify"]}

    def subgraph_node(self, state: WorkflowGraphState) -> Dict[str, Any]:
        subgraph = build_subgraph(state["summary"], state["question"], state["plan"])
        return {"subgraph": subgraph, "debug_events": ["subgraph"]}

    def retrieve_docs_node(self, state: WorkflowGraphState) -> Dict[str, Any]:
        docs_chunks = self._workflow.retrieve_docs_for_subgraph(
            state["question"],
            state["summary"],
            state["subgraph"],
            request_id=state.get("request_id"),
        )
        return {"docs_chunks": docs_chunks, "debug_events": ["retrieve_docs"]}

    def analyze_nodes_node(self, state: WorkflowGraphState) -> Dict[str, Any]:
        findings = self._workflow.analyze_nodes(
            question=state["question"],
            summary=state["summary"],
            subgraph=state["subgraph"],
            docs_chunks=state["docs_chunks"],
            plan=state["plan"],
            requested_model=state.get("request_model"),
            request_id=state.get("request_id"),
        )
        return {"findings": findings, "debug_events": ["analyze_nodes"]}

    def build_prompt_node(self, state: WorkflowGraphState) -> Dict[str, Any]:
        def _build(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
            return build_synthesis_messages(
                question=state["question"],
                plan=state["plan"],
                summary=state["summary"],
                subgraph=state["subgraph"],
                findings=state["findings"],
                docs_chunks=chunks,
                chat_context=state.get("chat_context") or "",
            )

        llm_messages, docs_chunks, prompt_budget = _apply_prompt_budget_with_docs_fallback(
            docs_chunks=state["docs_chunks"],
            message_builder=_build,
        )
        refs = collect_references(docs_chunks)
        return {
            "llm_messages": llm_messages,
            "docs_chunks": docs_chunks,
            "prompt_budget": prompt_budget,
            "refs": refs,
            "debug_events": ["build_prompt"],
        }


def scope_route(state: WorkflowGraphState) -> str:
    plan = state["plan"]
    if plan.scope == PlanScope.no_workflow:
        return "no_workflow"
    if plan.need_clarification:
        return "clarify"
    return "subgraph"
