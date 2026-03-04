from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .nodes import WorkflowNodes, scope_route
from .state import WorkflowGraphState

try:  # Optional until langgraph dependency is installed.
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency fallback
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


logger = logging.getLogger("n8n-assistant")


class WorkflowGraphRuntime:
    """Runs workflow-aware orchestration through LangGraph with sequential fallback."""

    def __init__(self, workflow_service: Any) -> None:
        self._nodes = WorkflowNodes(workflow_service)
        self._graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(WorkflowGraphState)
        graph.add_node("fetch_summary", self._nodes.fetch_summary_node)
        graph.add_node("scope_plan", self._nodes.scope_plan_node)
        graph.add_node("no_workflow", self._nodes.no_workflow_node)
        graph.add_node("clarify", self._nodes.clarify_node)
        graph.add_node("subgraph", self._nodes.subgraph_node)
        graph.add_node("retrieve_docs", self._nodes.retrieve_docs_node)
        graph.add_node("analyze_nodes", self._nodes.analyze_nodes_node)
        graph.add_node("build_prompt", self._nodes.build_prompt_node)

        graph.add_edge(START, "fetch_summary")
        graph.add_edge("fetch_summary", "scope_plan")
        graph.add_conditional_edges(
            "scope_plan",
            scope_route,
            {
                "no_workflow": "no_workflow",
                "clarify": "clarify",
                "subgraph": "subgraph",
            },
        )
        graph.add_edge("no_workflow", END)
        graph.add_edge("clarify", END)
        graph.add_edge("subgraph", "retrieve_docs")
        graph.add_edge("retrieve_docs", "analyze_nodes")
        graph.add_edge("analyze_nodes", "build_prompt")
        graph.add_edge("build_prompt", END)

        return graph.compile()

    def run(
        self,
        *,
        question: str,
        workflow_id: str,
        request_model: Optional[str],
        request_id: Optional[str],
        chat_context: str,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowGraphState:
        state: WorkflowGraphState = {
            "question": question,
            "workflow_id": workflow_id,
            "request_model": request_model,
            "request_id": request_id,
            "chat_context": chat_context,
            "fallback_to_docs_only": False,
            "debug_events": [],
        }

        if self._graph is None:
            return self._run_sequential(state)

        try:
            return self._graph.invoke(state, config=run_config)
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.warning("workflow graph fallback to sequential runtime: %s", str(exc))
            return self._run_sequential(state)

    def _run_sequential(self, state: WorkflowGraphState) -> WorkflowGraphState:
        current = dict(state)
        current.update(self._nodes.fetch_summary_node(current))
        current.update(self._nodes.scope_plan_node(current))

        scope_result = scope_route(current)
        if scope_result == "no_workflow":
            current.update(self._nodes.no_workflow_node(current))
            return current

        if scope_result == "clarify":
            current.update(self._nodes.clarify_node(current))
            return current

        current.update(self._nodes.subgraph_node(current))
        current.update(self._nodes.retrieve_docs_node(current))
        current.update(self._nodes.analyze_nodes_node(current))
        current.update(self._nodes.build_prompt_node(current))
        return current
