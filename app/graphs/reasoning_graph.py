from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..reasoning.pipeline import run_reasoning_pipeline
from ..reasoning.types import ReasoningPipelineResult
from .nodes import check_node, context_pack_node, finalize_node, plan_node, revise_node, route_node
from .state import ReasoningGraphState

try:  # Optional until langgraph dependency is installed.
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency fallback
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


logger = logging.getLogger("n8n-assistant")


def _check_route(state: ReasoningGraphState) -> str:
    checker = state.get("checker")
    if checker is None:
        return "finalize"

    has_errors = any(issue.severity == "error" for issue in checker.issues)
    if has_errors and not state.get("second_iteration_used", False):
        return "revise"
    return "finalize"


class ReasoningGraphRuntime:
    """Runs the reasoning pipeline through LangGraph with fallback to legacy orchestration."""

    def __init__(self) -> None:
        self._graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(ReasoningGraphState)
        graph.add_node("route", route_node)
        graph.add_node("context_pack", context_pack_node)
        graph.add_node("plan", plan_node)
        graph.add_node("check", check_node)
        graph.add_node("revise", revise_node)
        graph.add_node("finalize", finalize_node)

        graph.add_edge(START, "route")
        graph.add_edge("route", "context_pack")
        graph.add_edge("context_pack", "plan")
        graph.add_edge("plan", "check")
        graph.add_conditional_edges(
            "check",
            _check_route,
            {
                "revise": "revise",
                "finalize": "finalize",
            },
        )
        graph.add_edge("revise", "check")
        graph.add_edge("finalize", END)
        return graph.compile()

    def run(
        self,
        *,
        user_prompt: str,
        model: Optional[str],
        request_id: Optional[str],
        existing_workflow: Any,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> ReasoningPipelineResult:
        if self._graph is None:
            return run_reasoning_pipeline(
                user_prompt=user_prompt,
                model=model,
                request_id=request_id,
                existing_workflow=existing_workflow,
            )

        state: ReasoningGraphState = {
            "user_prompt": user_prompt,
            "model": model,
            "request_id": request_id,
            "existing_workflow": existing_workflow,
            "second_iteration_used": False,
            "attempts": 0,
            "debug_events": [],
        }

        try:
            result_state = self._graph.invoke(state, config=run_config)
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.warning("reasoning graph fallback to legacy pipeline: %s", str(exc))
            return run_reasoning_pipeline(
                user_prompt=user_prompt,
                model=model,
                request_id=request_id,
                existing_workflow=existing_workflow,
            )

        return ReasoningPipelineResult(
            plan=result_state["plan"],
            checker=result_state["checker"],
            router=result_state["router_output"],
            context_pack=result_state["context_pack"],
            second_iteration_used=bool(result_state.get("second_iteration_used", False)),
            attempts=int(result_state.get("attempts", 1)),
            metadata=dict(result_state.get("metadata") or {}),
        )
