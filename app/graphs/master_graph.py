from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .reasoning_graph import ReasoningGraphRuntime
from .state import MasterGraphState
from .workflow_graph import WorkflowGraphRuntime

try:  # Optional until langgraph dependency is installed.
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency fallback
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


logger = logging.getLogger("n8n-assistant")


def _route_mode(state: MasterGraphState) -> str:
    mode = state.get("mode")
    if mode == "workflow":
        return "workflow"
    return "reasoning"


class MasterGraphRuntime:
    """Master runtime that delegates to ReasoningGraph or WorkflowGraph."""

    def __init__(self, workflow_service: Any) -> None:
        self.reasoning_runtime = ReasoningGraphRuntime()
        self.workflow_runtime = WorkflowGraphRuntime(workflow_service)
        self._graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(MasterGraphState)
        graph.add_node("reasoning", self._reasoning_node)
        graph.add_node("workflow", self._workflow_node)
        graph.add_conditional_edges(
            START,
            _route_mode,
            {
                "reasoning": "reasoning",
                "workflow": "workflow",
            },
        )
        graph.add_edge("reasoning", END)
        graph.add_edge("workflow", END)
        return graph.compile()

    def _reasoning_node(self, state: MasterGraphState) -> Dict[str, Any]:
        input_state = state.get("reasoning_input") or {}
        run_config = state.get("run_config") if isinstance(state.get("run_config"), dict) else None
        result = self.reasoning_runtime.run(
            user_prompt=str(input_state.get("user_prompt") or ""),
            model=input_state.get("model"),
            request_id=input_state.get("request_id"),
            existing_workflow=input_state.get("existing_workflow"),
            conversation_context=(
                input_state.get("conversation_context")
                if isinstance(input_state.get("conversation_context"), list)
                else None
            ),
            active_workflow_context=(
                input_state.get("active_workflow_context")
                if isinstance(input_state.get("active_workflow_context"), dict)
                else None
            ),
            run_config=run_config,
        )
        return {"reasoning_result": result}

    def _workflow_node(self, state: MasterGraphState) -> Dict[str, Any]:
        input_state = state.get("workflow_input") or {}
        run_config = state.get("run_config") if isinstance(state.get("run_config"), dict) else None
        result = self.workflow_runtime.run(
            question=str(input_state.get("question") or ""),
            workflow_id=str(input_state.get("workflow_id") or ""),
            request_model=input_state.get("request_model"),
            request_id=input_state.get("request_id"),
            chat_context=str(input_state.get("chat_context") or ""),
            run_config=run_config,
        )
        return {"workflow_result": result}

    def run(self, state: MasterGraphState) -> MasterGraphState:
        if self._graph is None:
            routed = _route_mode(state)
            if routed == "workflow":
                return {**state, **self._workflow_node(state)}
            return {**state, **self._reasoning_node(state)}

        try:
            return self._graph.invoke(state)
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.warning("master graph fallback to direct routing: %s", str(exc))
            routed = _route_mode(state)
            if routed == "workflow":
                return {**state, **self._workflow_node(state)}
            return {**state, **self._reasoning_node(state)}

    def run_reasoning(
        self,
        *,
        user_prompt: str,
        model: Optional[str],
        request_id: Optional[str],
        existing_workflow: Any,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        active_workflow_context: Optional[Dict[str, Any]] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        output = self.run(
            {
                "mode": "reasoning",
                "reasoning_input": {
                    "user_prompt": user_prompt,
                    "model": model,
                    "request_id": request_id,
                    "existing_workflow": existing_workflow,
                    "conversation_context": conversation_context or [],
                    "active_workflow_context": active_workflow_context or {},
                },
                "run_config": run_config,
            }
        )
        return output.get("reasoning_result")

    def run_workflow(
        self,
        *,
        question: str,
        workflow_id: str,
        request_model: Optional[str],
        request_id: Optional[str],
        chat_context: str,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        output = self.run(
            {
                "mode": "workflow",
                "workflow_input": {
                    "question": question,
                    "workflow_id": workflow_id,
                    "request_model": request_model,
                    "request_id": request_id,
                    "chat_context": chat_context,
                },
                "run_config": run_config,
            }
        )
        return output.get("workflow_result")
