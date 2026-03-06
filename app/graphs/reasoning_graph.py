from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..features.reasoning.multi_agent_contracts import (
    EntryIntent,
    MultiAgentGraphResult,
)
from .multi_agent_state import MultiAgentGraphState
from .nodes.commercial_agent import commercial_agent_node
from .nodes.multi_agent_router import route_entry_intent
from .nodes.multi_agent_stubs import (
    consultant_agent_node,
    engineer_agent_node,
    product_manager_agent_node,
    qa_agent_node,
)

try:  # Optional until langgraph dependency is installed.
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency fallback
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


logger = logging.getLogger("n8n-assistant")


def _route_after_entry(state: MultiAgentGraphState) -> str:
    target = state.get("target_stage")
    if target == "commercial_agent":
        return "commercial_agent"
    if target == "consultant_agent":
        return "consultant_agent"
    if target == "product_manager_agent":
        return "product_manager_agent"
    if target == "engineer_agent":
        return "engineer_agent"
    if target == "qa_agent":
        return "qa_agent"
    return "unknown"


class ReasoningGraphRuntime:
    """Milestone 1 entry graph for multi-agent routing."""

    def __init__(self) -> None:
        self._graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(MultiAgentGraphState)
        graph.add_node("entry_router", self._entry_router_node)
        graph.add_node("commercial_agent", commercial_agent_node)
        graph.add_node("consultant_agent", consultant_agent_node)
        graph.add_node("product_manager_agent", product_manager_agent_node)
        graph.add_node("engineer_agent", engineer_agent_node)
        graph.add_node("qa_agent", qa_agent_node)

        graph.add_edge(START, "entry_router")
        graph.add_conditional_edges(
            "entry_router",
            _route_after_entry,
            {
                "commercial_agent": "commercial_agent",
                "consultant_agent": "consultant_agent",
                "product_manager_agent": "product_manager_agent",
                "engineer_agent": "engineer_agent",
                "qa_agent": "qa_agent",
                "unknown": END,
            },
        )
        graph.add_edge("commercial_agent", END)
        graph.add_edge("consultant_agent", END)
        graph.add_edge("product_manager_agent", END)
        graph.add_edge("engineer_agent", END)
        graph.add_edge("qa_agent", END)
        return graph.compile()

    def _entry_router_node(self, state: MultiAgentGraphState) -> Dict[str, Any]:
        decision = route_entry_intent(
            user_query=state.get("user_query") or "",
            model=state.get("workflow_context", {}).get("model"),
            request_id=state.get("workflow_context", {}).get("request_id"),
        )
        return {
            "entry_intent": decision.entry_intent,
            "target_stage": decision.target_stage,
            "confidence": decision.confidence,
            "routing_signals": list(decision.routing_signals),
            "missing_user_inputs": list(decision.missing_user_inputs),
            "current_stage": None,
        }

    def _run_sequential(self, state: MultiAgentGraphState) -> MultiAgentGraphState:
        current = dict(state)
        current.update(self._entry_router_node(current))
        route = _route_after_entry(current)
        if route == "commercial_agent":
            current.update(commercial_agent_node(current))
        elif route == "consultant_agent":
            current.update(consultant_agent_node(current))
        elif route == "product_manager_agent":
            current.update(product_manager_agent_node(current))
        elif route == "engineer_agent":
            current.update(engineer_agent_node(current))
        elif route == "qa_agent":
            current.update(qa_agent_node(current))
        return current

    @staticmethod
    def _build_result(state: MultiAgentGraphState) -> MultiAgentGraphResult:
        target_stage = state.get("target_stage")
        current_stage = state.get("current_stage")
        selected_use_case = state.get("selected_use_case")
        if current_stage == "commercial_agent" and selected_use_case is None:
            status = "unknown_terminal"
        else:
            status = "stub_routed" if target_stage is not None and current_stage else "unknown_terminal"
        entry_intent = state.get("entry_intent") or EntryIntent.unknown
        return MultiAgentGraphResult(
            user_query=state.get("user_query") or "",
            entry_intent=entry_intent,
            target_stage=target_stage,
            confidence=float(state.get("confidence", 0.0)),
            routing_signals=list(state.get("routing_signals") or []),
            current_stage=current_stage,
            missing_user_inputs=list(state.get("missing_user_inputs") or []),
            business_context_summary=state.get("business_context_summary"),
            discovered_use_cases=list(state.get("discovered_use_cases") or []),
            selected_use_case=selected_use_case,
            alternative_use_cases=list(state.get("alternative_use_cases") or []),
            selection_reason=state.get("selection_reason"),
            qa_enabled=bool(state.get("qa_enabled", False)),
            needs_replan=bool(state.get("needs_replan", False)),
            status=status,
        )

    def run(
        self,
        *,
        user_prompt: str,
        model: Optional[str],
        request_id: Optional[str],
        existing_workflow: Any,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> MultiAgentGraphResult:
        _ = existing_workflow  # reserved for future milestones
        state: MultiAgentGraphState = {
            "user_query": user_prompt or "",
            "entry_intent": EntryIntent.unknown,
            "target_stage": None,
            "confidence": 0.0,
            "routing_signals": [],
            "current_stage": None,
            "business_context_summary": None,
            "discovered_use_cases": [],
            "selected_use_case": None,
            "alternative_use_cases": [],
            "selection_reason": None,
            "workflow_context": {"model": model, "request_id": request_id},
            "architecture_plan": {},
            "missing_user_inputs": [],
            "qa_enabled": True,
            "qa_result": {},
            "needs_replan": False,
            "final_workflow_json": {},
        }

        if self._graph is None:
            result_state = self._run_sequential(state)
            return self._build_result(result_state)

        try:
            result_state = self._graph.invoke(state, config=run_config)
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.warning("reasoning graph fallback to sequential runtime: %s", str(exc))
            result_state = self._run_sequential(state)

        return self._build_result(result_state)
