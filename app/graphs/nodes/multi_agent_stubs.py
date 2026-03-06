from __future__ import annotations

from typing import Any, Dict

from ..multi_agent_state import MultiAgentGraphState


def _entered_signal(state: MultiAgentGraphState, signal: str) -> Dict[str, Any]:
    current_signals = list(state.get("routing_signals") or [])
    current_signals.append(signal)
    return {"routing_signals": current_signals}


def commercial_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    updates = _entered_signal(state, "entered_commercial_agent")
    updates["current_stage"] = "commercial_agent"
    return updates


def consultant_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    updates = _entered_signal(state, "entered_consultant_agent")
    updates["current_stage"] = "consultant_agent"
    return updates


def product_manager_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    updates = _entered_signal(state, "entered_product_manager_agent")
    updates["current_stage"] = "product_manager_agent"
    return updates


def engineer_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    updates = _entered_signal(state, "entered_engineer_agent")
    updates["current_stage"] = "engineer_agent"
    return updates


def qa_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    updates = _entered_signal(state, "entered_qa_agent")
    updates["current_stage"] = "qa_agent"
    return updates

