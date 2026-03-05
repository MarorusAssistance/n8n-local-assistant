from __future__ import annotations

from typing import Any, Dict

from ...reasoning.checker import check_plan
from ...reasoning.context_pack import build_context_pack
from ...reasoning.planner import plan_workflow, revise_plan
from ...reasoning.router import route_prompt
from ..state import ReasoningGraphState


def route_node(state: ReasoningGraphState) -> Dict[str, Any]:
    output = route_prompt(
        user_prompt=state["user_prompt"],
        existing_workflow=state.get("existing_workflow"),
        model=state.get("model"),
        request_id=state.get("request_id"),
    )
    return {"router_output": output, "debug_events": ["route"]}


def context_pack_node(state: ReasoningGraphState) -> Dict[str, Any]:
    context_pack = build_context_pack(
        router_output=state["router_output"],
        user_prompt=state["user_prompt"],
        request_id=state.get("request_id"),
    )
    return {"context_pack": context_pack, "debug_events": ["context_pack"]}


def plan_node(state: ReasoningGraphState) -> Dict[str, Any]:
    plan = plan_workflow(
        user_prompt=state["user_prompt"],
        router_output=state["router_output"],
        context_pack=state["context_pack"],
        model=state.get("model"),
        request_id=state.get("request_id"),
    )
    return {"plan": plan, "attempts": 1, "debug_events": ["plan"]}


def check_node(state: ReasoningGraphState) -> Dict[str, Any]:
    checker = check_plan(
        plan=state["plan"],
        router_output=state["router_output"],
        context_pack=state["context_pack"],
    )
    return {"checker": checker, "debug_events": ["check"]}


def revise_node(state: ReasoningGraphState) -> Dict[str, Any]:
    revised_plan = revise_plan(
        user_prompt=state["user_prompt"],
        router_output=state["router_output"],
        context_pack=state["context_pack"],
        current_plan=state["plan"],
        checker_issues=state["checker"].issues,
        model=state.get("model"),
        request_id=state.get("request_id"),
    )
    return {
        "plan": revised_plan,
        "second_iteration_used": True,
        "attempts": 2,
        "debug_events": ["revise"],
    }


def finalize_node(state: ReasoningGraphState) -> Dict[str, Any]:
    checker = state["checker"]
    errors = sum(1 for issue in checker.issues if issue.severity == "error")
    warns = sum(1 for issue in checker.issues if issue.severity == "warn")
    return {"metadata": {"errors": errors, "warns": warns}, "debug_events": ["finalize"]}
