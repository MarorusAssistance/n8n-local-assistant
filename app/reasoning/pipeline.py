from __future__ import annotations

import logging
from typing import Any, Optional

from .checker import check_plan
from .context_pack import build_context_pack
from .planner import plan_workflow, revise_plan
from .router import route_prompt
from .types import ReasoningPipelineResult

trace_logger = logging.getLogger("n8n-assistant.trace")


def run_reasoning_pipeline(
    user_prompt: str,
    *,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
    existing_workflow: Any = None,
) -> ReasoningPipelineResult:
    router_output = route_prompt(
        user_prompt=user_prompt,
        existing_workflow=existing_workflow,
        model=model,
    )
    context_pack = build_context_pack(
        router_output=router_output,
        user_prompt=user_prompt,
        request_id=request_id,
    )

    initial_plan = plan_workflow(
        user_prompt=user_prompt,
        router_output=router_output,
        context_pack=context_pack,
        model=model,
    )
    checker_result = check_plan(
        plan=initial_plan,
        router_output=router_output,
        context_pack=context_pack,
    )

    second_iteration_used = False
    final_plan = initial_plan
    final_checker = checker_result
    attempts = 1

    if any(issue.severity == "error" for issue in checker_result.issues):
        second_iteration_used = True
        attempts = 2
        revised_plan = revise_plan(
            user_prompt=user_prompt,
            router_output=router_output,
            context_pack=context_pack,
            current_plan=initial_plan,
            checker_issues=checker_result.issues,
            model=model,
        )
        final_plan = revised_plan
        final_checker = check_plan(
            plan=revised_plan,
            router_output=router_output,
            context_pack=context_pack,
        )

    errors = sum(1 for issue in final_checker.issues if issue.severity == "error")
    warns = sum(1 for issue in final_checker.issues if issue.severity == "warn")
    trace_logger.info(
        (
            "reasoning pipeline: id=%s intent=%s node_cards=%d doc_chunks=%d "
            "max_tokens=%d estimated_tokens=%d second_iteration=%s attempts=%d errors=%d warns=%d"
        ),
        request_id or "-",
        router_output.intent,
        len(context_pack.nodeCards),
        len(context_pack.docChunks),
        context_pack.budget.maxContextTokens,
        context_pack.budget.estimatedTokens,
        second_iteration_used,
        attempts,
        errors,
        warns,
    )

    return ReasoningPipelineResult(
        plan=final_plan,
        checker=final_checker,
        router=router_output,
        context_pack=context_pack,
        second_iteration_used=second_iteration_used,
        attempts=attempts,
        metadata={
            "errors": errors,
            "warns": warns,
        },
    )
