from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from ..llm import chat_completion, resolve_model
from .types import CheckerIssue, ContextPack, PlanSpec, PlanStep, RouterOutput

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

_PLANNER_SYSTEM_PROMPT = (
    "You are a strict n8n planning assistant. "
    "You must output ONLY valid JSON for PlanSpec. "
    "Never invent node types outside the allowed list."
)


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _fallback_plan(reason: str) -> PlanSpec:
    return PlanSpec(
        summary="Unable to produce a reliable plan from current context.",
        steps=[],
        dataFlowNotes=[],
        questionsForUser=[reason],
    )


def _normalize_plan(plan: PlanSpec, allowed_node_types: Set[str]) -> PlanSpec:
    if not allowed_node_types:
        questions = list(plan.questionsForUser)
        questions.append("No node cards were retrieved; please provide more context.")
        return PlanSpec(
            summary=plan.summary,
            steps=[],
            dataFlowNotes=plan.dataFlowNotes,
            questionsForUser=questions,
        )

    cleaned_steps: List[PlanStep] = []
    invalid_types: List[str] = []
    seen_ids = set()

    for step in plan.steps:
        if step.nodeType not in allowed_node_types:
            invalid_types.append(step.nodeType)
            continue
        step_id = step.id.strip() or f"step_{len(cleaned_steps) + 1}"
        if step_id in seen_ids:
            step_id = f"{step_id}_{len(cleaned_steps) + 1}"
        seen_ids.add(step_id)
        cleaned_steps.append(
            PlanStep(
                id=step_id,
                nodeType=step.nodeType,
                purpose=step.purpose,
                inputs=step.inputs,
                outputs=step.outputs,
                credentialsNeeded=step.credentialsNeeded,
                notes=step.notes,
            )
        )

    questions = list(plan.questionsForUser)
    for node_type in invalid_types:
        questions.append(
            f"Node type '{node_type}' is not available in retrieved node cards. "
            "Please clarify required integration."
        )

    if not cleaned_steps and not questions:
        questions.append("Missing information to generate a node-constrained plan.")

    return PlanSpec(
        summary=plan.summary,
        steps=cleaned_steps,
        dataFlowNotes=plan.dataFlowNotes,
        questionsForUser=questions,
    )


def _planner_user_prompt(
    *,
    user_prompt: str,
    router_output: RouterOutput,
    context_pack: ContextPack,
    allowed_node_types: Sequence[str],
) -> str:
    payload = {
        "userPrompt": user_prompt,
        "routerOutput": router_output.model_dump(),
        "contextPack": context_pack.model_dump(exclude_none=True),
        "allowedNodeTypes": list(allowed_node_types),
    }
    return (
        "Return ONLY valid JSON for this schema:\n"
        "{\n"
        '  "summary": "string",\n'
        '  "steps": [\n'
        "    {\n"
        '      "id": "string",\n'
        '      "nodeType": "string",\n'
        '      "purpose": "string",\n'
        '      "inputs": ["string"],\n'
        '      "outputs": ["string"],\n'
        '      "credentialsNeeded": ["string"],\n'
        '      "notes": "string"\n'
        "    }\n"
        "  ],\n"
        '  "dataFlowNotes": ["string"],\n'
        '  "questionsForUser": ["string"]\n'
        "}\n\n"
        "Rules:\n"
        "- Use nodeType ONLY from allowedNodeTypes.\n"
        "- If info is missing, add questionsForUser instead of inventing.\n"
        "- Keep plan compact.\n\n"
        f"Input payload:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _planner_revision_prompt(
    *,
    user_prompt: str,
    router_output: RouterOutput,
    context_pack: ContextPack,
    current_plan: PlanSpec,
    checker_issues: Sequence[CheckerIssue],
    allowed_node_types: Sequence[str],
) -> str:
    payload = {
        "userPrompt": user_prompt,
        "routerOutput": router_output.model_dump(),
        "contextPack": context_pack.model_dump(exclude_none=True),
        "currentPlan": current_plan.model_dump(exclude_none=True),
        "checkerIssues": [issue.model_dump(exclude_none=True) for issue in checker_issues],
        "allowedNodeTypes": list(allowed_node_types),
    }
    return (
        "Fix the current PlanSpec based on checker issues.\n"
        "Return ONLY valid JSON with the same PlanSpec schema.\n"
        "Rules:\n"
        "- Keep nodeType ONLY from allowedNodeTypes.\n"
        "- Do not add prose.\n"
        "- Preserve valid steps when possible.\n\n"
        f"Input payload:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _run_planner_call(prompt: str, model: Optional[str]) -> PlanSpec:
    resolved_model = resolve_model(model)
    response = chat_completion(
        [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=resolved_model,
        temperature=0.1,
    )
    content = (
        response.model_dump().get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    payload = _extract_json_payload(str(content))
    if payload is None:
        raise ValueError("planner returned invalid json")
    return PlanSpec.model_validate(payload)


def plan_workflow(
    user_prompt: str,
    router_output: RouterOutput,
    context_pack: ContextPack,
    model: Optional[str] = None,
) -> PlanSpec:
    allowed_node_types = [card.type for card in context_pack.nodeCards if card.type]
    try:
        raw_plan = _run_planner_call(
            _planner_user_prompt(
                user_prompt=user_prompt,
                router_output=router_output,
                context_pack=context_pack,
                allowed_node_types=allowed_node_types,
            ),
            model=model,
        )
    except Exception:
        return _fallback_plan(
            "Planner failed to produce valid JSON. Please clarify expected trigger and target nodes."
        )
    return _normalize_plan(raw_plan, allowed_node_types=set(allowed_node_types))


def revise_plan(
    user_prompt: str,
    router_output: RouterOutput,
    context_pack: ContextPack,
    current_plan: PlanSpec,
    checker_issues: Sequence[CheckerIssue],
    model: Optional[str] = None,
) -> PlanSpec:
    allowed_node_types = [card.type for card in context_pack.nodeCards if card.type]
    try:
        raw_plan = _run_planner_call(
            _planner_revision_prompt(
                user_prompt=user_prompt,
                router_output=router_output,
                context_pack=context_pack,
                current_plan=current_plan,
                checker_issues=checker_issues,
                allowed_node_types=allowed_node_types,
            ),
            model=model,
        )
    except Exception:
        return _normalize_plan(current_plan, set(allowed_node_types))
    return _normalize_plan(raw_plan, allowed_node_types=set(allowed_node_types))
