from __future__ import annotations

import re
from typing import Dict, List, Set

from .types import CheckerIssue, CheckerResult, ContextPack, PlanSpec, RouterOutput

_TRIGGER_HINTS = ("trigger", "webhook", "schedule", "cron", "manual")
_BRANCH_HINTS_RE = re.compile(r"\b(if|condicion|condición|condition|branch|route|ruta|rama)\b")
_IF_SWITCH_HINTS = ("if", "switch", "router")


def _is_trigger_step(node_type: str, purpose: str, notes: str) -> bool:
    text = f"{node_type} {purpose} {notes}".lower()
    return any(token in text for token in _TRIGGER_HINTS)


def _join_questions(plan: PlanSpec) -> str:
    return " ".join(plan.questionsForUser).lower()


def check_plan(
    plan: PlanSpec,
    router_output: RouterOutput,
    context_pack: ContextPack,
) -> CheckerResult:
    issues: List[CheckerIssue] = []
    allowed_types: Set[str] = {card.type for card in context_pack.nodeCards if card.type}
    required_creds_by_type: Dict[str, List[str]] = {
        card.type: list(card.requiredCredentials) for card in context_pack.nodeCards
    }

    if not plan.steps:
        issues.append(
            CheckerIssue(
                severity="error",
                code="empty_steps",
                msg="PlanSpec has no steps.",
            )
        )

    seen_ids: Set[str] = set()
    duplicate_ids: Set[str] = set()
    has_trigger = False
    has_if_or_switch = False
    has_branching_signal = False
    questions_text = _join_questions(plan)

    for step in plan.steps:
        step_id = (step.id or "").strip()
        if step_id in seen_ids:
            duplicate_ids.add(step_id)
        seen_ids.add(step_id)

        if step.nodeType not in allowed_types:
            issues.append(
                CheckerIssue(
                    severity="error",
                    code="unknown_node_type",
                    msg=f"Step uses nodeType outside nodeCards: {step.nodeType}",
                    stepId=step.id,
                )
            )

        if _is_trigger_step(step.nodeType, step.purpose, step.notes or ""):
            has_trigger = True

        lowered_type = step.nodeType.lower()
        if any(token in lowered_type for token in _IF_SWITCH_HINTS):
            has_if_or_switch = True

        purpose_and_notes = f"{step.purpose} {step.notes or ''}"
        if _BRANCH_HINTS_RE.search(purpose_and_notes.lower()):
            has_branching_signal = True

        required = required_creds_by_type.get(step.nodeType, [])
        if required:
            provided = {item for item in step.credentialsNeeded if item}
            missing = [
                cred
                for cred in required
                if cred not in provided and cred.lower() not in questions_text
            ]
            if missing:
                issues.append(
                    CheckerIssue(
                        severity="error",
                        code="missing_credentials",
                        msg=(
                            "Missing required credentials for step "
                            f"'{step.id}': {', '.join(missing)}"
                        ),
                        stepId=step.id,
                    )
                )

    for duplicate in sorted(duplicate_ids):
        issues.append(
            CheckerIssue(
                severity="error",
                code="duplicate_step_id",
                msg=f"Duplicate step id: {duplicate}",
                stepId=duplicate,
            )
        )

    if router_output.intent == "create" and not has_trigger:
        issues.append(
            CheckerIssue(
                severity="error",
                code="missing_trigger_step",
                msg="Create intent requires a trigger-like step.",
            )
        )

    if has_branching_signal and not has_if_or_switch:
        issues.append(
            CheckerIssue(
                severity="warn",
                code="branching_without_if_switch",
                msg="Branching language detected, but no IF/Switch node found.",
            )
        )

    has_errors = any(issue.severity == "error" for issue in issues)
    return CheckerResult(ok=not has_errors, issues=issues)
