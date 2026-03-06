from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    EntryIntent,
    EntryRouterDecision,
)
from ...llm import get_langchain_chat_model, resolve_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event

logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")

_MIN_CONFIDENCE = 0.45

_FIX_PATTERNS = (
    r"\bfix\b",
    r"\bdebug\b",
    r"\brepair\b",
    r"\berror\b",
    r"\bfails?\b",
    r"\bfailure\b",
    r"\bbug\b",
    r"\bexception\b",
    r"\bnot working\b",
    r"\bbroken\b",
    r"\btimeout\b",
)
_EDIT_PATTERNS = (
    r"\bmodify\b",
    r"\bchange\b",
    r"\bupdate\b",
    r"\bedit\b",
    r"\bextend\b",
    r"\badjust\b",
    r"\badd\b",
    r"\bremove\b",
    r"\brefactor\b",
)
_EXISTING_WORKFLOW_PATTERNS = (
    r"\bexisting workflow\b",
    r"\bcurrent workflow\b",
    r"\bthis workflow\b",
    r"\bworkflow id\b",
    r"\bnode\b",
)
_BUILD_PATTERNS = (
    r"\bcreate\b",
    r"\bbuild\b",
    r"\bfrom scratch\b",
    r"\bnew workflow\b",
    r"\bnew automation\b",
    r"\bdesign a workflow\b",
    r"\bset up a workflow\b",
)
_DISCOVERY_PATTERNS = (
    r"\bbusiness process\b",
    r"\bautomation opportunities\b",
    r"\bopportunities to automate\b",
    r"\bdiscovery session\b",
    r"\bconsultant\b",
    r"\bclient\b",
    r"\boperation(s)? team\b",
    r"\bwhere should we automate\b",
    r"\bwhich processes\b",
)
_INFORMATION_PATTERNS = (
    r"\bwhat is\b",
    r"\bhow does\b",
    r"\bexplain\b",
    r"\bcompare\b",
    r"\brecommend\b",
    r"\bbest practice\b",
    r"\bpros and cons\b",
    r"\bguidance\b",
)
_MULTITURN_MARKERS = (r"\buser:\b", r"\bassistant:\b", r"\bclient:\b", r"\bconsultant:\b")


def _count_matches(text: str, patterns: Tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def _route_for_intent(intent: EntryIntent) -> Optional[AgentStage]:
    if intent == EntryIntent.business_discovery_conversation:
        return AgentStage.commercial_agent
    if intent == EntryIntent.workflow_build_request:
        return AgentStage.product_manager_agent
    if intent == EntryIntent.workflow_edit_request:
        return AgentStage.engineer_agent
    if intent == EntryIntent.workflow_fix_request:
        return AgentStage.qa_agent
    if intent == EntryIntent.information_request:
        return AgentStage.consultant_agent
    return None


def _heuristic_decision(user_query: str) -> EntryRouterDecision:
    text = (user_query or "").strip()
    lowered = text.lower()
    signals: List[str] = []
    missing: List[str] = []

    fix_score = _count_matches(lowered, _FIX_PATTERNS)
    edit_score = _count_matches(lowered, _EDIT_PATTERNS)
    existing_score = _count_matches(lowered, _EXISTING_WORKFLOW_PATTERNS)
    build_score = _count_matches(lowered, _BUILD_PATTERNS)
    discovery_score = _count_matches(lowered, _DISCOVERY_PATTERNS)
    info_score = _count_matches(lowered, _INFORMATION_PATTERNS)
    multiturn_score = _count_matches(lowered, _MULTITURN_MARKERS)

    if multiturn_score:
        signals.append("multi_turn_conversation")
    if fix_score:
        signals.append("fix_signals_detected")
    if edit_score:
        signals.append("edit_signals_detected")
    if build_score:
        signals.append("build_signals_detected")
    if discovery_score:
        signals.append("business_discovery_signals_detected")
    if info_score:
        signals.append("information_signals_detected")
    if existing_score:
        signals.append("existing_workflow_signals_detected")

    intent = EntryIntent.unknown
    confidence = 0.35

    if fix_score > 0 and (fix_score >= edit_score):
        intent = EntryIntent.workflow_fix_request
        confidence = min(0.95, 0.62 + 0.08 * fix_score + 0.05 * existing_score)
        if existing_score == 0:
            missing.append("Please specify the affected workflow or node.")
    elif (edit_score > 0 and existing_score > 0) or ("existing workflow" in lowered and edit_score > 0):
        intent = EntryIntent.workflow_edit_request
        confidence = min(0.92, 0.58 + 0.08 * edit_score + 0.06 * existing_score)
    elif build_score > 0:
        intent = EntryIntent.workflow_build_request
        confidence = min(0.9, 0.56 + 0.08 * build_score)
    elif discovery_score > 0 and info_score == 0 and build_score == 0 and fix_score == 0:
        intent = EntryIntent.business_discovery_conversation
        confidence = min(0.88, 0.55 + 0.09 * discovery_score)
    elif info_score > 0 and fix_score == 0 and edit_score == 0 and build_score == 0:
        intent = EntryIntent.information_request
        confidence = min(0.9, 0.55 + 0.08 * info_score)

    if multiturn_score > 0 and any(
        token in lowered for token in ("error", "node", "workflow", "timeout", "credentials")
    ):
        signals.append("technical_multiturn_detected")
        if intent == EntryIntent.business_discovery_conversation:
            intent = EntryIntent.information_request
            confidence = max(confidence, 0.58)

    if len(text) < 12:
        missing.append("Please provide more detail about your goal.")
        confidence = min(confidence, 0.4)

    if intent == EntryIntent.workflow_edit_request and existing_score == 0:
        missing.append("Please specify which existing workflow should be modified.")
        confidence = min(confidence, 0.43)

    if intent == EntryIntent.workflow_fix_request and existing_score == 0:
        confidence = min(confidence, 0.52)

    if confidence < _MIN_CONFIDENCE:
        intent = EntryIntent.unknown

    target_stage = _route_for_intent(intent)
    if intent == EntryIntent.unknown:
        signals.append("fallback_unknown")
        target_stage = None

    return EntryRouterDecision(
        entry_intent=intent,
        target_stage=target_stage,
        confidence=round(max(0.0, min(confidence, 1.0)), 2),
        routing_signals=signals,
        missing_user_inputs=missing,
    )


def _router_prompts(user_query: str) -> Tuple[str, str]:
    system_prompt = (
        "You are the entry router for a multi-agent automation assistant. "
        "Classify user intent into the exact taxonomy and output a strict structured object."
    )
    user_prompt = (
        "Classify the user request with these intents exactly:\n"
        "- business_discovery_conversation\n"
        "- workflow_build_request\n"
        "- workflow_edit_request\n"
        "- workflow_fix_request\n"
        "- information_request\n"
        "- unknown\n\n"
        "Rules:\n"
        "- Prioritize workflow_fix_request over workflow_edit_request when both are present.\n"
        "- Multi-turn text is not automatically business discovery.\n"
        "- If confidence is low, return unknown.\n"
        "- target_stage must map as: business_discovery->commercial_agent, "
        "build->product_manager_agent, edit->engineer_agent, fix->qa_agent, "
        "information->consultant_agent, unknown->null.\n"
        "- Include concise routing_signals and missing_user_inputs arrays.\n\n"
        f"User query:\n{user_query}"
    )
    return system_prompt, user_prompt


def _classify_with_structured_output(
    user_query: str,
    model: Optional[str],
    request_id: Optional[str],
) -> EntryRouterDecision:
    chat_model = get_langchain_chat_model(model=model, temperature=0.0)
    if chat_model is None:
        raise RuntimeError("langchain chat model unavailable")

    system_prompt, user_prompt = _router_prompts(user_query)
    resolved_model = resolve_model(model)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.entry_router.structured",
        model=resolved_model,
        messages=prompt_messages,
        params={"temperature": 0.0},
    )

    structured_llm = chat_model.with_structured_output(EntryRouterDecision)
    started = time.perf_counter()
    output = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    serialized = output.model_dump(exclude_none=True) if isinstance(output, EntryRouterDecision) else output
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.entry_router.structured",
        model=resolved_model,
        latency_ms=latency_ms,
        content=json.dumps(serialized, ensure_ascii=False),
        usage=None,
        extra={"temperature": 0.0},
    )
    if isinstance(output, EntryRouterDecision):
        return output
    return EntryRouterDecision.model_validate(output)


def _apply_guardrails(user_query: str, decision: EntryRouterDecision) -> EntryRouterDecision:
    lowered = (user_query or "").lower()
    routing_signals = list(decision.routing_signals or [])
    missing = list(decision.missing_user_inputs or [])

    has_fix = _count_matches(lowered, _FIX_PATTERNS) > 0
    has_edit = _count_matches(lowered, _EDIT_PATTERNS) > 0

    intent = decision.entry_intent
    confidence = float(decision.confidence)

    if has_fix and has_edit and intent == EntryIntent.workflow_edit_request:
        intent = EntryIntent.workflow_fix_request
        confidence = max(confidence, 0.65)
        routing_signals.append("guardrail_fix_priority_over_edit")

    if intent == EntryIntent.unknown:
        return EntryRouterDecision(
            entry_intent=EntryIntent.unknown,
            target_stage=None,
            confidence=max(0.0, min(confidence, 1.0)),
            routing_signals=routing_signals,
            missing_user_inputs=missing,
        )

    if confidence < _MIN_CONFIDENCE:
        routing_signals.append("guardrail_low_confidence_to_unknown")
        return EntryRouterDecision(
            entry_intent=EntryIntent.unknown,
            target_stage=None,
            confidence=max(0.0, min(confidence, 1.0)),
            routing_signals=routing_signals,
            missing_user_inputs=missing,
        )

    return EntryRouterDecision(
        entry_intent=intent,
        target_stage=_route_for_intent(intent),
        confidence=max(0.0, min(confidence, 1.0)),
        routing_signals=routing_signals,
        missing_user_inputs=missing,
    )


def route_entry_intent(
    user_query: str,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> EntryRouterDecision:
    try:
        decision = _classify_with_structured_output(
            user_query=user_query,
            model=model,
            request_id=request_id,
        )
    except Exception as exc:
        logger.warning("entry router structured output failed, using heuristics: %s", str(exc))
        decision = _heuristic_decision(user_query)

    return _apply_guardrails(user_query, decision)
