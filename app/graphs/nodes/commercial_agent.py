from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    BusinessContextSummary,
    UseCase,
)
from ...llm import get_langchain_chat_model, resolve_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event
from ..multi_agent_state import MultiAgentGraphState

logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")


class CommercialUseCaseCandidate(BaseModel):
    title: str
    business_problem: str
    desired_outcome: str
    expected_value: str
    feasibility: str
    value_score: float = Field(ge=0.0, le=5.0)
    clarity_score: float = Field(ge=0.0, le=5.0)
    actionability_score: float = Field(ge=0.0, le=5.0)


class CommercialDiscoveryOutput(BaseModel):
    business_context_summary: BusinessContextSummary = Field(default_factory=BusinessContextSummary)
    candidates: List[CommercialUseCaseCandidate] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


_AUTOMATION_CUES = (
    "automation",
    "automate",
    "manual",
    "process",
    "bottleneck",
    "inefficient",
    "repetitive",
    "follow up",
    "approval",
    "handoff",
    "reporting",
    "ticket",
    "invoice",
    "onboarding",
)

_VALUE_HIGH = (
    "revenue",
    "sales",
    "customer churn",
    "compliance",
    "payment",
    "invoice",
    "cashflow",
    "sla",
    "penalty",
)
_VALUE_MEDIUM = (
    "manual",
    "delay",
    "support",
    "ticket",
    "report",
    "backlog",
    "follow-up",
)

_FEASIBILITY_HIGH = ("high", "easy", "quick", "simple", "feasible", "short")
_FEASIBILITY_LOW = ("low", "hard", "complex", "difficult", "long")

_TECHNICAL_TERMS = (
    "n8n",
    "node",
    "credential",
    "webhook",
    "json",
    "api key",
    "oauth",
    "switch",
    "if node",
)


def _commercial_prompts(user_query: str) -> Tuple[str, str]:
    system_prompt = (
        "You are a commercial discovery specialist for automation opportunities. "
        "Extract business use-case candidates only. Do not design technical architecture."
    )
    user_prompt = (
        "Return structured output with:\n"
        "- business_context_summary: short structured business context\n"
        "- candidates: normalized automation use-case candidates\n"
        "- notes: optional diagnostics\n\n"
        "Rules:\n"
        "- Focus on business value first.\n"
        "- Do not mention n8n nodes, credentials, implementation parameters, or workflow JSON.\n"
        "- Keep candidates specific and actionable.\n"
        "- Provide scores in [0..5]: value_score, clarity_score, actionability_score.\n"
        "- feasibility should be textual (high/medium/low + short reason).\n"
        "- If nothing actionable exists, return an empty candidates list.\n\n"
        f"Business input:\n{user_query}"
    )
    return system_prompt, user_prompt


def _analyze_with_structured_output(
    user_query: str,
    model: Optional[str],
    request_id: Optional[str],
) -> CommercialDiscoveryOutput:
    chat_model = get_langchain_chat_model(model=model, temperature=0.1)
    if chat_model is None:
        raise RuntimeError("langchain chat model unavailable")

    system_prompt, user_prompt = _commercial_prompts(user_query)
    resolved_model = resolve_model(model)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.commercial.structured",
        model=resolved_model,
        messages=prompt_messages,
        params={"temperature": 0.1},
    )

    structured_llm = chat_model.with_structured_output(CommercialDiscoveryOutput)
    started = time.perf_counter()
    output = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    serialized = (
        output.model_dump(exclude_none=True)
        if isinstance(output, CommercialDiscoveryOutput)
        else output
    )
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.commercial.structured",
        model=resolved_model,
        latency_ms=latency_ms,
        content=json.dumps(serialized, ensure_ascii=False),
        usage=None,
        extra={"temperature": 0.1},
    )
    if isinstance(output, CommercialDiscoveryOutput):
        return output
    return CommercialDiscoveryOutput.model_validate(output)


def _first_sentence(text: str) -> str:
    parts = re.split(r"[.!?]\s+", text.strip())
    return (parts[0] if parts else text).strip()


def _infer_value_score(text: str) -> float:
    lowered = text.lower()
    if any(token in lowered for token in _VALUE_HIGH):
        return 4.5
    if any(token in lowered for token in _VALUE_MEDIUM):
        return 3.6
    return 3.0


def _heuristic_discovery(user_query: str) -> CommercialDiscoveryOutput:
    text = (user_query or "").strip()
    lowered = text.lower()

    summary = BusinessContextSummary(
        process_scope=_first_sentence(text),
        pain_points=[
            clue
            for clue in ("manual work", "delays", "errors", "slow response")
            if clue.split()[0] in lowered
        ],
        desired_outcomes=[
            clue
            for clue in ("faster execution", "less manual work", "higher consistency")
            if clue.split()[0] in lowered
        ],
        constraints=[],
    )

    if sum(1 for cue in _AUTOMATION_CUES if cue in lowered) < 2:
        return CommercialDiscoveryOutput(
            business_context_summary=summary,
            candidates=[],
            notes=["input_too_vague_for_actionable_use_cases"],
        )

    chunks = re.split(r"\n|;|\.| and ", text)
    candidates: List[CommercialUseCaseCandidate] = []
    for chunk in chunks:
        fragment = chunk.strip(" -\t")
        if len(fragment) < 18:
            continue
        if not any(cue in fragment.lower() for cue in _AUTOMATION_CUES):
            continue
        title_words = fragment.split()[:8]
        title = " ".join(title_words).strip().capitalize()
        value_score = _infer_value_score(fragment)
        candidate = CommercialUseCaseCandidate(
            title=title or "Business automation use case",
            business_problem=fragment,
            desired_outcome=f"Automate and standardize: {fragment}",
            expected_value="Reduce manual effort and improve business throughput.",
            feasibility="medium: can be piloted incrementally.",
            value_score=value_score,
            clarity_score=3.0,
            actionability_score=3.2,
        )
        candidates.append(candidate)
        if len(candidates) >= 5:
            break

    return CommercialDiscoveryOutput(
        business_context_summary=summary,
        candidates=candidates,
        notes=[],
    )


def analyze_commercial_discovery(
    user_query: str,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> CommercialDiscoveryOutput:
    try:
        return _analyze_with_structured_output(user_query, model=model, request_id=request_id)
    except Exception as exc:
        logger.warning("commercial structured extraction failed, using fallback: %s", str(exc))
        return _heuristic_discovery(user_query)


def _feasibility_score(feasibility: str) -> float:
    lowered = (feasibility or "").lower()
    if any(token in lowered for token in _FEASIBILITY_HIGH):
        return 4.2
    if any(token in lowered for token in _FEASIBILITY_LOW):
        return 1.8
    return 3.0


def _compact(text: str, max_chars: int = 220) -> str:
    value = " ".join((text or "").split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _remove_technical_artifacts(text: str) -> str:
    cleaned = text
    for token in _TECHNICAL_TERMS:
        cleaned = re.sub(re.escape(token), "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def _candidate_valid(candidate: CommercialUseCaseCandidate) -> bool:
    if not candidate.title.strip():
        return False
    if len(candidate.business_problem.strip()) < 18:
        return False
    if len(candidate.desired_outcome.strip()) < 12:
        return False
    if len(candidate.expected_value.strip()) < 8:
        return False
    return True


def rank_and_select_use_cases(
    output: CommercialDiscoveryOutput,
) -> Tuple[List[UseCase], Optional[UseCase], List[UseCase], str]:
    valid_candidates: List[Tuple[CommercialUseCaseCandidate, int]] = []
    seen = set()
    for idx, candidate in enumerate(output.candidates):
        if not _candidate_valid(candidate):
            continue
        key = (
            candidate.title.strip().lower(),
            candidate.business_problem.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        valid_candidates.append((candidate, idx))

    if not valid_candidates:
        return (
            [],
            None,
            [],
            "No actionable use case was extracted from the current business description.",
        )

    ranked: List[Tuple[float, float, float, int, CommercialUseCaseCandidate]] = []
    for candidate, idx in valid_candidates:
        feasibility_score = _feasibility_score(candidate.feasibility)
        priority_score = (
            (candidate.value_score * 0.55)
            + (candidate.clarity_score * 0.20)
            + (feasibility_score * 0.15)
            + (candidate.actionability_score * 0.10)
        ) * 20.0
        ranked.append(
            (
                priority_score,
                candidate.value_score,
                candidate.clarity_score,
                -idx,
                candidate,
            )
        )

    ranked.sort(reverse=True)

    discovered: List[UseCase] = []
    for position, (priority_score, _value_score, _clarity_score, _neg_index, candidate) in enumerate(ranked, start=1):
        discovered.append(
            UseCase(
                id=f"uc_{position}",
                title=_compact(_remove_technical_artifacts(candidate.title), max_chars=80),
                business_problem=_compact(_remove_technical_artifacts(candidate.business_problem)),
                desired_outcome=_compact(_remove_technical_artifacts(candidate.desired_outcome)),
                expected_value=_compact(_remove_technical_artifacts(candidate.expected_value)),
                feasibility=_compact(candidate.feasibility, max_chars=120),
                priority_score=round(priority_score, 2),
                why_selected="",
            )
        )

    selected = discovered[0]
    selected.why_selected = (
        "Selected because it maximizes expected business value while remaining clear and actionable for an initial version."
    )
    alternatives = [item for item in discovered[1:]]
    for alternative in alternatives:
        alternative.why_selected = "Not selected in this iteration."

    selection_reason = (
        f"Selected '{selected.title}' as primary because it has the highest value-first priority score "
        f"({selected.priority_score}) and clear business outcome."
    )
    return discovered, selected, alternatives, selection_reason


def commercial_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    workflow_context = state.get("workflow_context") or {}
    model = workflow_context.get("model")
    request_id = workflow_context.get("request_id")

    discovery = analyze_commercial_discovery(
        user_query=state.get("user_query") or "",
        model=model if isinstance(model, str) or model is None else None,
        request_id=request_id if isinstance(request_id, str) or request_id is None else None,
    )
    discovered, selected, alternatives, selection_reason = rank_and_select_use_cases(discovery)

    routing_signals = list(state.get("routing_signals") or [])
    routing_signals.append("entered_commercial_agent")
    routing_signals.append(f"commercial_candidates:{len(discovered)}")
    missing_user_inputs = list(state.get("missing_user_inputs") or [])

    target_stage: Optional[AgentStage]
    if selected is not None:
        routing_signals.append(f"selected_use_case:{selected.id}")
        routing_signals.append("handoff_ready_product_manager")
        target_stage = AgentStage.product_manager_agent
    else:
        routing_signals.append("no_actionable_use_case")
        target_stage = None
        if not missing_user_inputs:
            missing_user_inputs.append(
                "Please describe one concrete business process, the pain point, and the desired outcome."
            )

    trace_logger.info(
        "commercial discovery: request_id=%s candidates=%d selected=%s top_score=%s",
        request_id or "-",
        len(discovered),
        selected.id if selected else "-",
        selected.priority_score if selected else "-",
    )

    return {
        "current_stage": "commercial_agent",
        "business_context_summary": discovery.business_context_summary,
        "discovered_use_cases": discovered,
        "selected_use_case": selected,
        "alternative_use_cases": alternatives,
        "selection_reason": selection_reason,
        "routing_signals": routing_signals,
        "missing_user_inputs": missing_user_inputs,
        "target_stage": target_stage,
    }

