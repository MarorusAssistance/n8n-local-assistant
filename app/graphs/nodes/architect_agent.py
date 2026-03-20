from __future__ import annotations

import html
import hashlib
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ...config import settings
from ...db import query_definition_chunks_by_entity, query_related_definition_chunks
from ...doc_links import derive_doc_page_key
from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectClarificationState,
    ArchitectClarificationTurn,
    ArchitectNodeCandidate,
    ArchitectStageSearchState,
    ArchitectStageSelection,
    ArchitectStatus,
    ArchitecturePlan,
    ArchitectureStage,
    DecisionSlot,
    MissingUserInput,
    ProposedNode,
    StageKind,
    UseCase,
    WorkflowContext,
    WorkflowDraft,
    WorkflowDraftConnection,
    WorkflowDraftNode,
    WorkflowVersion,
)
from ...llm import get_langchain_chat_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event, emit_trace_event
from ...rag import retrieve_context
from ..multi_agent_state import MultiAgentGraphState
from .engineer_agent import _append_version, _draft_to_final_workflow_json, _persist_workflow_candidate
from .question_utils import (
    build_decision_slot,
    infer_decision_slot_key,
    is_question_rephrase_request,
    localize_question_list,
    localize_question_text,
    merge_decision_slots,
    pending_slot_questions,
    resolve_pending_slots_from_answer,
)


logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")

_API_DOCS_SOURCE = "n8n-docs"
_MAX_STAGE_RETRIEVAL_PASSES = 2
_MAX_USER_CLARIFICATIONS = 3
_DEFAULT_TOP_K = 8
_MAX_LINKING_PAGE_KEYS = 12
_MAX_LINKING_TRACE_PAGES = 16

_GENERIC_DOC_HINTS = (
    "privacy",
    "release",
    "starter-kit",
    "starter kit",
    "tutorial",
    "course",
    "courses",
    "learn/",
    "ai workflow builder",
    "snippet",
    "snippets",
    "evaluation",
    "evaluations",
)
_INTEGRATION_DOC_HINTS = (
    "/integrations/",
    "/builtin/",
    "n8n-nodes-base.",
    "@n8n/n8n-nodes-",
)
_STAGE_ROLE_HINTS: Dict[str, Tuple[str, ...]] = {
    "trigger_intake": (
        "trigger",
        "intake",
        "incoming",
        "inbound",
        "receive",
        "receiving",
        "listen",
        "watch",
        "monitor",
        "consume",
        "captur",
        "recibir",
        "consum",
        "escuch",
        "monitoriz",
    ),
    "fetch_read": (
        "fetch",
        "read",
        "retrieve",
        "pull",
        "download",
        "query",
        "load",
        "leer",
        "obtener",
        "consult",
        "descarg",
        "recuper",
    ),
    "transform_process": (
        "transform",
        "process",
        "normalize",
        "parse",
        "extract",
        "enrich",
        "clean",
        "modify",
        "convert",
        "transform",
        "normaliz",
        "parse",
        "extra",
        "enriqu",
        "limpi",
        "convier",
        "code",
    ),
    "classify_decision": (
        "classify",
        "classification",
        "categor",
        "label",
        "triage",
        "score",
        "rank",
        "priority",
        "decision",
        "urgency",
        "clasific",
        "etiquet",
        "prioriz",
        "decis",
        "urgencia",
    ),
    "route_branch": (
        "route",
        "branch",
        "switch",
        "split",
        "if ",
        "condition",
        "router",
        "enrut",
        "ramific",
        "condicion",
        "deriv",
    ),
    "apply_update_source": (
        "apply",
        "update",
        "label",
        "tag",
        "move",
        "archive",
        "mark",
        "folder",
        "status",
        "apply to source",
        "etiquet",
        "mover",
        "archiv",
        "marcar",
        "actualiz",
        "carpeta",
    ),
    "persist_store": (
        "store",
        "persist",
        "save",
        "write",
        "append",
        "record",
        "database",
        "sheet",
        "table",
        "log",
        "guardar",
        "almacen",
        "persist",
        "escri",
        "anad",
        "añad",
        "registra",
    ),
    "notify_output": (
        "notify",
        "alert",
        "send",
        "reply",
        "respond",
        "post",
        "publish",
        "message",
        "email send",
        "notific",
        "alert",
        "avis",
        "envi",
        "respon",
        "public",
        "mensaje",
    ),
}
_SOURCE_SPECIFIC_HINTS = (
    "email",
    "gmail",
    "imap",
    "outlook",
    "inbox",
    "slack",
    "discord",
    "telegram",
    "webhook",
    "rss",
    "calendar",
    "sheet",
    "airtable",
    "notion",
    "drive",
    "postgres",
    "mysql",
    "database",
    "hubspot",
    "salesforce",
)
_ANALYSIS_ONLY_PLAN_HINTS = (
    "only",
    "just",
    "solo",
    "solamente",
    "simplemente",
    "únicamente",
    "unicamente",
    "report",
    "show",
    "display",
    "return",
    "output",
    "expose",
    "mostrar",
    "devolver",
)
_NODE_DEFINITION_SOURCE = settings.LINKED_DEFS_NODES_SOURCE or "n8n-nodes"
_AI_CLASSIFICATION_FALLBACK_NODE_TYPES = (
    "n8n-nodes-base.ollama",
    "n8n-nodes-base.mistralAi",
    "n8n-nodes-base.openAi",
    "n8n-nodes-base.aiTransform",
)
_SOURCE_UPDATE_GMAIL_FALLBACK_NODE_TYPES = (
    "n8n-nodes-base.gmail",
)

_LINE_VALUE_RE = re.compile(r"^\s*([^:]+):\s*(.*)$")
_CONNECTOR_TYPE_RE = re.compile(r"""type['"]?\s*:\s*['"]([^'"]+)['"]""", re.IGNORECASE)
_CANONICAL_CONNECTOR_RE = re.compile(r"^(main|ai_[a-z0-9_]+|[a-z][a-z0-9_]*)$", re.IGNORECASE)


class _StageSelectionOutput(BaseModel):
    selected_node_types: List[str] = Field(default_factory=list)
    rationale: str = ""
    needs_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)


class _WorkflowNodeBlueprint(BaseModel):
    node_id: str
    name: str
    node_type: str
    type_version: int = Field(default=1, ge=1)
    stage_id: str
    purpose: str = ""
    depends_on: List[str] = Field(default_factory=list)


class _WorkflowConnectionBlueprint(BaseModel):
    source_node_id: str
    target_node_id: str
    type: str = "main"
    index: int = Field(default=0, ge=0)


class _WorkflowBlueprintOutput(BaseModel):
    workflow_name: str
    summary: str = ""
    nodes: List[_WorkflowNodeBlueprint] = Field(default_factory=list)
    connections: List[_WorkflowConnectionBlueprint] = Field(default_factory=list)


def _sanitize_text(text: Any) -> str:
    raw = html.unescape(str(text or ""))
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def _compact(text: Any, max_chars: int = 320) -> str:
    value = " ".join(_sanitize_text(text).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _contains_pattern(text: Any, pattern: str) -> bool:
    normalized_text = _sanitize_text(text).lower()
    normalized_pattern = _sanitize_text(pattern).lower()
    if not normalized_text or not normalized_pattern:
        return False
    if any(token in normalized_pattern for token in (" ", ".", "@", "-", "/")):
        return normalized_pattern in normalized_text
    return re.search(rf"\b{re.escape(normalized_pattern)}[a-z0-9_]*\b", normalized_text) is not None


def _safe_list(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = _sanitize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _plan_explicitly_allows_terminal_analysis(plan: ArchitecturePlan) -> bool:
    combined = _combined_text(
        plan.title,
        plan.business_objective,
        plan.desired_outcome,
        plan.workflow_summary,
    )
    if not combined:
        return False
    if any(
        _contains_pattern(combined, token)
        for token in (
            "save",
            "store",
            "persist",
            "write",
            "append",
            "notify",
            "alert",
            "message",
            "send",
            "reply",
            "route",
            "branch",
            "move",
            "guardar",
            "almacen",
            "persist",
            "escrib",
            "notific",
            "avis",
            "enviar",
            "enrut",
            "ramific",
            "mover",
            "etiquet",
        )
    ):
        return False
    has_analysis_intent = any(
        _contains_pattern(combined, token)
        for token in (
            "classify",
            "classification",
            "analyze",
            "analyse",
            "score",
            "decide",
            "report",
            "show",
            "display",
            "return",
            "output",
            "clasific",
            "analiza",
            "puntu",
            "decid",
            "mostrar",
            "devolver",
        )
    )
    has_analysis_only_hint = any(
        _contains_pattern(combined, token) for token in _ANALYSIS_ONLY_PLAN_HINTS
    )
    return has_analysis_intent and has_analysis_only_hint


def _candidate_trace_summary(
    candidate: ArchitectNodeCandidate,
    *,
    rejected_reasons: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "node_type": candidate.node_type,
        "display_name": candidate.display_name,
        "usage_mode": candidate.usage_mode,
        "usable_as_tool": candidate.usable_as_tool,
        "has_main_input": candidate.has_main_input,
        "input_connection_types": list(candidate.input_connection_types),
        "output_connection_types": list(candidate.output_connection_types),
        "type_version": candidate.type_version,
        "rerank_confidence": candidate.rerank_confidence,
        "link_confidence": candidate.link_confidence,
        "limitations": list(candidate.limitations),
        "rejected_reasons": list(rejected_reasons or []),
        "evidence_refs": list(candidate.evidence_refs[:3]),
    }


def _selection_trace_summary(selection: ArchitectStageSelection) -> Dict[str, Any]:
    return {
        "stage_id": selection.stage_id,
        "selected_node_types": list(selection.selected_node_types),
        "passes_used": selection.passes_used,
        "blocked": selection.blocked,
        "missing_information": list(selection.missing_information),
        "selected_nodes": [_candidate_trace_summary(item) for item in selection.selected_nodes],
        "rationale": selection.rationale,
    }


def _blueprint_trace_summary(blueprint: _WorkflowBlueprintOutput) -> Dict[str, Any]:
    return {
        "workflow_name": blueprint.workflow_name,
        "summary": blueprint.summary,
        "node_count": len(blueprint.nodes),
        "connection_count": len(blueprint.connections),
        "nodes": [
            {
                "node_id": node.node_id,
                "name": node.name,
                "node_type": node.node_type,
                "type_version": node.type_version,
                "stage_id": node.stage_id,
                "depends_on": list(node.depends_on),
            }
            for node in blueprint.nodes
        ],
        "connections": [
            {
                "source_node_id": connection.source_node_id,
                "target_node_id": connection.target_node_id,
                "type": connection.type,
                "index": connection.index,
            }
            for connection in blueprint.connections
        ],
    }


def _runtime_context(state: MultiAgentGraphState) -> Tuple[Optional[str], Optional[str]]:
    runtime_context = state.get("runtime_context")
    if not isinstance(runtime_context, dict):
        runtime_context = state.get("workflow_context")
    if not isinstance(runtime_context, dict):
        return None, None
    model = runtime_context.get("model")
    request_id = runtime_context.get("request_id")
    return (
        model if isinstance(model, str) or model is None else None,
        request_id if isinstance(request_id, str) or request_id is None else None,
    )


def _normalize_model(value: Any, model_cls: Any) -> Optional[Any]:
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        try:
            return model_cls.model_validate(value)
        except Exception:
            return None
    return None


def _normalize_list(values: Any, model_cls: Any) -> List[Any]:
    if not isinstance(values, list):
        return []
    output: List[Any] = []
    for value in values:
        model = _normalize_model(value, model_cls)
        if model is not None:
            output.append(model)
    return output


def _normalize_status(value: Any) -> Optional[ArchitectStatus]:
    if isinstance(value, ArchitectStatus):
        return value
    if isinstance(value, str):
        try:
            return ArchitectStatus(value)
        except ValueError:
            return None
    return None


def _normalize_use_case(value: Any) -> Optional[UseCase]:
    return _normalize_model(value, UseCase)


def _request_context_from_plan(plan: ArchitecturePlan) -> str:
    business_objective = _sanitize_text(plan.business_objective)
    desired_outcome = _sanitize_text(plan.desired_outcome)
    prefixes = (
        "User requested a new workflow for:",
        "Deliver an abstract workflow plan that satisfies:",
    )
    for value in (business_objective, desired_outcome):
        for prefix in prefixes:
            if value.startswith(prefix):
                extracted = _sanitize_text(value[len(prefix) :])
                if extracted:
                    return extracted
    return _sanitize_text(plan.title) or business_objective or desired_outcome or _sanitize_text(plan.workflow_summary)


def _request_context_query(
    *,
    state: MultiAgentGraphState,
    plan: ArchitecturePlan,
    current_user_query: str,
) -> str:
    explicit = _sanitize_text(state.get("request_context_query"))
    if explicit:
        return _compact(explicit, max_chars=420)
    derived = _request_context_from_plan(plan)
    if derived:
        return _compact(derived, max_chars=420)
    return _compact(current_user_query, max_chars=420)


def _record_clarification_answer(
    clarification_state: ArchitectClarificationState,
    user_query: str,
) -> ArchitectClarificationState:
    pending = list(clarification_state.pending_questions)
    pending_slots = list(clarification_state.pending_slots)
    resolved_slots = list(clarification_state.resolved_slots)
    turns = list(clarification_state.turns)
    if not pending and not pending_slots:
        return clarification_state
    answer = _compact(user_query, max_chars=320)
    remaining_slots, newly_resolved = resolve_pending_slots_from_answer(
        pending_slots,
        answer_text=answer,
    )
    if not pending_slots:
        for question in pending:
            for idx in range(len(turns) - 1, -1, -1):
                if turns[idx].question == question and not turns[idx].answer:
                    turns[idx] = turns[idx].model_copy(update={"answer": answer})
                    break
            else:
                turns.append(ArchitectClarificationTurn(question=question, answer=answer))
    for slot in newly_resolved:
        for idx in range(len(turns) - 1, -1, -1):
            if turns[idx].slot_key == slot.slot_key and not turns[idx].answer:
                turns[idx] = turns[idx].model_copy(update={"answer": answer})
                break
    if pending_slots and not newly_resolved and len(pending_slots) == 1:
        slot = pending_slots[0].model_copy(update={"answer": answer, "answer_status": "resolved"})
        newly_resolved = [slot]
        remaining_slots = []
    return clarification_state.model_copy(
        update={
            "pending_questions": pending_slot_questions(remaining_slots) if pending_slots else [],
            "pending_slots": remaining_slots,
            "resolved_slots": merge_decision_slots(resolved_slots, newly_resolved),
            "turns": turns,
        }
    )


def _rephrase_pending_architect_questions(
    clarification_state: ArchitectClarificationState,
    *,
    user_query: str,
    plan: ArchitecturePlan,
) -> ArchitectClarificationState:
    pending_slots = list(clarification_state.pending_slots)
    localized_pending = localize_question_list(
        clarification_state.pending_questions,
        user_query=user_query,
        context_texts=[
            plan.title,
            plan.business_objective,
            plan.desired_outcome,
            plan.workflow_summary,
        ],
    )
    unanswered_indexes = [idx for idx, turn in enumerate(clarification_state.turns) if not turn.answer]
    turns = list(clarification_state.turns)
    for idx, question in zip(unanswered_indexes, localized_pending):
        turns[idx] = turns[idx].model_copy(update={"question": question})
    localized_slots = [
        slot.model_copy(
            update={
                "question_text": localize_question_text(
                    slot.question_text or slot.question_intent or slot.slot_key,
                    user_query=user_query,
                    context_texts=[
                        plan.title,
                        plan.business_objective,
                        plan.desired_outcome,
                        plan.workflow_summary,
                    ],
                )
            }
        )
        for slot in pending_slots
    ]
    return clarification_state.model_copy(
        update={"pending_questions": localized_pending, "pending_slots": localized_slots, "turns": turns}
    )


def _invoke_structured_output(
    *,
    system_prompt: str,
    user_prompt: str,
    output_model: Any,
    model: Optional[str],
    request_id: Optional[str],
    temperature: float,
    stage: str,
) -> Any:
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError("No model configured for architect structured output")

    llm = get_langchain_chat_model(model=model, temperature=temperature)
    if llm is None:
        raise RuntimeError("LangChain chat model is unavailable")

    try:
        structured = llm.with_structured_output(output_model)
    except Exception as exc:
        raise RuntimeError(f"Structured output unavailable: {exc}") from exc

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage=stage,
        model=model,
        messages=messages,
        estimated_tokens=0,
        params={"temperature": temperature, "structured": True},
    )
    response = structured.invoke(messages)
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage=stage,
        model=model,
        latency_ms=None,
        content=(
            response.model_dump_json(exclude_none=True)
            if hasattr(response, "model_dump_json")
            else str(response)
        ),
        usage=None,
        extra={"structured": True},
    )
    return response


def _field_value(text: str, field_name: str) -> str:
    target = field_name.lower().strip()
    for line in _sanitize_text(text).splitlines():
        match = _LINE_VALUE_RE.match(line)
        if not match:
            continue
        key = str(match.group(1) or "").strip().lower()
        if key == target:
            return str(match.group(2) or "").strip()
    return ""


def _parse_bool(value: str) -> Optional[bool]:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def _parse_list(value: str) -> List[str]:
    raw = str(value or "").strip()
    if not raw or raw in {"-", "(none)", "none"}:
        return []
    connector_types = _extract_connector_types(raw)
    if connector_types:
        return connector_types
    normalized = raw.strip("[]")
    parts = [item.strip() for item in normalized.split(",")]
    return [item for item in parts if item and item not in {"-"}]


def _normalize_connector_type(value: Any) -> str:
    token = str(value or "").strip().strip("'\"{}[]()")
    token = token.replace("-", "_").replace(" ", "")
    return token.lower()


def _extract_connector_types(text: Any) -> List[str]:
    raw = _sanitize_text(text)
    if not raw:
        return []

    seen = set()
    output: List[str] = []

    def _append(token: Any) -> None:
        normalized = _normalize_connector_type(token)
        if not normalized or normalized in {"type", "displayname", "display_name"}:
            return
        if not _CANONICAL_CONNECTOR_RE.match(normalized):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        output.append(normalized)

    for match in _CONNECTOR_TYPE_RE.finditer(raw):
        _append(match.group(1))

    if output:
        return output

    for fragment in raw.strip("[]").split(","):
        _append(fragment)
    return output


def _parse_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


def _normalize_confidence(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if 0.0 <= parsed <= 1.0:
        return parsed
    return 1.0 / (1.0 + math.exp(-parsed))


def _extract_doc_page_keys(chunks: Sequence[Dict[str, Any]], max_docs: int) -> List[str]:
    if max_docs <= 0:
        return []
    page_keys: List[str] = []
    seen = set()
    for chunk in list(chunks)[:max_docs]:
        metadata = chunk.get("metadata")
        metadata_map = metadata if isinstance(metadata, dict) else {}
        row_url = str(chunk.get("url") or "")
        page_key, _, _ = derive_doc_page_key(metadata_map, row_url=row_url)
        if not page_key or page_key in seen:
            continue
        seen.add(page_key)
        page_keys.append(page_key)
    return page_keys


def _combined_text(*values: Any) -> str:
    return " ".join(_sanitize_text(value).lower() for value in values if _sanitize_text(value))


def _expanded_identifier_text(*values: Any) -> str:
    parts: List[str] = []
    for value in values:
        text = _sanitize_text(value)
        if not text:
            continue
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        text = text.replace(".", " ").replace("_", " ").replace("-", " ")
        parts.append(text.lower())
    return " ".join(parts)


def _infer_stage_roles(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    stage_requires_trigger: bool,
) -> List[str]:
    if stage.stage_kind is not None:
        roles = [stage.stage_kind.value]
        if stage_requires_trigger and stage.stage_kind != StageKind.trigger_intake:
            roles.insert(0, StageKind.trigger_intake.value)
        return _safe_list(roles)
    core_text = _combined_text(stage.name, stage.purpose)
    context_text = _combined_text(
        " ".join(stage.required_capabilities),
        " ".join(stage.expected_inputs),
        " ".join(stage.expected_outputs),
        " ".join(stage.success_criteria),
        stage.notes or "",
    )
    plan_text = _combined_text(
        plan.workflow_summary,
        plan.business_objective,
        plan.desired_outcome,
    )
    scores: Dict[str, float] = {}
    for role_name, hints in _STAGE_ROLE_HINTS.items():
        score = 0.0
        if stage_requires_trigger and role_name == "trigger_intake":
            score += 10.0
        for hint in hints:
            if _contains_pattern(core_text, hint):
                score += 3.0
            elif _contains_pattern(context_text, hint):
                score += 1.2
            elif not context_text and _contains_pattern(plan_text, hint):
                score += 0.5
        if score > 0:
            scores[role_name] = score
    if not scores:
        return ["trigger_intake"] if stage_requires_trigger else ["transform_process"]
    return [item[0] for item in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _evidence_fingerprint(
    *,
    selected_page_keys: Sequence[str],
    valid_candidates: Sequence[ArchitectNodeCandidate],
) -> str:
    payload = "|".join(
        list(selected_page_keys)
        + sorted(candidate.node_type for candidate in valid_candidates)
    )
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _candidate_roles(candidate: ArchitectNodeCandidate) -> List[str]:
    roles: List[str] = []
    if _is_trigger_candidate(candidate):
        roles.append("trigger_intake")
    combined = _combined_text(
        candidate.node_type,
        candidate.display_name or "",
        candidate.capability_summary,
        " ".join(candidate.limitations),
        _expanded_identifier_text(candidate.node_type, candidate.display_name or ""),
    )
    for role_name, hints in _STAGE_ROLE_HINTS.items():
        if role_name == "trigger_intake":
            continue
        if any(_contains_pattern(combined, hint) for hint in hints):
            roles.append(role_name)
    ordered_roles: List[str] = []
    seen = set()
    for role in roles:
        if role in seen:
            continue
        seen.add(role)
        ordered_roles.append(role)
    return ordered_roles


def _candidate_identity_text(candidate: ArchitectNodeCandidate) -> str:
    return _combined_text(
        candidate.node_type,
        candidate.display_name or "",
        _expanded_identifier_text(candidate.node_type, candidate.display_name or ""),
    )


def _candidate_full_text(candidate: ArchitectNodeCandidate) -> str:
    return _combined_text(
        _candidate_identity_text(candidate),
        candidate.capability_summary,
        " ".join(candidate.limitations),
    )


def _candidate_is_generic_scheduler(candidate: ArchitectNodeCandidate) -> bool:
    combined = _candidate_full_text(candidate)
    if not any(token in combined for token in ("cron", "schedule", "interval", "time-based")):
        return False
    return not any(token in combined for token in _SOURCE_SPECIFIC_HINTS)


def _stage_requests_scheduled_polling(*texts: Any) -> bool:
    combined = _combined_text(*texts)
    return any(
        token in combined
        for token in (
            "schedule",
            "scheduled",
            "cron",
            "poll",
            "polling",
            "interval",
            "hourly",
            "daily",
            "every minute",
            "every hour",
            "batch",
            "periodic",
            "cada minuto",
            "cada hora",
            "por lotes",
        )
    )


def _stage_mentions_source_specific_intake(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
) -> bool:
    combined = _combined_text(
        stage.name,
        stage.purpose,
        " ".join(stage.required_capabilities),
        " ".join(stage.expected_inputs),
        " ".join(stage.expected_outputs),
        plan.workflow_summary,
        plan.business_objective,
        plan.desired_outcome,
    )
    return any(token in combined for token in _SOURCE_SPECIFIC_HINTS)


def _candidate_looks_ai_capable(candidate: ArchitectNodeCandidate) -> bool:
    identity = _candidate_identity_text(candidate)
    combined = _candidate_full_text(candidate)
    has_ai_identity = any(
        token in identity
        for token in (
            "openai",
            "ollama",
            "mistral",
            "anthropic",
            "gemini",
            "llm",
            "aitransform",
            "textclassifier",
            "text classifier",
            "basicllmchain",
            "open ai",
            "language model",
            "classifier",
            "transform with ai",
            "lmopenai",
            "sentimentanalysis",
            "sentiment analysis",
        )
    )
    if has_ai_identity:
        return True
    if _is_trigger_candidate(candidate):
        return False
    return _candidate_requires_non_main_inputs(candidate) and any(
        token in combined
        for token in (
            "language model",
            "openai",
            "ollama",
            "mistral",
            "anthropic",
            "gemini",
            "llm",
            "text classifier",
            "sentiment analysis",
        )
    )


def _candidate_is_generic_utility_node(candidate: ArchitectNodeCandidate) -> bool:
    identity = _candidate_identity_text(candidate)
    return any(
        token in identity
        for token in (
            "markdown",
            "wait",
            "manual",
            "noop",
        )
    )


def _stage_requests_label_application(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
) -> bool:
    combined = _combined_text(
        stage.name,
        stage.purpose,
        stage.business_effect or "",
        stage.user_visible_goal or "",
        " ".join(stage.required_capabilities),
        " ".join(stage.expected_outputs),
        " ".join(stage.success_criteria),
        stage.notes or "",
        plan.workflow_summary,
        plan.business_objective,
        plan.desired_outcome,
    )
    return any(token in combined for token in ("label", "tag", "etiquet", "review", "priority label"))


def _candidate_matches_classification_stage(
    *,
    candidate: ArchitectNodeCandidate,
    requires_ai: bool,
) -> bool:
    if _is_trigger_candidate(candidate):
        return False
    identity = _candidate_identity_text(candidate)
    summary = _sanitize_text(candidate.capability_summary).lower()
    has_classification_identity = any(
        token in identity
        for token in (
            "classif",
            "classifier",
            "urgenc",
            "priority",
            "score",
            "triage",
            "categor",
            "semantic",
            "sentimentanalysis",
            "textclassifier",
        )
    )
    has_classification_summary = any(
        token in summary
        for token in (
            "classif",
            "classifier",
            "urgenc",
            "priority",
            "triage",
            "categor",
            "semantic",
            "sentiment analysis",
        )
    )
    if requires_ai:
        if _candidate_is_generic_utility_node(candidate):
            return False
        if not _candidate_looks_ai_capable(candidate):
            return False
        if _candidate_requires_non_main_inputs(candidate) or not _candidate_has_main_output(candidate):
            return False
        return True
    return (
        has_classification_identity
        or has_classification_summary
        or (_candidate_looks_ai_capable(candidate) and not _candidate_is_generic_utility_node(candidate))
        or ("code" in identity and has_classification_summary)
    )


def _candidate_matches_source_update_stage(
    *,
    candidate: ArchitectNodeCandidate,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
) -> bool:
    if _candidate_is_generic_utility_node(candidate):
        return False
    if _is_trigger_candidate(candidate):
        return False
    identity = _candidate_identity_text(candidate)
    summary = _sanitize_text(candidate.capability_summary).lower()
    if _stage_requests_label_application(stage=stage, plan=plan):
        has_label_signal = any(
            token in f"{identity} {summary}"
            for token in ("label", "tag", "addlabel", "add label", "etiquet")
        )
        target_entity = _sanitize_text(stage.target_entity).lower()
        gmail_message_signal = "gmail" in identity and target_entity.startswith("gmail")
        return has_label_signal or gmail_message_signal
    return any(
        token in f"{identity} {summary}"
        for token in ("update", "status", "mark", "folder", "move", "archive", "label", "tag")
    )


def _page_selection_terms(
    *,
    user_query: str,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    previous_selections: Sequence[ArchitectStageSelection],
) -> List[str]:
    candidates: List[str] = []
    for value in (
        user_query,
        stage.name,
        stage.purpose,
        " ".join(stage.required_capabilities),
        " ".join(stage.expected_inputs),
        " ".join(stage.expected_outputs),
        plan.workflow_summary,
        plan.business_objective,
        plan.desired_outcome,
    ):
        text = _combined_text(value)
        for token in re.findall(r"[a-z0-9_@.\-]{3,}", text):
            candidates.append(token)
    for selection in previous_selections[-2:]:
        candidates.extend(
            re.findall(
                r"[a-z0-9_@.\-]{3,}",
                _combined_text(
                    selection.stage_id,
                    " ".join(selection.selected_node_types),
                    selection.rationale,
                ),
            )
        )
    stopwords = {
        "the",
        "and",
        "that",
        "with",
        "from",
        "this",
        "workflow",
        "stage",
        "using",
        "para",
        "que",
        "con",
        "una",
        "por",
        "del",
        "los",
        "las",
    }
    output: List[str] = []
    seen = set()
    for token in candidates:
        normalized = token.strip().lower()
        if not normalized or normalized in stopwords or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _page_selection_details(
    *,
    page_key: str,
    page_chunks: Sequence[Dict[str, Any]],
    selection_terms: Sequence[str],
    stage_roles: Sequence[str],
    stage_requires_trigger: bool,
) -> Dict[str, Any]:
    combined = _combined_text(
        page_key,
        " ".join(str(chunk.get("title") or "") for chunk in page_chunks[:2]),
        " ".join(str(chunk.get("url") or "") for chunk in page_chunks[:2]),
        " ".join(str(chunk.get("text") or "")[:400] for chunk in page_chunks[:2]),
    )
    score = 0.0
    reasons: List[str] = []

    rerank_scores = [
        _normalize_confidence(chunk.get("rerank_score"))
        for chunk in page_chunks
        if _normalize_confidence(chunk.get("rerank_score")) is not None
    ]
    if rerank_scores:
        score += max(rerank_scores) * 5.0
        reasons.append("rerank_signal")

    if any(hint in combined for hint in _INTEGRATION_DOC_HINTS):
        score += 2.5
        reasons.append("integration_doc")
    if any(hint in combined for hint in _GENERIC_DOC_HINTS):
        score -= 4.5
        reasons.append("generic_doc_penalty")

    lexical_matches = [term for term in selection_terms if term in combined]
    if lexical_matches:
        score += min(len(lexical_matches), 8) * 0.7
        reasons.append("lexical_match")

    if stage_requires_trigger and any(_contains_pattern(combined, hint) for hint in _STAGE_ROLE_HINTS["trigger_intake"]):
        score += 1.5
        reasons.append("trigger_role_match")
    if "classify_decision" in stage_roles and any(_contains_pattern(combined, hint) for hint in _STAGE_ROLE_HINTS["classify_decision"]):
        score += 1.3
        reasons.append("classification_role_match")
    if "persist_store" in stage_roles and any(_contains_pattern(combined, hint) for hint in _STAGE_ROLE_HINTS["persist_store"]):
        score += 1.3
        reasons.append("persistence_role_match")

    if any(token in combined for token in _SOURCE_SPECIFIC_HINTS):
        score += 1.2
        reasons.append("source_specific_signal")

    return {
        "page_key": page_key,
        "score": round(score, 4),
        "reasons": _safe_list(reasons),
    }


def _select_linking_page_keys(
    *,
    docs_chunks: Sequence[Dict[str, Any]],
    user_query: str,
    plan: ArchitecturePlan,
    stage: ArchitectureStage,
    previous_selections: Sequence[ArchitectStageSelection],
    stage_requires_trigger: bool,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = _docs_by_page_key(docs_chunks)
    if not grouped:
        return [], [], []

    stage_roles = _infer_stage_roles(
        stage=stage,
        plan=plan,
        stage_requires_trigger=stage_requires_trigger,
    )
    selection_terms = _page_selection_terms(
        user_query=user_query,
        stage=stage,
        plan=plan,
        previous_selections=previous_selections,
    )

    scored_pages = [
        _page_selection_details(
            page_key=page_key,
            page_chunks=page_chunks,
            selection_terms=selection_terms,
            stage_roles=stage_roles,
            stage_requires_trigger=stage_requires_trigger,
        )
        for page_key, page_chunks in grouped.items()
    ]
    scored_pages.sort(key=lambda item: (-float(item["score"]), str(item["page_key"])))

    limit = min(_MAX_LINKING_PAGE_KEYS, max(8, len(grouped)))
    selected = scored_pages[:limit]
    discarded = scored_pages[limit:_MAX_LINKING_TRACE_PAGES]
    selected_page_keys = [str(item["page_key"]) for item in selected]
    return selected_page_keys, selected, discarded


def _build_stage_query(
    *,
    user_query: str,
    plan: ArchitecturePlan,
    stage: ArchitectureStage,
    previous_selections: Sequence[ArchitectStageSelection],
    clarification_state: ArchitectClarificationState,
    pass_index: int,
) -> str:
    selected_summary = []
    for selection in previous_selections:
        if selection.selected_node_types:
            selected_summary.append(
                f"{selection.stage_id}: {', '.join(selection.selected_node_types)}"
            )
    clarification_lines = [
        f"Question: {turn.question}\nAnswer: {turn.answer}"
        for turn in clarification_state.turns
        if turn.answer
    ]
    pass_hint_parts: List[str] = []
    if stage.stage_kind == StageKind.classify_decision and _user_explicitly_requested_ai(
        plan.business_objective,
        plan.desired_outcome,
        plan.workflow_summary,
        stage.name,
        stage.purpose,
        " ".join(stage.required_capabilities),
    ):
        pass_hint_parts.append(
            "Prefer AI-capable main-path classification nodes such as OpenAI, Ollama, Mistral, AI Transform, or text-classifier style nodes. Reject plain triggers, inbound listeners, and unrelated email provider nodes for classification."
        )
    if stage.stage_kind == StageKind.apply_update_source:
        pass_hint_parts.append(
            "Prefer nodes that mutate the source item directly, such as adding labels or tags, moving folders, archiving, or updating source-side state. Reject trigger-only or read-only nodes."
        )
    if pass_index == 2:
        pass_hint_parts.append(
            "Treat workflow compatibility as a hard constraint. Preserve the exact source system, trigger style, processing mode, and final outcome named by the user instead of forcing a near match."
        )
    pass_hint = " ".join(pass_hint_parts)
    return (
        f"Original workflow request: {_compact(user_query, max_chars=420)}\n"
        f"Workflow objective: {plan.business_objective}\n"
        f"Desired outcome: {plan.desired_outcome}\n"
        f"Workflow summary: {plan.workflow_summary}\n"
        f"Current stage id: {stage.id}\n"
        f"Current stage name: {stage.name}\n"
        f"Stage purpose: {stage.purpose}\n"
        f"Stage required capabilities: {', '.join(stage.required_capabilities) or '-'}\n"
        f"Stage expected inputs: {', '.join(stage.expected_inputs) or '-'}\n"
        f"Stage expected outputs: {', '.join(stage.expected_outputs) or '-'}\n"
        f"Stage dependencies: {', '.join(stage.dependencies) or '-'}\n"
        f"Previous stage node selections: {' | '.join(selected_summary) or '-'}\n"
        f"Clarifications already provided:\n{chr(10).join(clarification_lines) if clarification_lines else '-'}\n"
        f"Retrieval focus hint: {pass_hint or 'Find the best standard n8n nodes for this stage.'}"
    )


def _docs_by_page_key(chunks: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata")
        metadata_map = metadata if isinstance(metadata, dict) else {}
        row_url = str(chunk.get("url") or "")
        page_key, _, _ = derive_doc_page_key(metadata_map, row_url=row_url)
        if not page_key:
            continue
        output.setdefault(page_key, []).append(chunk)
    return output


def _usage_mode_from_connectors(
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
    usable_as_tool: Optional[bool],
) -> str:
    normalized_inputs = [_normalize_connector_type(item) for item in inputs if _normalize_connector_type(item)]
    normalized_outputs = [_normalize_connector_type(item) for item in outputs if _normalize_connector_type(item)]
    has_main_input = "main" in normalized_inputs
    has_main_output = "main" in normalized_outputs
    has_ai_connectors = any(item.startswith("ai_") for item in normalized_inputs + normalized_outputs)

    if has_main_input and usable_as_tool:
        return "both"
    if has_main_input or (not normalized_inputs and has_main_output):
        return "action_only"
    if usable_as_tool or has_ai_connectors:
        return "tool_only"
    return "unknown"


def _is_trigger_candidate(candidate: ArchitectNodeCandidate) -> bool:
    rationale = " ".join(
        [
            candidate.capability_summary,
            " ".join(candidate.limitations),
            candidate.display_name or "",
            candidate.node_type,
        ]
    ).lower()
    positive_tokens = (
        "trigger",
        "webhook",
        "listener",
        "listen",
        "poll",
        "polling",
        "watch",
        "receive",
        "incoming",
        "inbound",
        "new email",
        "new message",
    )
    negative_tokens = (
        "send",
        "sending",
        "reply",
        "respond",
        "response",
        "post ",
        "post-",
        "dispatch",
        "outbound",
        "outgoing",
    )
    if any(token in rationale for token in negative_tokens):
        return False
    if any(token in rationale for token in positive_tokens):
        return (
            candidate.has_main_input is False
            and "main" in [_normalize_connector_type(item) for item in candidate.output_connection_types]
        )
    return False


def _candidate_requires_non_main_inputs(candidate: ArchitectNodeCandidate) -> bool:
    inputs = [_normalize_connector_type(item) for item in candidate.input_connection_types]
    return any(item and item != "main" for item in inputs)


def _candidate_has_main_output(candidate: ArchitectNodeCandidate) -> bool:
    outputs = [_normalize_connector_type(item) for item in candidate.output_connection_types]
    return "main" in outputs


def _user_explicitly_requested_ai(*texts: Any) -> bool:
    combined = " ".join(_sanitize_text(text).lower() for text in texts if _sanitize_text(text))
    return any(
        token in combined
        for token in (
            " ai ",
            " llm",
            "openai",
            "language model",
            "gpt",
            "chatgpt",
            "machine learning",
            "ml model",
        )
    ) or combined.startswith("ai ")


def _stage_prefers_rule_based(*texts: Any) -> bool:
    combined = " ".join(_sanitize_text(text).lower() for text in texts if _sanitize_text(text))
    return any(
        token in combined
        for token in (
            "heuristic",
            "heuristics",
            "rule-based",
            "rule based",
            "rules-based",
            "rules based",
            "if/then",
            "if then",
            "deterministic",
            "without ai",
        )
    )


def _candidate_looks_like_ai_classifier(candidate: ArchitectNodeCandidate) -> bool:
    combined = " ".join(
        [
            candidate.node_type,
            candidate.display_name or "",
            candidate.capability_summary,
        ]
    ).lower()
    if _candidate_requires_non_main_inputs(candidate):
        return True
    return any(
        token in combined
        for token in (" ai ", " llm", "language model", "openai", "textclassifier", "text classifier")
    )


def _candidate_rejection_reasons(
    *,
    candidate: ArchitectNodeCandidate,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    stage_requires_trigger: bool,
) -> List[str]:
    reasons: List[str] = []
    combined = _candidate_full_text(candidate)
    stage_roles = _infer_stage_roles(
        stage=stage,
        plan=plan,
        stage_requires_trigger=stage_requires_trigger,
    )
    candidate_roles = _candidate_roles(candidate)
    dominant_stage_role = stage_roles[0] if stage_roles else None
    requires_ai = _user_explicitly_requested_ai(
        plan.business_objective,
        plan.desired_outcome,
        plan.workflow_summary,
        stage.name,
        stage.purpose,
        " ".join(stage.required_capabilities),
    )
    compatible_roles: Dict[str, set[str]] = {
        "trigger_intake": {"trigger_intake", "fetch_read"},
        "fetch_read": {"fetch_read", "trigger_intake"},
        "transform_process": {"transform_process", "classify_decision"},
        "classify_decision": {"classify_decision", "transform_process"},
        "route_branch": {"route_branch", "classify_decision"},
        "apply_update_source": {"apply_update_source", "persist_store"},
        "persist_store": {"persist_store", "apply_update_source"},
        "notify_output": {"notify_output"},
    }
    if dominant_stage_role == "classify_decision" and _candidate_matches_classification_stage(
        candidate=candidate,
        requires_ai=requires_ai,
    ):
        candidate_roles = _safe_list(list(candidate_roles) + ["classify_decision"])
    if dominant_stage_role == "apply_update_source" and _candidate_matches_source_update_stage(
        candidate=candidate,
        stage=stage,
        plan=plan,
    ):
        candidate_roles = _safe_list(list(candidate_roles) + ["apply_update_source"])

    if candidate.usage_mode == "tool_only":
        reasons.append("tool_only_candidate")
    if _candidate_requires_non_main_inputs(candidate):
        reasons.append("requires_non_main_input_connectors")
    if not _candidate_has_main_output(candidate):
        reasons.append("missing_main_output")
    if stage_requires_trigger and not _is_trigger_candidate(candidate):
        reasons.append("not_a_valid_inbound_trigger")
    if stage_requires_trigger and any(
        token in combined for token in ("send", "reply", "respond", "dispatch", "post ")
    ):
        reasons.append("outbound_nodes_are_invalid_for_intake")
    if (
        stage_requires_trigger
        and _stage_mentions_source_specific_intake(stage=stage, plan=plan)
        and not _stage_requests_scheduled_polling(
            stage.name,
            stage.purpose,
            " ".join(stage.required_capabilities),
            " ".join(stage.expected_inputs),
            " ".join(stage.expected_outputs),
            plan.workflow_summary,
            plan.business_objective,
            plan.desired_outcome,
        )
        and _candidate_is_generic_scheduler(candidate)
    ):
        reasons.append("generic_scheduler_rejected_for_source_specific_trigger")
    if dominant_stage_role == "classify_decision":
        if _is_trigger_candidate(candidate):
            reasons.append("trigger_invalid_for_classification")
        if not _candidate_matches_classification_stage(candidate=candidate, requires_ai=requires_ai):
            reasons.append("not_a_classification_node")
    if dominant_stage_role == "apply_update_source":
        if _is_trigger_candidate(candidate):
            reasons.append("trigger_invalid_for_apply_update")
        if not _candidate_matches_source_update_stage(candidate=candidate, stage=stage, plan=plan):
            reasons.append("not_a_source_update_node")
    if dominant_stage_role:
        allowed_roles = compatible_roles.get(dominant_stage_role, {dominant_stage_role})
        if candidate_roles and not set(candidate_roles).intersection(allowed_roles):
            reasons.append("stage_role_mismatch")
        if not candidate_roles and dominant_stage_role not in {"transform_process"}:
            reasons.append("stage_role_mismatch")

    if _stage_prefers_rule_based(
        stage.name,
        stage.purpose,
        " ".join(stage.required_capabilities),
        " ".join(stage.expected_inputs),
        " ".join(stage.expected_outputs),
        plan.workflow_summary,
        plan.business_objective,
        plan.desired_outcome,
    ) and not _user_explicitly_requested_ai(
        plan.business_objective,
        plan.desired_outcome,
        plan.workflow_summary,
        stage.purpose,
    ):
        if _candidate_looks_like_ai_classifier(candidate):
            reasons.append("rule_based_stage_rejects_ai_candidate")
    if requires_ai and not _candidate_looks_ai_capable(candidate):
        reasons.append("stage_requires_ai_capable_candidate")

    return reasons


def _candidate_summary(
    *,
    linked_chunks: Sequence[Dict[str, Any]],
    doc_chunks: Sequence[Dict[str, Any]],
) -> str:
    descriptions: List[str] = []
    for chunk in linked_chunks:
        text = _sanitize_text(chunk.get("text") or "")
        description = _field_value(text, "Description")
        action = _field_value(text, "Action")
        if description:
            descriptions.append(description)
        elif action:
            descriptions.append(action)
    for chunk in list(doc_chunks)[:2]:
        snippet = _compact(chunk.get("text") or "", max_chars=220)
        if snippet:
            descriptions.append(snippet)
    return _compact(" ".join(descriptions), max_chars=320)


def _recent_selection_context(
    *,
    plan: ArchitecturePlan,
    previous_selections: Sequence[ArchitectStageSelection],
    limit: int = 3,
) -> str:
    if not previous_selections:
        return "-"

    stage_map = {stage.id: stage for stage in plan.stages}
    lines: List[str] = []
    for selection in list(previous_selections)[-limit:]:
        stage = stage_map.get(selection.stage_id)
        stage_label = stage.name if stage is not None else selection.stage_id
        node_parts = []
        for candidate in selection.selected_nodes[:3]:
            node_parts.append(
                (
                    f"{candidate.node_type} "
                    f"(inputs={candidate.input_connection_types or ['-']}, "
                    f"outputs={candidate.output_connection_types or ['-']}, "
                    f"usage={candidate.usage_mode})"
                )
            )
        lines.append(
            _compact(
                f"{selection.stage_id} / {stage_label}: "
                f"{' | '.join(node_parts) or ', '.join(selection.selected_node_types) or '-'}; "
                f"rationale={selection.rationale or '-'}",
                max_chars=320,
            )
        )
    return "\n".join(lines) if lines else "-"


def _stage_neighborhood_context(
    *,
    plan: ArchitecturePlan,
    stage_selections: Sequence[ArchitectStageSelection],
    limit: int = 3,
) -> str:
    if not stage_selections:
        return "-"

    stage_map = {stage.id: stage for stage in plan.stages}
    lines: List[str] = []
    for index, selection in enumerate(stage_selections):
        stage = stage_map.get(selection.stage_id)
        if stage is None:
            continue
        upstream = _recent_selection_context(
            plan=plan,
            previous_selections=list(stage_selections[:index]),
            limit=limit,
        )
        downstream = []
        for future in list(stage_selections[index + 1 : index + 1 + limit]):
            future_stage = stage_map.get(future.stage_id)
            downstream.append(
                f"{future.stage_id}/{future_stage.name if future_stage is not None else future.stage_id}: "
                f"{', '.join(future.selected_node_types) or '-'}"
            )
        lines.append(
            "\n".join(
                [
                    f"Stage neighborhood for {selection.stage_id} / {stage.name}:",
                    f"Current stage intent: {stage.purpose}",
                    f"Recent upstream selected nodes:\n{upstream}",
                    f"Nearest downstream stage bundles: {' | '.join(downstream) or '-'}",
                ]
            )
        )
    return "\n\n".join(lines) if lines else "-"


def _candidate_from_group(
    *,
    stage_id: str,
    node_type: str,
    linked_chunks: Sequence[Dict[str, Any]],
    doc_chunks_by_page: Dict[str, List[Dict[str, Any]]],
) -> ArchitectNodeCandidate:
    display_name = ""
    version = 1
    inputs: List[str] = []
    outputs: List[str] = []
    usable_as_tool: Optional[bool] = None
    evidence_ids: List[str] = []
    evidence_refs: List[str] = []
    link_confidence: Optional[float] = None
    rerank_confidence: Optional[float] = None
    doc_chunks: List[Dict[str, Any]] = []
    limitations: List[str] = []

    for chunk in linked_chunks:
        metadata = chunk.get("metadata")
        metadata_map = metadata if isinstance(metadata, dict) else {}
        text = _sanitize_text(chunk.get("text") or "")
        display_name = display_name or str(
            metadata_map.get("displayName") or _field_value(text, "Display Name") or ""
        ).strip()
        version = _parse_int(metadata_map.get("version") or _field_value(text, "Version"), default=version)
        inputs = inputs or _parse_list(_field_value(text, "Inputs"))
        outputs = outputs or _parse_list(_field_value(text, "Outputs"))
        parsed_tool = _parse_bool(_field_value(text, "Usable As Tool"))
        if usable_as_tool is None and parsed_tool is not None:
            usable_as_tool = parsed_tool
        doc_id = str(chunk.get("doc_id") or "").strip()
        if doc_id:
            evidence_ids.append(doc_id)
        ref = str(chunk.get("url") or chunk.get("title") or chunk.get("section") or doc_id or "").strip()
        if ref:
            evidence_refs.append(ref)
        if chunk.get("link_confidence") is not None:
            value = _normalize_confidence(chunk.get("link_confidence"))
            if value is not None:
                link_confidence = max(link_confidence or 0.0, value)
        page_key = str(chunk.get("link_doc_page_key") or "").strip()
        if page_key:
            related_docs = doc_chunks_by_page.get(page_key) or []
            doc_chunks.extend(related_docs)
            for doc in related_docs:
                score = doc.get("rerank_score")
                if score is None:
                    continue
                normalized_score = _normalize_confidence(score)
                if normalized_score is not None:
                    rerank_confidence = max(rerank_confidence or 0.0, normalized_score)

    usage_mode = _usage_mode_from_connectors(
        inputs=inputs,
        outputs=outputs,
        usable_as_tool=usable_as_tool,
    )
    if usage_mode == "tool_only":
        limitations.append("This candidate only exposes AI-tool style connectors for the current evidence.")
    if _candidate_requires_non_main_inputs(
        ArchitectNodeCandidate(
            node_type=node_type,
            input_connection_types=list(inputs),
            output_connection_types=list(outputs),
        )
    ):
        limitations.append("This candidate requires non-main input connectors and is invalid in v1 classic workflows.")
    if "main" not in [_normalize_connector_type(item) for item in outputs]:
        limitations.append("This candidate does not expose a standard main output in the current evidence.")

    return ArchitectNodeCandidate(
        node_type=node_type,
        display_name=display_name or node_type,
        stage_id=stage_id,
        capability_summary=_candidate_summary(linked_chunks=linked_chunks, doc_chunks=doc_chunks),
        limitations=_safe_list(limitations),
        rationale="Evidence-backed node candidate extracted from API docs and linked node definitions.",
        usage_mode=usage_mode,  # type: ignore[arg-type]
        usable_as_tool=usable_as_tool,
        has_main_input=("main" in [_normalize_connector_type(item) for item in inputs]) if inputs else False,
        input_connection_types=list(inputs),
        output_connection_types=list(outputs),
        evidence_chunk_ids=_safe_list(evidence_ids),
        evidence_refs=_safe_list(evidence_refs),
        rerank_confidence=rerank_confidence,
        link_confidence=link_confidence,
        type_version=version,
    )


def _build_candidates(
    *,
    stage_id: str,
    docs_chunks: Sequence[Dict[str, Any]],
    linked_chunks: Sequence[Dict[str, Any]],
) -> List[ArchitectNodeCandidate]:
    docs_by_page = _docs_by_page_key(docs_chunks)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in linked_chunks:
        if str(chunk.get("linked_def_type") or "").strip().lower() != "node":
            continue
        node_type = str(chunk.get("linked_entity_id") or "").strip()
        if not node_type:
            continue
        grouped.setdefault(node_type, []).append(chunk)

    output: List[ArchitectNodeCandidate] = []
    for node_type, node_chunks in grouped.items():
        output.append(
            _candidate_from_group(
                stage_id=stage_id,
                node_type=node_type,
                linked_chunks=node_chunks,
                doc_chunks_by_page=docs_by_page,
            )
        )
    output.sort(
        key=lambda item: (
            0 if item.usage_mode != "tool_only" else 1,
            -(item.rerank_confidence or 0.0),
            -(item.link_confidence or 0.0),
            item.node_type,
        )
    )
    return output


def _candidate_from_definition_lookup(
    *,
    stage_id: str,
    node_type: str,
) -> Optional[ArchitectNodeCandidate]:
    rows = query_definition_chunks_by_entity(
        entity_key="nodeType",
        entity_id=node_type,
        source_value=_NODE_DEFINITION_SOURCE,
    )
    if not rows:
        rows = query_definition_chunks_by_entity(
            entity_key="nodeType",
            entity_id=node_type,
            source_value=None,
        )
    if not rows:
        return None

    prepared_rows: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        url = str(row.get("url") or "")
        page_key, _, _ = derive_doc_page_key(metadata, row_url=url)
        prepared_rows.append(
            {
                **dict(row),
                "linked_def_type": "node",
                "linked_entity_id": node_type,
                "link_doc_page_key": page_key,
                "link_confidence": 1.0,
            }
        )

    return _candidate_from_group(
        stage_id=stage_id,
        node_type=node_type,
        linked_chunks=prepared_rows,
        doc_chunks_by_page={},
    )


def _stage_standard_fallback_node_types(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    stage_requires_trigger: bool,
) -> List[str]:
    if stage_requires_trigger:
        return []

    if stage.stage_kind == StageKind.classify_decision and _user_explicitly_requested_ai(
        plan.business_objective,
        plan.desired_outcome,
        plan.workflow_summary,
        stage.name,
        stage.purpose,
        " ".join(stage.required_capabilities),
        stage.notes or "",
    ):
        return list(_AI_CLASSIFICATION_FALLBACK_NODE_TYPES)

    if (
        stage.stage_kind == StageKind.apply_update_source
        and _stage_requests_label_application(stage=stage, plan=plan)
        and _sanitize_text(stage.target_entity).lower().startswith("gmail")
    ):
        return list(_SOURCE_UPDATE_GMAIL_FALLBACK_NODE_TYPES)

    return []


def _augment_candidates_with_definition_fallbacks(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    stage_requires_trigger: bool,
    candidates: Sequence[ArchitectNodeCandidate],
) -> List[ArchitectNodeCandidate]:
    output = list(candidates)
    for node_type in _stage_standard_fallback_node_types(
        stage=stage,
        plan=plan,
        stage_requires_trigger=stage_requires_trigger,
    ):
        clean_candidate = _candidate_from_definition_lookup(stage_id=stage.id, node_type=node_type)
        if clean_candidate is None:
            continue
        clean_rejections = _candidate_rejection_reasons(
            candidate=clean_candidate,
            stage=stage,
            plan=plan,
            stage_requires_trigger=stage_requires_trigger,
        )
        existing_same_type = [item for item in output if item.node_type == node_type]
        if not existing_same_type:
            output.append(clean_candidate)
            continue
        existing_rejection_lists = [
            _candidate_rejection_reasons(
                candidate=item,
                stage=stage,
                plan=plan,
                stage_requires_trigger=stage_requires_trigger,
            )
            for item in existing_same_type
        ]
        existing_has_valid = any(not reasons for reasons in existing_rejection_lists)
        clean_is_valid = not clean_rejections
        if clean_is_valid and not existing_has_valid:
            output = [item for item in output if item.node_type != node_type]
            output.append(clean_candidate)
            continue
        if clean_is_valid:
            continue
        existing_best_rejection_count = min((len(reasons) for reasons in existing_rejection_lists), default=999)
        if len(clean_rejections) < existing_best_rejection_count:
            output = [item for item in output if item.node_type != node_type]
            output.append(clean_candidate)
    output.sort(
        key=lambda item: (
            0 if item.usage_mode != "tool_only" else 1,
            -(item.rerank_confidence or 0.0),
            -(item.link_confidence or 0.0),
            item.node_type,
        )
    )
    return output


def _semantic_architect_question(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    stage_requires_trigger: bool,
) -> str:
    stage_kind = stage.stage_kind or (
        StageKind(_infer_stage_roles(stage=stage, plan=plan, stage_requires_trigger=stage_requires_trigger)[0])
        if _infer_stage_roles(stage=stage, plan=plan, stage_requires_trigger=stage_requires_trigger)
        else StageKind.transform_process
    )
    if stage_requires_trigger or stage_kind == StageKind.trigger_intake:
        return "Que sistema u evento concreto debe iniciar este workflow?"
    if stage_kind == StageKind.classify_decision:
        return (
            f"La etapa '{stage.name}' necesita una decision semantica mas concreta. "
            "Debe clasificar con IA, con reglas deterministas o con otro criterio?"
        )
    if stage_kind == StageKind.apply_update_source:
        return (
            f"Como debe aplicarse el resultado de la etapa '{stage.name}' sobre el sistema origen: "
            "etiquetar, mover, actualizar estado o realizar otra accion concreta?"
        )
    if stage_kind == StageKind.persist_store:
        return f"Donde debe guardarse o persistirse el resultado de la etapa '{stage.name}'?"
    if stage_kind == StageKind.notify_output:
        return f"Como debe notificarse o publicarse el resultado de la etapa '{stage.name}'?"
    if stage_kind == StageKind.route_branch:
        return f"Que accion concreta debe ocurrir despues de la decision de la etapa '{stage.name}'?"
    return (
        f"La etapa '{stage.name}' necesita una definicion semantica mas concreta. "
        "Que resultado operable debe producir exactamente?"
    )


def _selection_failure_question(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    stage_requires_trigger: bool,
    selected_node_types: Sequence[str],
) -> str:
    if selected_node_types:
        return (
            f"La seleccion devuelta para la etapa '{stage.name}' no es compatible con los candidatos validos del workflow. "
            + _semantic_architect_question(
                stage=stage,
                plan=plan,
                stage_requires_trigger=stage_requires_trigger,
            )
        )
    return _semantic_architect_question(
        stage=stage,
        plan=plan,
        stage_requires_trigger=stage_requires_trigger,
    )


def _extract_stage_named_token(*texts: Any) -> Optional[str]:
    combined = " ".join(_sanitize_text(text) for text in texts if _sanitize_text(text))
    if not combined:
        return None
    patterns = (
        r"(?:label|tag|etiqueta|etiquetar)\s+(?:named|called|llamada)?\s*[\"']([^\"']{2,48})[\"']",
        r"(?:label|tag|etiqueta|etiquetar)\s+(?:named|called|llamada)?\s+([A-Z][A-Za-z0-9 _-]{1,48})",
        r"[\"']([^\"']{2,48})[\"']\s+(?:label|tag|etiqueta)",
    )
    for pattern in patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if not match:
            continue
        value = _compact(match.group(1), max_chars=64).strip()
        if value:
            return value
    return None


def _derive_operation_hints(
    *,
    stage: ArchitectureStage,
    candidate: Optional[ArchitectNodeCandidate],
    plan: ArchitecturePlan,
) -> Dict[str, Any]:
    stage_kind = stage.stage_kind.value if isinstance(stage.stage_kind, StageKind) else str(stage.stage_kind or "")
    combined = " ".join(
        _sanitize_text(item)
        for item in (
            stage.name,
            stage.purpose,
            stage.business_effect or "",
            stage.target_entity or "",
            stage.user_visible_goal or "",
            " ".join(stage.required_capabilities),
            " ".join(stage.expected_inputs),
            " ".join(stage.expected_outputs),
            " ".join(stage.success_criteria),
            stage.notes or "",
            plan.workflow_summary,
        )
        if _sanitize_text(item)
    ).lower()
    action_text = " ".join(
        _sanitize_text(item)
        for item in (
            stage.name,
            stage.purpose,
            stage.business_effect or "",
            stage.user_visible_goal or "",
            " ".join(stage.success_criteria),
            stage.notes or "",
        )
        if _sanitize_text(item)
    ).lower()
    hints: Dict[str, Any] = {
        "stage_kind": stage_kind,
        "business_effect": _compact(stage.business_effect or stage.purpose, max_chars=220),
        "target_entity": stage.target_entity or "",
        "user_visible_goal": _compact(stage.user_visible_goal or stage.name, max_chars=220),
    }
    if candidate is not None:
        hints["candidate_display_name"] = candidate.display_name or ""

    if stage.stage_kind == StageKind.apply_update_source:
        candidate_name = " ".join(
            part
            for part in (
                candidate.node_type if candidate is not None else "",
                candidate.display_name if candidate is not None else "",
            )
            if part
        ).lower()
        generic_apply_candidate = not any(
            token in candidate_name
            for token in ("label", "tag", "move", "archive", "mark", "folder", "status", "update")
        )
        if generic_apply_candidate:
            hints["require_action_selection"] = True
            hints["allow_inferred_parameter_keys"] = ["resource", "operation", "action"]
            hints["required_parameter_keys"] = ["resource", "operation"]
        if any(token in action_text for token in ("label", "tag", "etiquet")):
            hints["semantic_action"] = "apply_label"
            hints["selector_guidance"] = (
                "Configure this node to apply a label or tag to the source item so the result becomes visible in the source system."
            )
            if any(token in combined for token in ("gmail", "email", "correo", "message", "mensaje")):
                hints["preferred_resource"] = "message"
            label_name = _extract_stage_named_token(
                stage.name,
                stage.purpose,
                stage.business_effect or "",
                stage.user_visible_goal or "",
                " ".join(stage.expected_outputs),
                " ".join(stage.success_criteria),
            )
            if label_name:
                hints["target_label_name"] = label_name
        elif any(token in action_text for token in ("move", "folder", "carpeta", "mover")):
            hints["semantic_action"] = "move_item"
            hints["selector_guidance"] = "Configure this node to move the source item to the appropriate folder or location."
        elif any(token in action_text for token in ("archive", "archiv")):
            hints["semantic_action"] = "archive_item"
            hints["selector_guidance"] = "Configure this node to archive the source item."
        elif any(token in action_text for token in ("status", "mark", "marcar", "read", "unread", "state")):
            hints["semantic_action"] = "update_item_status"
            hints["selector_guidance"] = "Configure this node to update the status of the source item."
        else:
            hints["selector_guidance"] = (
                "Configure this node to apply the workflow result back onto the source system with a concrete action."
            )
    elif stage.stage_kind == StageKind.classify_decision:
        hints["semantic_action"] = "classify_payload"
        if _user_explicitly_requested_ai(
            plan.business_objective,
            plan.desired_outcome,
            plan.workflow_summary,
            stage.name,
            stage.purpose,
            " ".join(stage.required_capabilities),
        ):
            hints["classification_method"] = "ai"
        elif _stage_prefers_rule_based(
            stage.name,
            stage.purpose,
            " ".join(stage.required_capabilities),
            " ".join(stage.expected_inputs),
            " ".join(stage.expected_outputs),
            plan.workflow_summary,
            plan.business_objective,
            plan.desired_outcome,
        ):
            hints["classification_method"] = "rule_based"
    elif stage.stage_kind == StageKind.trigger_intake:
        hints["semantic_action"] = "receive_incoming_item"
    elif stage.stage_kind == StageKind.persist_store:
        hints["semantic_action"] = "persist_result"
    elif stage.stage_kind == StageKind.notify_output:
        hints["semantic_action"] = "notify_result"
    elif stage.stage_kind == StageKind.route_branch:
        hints["semantic_action"] = "route_result"

    return {key: value for key, value in hints.items() if value not in ("", [], {}, None)}


def _validate_operation_hints(
    *,
    plan: ArchitecturePlan,
    proposed_nodes: Sequence[ProposedNode],
) -> List[Tuple[str, str]]:
    by_stage: Dict[str, List[ProposedNode]] = {}
    for node in proposed_nodes:
        if not node.stage_id:
            continue
        by_stage.setdefault(node.stage_id, []).append(node)

    issues: List[Tuple[str, str]] = []
    for stage in plan.stages:
        if stage.stage_kind != StageKind.apply_update_source:
            continue
        stage_nodes = by_stage.get(stage.id, [])
        if not stage_nodes:
            continue
        if any(item.implementation_hints.get("semantic_action") for item in stage_nodes):
            continue
        issues.append(
            (
                stage.id,
                (
                    f"La etapa '{stage.name}' necesita una accion operable concreta sobre el sistema origen. "
                    "Debo etiquetar, mover, archivar o actualizar el item fuente?"
                ),
            )
        )
    return issues


def _fallback_candidate_score(
    *,
    candidate: ArchitectNodeCandidate,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    stage_requires_trigger: bool,
) -> float:
    score = float(candidate.rerank_confidence or 0.0) + float(candidate.link_confidence or 0.0)
    combined = _candidate_full_text(candidate)
    if stage_requires_trigger and _is_trigger_candidate(candidate):
        score += 6.0
    if stage.stage_kind == StageKind.classify_decision:
        if _candidate_matches_classification_stage(
            candidate=candidate,
            requires_ai=_user_explicitly_requested_ai(
                plan.business_objective,
                plan.desired_outcome,
                plan.workflow_summary,
                stage.name,
                stage.purpose,
                " ".join(stage.required_capabilities),
            ),
        ):
            score += 5.0
        if _is_trigger_candidate(candidate):
            score -= 8.0
    if stage.stage_kind == StageKind.apply_update_source:
        if _candidate_matches_source_update_stage(candidate=candidate, stage=stage, plan=plan):
            score += 5.0
        if _stage_requests_label_application(stage=stage, plan=plan) and any(
            token in combined for token in ("label", "tag", "addlabel", "add label", "etiquet")
        ):
            score += 3.0
        if "gmail" in combined and _sanitize_text(stage.target_entity).lower().startswith("gmail"):
            score += 2.0
        if _is_trigger_candidate(candidate):
            score -= 8.0
    return score


def _fallback_stage_selection(
    *,
    stage: ArchitectureStage,
    plan: ArchitecturePlan,
    candidates: Sequence[ArchitectNodeCandidate],
    stage_requires_trigger: bool,
) -> _StageSelectionOutput:
    filtered = [candidate for candidate in candidates if candidate.usage_mode != "tool_only"]
    if stage_requires_trigger:
        filtered = [candidate for candidate in filtered if _is_trigger_candidate(candidate)]
    if not filtered:
        return _StageSelectionOutput(
            selected_node_types=[],
            rationale="No standard workflow candidate matched the stage constraints.",
            needs_clarification=True,
            clarification_questions=[
                _semantic_architect_question(
                    stage=stage,
                    plan=plan,
                    stage_requires_trigger=stage_requires_trigger,
                )
            ],
        )
    best = max(
        filtered,
        key=lambda item: _fallback_candidate_score(
            candidate=item,
            stage=stage,
            plan=plan,
            stage_requires_trigger=stage_requires_trigger,
        ),
    )
    return _StageSelectionOutput(
        selected_node_types=[best.node_type],
        rationale=f"Selected '{best.display_name or best.node_type}' as the best available standard workflow candidate.",
        needs_clarification=False,
        clarification_questions=[],
    )


def _select_stage_nodes_with_structured_output(
    *,
    user_query: str = "",
    plan: ArchitecturePlan,
    stage: ArchitectureStage,
    candidates: Sequence[ArchitectNodeCandidate],
    previous_selections: Sequence[ArchitectStageSelection],
    stage_requires_trigger: bool,
    model: Optional[str],
    request_id: Optional[str],
) -> _StageSelectionOutput:
    if not candidates:
        return _fallback_stage_selection(
            stage=stage,
            plan=plan,
            candidates=candidates,
            stage_requires_trigger=stage_requires_trigger,
        )

    selected_summary = [
        f"{selection.stage_id}: {', '.join(selection.selected_node_types)}"
        for selection in previous_selections
        if selection.selected_node_types
    ]
    recent_context = _recent_selection_context(
        plan=plan,
        previous_selections=previous_selections,
        limit=3,
    )
    candidate_lines = []
    for candidate in candidates[:12]:
        rejected_reasons = _candidate_rejection_reasons(
            candidate=candidate,
            stage=stage,
            plan=plan,
            stage_requires_trigger=stage_requires_trigger,
        )
        candidate_lines.append(
            "\n".join(
                [
                    f"- Node Type: {candidate.node_type}",
                    f"  Display Name: {candidate.display_name or '-'}",
                    f"  Usage Mode: {candidate.usage_mode}",
                    f"  Usable As Tool: {candidate.usable_as_tool}",
                    f"  Has Main Input: {candidate.has_main_input}",
                    f"  Input Connection Types: {', '.join(candidate.input_connection_types) or '-'}",
                    f"  Output Connection Types: {', '.join(candidate.output_connection_types) or '-'}",
                    f"  Type Version: {candidate.type_version}",
                    f"  Capability Summary: {candidate.capability_summary or '-'}",
                    f"  Limitations: {', '.join(candidate.limitations) or '-'}",
                    f"  Hard Reject Reasons In v1: {', '.join(rejected_reasons) or '-'}",
                    f"  Evidence Refs: {', '.join(candidate.evidence_refs[:3]) or '-'}",
                ]
            )
        )

    system_prompt = (
        "You are architect_agent for an n8n workflow assistant. "
        "Select the best standard n8n nodes for one workflow stage using only the candidates provided. "
        "Do not invent node types. "
        "This release only supports classic workflows with main-only compatible nodes. "
        "Treat the provided hard constraints as mandatory, not preferences. "
        "Reason like a workflow architect, not like a semantic search engine: choose nodes by structural role in the workflow."
    )
    user_prompt = (
        f"Workflow objective: {plan.business_objective}\n"
        f"Desired outcome: {plan.desired_outcome}\n"
        f"Workflow summary: {plan.workflow_summary}\n"
        f"Current stage id: {stage.id}\n"
        f"Current stage name: {stage.name}\n"
        f"Current stage purpose: {stage.purpose}\n"
        f"Current stage required capabilities: {', '.join(stage.required_capabilities) or '-'}\n"
        f"Current stage expected inputs: {', '.join(stage.expected_inputs) or '-'}\n"
        f"Current stage expected outputs: {', '.join(stage.expected_outputs) or '-'}\n"
        f"Previous selected stages: {' | '.join(selected_summary) or '-'}\n\n"
        f"Recent upstream selected nodes and connectors:\n{recent_context}\n\n"
        "Candidate nodes:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "Rules:\n"
        "- Select only from the listed node types.\n"
        "- Prefer the smallest bundle that fully satisfies the stage without breaking the overall workflow.\n"
        "- Ask only semantic clarification questions by default. Do not ask the user to choose a concrete node or app unless they explicitly requested technical control.\n"
        "- Keep the overall workflow coherent from start to finish.\n"
        "- Use the recent upstream selected nodes as hard context for compatibility, I/O continuity, and realistic sequencing.\n"
        "- Avoid selecting a node that would make the previous 2 to 3 stages impossible to connect coherently.\n"
        "- Decide by functional role, not just semantic similarity. Distinguish trigger, poller, classifier, model provider, transformer, router, and sink roles.\n"
        "- Reject provider-only or infrastructure-only nodes when the stage needs a complete business operation node.\n"
        "- Prefer nodes whose main inputs and outputs naturally match the stage I/O and the nearest upstream node outputs.\n"
        "- If a candidate would require an extra hidden node, hidden model attachment, or hidden auxiliary connection to work, reject it in this selection step.\n"
        "- For the first stage, only explicit inbound trigger/listener/polling nodes are valid.\n"
        "- Send, post, respond, reply, and dispatch nodes are invalid for intake stages.\n"
        "- Reject any candidate with hard reject reasons in v1.\n"
        "- In v1, reject any node that requires ai_languageModel, ai_tool, ai_memory, or any other non-main required connector.\n"
        "- If the stage is heuristic or rule-based, reject AI or LLM classifier nodes unless the user explicitly asked for AI.\n"
        f"- Stage requires trigger-capable node: {stage_requires_trigger}.\n"
        "- If you ask a clarification question, write it in the same language as the user's request.\n"
        "- Before selecting, perform a private compatibility check against: stage intent, stage I/O, previous 2 to 3 stages, and likely next-stage connectivity.\n"
        "- If no candidate fits confidently, set needs_clarification=true and ask only the minimum question needed.\n"
    )
    try:
        output = _invoke_structured_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=_StageSelectionOutput,
            model=model,
            request_id=request_id,
            temperature=0.1,
            stage="multi_agent.architect.stage_selection",
        )
        if not output.selected_node_types and not output.needs_clarification:
            return _fallback_stage_selection(
                stage=stage,
                plan=plan,
                candidates=candidates,
                stage_requires_trigger=stage_requires_trigger,
            )
        if output.clarification_questions:
            output = output.model_copy(
                update={
                    "clarification_questions": localize_question_list(
                        output.clarification_questions,
                        user_query=user_query,
                        context_texts=[
                            plan.title,
                            plan.business_objective,
                            plan.desired_outcome,
                            stage.name,
                            stage.purpose,
                        ],
                    )
                }
            )
        return output
    except Exception as exc:
        logger.warning("architect stage selection fallback to deterministic selection: %s", str(exc))
        return _fallback_stage_selection(
            stage=stage,
            plan=plan,
            candidates=candidates,
            stage_requires_trigger=stage_requires_trigger,
        )


def _short_type(node_type: str) -> str:
    value = str(node_type or "").strip()
    if not value:
        return "node"
    return value.split(".")[-1]


def _slugify_identifier(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return token or "node"


def _canonical_blueprint_nodes(
    *,
    stage_selections: Sequence[ArchitectStageSelection],
) -> List[_WorkflowNodeBlueprint]:
    seeds: List[_WorkflowNodeBlueprint] = []
    used_ids: set[str] = set()
    used_names: set[str] = set()

    for selection in stage_selections:
        for index, candidate in enumerate(selection.selected_nodes, start=1):
            base_name = str(candidate.display_name or _short_type(candidate.node_type)).strip() or _short_type(
                candidate.node_type
            )
            base_id = f"{selection.stage_id}_{_slugify_identifier(base_name)}"
            node_id = base_id
            duplicate = 2
            while node_id in used_ids:
                node_id = f"{base_id}_{duplicate}"
                duplicate += 1
            name = base_name
            while name in used_names:
                name = f"{base_name} {duplicate - 1}"
                duplicate += 1
            used_ids.add(node_id)
            used_names.add(name)
            seeds.append(
                _WorkflowNodeBlueprint(
                    node_id=node_id,
                    name=name,
                    node_type=candidate.node_type,
                    type_version=max(1, candidate.type_version),
                    stage_id=selection.stage_id,
                    purpose=candidate.rationale or selection.rationale,
                    depends_on=[],
                )
            )
    return seeds


def _normalize_node_ref(value: str, alias_map: Dict[str, str]) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw or raw in {"-", "null", "none"}:
        return None
    return alias_map.get(raw, raw)


def _normalize_workflow_blueprint_output(
    *,
    output: _WorkflowBlueprintOutput,
    stage_selections: Sequence[ArchitectStageSelection],
) -> _WorkflowBlueprintOutput:
    seeds = _canonical_blueprint_nodes(stage_selections=stage_selections)
    if not seeds:
        return output

    seed_by_key = {(item.stage_id, item.node_type): item for item in seeds}
    alias_map: Dict[str, str] = {}
    normalized_nodes: List[_WorkflowNodeBlueprint] = []

    for node in output.nodes:
        seed = seed_by_key.get((node.stage_id, node.node_type))
        if seed is None:
            normalized_nodes.append(node)
            alias_map[str(node.node_id)] = str(node.node_id)
            continue
        alias_map[str(node.node_id)] = seed.node_id
        alias_map[str(seed.node_id)] = seed.node_id
        alias_map[str(node.name)] = seed.node_id
        normalized_nodes.append(
            seed.model_copy(
                update={
                    "purpose": node.purpose or seed.purpose,
                    "depends_on": list(node.depends_on),
                }
            )
        )

    node_ids = {node.node_id for node in normalized_nodes}
    normalized_connections: List[_WorkflowConnectionBlueprint] = []
    for connection in output.connections:
        source_id = _normalize_node_ref(connection.source_node_id, alias_map)
        target_id = _normalize_node_ref(connection.target_node_id, alias_map)
        if not source_id or not target_id:
            continue
        if source_id not in node_ids or target_id not in node_ids:
            continue
        normalized_connections.append(
            _WorkflowConnectionBlueprint(
                source_node_id=source_id,
                target_node_id=target_id,
                type=connection.type,
                index=connection.index,
            )
        )

    normalized_nodes = [
        node.model_copy(
            update={
                "depends_on": [
                    dep
                    for dep in [
                        _normalize_node_ref(item, alias_map) for item in list(node.depends_on)
                    ]
                    if dep and dep in node_ids
                ]
            }
        )
        for node in normalized_nodes
    ]

    return _WorkflowBlueprintOutput(
        workflow_name=output.workflow_name,
        summary=output.summary,
        nodes=normalized_nodes,
        connections=normalized_connections,
    )


def _fallback_workflow_blueprint(
    *,
    plan: ArchitecturePlan,
    stage_selections: Sequence[ArchitectStageSelection],
) -> _WorkflowBlueprintOutput:
    nodes: List[_WorkflowNodeBlueprint] = _canonical_blueprint_nodes(stage_selections=stage_selections)
    connections: List[_WorkflowConnectionBlueprint] = []
    stage_primary_node_ids: Dict[str, str] = {}
    for node in nodes:
        stage_primary_node_ids.setdefault(node.stage_id, node.node_id)

    nodes_by_stage = {node.stage_id: node for node in nodes}
    stage_order = [selection.stage_id for selection in stage_selections]
    stage_map = {stage.id: stage for stage in plan.stages}
    for index, selection in enumerate(stage_selections):
        node = nodes_by_stage.get(selection.stage_id)
        if node is None:
            continue
        stage = stage_map.get(selection.stage_id)
        deps = [
            stage_primary_node_ids[dependency]
            for dependency in list(stage.dependencies if stage is not None else [])
            if dependency in stage_primary_node_ids
        ]
        if not deps and index > 0:
            previous_stage_id = stage_order[index - 1]
            previous_node_id = stage_primary_node_ids.get(previous_stage_id)
            if previous_node_id:
                deps = [previous_node_id]
        node.depends_on = deps
        for dep in deps:
            connections.append(
                _WorkflowConnectionBlueprint(
                    source_node_id=dep,
                    target_node_id=node.node_id,
                )
            )

    return _WorkflowBlueprintOutput(
        workflow_name=plan.title or "Architect Workflow Draft",
        summary=plan.workflow_summary,
        nodes=nodes,
        connections=connections,
    )


def _build_workflow_blueprint_with_structured_output(
    *,
    plan: ArchitecturePlan,
    stage_selections: Sequence[ArchitectStageSelection],
    model: Optional[str],
    request_id: Optional[str],
) -> _WorkflowBlueprintOutput:
    if not stage_selections:
        return _WorkflowBlueprintOutput(
            workflow_name=plan.title or "Architect Workflow Draft",
            summary=plan.workflow_summary,
            nodes=[],
            connections=[],
        )

    selection_lines = []
    allowed_blueprint_nodes = _canonical_blueprint_nodes(stage_selections=stage_selections)
    for selection in stage_selections:
        stage_nodes = []
        for candidate in selection.selected_nodes:
            stage_nodes.append(
                f"{candidate.node_type} (name={candidate.display_name or candidate.node_type}, "
                f"typeVersion={candidate.type_version}, usage_mode={candidate.usage_mode}, "
                f"has_main_input={candidate.has_main_input}, "
                f"inputs={candidate.input_connection_types}, outputs={candidate.output_connection_types}, "
                f"rationale={candidate.rationale or candidate.capability_summary})"
            )
        selection_lines.append(
            "\n".join(
                [
                    f"Stage ID: {selection.stage_id}",
                    f"Selected Node Types: {', '.join(selection.selected_node_types) or '-'}",
                    f"Stage Rationale: {selection.rationale}",
                    f"Nodes: {' | '.join(stage_nodes) or '-'}",
                ]
            )
        )
    neighborhood_context = _stage_neighborhood_context(
        plan=plan,
        stage_selections=stage_selections,
        limit=3,
    )
    blueprint_seed_lines = [
        "\n".join(
            [
                f"- Exact node_id: {item.node_id}",
                f"  Exact name: {item.name}",
                f"  Node type: {item.node_type}",
                f"  Stage id: {item.stage_id}",
                f"  type_version: {item.type_version}",
            ]
        )
        for item in allowed_blueprint_nodes
    ]

    system_prompt = (
        "You are architect_agent for an n8n workflow assistant. "
        "Build the first structural n8n workflow draft using only the selected node candidates. "
        "Do not generate parameter values, credentials, AI tool connectors, unsupported connection types, "
        "or invalid edges. Use only classic main-only workflow structure. "
        "Reason like a workflow architect assembling a valid DAG, not like a text generator."
    )
    user_prompt = (
        f"Workflow title: {plan.title}\n"
        f"Business objective: {plan.business_objective}\n"
        f"Desired outcome: {plan.desired_outcome}\n"
        f"Workflow summary: {plan.workflow_summary}\n\n"
        "Selected stage bundles:\n"
        f"{chr(10).join(selection_lines)}\n\n"
        "Allowed blueprint nodes (use these exact ids and names in the output):\n"
        f"{chr(10).join(blueprint_seed_lines)}\n\n"
        "Stage neighborhood context:\n"
        f"{neighborhood_context}\n\n"
        "Rules:\n"
        "- Use only the selected node types.\n"
        "- Materialize only the allowed blueprint nodes listed above.\n"
        "- Use the exact node_id and exact name for each allowed node; do not rename them and do not invent additional ids.\n"
        "- Build a coherent draft for a classic non-AI workflow.\n"
        "- The draft must contain at least one trigger/start node.\n"
        "- Use only main connections.\n"
        "- Validate every edge before emitting it.\n"
        "- When deciding each edge, consider the whole workflow objective, each stage I/O, and the nearest upstream selected nodes.\n"
        "- Keep adjacency coherent: each node must make sense given the 2 to 3 closest previous nodes, not only the stage order.\n"
        "- Do not connect through main into a node that requires unsupported auxiliary connectors.\n"
        "- Do not force a linear chain only because the stage list is linear.\n"
        "- If a selected node cannot participate in a classic main-only flow, return no invalid edge for it.\n"
        "- Return unique node ids and names.\n"
        "- type_version must come from the selected candidate metadata.\n"
        "- Keep the node order and dependencies coherent with the stage order.\n"
        "- Never reference a source_node_id or target_node_id that is not present in the returned nodes list.\n"
        "- Never use placeholder ids such as '-', '', null, none, output, end, terminal, or similar.\n"
        "- If a node has no valid downstream target, emit no connection for it.\n"
        "- depends_on must reference only existing allowed node ids, or be empty.\n"
        "- Do not invent helper nodes that were not selected.\n"
        "- Do not assume hidden model wiring, hidden tools, hidden memory, or hidden side connections.\n"
        "- Prefer the minimum valid edge set that preserves the workflow logic.\n"
        "- Before returning, perform a private self-check: every dependency references an existing node id, every connection references existing node ids, every target can accept main, and the full graph stays coherent end-to-end.\n"
        "- Parameters remain empty at this stage.\n"
    )
    try:
        output = _invoke_structured_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=_WorkflowBlueprintOutput,
            model=model,
            request_id=request_id,
            temperature=0.1,
            stage="multi_agent.architect.workflow_blueprint",
        )
        if not output.nodes:
            return _fallback_workflow_blueprint(plan=plan, stage_selections=stage_selections)
        return _normalize_workflow_blueprint_output(
            output=output,
            stage_selections=stage_selections,
        )
    except Exception as exc:
        logger.warning("architect workflow blueprint fallback to deterministic draft: %s", str(exc))
        return _fallback_workflow_blueprint(plan=plan, stage_selections=stage_selections)


def _validate_stage_selection_coverage(
    *,
    plan: ArchitecturePlan,
    stage_selections: Sequence[ArchitectStageSelection],
) -> List[str]:
    issues: List[str] = []
    selection_map = {selection.stage_id: selection for selection in stage_selections}
    for stage in plan.stages:
        selection = selection_map.get(stage.id)
        if selection is None:
            issues.append(f"Architect did not materialize a node selection for stage '{stage.id}'.")
            continue
        if not selection.selected_nodes or not selection.selected_node_types:
            issues.append(f"Architect returned an empty node bundle for stage '{stage.id}'.")
    return _safe_list(issues)


def _build_missing_input(
    *,
    stage_id: str,
    message: str,
    slot_key: Optional[str] = None,
) -> MissingUserInput:
    question = _compact(message, max_chars=220)
    return MissingUserInput(
        input_id=f"handoff:{stage_id}:architect",
        input_key=f"handoff:{stage_id}:architect",
        missing_item=stage_id,
        reason=question,
        blocking_node_id=stage_id,
        slot_key=slot_key,
        category="decision" if slot_key else "handoff",
        question=question,
    )


def _plan_terminal_outcome_gap_question(plan: ArchitecturePlan) -> Optional[str]:
    if not plan.stages or _plan_explicitly_allows_terminal_analysis(plan):
        return None
    outgoing_stage_ids = {
        item.source_stage_id
        for item in plan.data_flow
        if _sanitize_text(item.source_stage_id)
    }
    sink_stages = [stage for stage in plan.stages if stage.id not in outgoing_stage_ids]
    if not sink_stages:
        sink_stages = [plan.stages[-1]]
    for stage_index, stage in enumerate(sink_stages):
        roles = _infer_stage_roles(
            stage=stage,
            plan=plan,
            stage_requires_trigger=(stage_index == 0 and stage.id == plan.stages[0].id),
        )
        dominant_role = roles[0] if roles else None
        if dominant_role in {"classify_decision", "route_branch", "transform_process"}:
            return _compact(
                (
                    f"The workflow currently ends at stage '{stage.name}' without a clear final business outcome. "
                    "What should happen with that result next: apply it to the source item, save it somewhere, "
                    "notify someone, or route it to a concrete downstream action?"
                ),
                max_chars=220,
            )
    return None


def _resolved_terminal_outcome_answer(
    clarification_state: ArchitectClarificationState,
) -> Optional[str]:
    for slot in reversed(list(clarification_state.resolved_slots)):
        answer = _sanitize_text(slot.answer)
        if not answer:
            continue
        if slot.stage_id == "plan_terminal_outcome":
            return answer
        if slot.slot_key in {"result_application_mode", "storage_destination", "notification_policy", "workflow_goal"}:
            return answer
    return None


def _infer_terminal_outcome_mode(answer: str) -> Optional[str]:
    lowered = _sanitize_text(answer).lower()
    if not lowered:
        return None
    if any(token in lowered for token in ("nothing else", "nada mas", "nada más", "solo analizar", "just analyze", "analysis only")):
        return "analysis_only"
    if any(token in lowered for token in ("label", "labels", "tag", "tags", "etiquet", "visible", "filter", "filtrar", "apply", "aplicar", "gmail", "correo", "source item", "elemento origen")):
        return "apply_update_source"
    if any(token in lowered for token in ("save", "store", "persist", "guardar", "almacen", "sheet", "sheets", "database", "table")):
        return "persist_store"
    if any(token in lowered for token in ("notify", "notification", "alert", "notific", "avis")):
        return "notify_output"
    if any(token in lowered for token in ("route", "branch", "downstream action", "enrutar", "derivar", "ramificar")):
        return "route_branch"
    return None


def _refine_terminal_outcome_plan_from_slots(
    *,
    plan: ArchitecturePlan,
    clarification_state: ArchitectClarificationState,
) -> ArchitecturePlan:
    answer = _resolved_terminal_outcome_answer(clarification_state)
    if not answer:
        return plan
    outcome_mode = _infer_terminal_outcome_mode(answer)
    if outcome_mode in {None, "analysis_only"}:
        return plan

    outgoing_stage_ids = {
        item.source_stage_id
        for item in plan.data_flow
        if _sanitize_text(item.source_stage_id)
    }
    sink_indexes = [idx for idx, stage in enumerate(plan.stages) if stage.id not in outgoing_stage_ids]
    if not sink_indexes and plan.stages:
        sink_indexes = [len(plan.stages) - 1]
    if not sink_indexes:
        return plan

    sink_index = sink_indexes[-1]
    sink_stage = plan.stages[sink_index]
    combined = _combined_text(
        answer,
        plan.title,
        plan.business_objective,
        plan.desired_outcome,
        plan.workflow_summary,
    )
    target_entity = "gmail_message" if "gmail" in combined else ("email_message" if any(token in combined for token in ("email", "correo", "message", "mensaje")) else sink_stage.target_entity)
    note = _compact(f"Refined from user clarification: {answer}", max_chars=220)

    if outcome_mode == "apply_update_source":
        apply_label = any(token in combined for token in ("label", "tag", "etiquet", "visible", "filter", "filtrar"))
        updated_stage = sink_stage.model_copy(
            update={
                "name": "Apply Result to Source Item" if not apply_label else ("Apply Gmail Label" if target_entity == "gmail_message" else "Apply Source Label"),
                "purpose": (
                    "Apply the classification result back onto the same Gmail message as a visible label so it can be filtered later."
                    if apply_label and target_entity == "gmail_message"
                    else "Apply the workflow result back onto the original source item with a concrete source-side update."
                ),
                "stage_kind": StageKind.apply_update_source,
                "business_effect": (
                    "The original Gmail message is updated with the urgency label."
                    if apply_label and target_entity == "gmail_message"
                    else "The original source item reflects the workflow result."
                ),
                "target_entity": target_entity or "source_item",
                "user_visible_goal": (
                    "The urgency label is visible in Gmail so the user can filter emails later."
                    if apply_label and target_entity == "gmail_message"
                    else "The source system shows the applied workflow result."
                ),
                "required_capabilities": [
                    "Update the original source item",
                    "Apply the workflow result back onto that same item",
                ],
                "expected_inputs": ["Decision result or enriched payload", "Source item identifiers"],
                "expected_outputs": ["Updated source item"],
                "success_criteria": ["The workflow result is applied exactly once to the original source item."],
                "notes": _safe_list([sink_stage.notes or "", note])[-1] if sink_stage.notes else note,
            }
        )
    elif outcome_mode == "persist_store":
        updated_stage = sink_stage.model_copy(
            update={
                "name": "Persist Workflow Result",
                "purpose": "Persist the workflow result in the chosen downstream store.",
                "stage_kind": StageKind.persist_store,
                "business_effect": "The workflow result is stored for later access.",
                "target_entity": sink_stage.target_entity or "workflow_result",
                "user_visible_goal": "The workflow result is retained in downstream storage.",
                "required_capabilities": ["Persist the final workflow result"],
                "expected_inputs": ["Decision result or enriched payload"],
                "expected_outputs": ["Stored workflow result"],
                "success_criteria": ["The final workflow result is stored exactly once."],
                "notes": _safe_list([sink_stage.notes or "", note])[-1] if sink_stage.notes else note,
            }
        )
    elif outcome_mode == "notify_output":
        updated_stage = sink_stage.model_copy(
            update={
                "name": "Notify Workflow Result",
                "purpose": "Notify the chosen recipient or downstream system about the workflow result.",
                "stage_kind": StageKind.notify_output,
                "business_effect": "The workflow result is communicated downstream.",
                "target_entity": sink_stage.target_entity or "notification",
                "user_visible_goal": "The relevant recipient receives the workflow result.",
                "required_capabilities": ["Send the workflow result to a downstream recipient or system"],
                "expected_inputs": ["Decision result or enriched payload"],
                "expected_outputs": ["Notification or outbound action result"],
                "success_criteria": ["The workflow result is communicated exactly once."],
                "notes": _safe_list([sink_stage.notes or "", note])[-1] if sink_stage.notes else note,
            }
        )
    else:
        updated_stage = sink_stage.model_copy(
            update={
                "stage_kind": StageKind.route_branch,
                "notes": _safe_list([sink_stage.notes or "", note])[-1] if sink_stage.notes else note,
            }
        )

    updated_stages = list(plan.stages)
    updated_stages[sink_index] = updated_stage
    return plan.model_copy(update={"stages": updated_stages})


def _block_architect(
    *,
    architecture_plan: ArchitecturePlan,
    workflow_context: Optional[WorkflowContext],
    selections: List[ArchitectStageSelection],
    search_history: List[ArchitectStageSearchState],
    clarification_state: ArchitectClarificationState,
    architect_notes: List[str],
    question: str,
    stage_id: str,
    user_query: str = "",
    block_cause: str = "selection_failure",
) -> Dict[str, Any]:
    question_value = _compact(
        localize_question_text(
            question,
            user_query=user_query,
            context_texts=[
                architecture_plan.title,
                architecture_plan.business_objective,
                architecture_plan.desired_outcome,
                architecture_plan.workflow_summary,
            ],
        ),
        max_chars=220,
    )
    slot_key = infer_decision_slot_key(question_value, stage_name=stage_id)
    slot = build_decision_slot(
        slot_key=slot_key,
        owner_agent=AgentStage.architect_agent,
        question_text=question_value,
        stage_id=stage_id,
        question_intent=slot_key,
        user_query=user_query,
        context_texts=[
            architecture_plan.title,
            architecture_plan.business_objective,
            architecture_plan.desired_outcome,
            architecture_plan.workflow_summary,
        ],
    )
    updated_attempts = clarification_state.attempts_used + 1
    if updated_attempts > clarification_state.max_attempts:
        failure_question = (
            f"Architect stopped after {clarification_state.max_attempts} clarification attempts. {question_value}"
        )
        missing_details = [_build_missing_input(stage_id=stage_id, message=failure_question, slot_key=slot_key)]
        if workflow_context is None:
            workflow_context = WorkflowContext(use_case_id=architecture_plan.use_case_id)
        workflow_context.planning_ready = False
        workflow_context.handoff_target = None
        workflow_context.unresolved_inputs = [item.question for item in missing_details]
        workflow_context.pending_decision_slots = []
        workflow_context.resolved_decision_slots = list(clarification_state.resolved_slots)
        workflow_context.clarification_owner = AgentStage.architect_agent
        workflow_context.clarification_reason = block_cause
        workflow_context.last_block_cause = block_cause
        workflow_context.notes = _safe_list(
            list(workflow_context.notes) + ["architect_status=architect_failed_no_solution"]
        )
        return {
            "current_stage": "architect_agent",
            "target_stage": None,
            "architect_status": ArchitectStatus.architect_failed_no_solution,
            "architect_stage_search_history": search_history,
            "architect_stage_selections": selections,
            "architect_clarification_state": clarification_state.model_copy(
                update={"attempts_used": clarification_state.max_attempts}
            ),
            "architect_notes": _safe_list(architect_notes + [failure_question]),
            "missing_user_inputs": [item.question for item in missing_details],
            "missing_user_input_details": missing_details,
            "workflow_context": workflow_context,
            "pending_decision_slots": [],
            "resolved_decision_slots": list(clarification_state.resolved_slots),
            "clarification_owner": AgentStage.architect_agent,
            "clarification_reason": block_cause,
            "last_block_cause": block_cause,
            "stage_bundle_map": dict(workflow_context.stage_bundle_map),
            "evidence_fingerprints": dict(workflow_context.evidence_fingerprints),
        }

    turns = list(clarification_state.turns)
    turns.append(
        ArchitectClarificationTurn(
            stage_id=stage_id,
            slot_key=slot.slot_key,
            question=question_value,
            answer=None,
        )
    )
    clarification_state = clarification_state.model_copy(
        update={
            "attempts_used": updated_attempts,
            "pending_questions": [slot.question_text],
            "pending_slots": merge_decision_slots(clarification_state.pending_slots, [slot]),
            "turns": turns,
        }
    )
    missing_details = [_build_missing_input(stage_id=stage_id, message=slot.question_text, slot_key=slot.slot_key)]
    if workflow_context is None:
        workflow_context = WorkflowContext(use_case_id=architecture_plan.use_case_id)
    workflow_context.planning_ready = False
    workflow_context.handoff_target = None
    workflow_context.unresolved_inputs = [item.question for item in missing_details]
    workflow_context.pending_decision_slots = list(clarification_state.pending_slots)
    workflow_context.resolved_decision_slots = list(clarification_state.resolved_slots)
    workflow_context.clarification_owner = AgentStage.architect_agent
    workflow_context.clarification_reason = block_cause
    workflow_context.last_block_cause = block_cause
    workflow_context.notes = _safe_list(
        list(workflow_context.notes)
        + [f"architect_status={ArchitectStatus.architect_blocked_waiting_user.value}", f"blocked_stage={stage_id}"]
    )
    return {
        "current_stage": "architect_agent",
        "target_stage": None,
        "architect_status": ArchitectStatus.architect_blocked_waiting_user,
        "architect_stage_search_history": search_history,
        "architect_stage_selections": selections,
        "architect_clarification_state": clarification_state,
        "architect_notes": _safe_list(architect_notes + [question_value]),
        "missing_user_inputs": [item.question for item in missing_details],
        "missing_user_input_details": missing_details,
        "workflow_context": workflow_context,
        "pending_decision_slots": list(clarification_state.pending_slots),
        "resolved_decision_slots": list(clarification_state.resolved_slots),
        "clarification_owner": AgentStage.architect_agent,
        "clarification_reason": block_cause,
        "last_block_cause": block_cause,
        "stage_bundle_map": dict(workflow_context.stage_bundle_map),
        "evidence_fingerprints": dict(workflow_context.evidence_fingerprints),
    }


def _validate_blueprint(
    *,
    blueprint: _WorkflowBlueprintOutput,
    plan: ArchitecturePlan,
    stage_map: Dict[str, ArchitectureStage],
    stage_selection_map: Dict[str, ArchitectStageSelection],
) -> List[str]:
    issues: List[str] = []
    if not blueprint.nodes:
        return ["Workflow blueprint returned no nodes."]

    node_ids = [node.node_id for node in blueprint.nodes]
    node_names = [node.name for node in blueprint.nodes]
    if len(node_ids) != len(set(node_ids)):
        issues.append("Workflow blueprint returned duplicate node ids.")
    if len(node_names) != len(set(node_names)):
        issues.append("Workflow blueprint returned duplicate node names.")

    nodes_by_id = {node.node_id: node for node in blueprint.nodes}
    stage_node_ids: Dict[str, List[str]] = {}
    for node in blueprint.nodes:
        stage_node_ids.setdefault(node.stage_id, []).append(node.node_id)
        if node.stage_id not in stage_map:
            issues.append(f"Node '{node.node_id}' references unknown stage '{node.stage_id}'.")
            continue
        selection = stage_selection_map.get(node.stage_id)
        if selection is None:
            issues.append(f"Node '{node.node_id}' references stage '{node.stage_id}' without selection state.")
            continue
        if node.node_type not in selection.selected_node_types:
            issues.append(
                f"Node '{node.node_id}' uses node type '{node.node_type}' outside the selected stage candidates."
            )
        if node.type_version <= 0:
            issues.append(f"Node '{node.node_id}' returned invalid type_version.")
        candidate = next(
            (item for item in selection.selected_nodes if item.node_type == node.node_type),
            None,
        )
        if candidate is not None and _candidate_rejection_reasons(
            candidate=candidate,
            stage=stage_map[node.stage_id],
            plan=plan,
            stage_requires_trigger=(plan.stages and node.stage_id == plan.stages[0].id),
        ):
            issues.append(f"Node '{node.node_id}' is incompatible with v1 classic workflow constraints.")
    for stage in plan.stages:
        if stage.id not in stage_node_ids:
            issues.append(f"Workflow blueprint did not materialize required stage '{stage.id}'.")

    stage_connection_pairs = set()
    for connection in blueprint.connections:
        if connection.type != "main":
            issues.append("Workflow blueprint used non-main connection type in v1 architect flow.")
        if connection.source_node_id not in nodes_by_id or connection.target_node_id not in nodes_by_id:
            issues.append("Workflow blueprint returned a connection to a non-existent node.")
            continue
        source_node = nodes_by_id[connection.source_node_id]
        target_node = nodes_by_id[connection.target_node_id]
        stage_connection_pairs.add((source_node.stage_id, target_node.stage_id))
        selection = stage_selection_map.get(target_node.stage_id)
        candidate = None
        if selection is not None:
            candidate = next(
                (item for item in selection.selected_nodes if item.node_type == target_node.node_type),
                None,
            )
        if candidate is not None and _candidate_requires_non_main_inputs(candidate):
            issues.append(
                f"Connection into '{target_node.node_id}' is invalid because the target requires non-main input connectors."
            )

    for selection in stage_selection_map.values():
        selected_types = set(selection.selected_node_types)
        used_types = {node.node_type for node in blueprint.nodes if node.stage_id == selection.stage_id}
        if selected_types and not selected_types.issubset(used_types):
            issues.append(f"Stage '{selection.stage_id}' did not materialize every selected node type in the draft.")
    for node in blueprint.nodes:
        for dependency in node.depends_on:
            source_node = nodes_by_id.get(dependency)
            if source_node is not None:
                stage_connection_pairs.add((source_node.stage_id, node.stage_id))
    for flow in plan.data_flow:
        if (flow.source_stage_id, flow.target_stage_id) not in stage_connection_pairs:
            issues.append(
                f"Workflow blueprint does not connect required stage flow '{flow.source_stage_id}' -> '{flow.target_stage_id}'."
            )

    trigger_present = False
    for node in blueprint.nodes:
        selection = stage_selection_map.get(node.stage_id)
        if selection is None:
            continue
        candidate = next(
            (item for item in selection.selected_nodes if item.node_type == node.node_type),
            None,
        )
        if candidate is not None and _is_trigger_candidate(candidate):
            trigger_present = True
            break
    if not trigger_present:
        issues.append("Workflow blueprint returned no trigger-capable start node.")
    return issues


def _draft_from_blueprint(
    *,
    blueprint: _WorkflowBlueprintOutput,
    plan: ArchitecturePlan,
    stage_map: Dict[str, ArchitectureStage],
    stage_selection_map: Dict[str, ArchitectStageSelection],
) -> Tuple[WorkflowDraft, List[ProposedNode]]:
    draft_nodes: List[WorkflowDraftNode] = []
    proposed_nodes: List[ProposedNode] = []
    connection_entries: List[WorkflowDraftConnection] = []

    for idx, node in enumerate(blueprint.nodes):
        stage = stage_map[node.stage_id]
        purpose = _compact(node.purpose or stage.purpose, max_chars=220)
        position = [260 + (idx * 280), 300]
        selection = stage_selection_map[node.stage_id]
        candidate = next(
            (item for item in selection.selected_nodes if item.node_type == node.node_type),
            None,
        )
        implementation_hints = _derive_operation_hints(
            stage=stage,
            candidate=candidate,
            plan=plan,
        )
        draft_nodes.append(
            WorkflowDraftNode(
                node_id=node.node_id,
                name=node.name,
                node_type=node.node_type,
                type_version=max(1, node.type_version),
                purpose=purpose,
                stage_id=node.stage_id,
                parameters_known={},
                parameters_inferred={},
                parameters_unresolved=[],
                credential_refs={},
                expected_inputs=list(stage.expected_inputs),
                expected_outputs=list(stage.expected_outputs),
                dependencies=list(node.depends_on),
                position=position,
                notes=_safe_list(
                    [f"architect_stage={node.stage_id}"]
                    + (
                        ["architect_requires_material_action"]
                        if implementation_hints.get("require_action_selection")
                        else []
                    )
                ),
                implementation_hints=dict(implementation_hints),
            )
        )
        proposed_nodes.append(
            ProposedNode(
                node_id=node.node_id,
                node_type=node.node_type,
                stage_id=node.stage_id,
                purpose=purpose or selection.rationale or stage.purpose,
                depends_on=list(node.depends_on),
                expected_inputs=list(stage.expected_inputs),
                expected_outputs=list(stage.expected_outputs),
                usage_mode=(candidate.usage_mode if candidate else "unknown"),
                usable_as_tool=(candidate.usable_as_tool if candidate else None),
                has_main_input=(candidate.has_main_input if candidate else None),
                input_connection_types=(list(candidate.input_connection_types) if candidate else []),
                output_connection_types=(list(candidate.output_connection_types) if candidate else []),
                implementation_hints=dict(implementation_hints),
            )
        )

    if blueprint.connections:
        for connection in blueprint.connections:
            connection_entries.append(
                WorkflowDraftConnection(
                    source_node_id=connection.source_node_id,
                    target_node_id=connection.target_node_id,
                    source_output=connection.type,
                    target_input=connection.type,
                )
            )
    else:
        for idx, node in enumerate(blueprint.nodes[1:], start=1):
            deps = list(node.depends_on)
            if not deps and idx > 0:
                deps = [blueprint.nodes[idx - 1].node_id]
            for dep in deps:
                connection_entries.append(
                    WorkflowDraftConnection(
                        source_node_id=dep,
                        target_node_id=node.node_id,
                        source_output="main",
                        target_input="main",
                    )
                )

    draft = WorkflowDraft(
        name=blueprint.workflow_name or plan.title or "Architect Workflow Draft",
        use_case_id=plan.use_case_id,
        summary=blueprint.summary or plan.workflow_summary,
        nodes=draft_nodes,
        connections=connection_entries,
        metadata={
            "architect_stage_ids": [stage.id for stage in plan.stages],
            "architect_selected_node_types": [item.node_type for item in proposed_nodes],
            "stage_operation_hints": {
                item.stage_id: dict(item.implementation_hints)
                for item in proposed_nodes
                if item.stage_id
            },
        },
    )
    return draft, proposed_nodes


def architect_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    model, request_id = _runtime_context(state)
    routing_signals = list(state.get("routing_signals") or [])
    if "entered_architect_agent" not in routing_signals:
        routing_signals.append("entered_architect_agent")

    architecture_plan = _normalize_model(state.get("architecture_plan"), ArchitecturePlan)
    workflow_context = _normalize_model(state.get("workflow_context"), WorkflowContext)
    selected_use_case = _normalize_use_case(state.get("selected_use_case"))
    architect_status = _normalize_status(state.get("architect_status"))
    search_history = _normalize_list(state.get("architect_stage_search_history"), ArchitectStageSearchState)
    stage_selections = _normalize_list(state.get("architect_stage_selections"), ArchitectStageSelection)
    clarification_state = _normalize_model(
        state.get("architect_clarification_state"),
        ArchitectClarificationState,
    ) or ArchitectClarificationState(max_attempts=_MAX_USER_CLARIFICATIONS)
    architect_notes = list(state.get("architect_notes") or [])
    workflow_versions = _normalize_list(state.get("workflow_versions"), WorkflowVersion)
    user_query = str(state.get("user_query") or "")

    if architecture_plan is None:
        message = "Architect requires an abstract architecture_plan from product_manager_agent before grounding nodes."
        missing_detail = _build_missing_input(stage_id="architect", message=message, slot_key="planning_gap")
        return {
            "current_stage": "architect_agent",
            "target_stage": None,
            "architecture_plan": architecture_plan,
            "architect_status": ArchitectStatus.architect_failed_no_solution,
            "architect_notes": [message],
            "missing_user_inputs": [missing_detail.question],
            "missing_user_input_details": [missing_detail],
            "pending_decision_slots": [],
            "resolved_decision_slots": [],
            "clarification_owner": AgentStage.architect_agent,
            "clarification_reason": "planning_gap",
            "last_block_cause": "planning_gap",
            "workflow_persisted": False,
            "workflow_persist_action": None,
            "workflow_api_sync_result": {},
            "routing_signals": routing_signals,
        }

    if workflow_context is None:
        workflow_context = WorkflowContext(use_case_id=architecture_plan.use_case_id)

    if (
        architect_status == ArchitectStatus.architect_blocked_waiting_user
        and (clarification_state.pending_questions or clarification_state.pending_slots)
        and user_query
    ):
        if is_question_rephrase_request(user_query):
            clarification_state = _rephrase_pending_architect_questions(
                clarification_state,
                user_query=user_query,
                plan=architecture_plan,
            )
            pending_questions = pending_slot_questions(clarification_state.pending_slots) or list(
                clarification_state.pending_questions
            )
            pending_stage_ids = [
                turn.stage_id or "architect"
                for turn in clarification_state.turns
                if not turn.answer
            ]
            missing_details = [
                _build_missing_input(
                    stage_id=(pending_stage_ids[idx] if idx < len(pending_stage_ids) else "architect"),
                    message=question,
                )
                for idx, question in enumerate(pending_questions)
            ]
            workflow_context.planning_ready = False
            workflow_context.handoff_target = None
            workflow_context.unresolved_inputs = [item.question for item in missing_details]
            workflow_context.pending_decision_slots = list(clarification_state.pending_slots)
            workflow_context.resolved_decision_slots = list(clarification_state.resolved_slots)
            workflow_context.clarification_owner = AgentStage.architect_agent
            workflow_context.clarification_reason = "selection_failure"
            workflow_context.last_block_cause = "selection_failure"
            workflow_context.notes = _safe_list(
                list(workflow_context.notes) + ["architect_question_rephrased"]
            )
            routing_signals.append("architect_question_rephrased")
            return {
                "current_stage": "architect_agent",
                "target_stage": None,
                "architecture_plan": architecture_plan,
                "architect_status": ArchitectStatus.architect_blocked_waiting_user,
                "architect_stage_search_history": search_history,
                "architect_stage_selections": stage_selections,
                "architect_clarification_state": clarification_state,
                "architect_notes": _safe_list(list(state.get("architect_notes") or []) + ["Rephrased pending architect question for the user."]),
                "missing_user_inputs": [item.question for item in missing_details],
                "missing_user_input_details": missing_details,
                "workflow_context": workflow_context,
                "pending_decision_slots": list(clarification_state.pending_slots),
                "resolved_decision_slots": list(clarification_state.resolved_slots),
                "clarification_owner": AgentStage.architect_agent,
                "clarification_reason": "selection_failure",
                "last_block_cause": "selection_failure",
                "routing_signals": routing_signals,
            }
        clarification_state = _record_clarification_answer(clarification_state, user_query)
        emit_trace_event(
            trace_logger,
            event="clarification_slot_resolved",
            request_id=request_id,
            stage="multi_agent.architect",
            payload={
                "owner_agent": AgentStage.architect_agent.value,
                "resolved_slots": [
                    {
                        "slot_key": slot.slot_key,
                        "stage_id": slot.stage_id,
                        "answer_status": slot.answer_status.value,
                    }
                    for slot in clarification_state.resolved_slots
                ],
            },
        )

    user_query_value = _request_context_query(
        state=state,
        plan=architecture_plan,
        current_user_query=user_query,
    )
    architecture_plan = _refine_terminal_outcome_plan_from_slots(
        plan=architecture_plan,
        clarification_state=clarification_state,
    )
    stage_map = {stage.id: stage for stage in architecture_plan.stages}
    stage_selection_map = {item.stage_id: item for item in stage_selections}
    search_states = list(search_history)
    retrieval_cache: Dict[str, List[Dict[str, Any]]] = {}
    evidence_fingerprints = dict(workflow_context.evidence_fingerprints or {})
    terminal_outcome_question = _plan_terminal_outcome_gap_question(architecture_plan)
    if terminal_outcome_question:
        workflow_context.evidence_fingerprints = dict(evidence_fingerprints)
        updates = _block_architect(
            architecture_plan=architecture_plan,
            workflow_context=workflow_context,
            selections=stage_selections,
            search_history=search_states,
            clarification_state=clarification_state,
            architect_notes=architect_notes,
            question=terminal_outcome_question,
            stage_id="plan_terminal_outcome",
            user_query=user_query_value,
            block_cause="planning_gap",
        )
        updates["routing_signals"] = routing_signals
        emit_trace_event(
            trace_logger,
            event="architect_result",
            request_id=request_id,
            stage="multi_agent.architect",
            payload={
                "status": updates.get("architect_status"),
                "reason": "missing_terminal_business_outcome",
                "missing_user_inputs": updates.get("missing_user_inputs", []),
            },
        )
        return updates

    for stage_index, stage in enumerate(architecture_plan.stages):
        existing_selection = stage_selection_map.get(stage.id)
        if existing_selection and existing_selection.selected_node_types and not existing_selection.blocked:
            continue

        stage_requires_trigger = stage_index == 0
        final_selection: Optional[ArchitectStageSelection] = None
        selection_output = _StageSelectionOutput()
        previous_fingerprint: Optional[str] = None

        for pass_index in range(1, _MAX_STAGE_RETRIEVAL_PASSES + 1):
            stage_query = _build_stage_query(
                user_query=user_query_value,
                plan=architecture_plan,
                stage=stage,
                previous_selections=stage_selections,
                clarification_state=clarification_state,
                pass_index=pass_index,
            )
            normalized_query = " ".join(stage_query.lower().split())
            emit_trace_event(
                trace_logger,
                event="architect_stage_query_variant",
                request_id=request_id,
                stage="multi_agent.architect.stage_search",
                payload={
                    "stage_id": stage.id,
                    "query_variant": pass_index,
                    "query": stage_query,
                    "cache_hit": normalized_query in retrieval_cache,
                },
            )
            docs_chunks = retrieval_cache.get(normalized_query)
            if docs_chunks is None:
                docs_chunks = retrieve_context(
                    stage_query,
                    top_k=_DEFAULT_TOP_K,
                    request_id=request_id,
                    source_filter=_API_DOCS_SOURCE,
                )
                retrieval_cache[normalized_query] = docs_chunks
            page_keys, selected_page_details, discarded_page_details = _select_linking_page_keys(
                docs_chunks=docs_chunks,
                user_query=user_query_value,
                plan=architecture_plan,
                stage=stage,
                previous_selections=stage_selections,
                stage_requires_trigger=stage_requires_trigger,
            )
            emit_trace_event(
                trace_logger,
                event="architect_linking_page_selection",
                request_id=request_id,
                stage="multi_agent.architect.stage_search",
                payload={
                    "stage_id": stage.id,
                    "pass_index": pass_index,
                    "selected_page_keys": list(page_keys),
                    "selected_pages": selected_page_details,
                    "discarded_pages": discarded_page_details,
                },
            )
            linked_chunks = query_related_definition_chunks(page_keys, request_id=request_id) if page_keys else []
            candidates = _build_candidates(
                stage_id=stage.id,
                docs_chunks=docs_chunks,
                linked_chunks=linked_chunks,
            )
            candidates = _augment_candidates_with_definition_fallbacks(
                stage=stage,
                plan=architecture_plan,
                stage_requires_trigger=stage_requires_trigger,
                candidates=candidates,
            )
            top_rerank = max(
                [
                    score
                    for chunk in docs_chunks
                    for score in [_normalize_confidence(chunk.get("rerank_score"))]
                    if score is not None
                ]
                or [0.0]
            )
            search_state = ArchitectStageSearchState(
                stage_id=stage.id,
                pass_index=pass_index,
                query=stage_query,
                doc_chunk_ids=[str(chunk.get("doc_id") or "") for chunk in docs_chunks[:8] if str(chunk.get("doc_id") or "").strip()],
                candidate_node_types=[item.node_type for item in candidates[:12]],
                result_count=len(docs_chunks),
                top_rerank_confidence=(top_rerank if top_rerank > 0 else None),
                notes=["stage_requires_trigger=true"] if stage_requires_trigger else [],
            )
            search_states.append(search_state)

            emit_trace_event(
                trace_logger,
                event="architect_stage_search",
                request_id=request_id,
                stage="multi_agent.architect.stage_search",
                payload={
                    "stage_id": stage.id,
                    "pass_index": pass_index,
                    "query": stage_query,
                    "doc_chunk_ids": search_state.doc_chunk_ids,
                    "candidate_node_types": search_state.candidate_node_types,
                    "result_count": search_state.result_count,
                    "top_rerank_confidence": search_state.top_rerank_confidence,
                },
            )

            valid_candidates = [
                item
                for item in candidates
                if not _candidate_rejection_reasons(
                    candidate=item,
                    stage=stage,
                    plan=architecture_plan,
                    stage_requires_trigger=stage_requires_trigger,
                )
            ]
            rejected_candidates = [
                {
                    "node_type": item.node_type,
                    "rejected_reasons": _candidate_rejection_reasons(
                        candidate=item,
                        stage=stage,
                        plan=architecture_plan,
                        stage_requires_trigger=stage_requires_trigger,
                    ),
                }
                for item in candidates
                if item not in valid_candidates
            ]
            fingerprint = _evidence_fingerprint(
                selected_page_keys=page_keys,
                valid_candidates=valid_candidates,
            )
            evidence_fingerprints[stage.id] = fingerprint
            search_state.evidence_fingerprint = fingerprint
            search_state.query_variant = pass_index
            emit_trace_event(
                trace_logger,
                event="architect_stage_candidates",
                request_id=request_id,
                stage="multi_agent.architect.stage_selection",
                payload={
                    "stage_id": stage.id,
                    "pass_index": pass_index,
                    "stage_requires_trigger": stage_requires_trigger,
                    "candidate_count": len(candidates),
                    "valid_candidate_count": len(valid_candidates),
                    "valid_candidates": [
                        _candidate_trace_summary(item) for item in valid_candidates[:8]
                    ],
                    "rejected_candidates": rejected_candidates[:8],
                    "evidence_fingerprint": fingerprint,
                },
            )
            emit_trace_event(
                trace_logger,
                event="architect_evidence_fingerprint",
                request_id=request_id,
                stage="multi_agent.architect.stage_search",
                payload={
                    "stage_id": stage.id,
                    "query_variant": pass_index,
                    "selected_page_keys": list(page_keys),
                    "valid_candidate_node_types": [item.node_type for item in valid_candidates],
                    "evidence_fingerprint": fingerprint,
                },
            )

            if pass_index > 1 and previous_fingerprint == fingerprint:
                question = _semantic_architect_question(
                    stage=stage,
                    plan=architecture_plan,
                    stage_requires_trigger=stage_requires_trigger,
                )
                selection_output = _StageSelectionOutput(
                    selected_node_types=[],
                    rationale="Search frontier stayed stable across query variants; retry would not add new evidence.",
                    needs_clarification=True,
                    clarification_questions=[question],
                )
                emit_trace_event(
                    trace_logger,
                    event="architect_retry_stopped_stable_frontier",
                    request_id=request_id,
                    stage="multi_agent.architect.stage_search",
                    payload={
                        "stage_id": stage.id,
                        "query_variant": pass_index,
                        "evidence_fingerprint": fingerprint,
                        "question": question,
                    },
                )
                break
            previous_fingerprint = fingerprint

            selection_output = _select_stage_nodes_with_structured_output(
                user_query=user_query_value,
                plan=architecture_plan,
                stage=stage,
                candidates=valid_candidates,
                previous_selections=stage_selections,
                stage_requires_trigger=stage_requires_trigger,
                model=model,
                request_id=request_id,
            )
            selected_candidates = [
                item for item in valid_candidates if item.node_type in set(selection_output.selected_node_types)
            ]
            if selection_output.selected_node_types and not selected_candidates:
                emit_trace_event(
                    trace_logger,
                    event="architect_selection_failure",
                    request_id=request_id,
                    stage="multi_agent.architect.stage_selection",
                    payload={
                        "stage_id": stage.id,
                        "query_variant": pass_index,
                        "selected_node_types": list(selection_output.selected_node_types),
                        "valid_candidate_node_types": [item.node_type for item in valid_candidates],
                        "rationale": selection_output.rationale,
                    },
                )
                if len(valid_candidates) == 1:
                    selected_candidates = [valid_candidates[0]]
                    selection_output = selection_output.model_copy(
                        update={
                            "selected_node_types": [valid_candidates[0].node_type],
                            "rationale": (
                                selection_output.rationale
                                or "Applied deterministic fallback to the only compatible candidate."
                            ),
                            "needs_clarification": False,
                            "clarification_questions": [],
                        }
                    )
                else:
                    question = _selection_failure_question(
                        stage=stage,
                        plan=architecture_plan,
                        stage_requires_trigger=stage_requires_trigger,
                        selected_node_types=selection_output.selected_node_types,
                    )
                    selection_output = selection_output.model_copy(
                        update={
                            "selected_node_types": [],
                            "needs_clarification": True,
                            "clarification_questions": [question],
                        }
                    )
                    emit_trace_event(
                        trace_logger,
                        event="architect_retry_stopped_stable_frontier",
                        request_id=request_id,
                        stage="multi_agent.architect.stage_selection",
                        payload={
                            "stage_id": stage.id,
                            "query_variant": pass_index,
                            "reason": "selection_failure_outside_valid_candidate_set",
                            "question": question,
                        },
                    )
                    break
            emit_trace_event(
                trace_logger,
                event="architect_stage_selection_decision",
                request_id=request_id,
                stage="multi_agent.architect.stage_selection",
                payload={
                    "stage_id": stage.id,
                    "pass_index": pass_index,
                    "selected_node_types": list(selection_output.selected_node_types),
                    "needs_clarification": selection_output.needs_clarification,
                    "clarification_questions": list(selection_output.clarification_questions),
                    "selected_candidates": [
                        _candidate_trace_summary(item) for item in selected_candidates
                    ],
                    "rationale": selection_output.rationale,
                },
            )
            if (
                not selection_output.needs_clarification
                and selected_candidates
                and (not stage_requires_trigger or any(_is_trigger_candidate(item) for item in selected_candidates))
            ):
                final_selection = ArchitectStageSelection(
                    stage_id=stage.id,
                    selected_node_types=[item.node_type for item in selected_candidates],
                    selected_nodes=selected_candidates,
                    rationale=_compact(selection_output.rationale or stage.purpose, max_chars=260),
                    passes_used=pass_index,
                    blocked=False,
                    missing_information=[],
                )
                break

        if final_selection is None:
            question = ""
            if selection_output.clarification_questions:
                question = selection_output.clarification_questions[0]
            if not question:
                question = _semantic_architect_question(
                    stage=stage,
                    plan=architecture_plan,
                    stage_requires_trigger=stage_requires_trigger,
                )
            workflow_context.evidence_fingerprints = dict(evidence_fingerprints)
            updates = _block_architect(
                architecture_plan=architecture_plan,
                workflow_context=workflow_context,
                selections=stage_selections,
                search_history=search_states,
                clarification_state=clarification_state,
                architect_notes=architect_notes,
                question=question,
                stage_id=stage.id,
                user_query=user_query_value,
                block_cause=(
                    "selection_failure"
                    if selection_output.selected_node_types
                    else "search_failure"
                ),
            )
            updates["routing_signals"] = routing_signals
            emit_trace_event(
                trace_logger,
                event="architect_result",
                request_id=request_id,
                stage="multi_agent.architect",
                payload={
                    "status": updates.get("architect_status"),
                    "stage_id": stage.id,
                    "selected_stages": [item.stage_id for item in stage_selections],
                    "missing_user_inputs": updates.get("missing_user_inputs", []),
                },
            )
            return updates

        stage_selection_map[stage.id] = final_selection
        stage_selections = [item for item in stage_selections if item.stage_id != stage.id] + [final_selection]

    ordered_selections = [stage_selection_map[stage.id] for stage in architecture_plan.stages if stage.id in stage_selection_map]
    emit_trace_event(
        trace_logger,
        event="architect_stage_selection_summary",
        request_id=request_id,
        stage="multi_agent.architect",
        payload={
            "stage_count": len(ordered_selections),
            "selections": [_selection_trace_summary(item) for item in ordered_selections],
        },
    )
    stage_selection_issues = _validate_stage_selection_coverage(
        plan=architecture_plan,
        stage_selections=ordered_selections,
    )
    if stage_selection_issues:
        workflow_context.evidence_fingerprints = dict(evidence_fingerprints)
        emit_trace_event(
            trace_logger,
            event="architect_stage_selection_incomplete",
            request_id=request_id,
            stage="multi_agent.architect",
            payload={
                "issues": list(stage_selection_issues),
                "selections": [_selection_trace_summary(item) for item in ordered_selections],
            },
        )
        updates = _block_architect(
            architecture_plan=architecture_plan,
            workflow_context=workflow_context,
            selections=ordered_selections,
            search_history=search_states,
            clarification_state=clarification_state,
            architect_notes=architect_notes,
            question=stage_selection_issues[0],
            stage_id="stage_selection_coverage",
            user_query=user_query_value,
            block_cause="planning_gap",
        )
        updates["routing_signals"] = routing_signals
        emit_trace_event(
            trace_logger,
            event="architect_result",
            request_id=request_id,
            stage="multi_agent.architect",
            payload={
                "status": updates.get("architect_status"),
                "stage_selection_issues": list(stage_selection_issues),
            },
        )
        return updates
    blueprint = _build_workflow_blueprint_with_structured_output(
        plan=architecture_plan,
        stage_selections=ordered_selections,
        model=model,
        request_id=request_id,
    )
    emit_trace_event(
        trace_logger,
        event="architect_workflow_blueprint_generated",
        request_id=request_id,
        stage="multi_agent.architect.workflow_blueprint",
        payload=_blueprint_trace_summary(blueprint),
    )
    validation_issues = _validate_blueprint(
        blueprint=blueprint,
        plan=architecture_plan,
        stage_map=stage_map,
        stage_selection_map=stage_selection_map,
    )
    if validation_issues:
        workflow_context.evidence_fingerprints = dict(evidence_fingerprints)
        emit_trace_event(
            trace_logger,
            event="architect_workflow_blueprint_invalid",
            request_id=request_id,
            stage="multi_agent.architect.workflow_blueprint",
            payload={
                **_blueprint_trace_summary(blueprint),
                "validation_issues": validation_issues,
            },
        )
        updates = _block_architect(
            architecture_plan=architecture_plan,
            workflow_context=workflow_context,
            selections=ordered_selections,
            search_history=search_states,
            clarification_state=clarification_state,
            architect_notes=architect_notes,
            question=validation_issues[0],
            stage_id="workflow_blueprint",
            user_query=user_query_value,
            block_cause="selection_failure",
        )
        updates["routing_signals"] = routing_signals
        emit_trace_event(
            trace_logger,
            event="architect_result",
            request_id=request_id,
            stage="multi_agent.architect",
            payload={
                "status": updates.get("architect_status"),
                "validation_issues": validation_issues,
            },
        )
        return updates
    emit_trace_event(
        trace_logger,
        event="architect_workflow_blueprint_validated",
        request_id=request_id,
        stage="multi_agent.architect.workflow_blueprint",
        payload={
            "workflow_name": blueprint.workflow_name,
            "node_count": len(blueprint.nodes),
            "connection_count": len(blueprint.connections),
        },
    )

    draft, proposed_nodes = _draft_from_blueprint(
        blueprint=blueprint,
        plan=architecture_plan,
        stage_map=stage_map,
        stage_selection_map=stage_selection_map,
    )
    operation_hint_issues = _validate_operation_hints(
        plan=architecture_plan,
        proposed_nodes=proposed_nodes,
    )
    if operation_hint_issues:
        stage_id, question = operation_hint_issues[0]
        emit_trace_event(
            trace_logger,
            event="architect_stage_selection_incomplete",
            request_id=request_id,
            stage="multi_agent.architect.stage_selection",
            payload={
                "stage_id": stage_id,
                "reason": "missing_operational_action_hint",
                "question": question,
            },
        )
        return _block_architect(
            architecture_plan=architecture_plan,
            workflow_context=workflow_context,
            selections=stage_selections,
            search_history=search_history,
            clarification_state=clarification_state,
            architect_notes=architect_notes,
            question=question,
            stage_id=stage_id,
            user_query=user_query_value,
            block_cause="planning_gap",
        )
    stage_bundle_map = {
        stage.id: [node.node_id for node in draft.nodes if node.stage_id == stage.id]
        for stage in architecture_plan.stages
    }
    draft.metadata["stage_bundle_map"] = dict(stage_bundle_map)
    draft.metadata["architect_evidence_fingerprints"] = dict(evidence_fingerprints)
    workflow_versions = [item for item in workflow_versions if isinstance(item, WorkflowVersion)]
    if not workflow_versions:
        _append_version(workflow_versions, draft, reason="initialized architect workflow draft")
    else:
        _append_version(workflow_versions, draft, reason="updated architect workflow draft")
    final_workflow_json = _draft_to_final_workflow_json(draft)
    persist_payload = _persist_workflow_candidate(
        state=state,
        final_workflow_json=final_workflow_json,
        workflow_name=draft.name,
        request_id=request_id,
    )

    workflow_context.planning_ready = True
    workflow_context.handoff_target = AgentStage.engineer_agent
    workflow_context.required_node_types = [item.node_type for item in proposed_nodes]
    workflow_context.unresolved_inputs = []
    workflow_context.pending_decision_slots = []
    workflow_context.resolved_decision_slots = list(clarification_state.resolved_slots)
    workflow_context.clarification_owner = None
    workflow_context.clarification_reason = None
    workflow_context.last_block_cause = None
    workflow_context.stage_bundle_map = dict(stage_bundle_map)
    workflow_context.evidence_fingerprints = dict(evidence_fingerprints)
    workflow_context.notes = _safe_list(
        list(workflow_context.notes)
        + [
            f"architect_status={ArchitectStatus.architect_completed.value}",
            "handoff_target=engineer_agent",
        ]
    )

    architect_status = ArchitectStatus.architect_completed
    if "handoff_ready_engineer" not in routing_signals:
        routing_signals.append("handoff_ready_engineer")
    if persist_payload.get("workflow_persisted"):
        action = str(persist_payload.get("workflow_persist_action") or "").strip() or "persisted"
        signal = f"workflow_persist_{action}"
        if signal not in routing_signals:
            routing_signals.append(signal)

    architect_notes = _safe_list(
        architect_notes
        + [
            f"architect_selected_stages={len(ordered_selections)}",
            f"persist_action={persist_payload.get('workflow_persist_action') or 'none'}",
        ]
    )

    emit_trace_event(
        trace_logger,
        event="architect_result",
        request_id=request_id,
        stage="multi_agent.architect",
        payload={
            "status": architect_status.value,
            "selected_stage_count": len(ordered_selections),
            "selected_node_types": [item.node_type for item in proposed_nodes],
            "workflow_draft_nodes": [
                {
                    "node_id": item.node_id,
                    "name": item.name,
                    "node_type": item.node_type,
                    "stage_id": item.stage_id,
                    "dependencies": list(item.dependencies),
                }
                for item in draft.nodes
            ],
            "workflow_draft_connections": [
                {
                    "source_node_id": item.source_node_id,
                    "target_node_id": item.target_node_id,
                    "source_output": item.source_output,
                    "target_input": item.target_input,
                }
                for item in draft.connections
            ],
            "workflow_id": persist_payload.get("active_workflow_id"),
            "persist_action": persist_payload.get("workflow_persist_action"),
            "stage_bundle_map": dict(stage_bundle_map),
            "evidence_fingerprints": dict(evidence_fingerprints),
        },
    )

    return {
        "current_stage": "architect_agent",
        "target_stage": AgentStage.engineer_agent,
        "architecture_plan": architecture_plan,
        "architect_status": architect_status,
        "architect_stage_search_history": search_states,
        "architect_stage_selections": ordered_selections,
        "architect_clarification_state": ArchitectClarificationState(
            attempts_used=clarification_state.attempts_used,
            max_attempts=clarification_state.max_attempts,
            pending_questions=[],
            pending_slots=[],
            resolved_slots=list(clarification_state.resolved_slots),
            turns=list(clarification_state.turns),
        ),
        "architect_notes": architect_notes,
        "workflow_context": workflow_context,
        "pending_decision_slots": [],
        "resolved_decision_slots": list(clarification_state.resolved_slots),
        "clarification_owner": None,
        "clarification_reason": None,
        "last_block_cause": None,
        "stage_bundle_map": dict(stage_bundle_map),
        "evidence_fingerprints": dict(evidence_fingerprints),
        "proposed_nodes": proposed_nodes,
        "workflow_draft": draft,
        "workflow_versions": workflow_versions,
        "final_workflow_json": final_workflow_json,
        "missing_user_inputs": [],
        "missing_user_input_details": [],
        "routing_signals": routing_signals,
        **persist_payload,
    }
