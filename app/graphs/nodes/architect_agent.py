from __future__ import annotations

import html
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ...db import query_related_definition_chunks
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
    MissingUserInput,
    ProposedNode,
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


logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")

_API_DOCS_SOURCE = "n8n-docs"
_MAX_STAGE_RETRIEVAL_PASSES = 3
_MAX_USER_CLARIFICATIONS = 3
_DEFAULT_TOP_K = 8

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


def _record_clarification_answer(
    clarification_state: ArchitectClarificationState,
    user_query: str,
) -> ArchitectClarificationState:
    pending = list(clarification_state.pending_questions)
    turns = list(clarification_state.turns)
    if not pending:
        return clarification_state
    question = pending.pop(0)
    answer = _compact(user_query, max_chars=320)
    for idx in range(len(turns) - 1, -1, -1):
        if turns[idx].question == question and not turns[idx].answer:
            turns[idx] = turns[idx].model_copy(update={"answer": answer})
            break
    else:
        turns.append(ArchitectClarificationTurn(question=question, answer=answer))
    return clarification_state.model_copy(update={"pending_questions": pending, "turns": turns})


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
    pass_hint = ""
    if pass_index == 2:
        pass_hint = (
            "Prefer exact inbound trigger or main-path action nodes that match the user wording. "
            "Reject send/respond nodes for intake and reject nodes with auxiliary AI connectors."
        )
    elif pass_index >= 3:
        pass_hint = (
            "Treat workflow compatibility as a hard constraint. Preserve the exact source system, trigger style, "
            "processing mode, and final outcome named by the user instead of forcing a near match."
        )
    return (
        f"User request: {_compact(user_query, max_chars=420)}\n"
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
    combined = " ".join(
        [
            candidate.node_type,
            candidate.display_name or "",
            candidate.capability_summary,
            " ".join(candidate.limitations),
        ]
    ).lower()

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


def _fallback_stage_selection(
    *,
    stage: ArchitectureStage,
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
                f"Which concrete system or trigger should start the stage '{stage.name}'?"
                if stage_requires_trigger
                else f"Which concrete app or action should stage '{stage.name}' use?"
            ],
        )
    best = filtered[0]
    return _StageSelectionOutput(
        selected_node_types=[best.node_type],
        rationale=f"Selected '{best.display_name or best.node_type}' as the best available standard workflow candidate.",
        needs_clarification=False,
        clarification_questions=[],
    )


def _select_stage_nodes_with_structured_output(
    *,
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
                candidates=candidates,
                stage_requires_trigger=stage_requires_trigger,
            )
        return output
    except Exception as exc:
        logger.warning("architect stage selection fallback to deterministic selection: %s", str(exc))
        return _fallback_stage_selection(
            stage=stage,
            candidates=candidates,
            stage_requires_trigger=stage_requires_trigger,
        )


def _short_type(node_type: str) -> str:
    value = str(node_type or "").strip()
    if not value:
        return "node"
    return value.split(".")[-1]


def _fallback_workflow_blueprint(
    *,
    plan: ArchitecturePlan,
    stage_selections: Sequence[ArchitectStageSelection],
) -> _WorkflowBlueprintOutput:
    nodes: List[_WorkflowNodeBlueprint] = []
    connections: List[_WorkflowConnectionBlueprint] = []
    stage_primary_node_ids: Dict[str, str] = {}
    counter = 1
    for selection in stage_selections:
        for candidate in selection.selected_nodes:
            node_id = f"an_{counter}"
            node_name = f"{_short_type(candidate.node_type)}_{counter}"
            nodes.append(
                _WorkflowNodeBlueprint(
                    node_id=node_id,
                    name=node_name,
                    node_type=candidate.node_type,
                    type_version=max(1, candidate.type_version),
                    stage_id=selection.stage_id,
                    purpose=candidate.rationale or selection.rationale,
                    depends_on=[],
                )
            )
            stage_primary_node_ids.setdefault(selection.stage_id, node_id)
            counter += 1

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
        "Stage neighborhood context:\n"
        f"{neighborhood_context}\n\n"
        "Rules:\n"
        "- Use only the selected node types.\n"
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
        return output
    except Exception as exc:
        logger.warning("architect workflow blueprint fallback to deterministic draft: %s", str(exc))
        return _fallback_workflow_blueprint(plan=plan, stage_selections=stage_selections)


def _build_missing_input(
    *,
    stage_id: str,
    message: str,
) -> MissingUserInput:
    question = _compact(message, max_chars=220)
    return MissingUserInput(
        input_id=f"handoff:{stage_id}:architect",
        input_key=f"handoff:{stage_id}:architect",
        missing_item=stage_id,
        reason=question,
        blocking_node_id=stage_id,
        category="handoff",
        question=question,
    )


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
) -> Dict[str, Any]:
    question_value = _compact(question, max_chars=220)
    updated_attempts = clarification_state.attempts_used + 1
    if updated_attempts > clarification_state.max_attempts:
        failure_question = (
            f"Architect stopped after {clarification_state.max_attempts} clarification attempts. {question_value}"
        )
        missing_details = [_build_missing_input(stage_id=stage_id, message=failure_question)]
        if workflow_context is None:
            workflow_context = WorkflowContext(use_case_id=architecture_plan.use_case_id)
        workflow_context.planning_ready = False
        workflow_context.handoff_target = None
        workflow_context.unresolved_inputs = [item.question for item in missing_details]
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
        }

    turns = list(clarification_state.turns)
    turns.append(ArchitectClarificationTurn(stage_id=stage_id, question=question_value, answer=None))
    clarification_state = clarification_state.model_copy(
        update={
            "attempts_used": updated_attempts,
            "pending_questions": [question_value],
            "turns": turns,
        }
    )
    missing_details = [_build_missing_input(stage_id=stage_id, message=question_value)]
    if workflow_context is None:
        workflow_context = WorkflowContext(use_case_id=architecture_plan.use_case_id)
    workflow_context.planning_ready = False
    workflow_context.handoff_target = None
    workflow_context.unresolved_inputs = [item.question for item in missing_details]
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
    for node in blueprint.nodes:
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

    for connection in blueprint.connections:
        if connection.type != "main":
            issues.append("Workflow blueprint used non-main connection type in v1 architect flow.")
        if connection.source_node_id not in nodes_by_id or connection.target_node_id not in nodes_by_id:
            issues.append("Workflow blueprint returned a connection to a non-existent node.")
            continue
        target_node = nodes_by_id[connection.target_node_id]
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
                notes=[f"architect_stage={node.stage_id}"],
            )
        )
        selection = stage_selection_map[node.stage_id]
        candidate = next(
            (item for item in selection.selected_nodes if item.node_type == node.node_type),
            None,
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
        missing_detail = _build_missing_input(stage_id="architect", message=message)
        return {
            "current_stage": "architect_agent",
            "target_stage": None,
            "architect_status": ArchitectStatus.architect_failed_no_solution,
            "architect_notes": [message],
            "missing_user_inputs": [missing_detail.question],
            "missing_user_input_details": [missing_detail],
            "workflow_persisted": False,
            "workflow_persist_action": None,
            "workflow_api_sync_result": {},
            "routing_signals": routing_signals,
        }

    if workflow_context is None:
        workflow_context = WorkflowContext(use_case_id=architecture_plan.use_case_id)

    if architect_status == ArchitectStatus.architect_blocked_waiting_user and clarification_state.pending_questions and user_query:
        clarification_state = _record_clarification_answer(clarification_state, user_query)

    stage_map = {stage.id: stage for stage in architecture_plan.stages}
    stage_selection_map = {item.stage_id: item for item in stage_selections}
    search_states = list(search_history)
    user_query_value = _compact(
        selected_use_case.title if selected_use_case is not None and not user_query else user_query,
        max_chars=420,
    )

    for stage_index, stage in enumerate(architecture_plan.stages):
        existing_selection = stage_selection_map.get(stage.id)
        if existing_selection and existing_selection.selected_node_types and not existing_selection.blocked:
            continue

        stage_requires_trigger = stage_index == 0
        final_selection: Optional[ArchitectStageSelection] = None
        selection_output = _StageSelectionOutput()

        for pass_index in range(1, _MAX_STAGE_RETRIEVAL_PASSES + 1):
            stage_query = _build_stage_query(
                user_query=user_query_value or architecture_plan.title,
                plan=architecture_plan,
                stage=stage,
                previous_selections=stage_selections,
                clarification_state=clarification_state,
                pass_index=pass_index,
            )
            docs_chunks = retrieve_context(
                stage_query,
                top_k=_DEFAULT_TOP_K,
                request_id=request_id,
                source_filter=_API_DOCS_SOURCE,
            )
            page_keys = _extract_doc_page_keys(docs_chunks, max_docs=6)
            linked_chunks = query_related_definition_chunks(page_keys, request_id=request_id) if page_keys else []
            candidates = _build_candidates(
                stage_id=stage.id,
                docs_chunks=docs_chunks,
                linked_chunks=linked_chunks,
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

            selection_output = _select_stage_nodes_with_structured_output(
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
            if not question and stage_requires_trigger:
                question = (
                    f"I could not find a clear trigger-capable standard n8n node for stage '{stage.name}'. "
                    "Which system or event should start this workflow?"
                )
            elif not question:
                question = (
                    f"I could not find a clear standard n8n node for stage '{stage.name}'. "
                    "Which concrete app or action should this stage use?"
                )
            updates = _block_architect(
                architecture_plan=architecture_plan,
                workflow_context=workflow_context,
                selections=stage_selections,
                search_history=search_states,
                clarification_state=clarification_state,
                architect_notes=architect_notes,
                question=question,
                stage_id=stage.id,
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
    blueprint = _build_workflow_blueprint_with_structured_output(
        plan=architecture_plan,
        stage_selections=ordered_selections,
        model=model,
        request_id=request_id,
    )
    validation_issues = _validate_blueprint(
        blueprint=blueprint,
        plan=architecture_plan,
        stage_map=stage_map,
        stage_selection_map=stage_selection_map,
    )
    if validation_issues:
        updates = _block_architect(
            architecture_plan=architecture_plan,
            workflow_context=workflow_context,
            selections=ordered_selections,
            search_history=search_states,
            clarification_state=clarification_state,
            architect_notes=architect_notes,
            question=validation_issues[0],
            stage_id="workflow_blueprint",
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

    draft, proposed_nodes = _draft_from_blueprint(
        blueprint=blueprint,
        plan=architecture_plan,
        stage_map=stage_map,
        stage_selection_map=stage_selection_map,
    )
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
            "workflow_id": persist_payload.get("active_workflow_id"),
            "persist_action": persist_payload.get("workflow_persist_action"),
        },
    )

    return {
        "current_stage": "architect_agent",
        "target_stage": AgentStage.engineer_agent,
        "architect_status": architect_status,
        "architect_stage_search_history": search_states,
        "architect_stage_selections": ordered_selections,
        "architect_clarification_state": ArchitectClarificationState(
            attempts_used=clarification_state.attempts_used,
            max_attempts=clarification_state.max_attempts,
            pending_questions=[],
            turns=list(clarification_state.turns),
        ),
        "architect_notes": architect_notes,
        "workflow_context": workflow_context,
        "proposed_nodes": proposed_nodes,
        "workflow_draft": draft,
        "workflow_versions": workflow_versions,
        "final_workflow_json": final_workflow_json,
        "missing_user_inputs": [],
        "missing_user_input_details": [],
        "routing_signals": routing_signals,
        **persist_payload,
    }
