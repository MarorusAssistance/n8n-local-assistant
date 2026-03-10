from __future__ import annotations

import html
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ...config import settings
from ...doc_links import derive_doc_page_key
from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    EntryIntent,
    NodeRequirement,
    PMClarificationState,
    PMClarificationTurn,
    PMNodeCandidate,
    PMProgressState,
    PMStagePlan,
    PMStageSearchState,
    PMStageSelection,
    PMStatus,
    ProposedNode,
    UseCase,
    WorkflowContext,
)
from ...llm import get_langchain_chat_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event, emit_trace_event
from ...token_budget import estimate_messages_tokens
from ...workflow.retriever import retrieve_docs
from ..multi_agent_state import MultiAgentGraphState

logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")

_MAX_REQUIRED_NODES = 20
_NODE_TYPE_LINE_RE = re.compile(r"(?im)^Node Type:\s*(.+?)\s*$")
_USABLE_AS_TOOL_LINE_RE = re.compile(r"(?im)^Usable As Tool:\s*(true|false)\s*$")
_INPUTS_LINE_RE = re.compile(r"(?im)^Inputs:\s*(.+?)\s*$")


class NodeFunctionalSummary(BaseModel):
    node_type: str
    display_name: Optional[str] = None
    purpose: str = ""
    capabilities: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    usage_constraints: List[str] = Field(default_factory=list)
    data_io_hints: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)


class _StagePlanOutput(BaseModel):
    stages: List[PMStagePlan] = Field(default_factory=list)


class _CandidateSummaryItem(BaseModel):
    node_type: str
    capability_summary: str
    limitations: List[str] = Field(default_factory=list)


class _CandidateSummaryOutput(BaseModel):
    summaries: List[_CandidateSummaryItem] = Field(default_factory=list)


class _StageSelectionOutput(BaseModel):
    selected_node_types: List[str] = Field(default_factory=list)
    rationale: str = ""
    pm_fit_score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_information: List[str] = Field(default_factory=list)


@dataclass
class _NodeEvidenceAccumulator:
    node_type: str
    display_name: Optional[str] = None
    chunk_ids: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    strength: float = 0.0
    count: int = 0
    rerank_strength: float = 0.0
    rerank_count: int = 0
    usable_true_count: int = 0
    usable_false_count: int = 0
    main_input_count: int = 0
    ai_input_count: int = 0
    input_connection_types: List[str] = field(default_factory=list)
    all_texts: List[str] = field(default_factory=list)

    def add(
        self,
        *,
        chunk_id: str,
        ref: str,
        score: float,
        display_name: Optional[str],
        rerank_scores: List[float],
        usable_as_tool: Optional[bool],
        has_main_input: Optional[bool],
        has_ai_input: Optional[bool],
        input_types: List[str],
        text: str,
    ) -> None:
        if chunk_id and chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)
        if ref and ref not in self.refs:
            self.refs.append(ref)
        self.count += 1
        self.strength += max(0.0, score)
        for rerank_confidence in rerank_scores:
            self.rerank_count += 1
            self.rerank_strength += min(1.0, max(0.0, float(rerank_confidence)))
        if usable_as_tool is True:
            self.usable_true_count += 1
        elif usable_as_tool is False:
            self.usable_false_count += 1
        if has_main_input is True:
            self.main_input_count += 1
        if has_ai_input is True:
            self.ai_input_count += 1
        for item in input_types:
            if item and item not in self.input_connection_types:
                self.input_connection_types.append(item)
        if text:
            self.all_texts.append(text)
        if display_name and not self.display_name:
            self.display_name = display_name


def _pm_int(name: str, default: int) -> int:
    raw = getattr(settings, name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _pm_float(name: str, default: float) -> float:
    raw = getattr(settings, name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return min(1.0, max(0.0, value))


def _compact(text: str, max_chars: int = 220) -> str:
    value = " ".join(_sanitize_text(text).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _sanitize_text(text: Any) -> str:
    raw = html.unescape(str(text or ""))
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def _sanitize_ref_value(value: Any) -> str:
    text = _sanitize_text(value)
    lowered = text.lower()
    if lowered.startswith(("http://", "https://")):
        match = re.search(r"https?://[^\s\"'<>]+", text)
        if match:
            text = match.group(0)
    text = text.replace("&quot;", "")
    text = text.rstrip("\"'`])},; ")
    spillover_markers = (
        '"], "',
        '"}, "',
        "}, {",
        "], {",
        '", "node_type"',
        '", "confidence"',
    )
    for marker in spillover_markers:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].rstrip()
            break
    return _compact(text, max_chars=240)


def _safe_list(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        item = _sanitize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _chunk_metadata(chunk: Any) -> Dict[str, Any]:
    if not isinstance(chunk, dict):
        return {}
    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _is_explicit_node_evidence(chunk: Dict[str, Any]) -> bool:
    metadata = _chunk_metadata(chunk)
    context_kind = str(chunk.get("context_kind") or "").strip().lower()
    linked_type = str(chunk.get("linked_def_type") or "").strip().lower()
    kind = str(metadata.get("kind") or "").strip().upper()
    if context_kind == "linked_def" and linked_type == "node":
        return True
    if metadata.get("nodeType"):
        return True
    return kind.startswith("NODE_")


def _extract_node_type(chunk: Dict[str, Any]) -> Optional[str]:
    metadata = _chunk_metadata(chunk)
    node_type = str(metadata.get("nodeType") or "").strip()
    if node_type:
        return node_type
    context_kind = str(chunk.get("context_kind") or "").strip().lower()
    linked_type = str(chunk.get("linked_def_type") or "").strip().lower()
    if context_kind == "linked_def" and linked_type == "node":
        linked_entity = str(chunk.get("linked_entity_id") or "").strip()
        if linked_entity:
            return linked_entity
    text = str(chunk.get("text") or "")
    match = _NODE_TYPE_LINE_RE.search(text)
    if match:
        from_line = str(match.group(1) or "").strip()
        if from_line:
            return from_line
    return None


def _normalize_connection_type(value: Any) -> Optional[str]:
    text = _sanitize_text(value).strip("\"'`")
    if not text:
        return None
    lowered = text.lower()
    if "nodeconnectiontypes." in lowered:
        lowered = lowered.split("nodeconnectiontypes.", 1)[1]
    lowered = lowered.replace(" ", "").replace("-", "_")
    if lowered == "main":
        return "main"
    if lowered.startswith("ai"):
        return lowered
    return lowered


def _extract_input_connection_types(chunk: Dict[str, Any]) -> List[str]:
    metadata = _chunk_metadata(chunk)
    output: List[str] = []
    inputs = metadata.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            candidate = item
            if isinstance(item, dict):
                candidate = item.get("type") or item.get("name") or item.get("displayName")
            normalized = _normalize_connection_type(candidate)
            if normalized and normalized not in output:
                output.append(normalized)
        if output:
            return output

    text = str(chunk.get("text") or "")
    match = _INPUTS_LINE_RE.search(text)
    if not match:
        return output
    raw_line = str(match.group(1) or "").strip()
    if not raw_line:
        return output

    parsed_items: List[str] = []
    if raw_line.startswith("[") and raw_line.endswith("]"):
        candidate = raw_line
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, list):
                parsed_items = [str(item) for item in loaded]
        except Exception:
            parsed_items = re.findall(r"[A-Za-z0-9_.]+", candidate)
    else:
        parsed_items = re.findall(r"[A-Za-z0-9_.]+", raw_line)

    for item in parsed_items:
        normalized = _normalize_connection_type(item)
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _extract_usable_as_tool(chunk: Dict[str, Any]) -> Optional[bool]:
    metadata = _chunk_metadata(chunk)
    raw_value = metadata.get("usableAsTool")
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        lowered = raw_value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

    text = str(chunk.get("text") or "")
    match = _USABLE_AS_TOOL_LINE_RE.search(text)
    if not match:
        return None
    return str(match.group(1) or "").strip().lower() == "true"


def _classify_usage_mode(
    *,
    usable_as_tool: Optional[bool],
    has_main_input: Optional[bool],
    has_ai_input: Optional[bool],
) -> str:
    if has_main_input is True and usable_as_tool is True:
        return "both"
    if has_main_input is True:
        return "action_only"
    if has_main_input is False and has_ai_input is True:
        return "tool_only"
    if has_main_input is False and usable_as_tool is True:
        return "tool_only"
    return "unknown"


def _chunk_ref(chunk: Dict[str, Any]) -> str:
    for key in ("url", "title", "section", "doc_id"):
        value = _sanitize_ref_value(chunk.get(key))
        if value:
            return value
    return "unknown-reference"


def _normalize_rerank_score(raw_score: Any) -> Optional[float]:
    if raw_score is None:
        return None
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    if score > 50:
        return 1.0
    if score < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


def _chunk_doc_page_key(chunk: Dict[str, Any]) -> str:
    context_kind = str(chunk.get("context_kind") or "").strip().lower()
    if context_kind == "linked_def":
        explicit = str(chunk.get("link_doc_page_key") or "").strip()
        if explicit:
            return explicit
    metadata = _chunk_metadata(chunk)
    page_key, _, _ = derive_doc_page_key(metadata, row_url=str(chunk.get("url") or ""))
    return str(page_key or "").strip()


def _chunk_score(chunk: Dict[str, Any]) -> float:
    metadata = _chunk_metadata(chunk)
    score = 0.0
    context_kind = str(chunk.get("context_kind") or "").strip().lower()
    linked_type = str(chunk.get("linked_def_type") or "").strip().lower()
    kind = str(metadata.get("kind") or "").strip().upper()
    if context_kind == "linked_def" and linked_type == "node":
        score += 1.0
    if kind == "NODE_OVERVIEW":
        score += 0.9
    elif kind.startswith("NODE_"):
        score += 0.7
    score += min(1.0, max(0.0, float(chunk.get("link_confidence") or 0.0)))
    return score


def _docs_rerank_scores_by_page(chunks: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    by_page: Dict[str, List[float]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if str(chunk.get("context_kind") or "").strip().lower() == "linked_def":
            continue
        score = _normalize_rerank_score(chunk.get("rerank_score"))
        if score is None:
            continue
        page_key = _chunk_doc_page_key(chunk)
        if not page_key:
            continue
        by_page.setdefault(page_key, []).append(score)
    return by_page


def _collect_docs_rerank_scores(
    chunk: Dict[str, Any],
    docs_rerank_by_page: Dict[str, List[float]],
) -> List[float]:
    output: List[float] = []
    if str(chunk.get("context_kind") or "").strip().lower() != "linked_def":
        score = _normalize_rerank_score(chunk.get("rerank_score"))
        if score is not None:
            output.append(score)
        return output

    page_key = _chunk_doc_page_key(chunk)
    if page_key:
        output.extend(docs_rerank_by_page.get(page_key, []))
    return output


def _split_text_units(text: str) -> List[str]:
    normalized = _sanitize_text(text)
    if not normalized:
        return []
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", normalized)
    output: List[str] = []
    for part in parts:
        value = _compact(part, max_chars=420)
        if len(value) < 12:
            continue
        output.append(value)
    return output


def _extract_signal_items(
    texts: List[str],
    *,
    keywords: Tuple[str, ...],
    max_items: int = 4,
    max_chars: int = 180,
) -> List[str]:
    output: List[str] = []
    seen = set()
    for text in texts:
        for unit in _split_text_units(text):
            lowered = unit.lower()
            if not any(keyword in lowered for keyword in keywords):
                continue
            compacted = _compact(unit, max_chars=max_chars)
            if compacted in seen:
                continue
            seen.add(compacted)
            output.append(compacted)
            if len(output) >= max_items:
                return output
    return output


def _usage_constraints_from_accumulator(
    accumulator: _NodeEvidenceAccumulator,
    usage_mode: str,
) -> List[str]:
    constraints: List[str] = []
    if usage_mode == "tool_only":
        constraints.append("Use as AI tool sub-node only; do not place as a standard main-action step.")
    elif usage_mode == "action_only":
        constraints.append("Use as standard action/trigger node with main workflow connections.")
    elif usage_mode == "both":
        constraints.append("Can be used both as standard action node and as AI tool.")
    else:
        constraints.append("Usage mode is unclear from evidence; validate node placement.")
    if accumulator.input_connection_types:
        constraints.append(
            f"Observed input connection types: {', '.join(accumulator.input_connection_types[:4])}."
        )
    return _safe_list(constraints)


def _build_node_functional_summary(
    *,
    node_type: str,
    display_name: Optional[str],
    accumulator: _NodeEvidenceAccumulator,
    usage_mode: str,
) -> NodeFunctionalSummary:
    snippets: List[str] = []
    for text in accumulator.all_texts[:3]:
        compacted = _compact(_sanitize_text(text), max_chars=280)
        if compacted:
            snippets.append(compacted)

    purpose = ""
    preferred_markers = (
        "supports",
        "can ",
        "use",
        "trigger",
        "send",
        "receive",
        "read",
        "write",
        "execute",
        "transform",
    )
    for text in accumulator.all_texts:
        units = _split_text_units(text)
        for unit in units:
            lowered = unit.lower()
            if lowered.startswith(("kind:", "node type:", "inputs:", "outputs:")):
                continue
            if any(marker in lowered for marker in preferred_markers):
                purpose = unit
                break
            if not purpose:
                purpose = unit
        if purpose:
            break
    if not purpose:
        label = display_name or node_type
        purpose = f"Use '{label}' to implement one required capability of the selected workflow stage."

    capabilities = _extract_signal_items(
        accumulator.all_texts,
        keywords=(
            "can ",
            "supports",
            "operation",
            "action",
            "trigger",
            "receive",
            "send",
            "write",
            "read",
            "transform",
            "classif",
            "filter",
            "route",
        ),
    )
    if not capabilities and snippets:
        capabilities = snippets[:2]

    limitations = _extract_signal_items(
        accumulator.all_texts,
        keywords=(
            "requires",
            "must ",
            "only ",
            "cannot",
            "can't",
            "limit",
            "warning",
            "credential",
            "auth",
            "permission",
        ),
    )

    data_io_hints: List[str] = []
    if accumulator.input_connection_types:
        data_io_hints.append(f"Input types: {', '.join(accumulator.input_connection_types[:4])}.")
    data_io_hints.extend(
        _extract_signal_items(
            accumulator.all_texts,
            keywords=("input", "output", "payload", "data", "item"),
            max_items=3,
            max_chars=170,
        )
    )

    return NodeFunctionalSummary(
        node_type=node_type,
        display_name=display_name,
        purpose=_compact(purpose, max_chars=220),
        capabilities=_safe_list(capabilities[:4]),
        limitations=_safe_list(limitations[:3]),
        usage_constraints=_usage_constraints_from_accumulator(accumulator, usage_mode),
        data_io_hints=_safe_list(data_io_hints[:4]),
        evidence_refs=list(accumulator.refs[:6]),
        evidence_snippets=snippets,
    )


def _blended_confidence(evidence_confidence: float, rerank_confidence: Optional[float]) -> float:
    evidence = min(1.0, max(0.0, float(evidence_confidence)))
    if rerank_confidence is None:
        return evidence
    rerank = min(1.0, max(0.0, float(rerank_confidence)))
    blended = (0.75 * evidence) + (0.25 * rerank)
    return min(1.0, max(0.0, blended))


def _extract_required_nodes_with_summaries(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[NodeRequirement], Dict[str, NodeFunctionalSummary]]:
    by_type: Dict[str, _NodeEvidenceAccumulator] = {}
    docs_rerank_by_page = _docs_rerank_scores_by_page(chunks)

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if not _is_explicit_node_evidence(chunk):
            continue
        node_type = _extract_node_type(chunk)
        if not node_type:
            continue

        chunk_id = str(chunk.get("doc_id") or "").strip() or f"chunk-{len(by_type) + 1}"
        metadata = _chunk_metadata(chunk)
        display_name = str(metadata.get("displayName") or "").strip() or None
        ref = _chunk_ref(chunk)
        score = _chunk_score(chunk)
        rerank_scores = _collect_docs_rerank_scores(chunk, docs_rerank_by_page)
        input_types = _extract_input_connection_types(chunk)
        has_main_input = "main" in input_types if input_types else None
        has_ai_input = any(item.startswith("ai") for item in input_types) if input_types else None
        usable_as_tool = _extract_usable_as_tool(chunk)

        acc = by_type.get(node_type)
        if acc is None:
            acc = _NodeEvidenceAccumulator(node_type=node_type)
            by_type[node_type] = acc
        acc.add(
            chunk_id=chunk_id,
            ref=ref,
            score=score,
            display_name=display_name,
            rerank_scores=rerank_scores,
            usable_as_tool=usable_as_tool,
            has_main_input=has_main_input,
            has_ai_input=has_ai_input,
            input_types=input_types,
            text=_sanitize_text(chunk.get("text") or ""),
        )

    ranked = sorted(
        by_type.values(),
        key=lambda item: (
            -item.count,
            -item.strength,
            item.node_type.lower(),
        ),
    )

    requirements: List[NodeRequirement] = []
    node_summaries: Dict[str, NodeFunctionalSummary] = {}
    for item in ranked[:_MAX_REQUIRED_NODES]:
        avg_strength = item.strength / item.count if item.count > 0 else 0.0
        evidence_confidence = min(1.0, 0.25 + (0.12 * item.count) + (0.10 * avg_strength))
        rerank_confidence: Optional[float] = None
        if item.rerank_count > 0:
            rerank_confidence = item.rerank_strength / item.rerank_count

        blended_confidence = _blended_confidence(evidence_confidence, rerank_confidence)
        usable_as_tool: Optional[bool] = None
        if item.usable_true_count > 0 and item.usable_false_count == 0:
            usable_as_tool = True
        elif item.usable_false_count > 0 and item.usable_true_count == 0:
            usable_as_tool = False

        has_main_input: Optional[bool] = None
        if item.main_input_count > 0:
            has_main_input = True
        elif item.ai_input_count > 0:
            has_main_input = False

        usage_mode = _classify_usage_mode(
            usable_as_tool=usable_as_tool,
            has_main_input=has_main_input,
            has_ai_input=(item.ai_input_count > 0),
        )

        node_summary = _build_node_functional_summary(
            node_type=item.node_type,
            display_name=item.display_name,
            accumulator=item,
            usage_mode=usage_mode,
        )
        node_summaries[item.node_type] = node_summary

        requirements.append(
            NodeRequirement(
                node_type=item.node_type,
                display_name=item.display_name,
                why_required=node_summary.purpose,
                evidence_chunk_ids=item.chunk_ids[:8],
                evidence_refs=item.refs[:6],
                evidence_confidence=round(evidence_confidence, 2),
                rerank_confidence=(
                    round(rerank_confidence, 2) if isinstance(rerank_confidence, float) else None
                ),
                blended_confidence=round(blended_confidence, 2),
                usage_mode=usage_mode,
                usable_as_tool=usable_as_tool,
                has_main_input=has_main_input,
                input_connection_types=list(item.input_connection_types),
            )
        )

    return requirements, node_summaries


def extract_required_nodes_from_docs(chunks: List[Dict[str, Any]]) -> List[NodeRequirement]:
    requirements, _summaries = _extract_required_nodes_with_summaries(chunks)
    return requirements


def _is_agentic_tool_calling_use_case(use_case: UseCase) -> bool:
    text = " ".join(
        [
            str(use_case.title or ""),
            str(use_case.business_problem or ""),
            str(use_case.desired_outcome or ""),
            str(use_case.expected_value or ""),
        ]
    ).lower()
    hints = (
        "ai agent",
        "agentic",
        "tool calling",
        "tools agent",
        "as ai tool",
        "as a tool",
        "call n8n workflow tool",
    )
    return any(hint in text for hint in hints)


def _filter_required_nodes_for_usage(
    *,
    use_case: UseCase,
    required_nodes: List[NodeRequirement],
) -> Tuple[List[NodeRequirement], List[NodeRequirement]]:
    if not required_nodes:
        return [], []
    allow_tool_only = _is_agentic_tool_calling_use_case(use_case)
    kept: List[NodeRequirement] = []
    dropped: List[NodeRequirement] = []
    for node in required_nodes:
        if node.usage_mode == "tool_only" and not allow_tool_only:
            dropped.append(node)
            continue
        kept.append(node)
    return kept, dropped


def _runtime_context(state: MultiAgentGraphState) -> Tuple[Optional[str], Optional[str]]:
    runtime_context = state.get("runtime_context")
    if not isinstance(runtime_context, dict):
        runtime_context = state.get("workflow_context")
    if isinstance(runtime_context, dict):
        model = runtime_context.get("model")
        request_id = runtime_context.get("request_id")
        return (
            model if isinstance(model, str) or model is None else None,
            request_id if isinstance(request_id, str) or request_id is None else None,
        )
    return None, None


def _normalize_use_case(value: Any) -> Optional[UseCase]:
    if isinstance(value, UseCase):
        return value
    if isinstance(value, dict):
        try:
            return UseCase.model_validate(value)
        except Exception:
            return None
    return None


def _derive_use_case_from_direct_build_request(user_query: str) -> Optional[UseCase]:
    query = _compact(str(user_query or ""), max_chars=220)
    if len(query) < 10:
        return None
    return UseCase(
        id="direct_build_request",
        title=_compact(query, max_chars=80),
        business_problem=f"User requested a new workflow for: {query}",
        desired_outcome=f"Deliver a workflow architecture that satisfies: {query}",
        expected_value="Deliver a first production-ready workflow candidate.",
        feasibility="unknown",
        priority_score=70.0,
        why_selected="Direct workflow build request routed to product manager.",
    )


def _normalize_pm_status(value: Any) -> Optional[PMStatus]:
    if isinstance(value, PMStatus):
        return value
    if isinstance(value, str):
        try:
            return PMStatus(value)
        except ValueError:
            return None
    return None


def _normalize_model(value: Any, model_cls: Any) -> Optional[Any]:
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        try:
            return model_cls.model_validate(value)
        except Exception:
            return None
    return None


def _normalize_model_list(values: Any, model_cls: Any) -> List[Any]:
    if not isinstance(values, list):
        return []
    out: List[Any] = []
    for value in values:
        model = _normalize_model(value, model_cls)
        if model is not None:
            out.append(model)
    return out


def _problem_statement(use_case: UseCase) -> str:
    return (
        f"Use case title: {use_case.title}\n"
        f"Business problem: {use_case.business_problem}\n"
        f"Desired outcome: {use_case.desired_outcome}\n"
        f"Expected business value: {use_case.expected_value}"
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
        raise RuntimeError("No model configured for PM structured output")

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
    prompt_tokens = estimate_messages_tokens(messages)
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage=stage,
        model=model,
        messages=messages,
        estimated_tokens=prompt_tokens,
        params={"temperature": temperature, "structured": True},
    )

    response = structured.invoke(messages)
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage=stage,
        model=model,
        latency_ms=None,
        content=response.model_dump_json(exclude_none=True) if hasattr(response, "model_dump_json") else str(response),
        usage=None,
        extra={"structured": True},
    )
    return response


def _heuristic_stage_plan(use_case: UseCase) -> List[PMStagePlan]:
    return [
        PMStagePlan(
            id="stage_intake",
            name="Intake",
            objective="Capture the incoming trigger/event and normalize payload.",
            expected_inputs=["Business event or external trigger"],
            expected_outputs=["Normalized payload"],
            success_criteria=["Workflow receives event reliably", "Payload is normalized for downstream steps"],
            dependencies=[],
        ),
        PMStagePlan(
            id="stage_processing",
            name="Processing",
            objective="Apply business logic and prepare actions.",
            expected_inputs=["Normalized payload"],
            expected_outputs=["Actionable command payload"],
            success_criteria=["Business rules are represented", "Action payload is complete"],
            dependencies=["stage_intake"],
        ),
        PMStagePlan(
            id="stage_delivery",
            name="Delivery",
            objective="Execute final actions on target systems and record outcome.",
            expected_inputs=["Actionable command payload"],
            expected_outputs=["Delivery result"],
            success_criteria=["Target action executed", "Result is logged"],
            dependencies=["stage_processing"],
        ),
    ]


def _plan_stages_with_structured_output(
    *,
    use_case: UseCase,
    model: Optional[str],
    request_id: Optional[str],
) -> List[PMStagePlan]:
    system_prompt = (
        "You are a Product Manager for n8n workflow architecture. "
        "Create planning stages only (no node parameters, no workflow JSON). "
        "Each stage must be implementation-oriented and explicit."
    )
    user_prompt = (
        "Create a stage-first plan for this use case.\n"
        f"{_problem_statement(use_case)}\n\n"
        "Rules:\n"
        "- Produce 2 to 6 stages.\n"
        "- Every stage must define objective, expected inputs/outputs, and success criteria.\n"
        "- Keep planning level only; do not include secrets or exact node parameter values."
    )
    output = _invoke_structured_output(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=_StagePlanOutput,
        model=model,
        request_id=request_id,
        temperature=0.2,
        stage="multi_agent.product_manager.stage_plan",
    )
    stages = [stage for stage in output.stages if isinstance(stage, PMStagePlan)]
    return stages


def _build_stage_query(
    *,
    use_case: UseCase,
    stage: PMStagePlan,
    pass_index: int,
    prior_missing: List[str],
    clarification_turns: List[PMClarificationTurn],
) -> str:
    clarification_bits = [
        f"Q: {turn.question} A: {turn.answer}"
        for turn in clarification_turns
        if turn.stage_id == stage.id and turn.answer
    ]
    clarification_text = "\n".join(clarification_bits[:3])
    missing_text = "\n".join(prior_missing[:3])
    refine_hint = ""
    if pass_index > 1:
        refine_hint = (
            "Refine retrieval toward explicit n8n node documentation for this stage only. "
            "Favor chunks with concrete node capabilities/limitations."
        )

    return (
        "Retrieve explicit n8n node docs evidence for this workflow stage.\n"
        f"Use case: {use_case.title}\n"
        f"Business problem: {use_case.business_problem}\n"
        f"Desired outcome: {use_case.desired_outcome}\n"
        f"Stage id: {stage.id}\n"
        f"Stage objective: {stage.objective}\n"
        f"Stage expected inputs: {', '.join(stage.expected_inputs) or '-'}\n"
        f"Stage expected outputs: {', '.join(stage.expected_outputs) or '-'}\n"
        f"Previous gaps: {missing_text or '-'}\n"
        f"Clarifications: {clarification_text or '-'}\n"
        f"Pass: {pass_index}\n"
        f"{refine_hint}"
    )


def retrieve_pm_api_docs(
    use_case: UseCase,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return retrieve_docs(_problem_statement(use_case), request_id=request_id)


def _retrieve_stage_docs(
    *,
    use_case: UseCase,
    stage: PMStagePlan,
    pass_index: int,
    prior_missing: List[str],
    clarification_turns: List[PMClarificationTurn],
    request_id: Optional[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    query = _build_stage_query(
        use_case=use_case,
        stage=stage,
        pass_index=pass_index,
        prior_missing=prior_missing,
        clarification_turns=clarification_turns,
    )
    chunks = retrieve_docs(query, request_id=request_id)
    return query, chunks


def _build_stage_candidates(
    *,
    required_nodes: List[NodeRequirement],
    node_summaries: Dict[str, NodeFunctionalSummary],
) -> List[PMNodeCandidate]:
    candidates: List[PMNodeCandidate] = []
    for req in required_nodes:
        summary = node_summaries.get(req.node_type)
        capability_summary = summary.purpose if summary else req.why_required
        limitations = list(summary.limitations if summary else [])
        candidates.append(
            PMNodeCandidate(
                node_type=req.node_type,
                display_name=req.display_name,
                capability_summary=_compact(capability_summary, max_chars=280),
                limitations=limitations[:4],
                usage_mode=req.usage_mode,
                evidence_chunk_ids=list(req.evidence_chunk_ids),
                evidence_refs=list(req.evidence_refs),
                rerank_confidence=req.rerank_confidence,
                pm_fit_score=float(req.blended_confidence or req.evidence_confidence or 0.0),
                top_margin=None,
            )
        )
    return candidates


def _summarize_stage_candidates_with_structured_output(
    *,
    stage: PMStagePlan,
    candidates: List[PMNodeCandidate],
    use_case: UseCase,
    model: Optional[str],
    request_id: Optional[str],
) -> List[PMNodeCandidate]:
    if not candidates:
        return []
    system_prompt = (
        "You are a Product Manager assistant. "
        "Summarize node candidates based strictly on evidence."
    )
    candidate_lines = [
        (
            f"- {item.node_type} | capability={item.capability_summary} "
            f"| limitations={'; '.join(item.limitations) if item.limitations else '-'} "
            f"| usage_mode={item.usage_mode}"
        )
        for item in candidates
    ]
    user_prompt = (
        "Summarize each node candidate for this stage with concise capability summary and limitations.\n"
        f"Use case: {use_case.title}\n"
        f"Stage: {stage.id} - {stage.objective}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
    )
    output = _invoke_structured_output(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=_CandidateSummaryOutput,
        model=model,
        request_id=request_id,
        temperature=0.1,
        stage="multi_agent.product_manager.stage_candidate_summary",
    )

    by_type = {item.node_type: item for item in candidates}
    for summary in output.summaries:
        current = by_type.get(summary.node_type)
        if current is None:
            continue
        by_type[summary.node_type] = current.model_copy(
            update={
                "capability_summary": _compact(summary.capability_summary, max_chars=280),
                "limitations": _safe_list(summary.limitations)[:4],
            }
        )
    return list(by_type.values())


def _heuristic_stage_selection(
    candidates: List[PMNodeCandidate],
) -> _StageSelectionOutput:
    if not candidates:
        return _StageSelectionOutput(
            selected_node_types=[],
            rationale="No explicit API-doc node evidence available for this stage.",
            pm_fit_score=0.0,
            missing_information=["No explicit node evidence retrieved from API docs for this stage."],
        )

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item.pm_fit_score or 0.0),
            -(item.rerank_confidence or 0.0),
            item.node_type,
        ),
    )
    top = sorted_candidates[:2]
    best = top[0]
    return _StageSelectionOutput(
        selected_node_types=[item.node_type for item in top],
        rationale=(
            f"Selected '{best.node_type}' (and optional companion nodes) because it best matches stage objective."
        ),
        pm_fit_score=min(1.0, max(0.0, float(best.pm_fit_score or 0.0))),
        missing_information=[],
    )


def _select_stage_nodes_with_structured_output(
    *,
    stage: PMStagePlan,
    candidates: List[PMNodeCandidate],
    use_case: UseCase,
    model: Optional[str],
    request_id: Optional[str],
) -> _StageSelectionOutput:
    if not candidates:
        return _heuristic_stage_selection(candidates)

    system_prompt = (
        "You are a Product Manager selecting nodes for one workflow stage. "
        "Select only from provided candidates."
    )
    lines = [
        (
            f"- {item.node_type} | fit_hint={item.pm_fit_score:.2f} | rerank={item.rerank_confidence} "
            f"| capability={item.capability_summary} | limitations={'; '.join(item.limitations) if item.limitations else '-'}"
        )
        for item in candidates
    ]
    user_prompt = (
        "Select the best node bundle for this stage.\n"
        f"Use case: {use_case.title}\n"
        f"Stage id: {stage.id}\n"
        f"Stage objective: {stage.objective}\n"
        f"Expected inputs: {', '.join(stage.expected_inputs) or '-'}\n"
        f"Expected outputs: {', '.join(stage.expected_outputs) or '-'}\n"
        "Rules:\n"
        "- Select only node types listed in candidates.\n"
        "- Prefer business-fit and stage objective alignment.\n"
        "- If evidence is insufficient, return empty selected_node_types and explain missing information.\n"
        "Candidates:\n"
        + "\n".join(lines)
    )

    output = _invoke_structured_output(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=_StageSelectionOutput,
        model=model,
        request_id=request_id,
        temperature=0.1,
        stage="multi_agent.product_manager.stage_selection",
    )

    allowed = {item.node_type for item in candidates}
    selected = [node_type for node_type in output.selected_node_types if node_type in allowed]
    if not selected:
        return _heuristic_stage_selection(candidates).model_copy(
            update={
                "missing_information": _safe_list(output.missing_information) or [
                    "Node evidence was not strong enough for confident stage selection."
                ],
                "rationale": _compact(output.rationale or "Evidence too weak for confident selection.", max_chars=280),
                "pm_fit_score": float(output.pm_fit_score or 0.0),
            }
        )

    return _StageSelectionOutput(
        selected_node_types=selected,
        rationale=_compact(output.rationale, max_chars=320),
        pm_fit_score=float(output.pm_fit_score),
        missing_information=_safe_list(output.missing_information),
    )


def _top_margin(candidates: List[PMNodeCandidate]) -> Optional[float]:
    scores = sorted(
        [float(item.rerank_confidence) for item in candidates if item.rerank_confidence is not None],
        reverse=True,
    )
    if len(scores) < 2:
        return None
    return max(0.0, min(1.0, scores[0] - scores[1]))


def _stage_rerank_from_selected(selected_nodes: List[PMNodeCandidate]) -> Optional[float]:
    scores = [float(item.rerank_confidence) for item in selected_nodes if item.rerank_confidence is not None]
    if not scores:
        return None
    return max(0.0, min(1.0, max(scores)))


def _stage_gate_pass(
    *,
    pm_fit_score: float,
    rerank_confidence: Optional[float],
    top_margin: Optional[float],
) -> bool:
    fit_threshold = _pm_float("PM_FIT_SCORE_THRESHOLD", 0.70)
    rerank_threshold = _pm_float("PM_RERANK_THRESHOLD", 0.55)
    margin_threshold = _pm_float("PM_TOP_MARGIN_THRESHOLD", 0.10)

    if pm_fit_score < fit_threshold:
        return False
    rerank_ok = rerank_confidence is not None and rerank_confidence >= rerank_threshold
    margin_ok = top_margin is not None and top_margin >= margin_threshold
    return rerank_ok or margin_ok


def _stage_progress_summary(
    *,
    stage: PMStagePlan,
    pass_index: int,
    query: str,
    candidates: List[PMNodeCandidate],
    top_margin: Optional[float],
) -> PMStageSearchState:
    top_rerank = None
    if candidates:
        top_rerank = max(
            [item.rerank_confidence for item in candidates if item.rerank_confidence is not None],
            default=None,
        )
    return PMStageSearchState(
        stage_id=stage.id,
        pass_index=pass_index,
        query=_compact(query, max_chars=500),
        retrieved_chunk_count=len(candidates),
        candidate_node_types=[item.node_type for item in candidates],
        top_rerank_confidence=top_rerank,
        top_margin=top_margin,
    )


def _run_stage_selection_passes(
    *,
    use_case: UseCase,
    stage: PMStagePlan,
    model: Optional[str],
    request_id: Optional[str],
    clarification_state: PMClarificationState,
    trace_events: List[Dict[str, Any]],
) -> PMStageSelection:
    max_passes = _pm_int("PM_MAX_STAGE_RETRIEVAL_PASSES", 3)
    prior_missing: List[str] = []
    search_history: List[PMStageSearchState] = []
    best_selection: Optional[PMStageSelection] = None

    for pass_index in range(1, max_passes + 1):
        query, chunks = _retrieve_stage_docs(
            use_case=use_case,
            stage=stage,
            pass_index=pass_index,
            prior_missing=prior_missing,
            clarification_turns=clarification_state.turns,
            request_id=request_id,
        )

        required_nodes, node_summaries = _extract_required_nodes_with_summaries(chunks)
        required_nodes, dropped = _filter_required_nodes_for_usage(
            use_case=use_case,
            required_nodes=required_nodes,
        )
        if dropped:
            prior_missing = _safe_list(
                prior_missing
                + [
                    "Retrieved evidence is mostly tool-only nodes. Clarify if this stage is agent tool-calling or main-action workflow."
                ]
            )

        candidates = _build_stage_candidates(required_nodes=required_nodes, node_summaries=node_summaries)
        if candidates and isinstance(model, str) and model.strip():
            try:
                candidates = _summarize_stage_candidates_with_structured_output(
                    stage=stage,
                    candidates=candidates,
                    use_case=use_case,
                    model=model,
                    request_id=request_id,
                )
            except Exception as exc:
                logger.warning("pm stage candidate summary failed (stage=%s pass=%d): %s", stage.id, pass_index, str(exc))

        margin = _top_margin(candidates)
        search_state = _stage_progress_summary(
            stage=stage,
            pass_index=pass_index,
            query=query,
            candidates=candidates,
            top_margin=margin,
        )
        search_history.append(search_state)

        selection_output: _StageSelectionOutput
        if candidates and isinstance(model, str) and model.strip():
            try:
                selection_output = _select_stage_nodes_with_structured_output(
                    stage=stage,
                    candidates=candidates,
                    use_case=use_case,
                    model=model,
                    request_id=request_id,
                )
            except Exception as exc:
                logger.warning("pm stage selection failed (stage=%s pass=%d): %s", stage.id, pass_index, str(exc))
                selection_output = _heuristic_stage_selection(candidates)
        else:
            selection_output = _heuristic_stage_selection(candidates)

        selected_nodes = [
            item for item in candidates if item.node_type in set(selection_output.selected_node_types)
        ]
        for item in selected_nodes:
            item.top_margin = margin

        rerank_confidence = _stage_rerank_from_selected(selected_nodes)
        gate_passed = _stage_gate_pass(
            pm_fit_score=float(selection_output.pm_fit_score),
            rerank_confidence=rerank_confidence,
            top_margin=margin,
        )

        stage_selection = PMStageSelection(
            stage_id=stage.id,
            selected_node_types=[item.node_type for item in selected_nodes],
            selected_nodes=selected_nodes,
            rationale=_compact(selection_output.rationale, max_chars=320),
            pm_fit_score=float(selection_output.pm_fit_score),
            rerank_confidence=rerank_confidence,
            top_margin=margin,
            gate_passed=gate_passed,
            passes_used=pass_index,
            missing_information=_safe_list(selection_output.missing_information),
            search_history=list(search_history),
        )

        trace_events.append(
            {
                "stage_id": stage.id,
                "pass_index": pass_index,
                "query": query,
                "retrieved_chunks": len(chunks),
                "candidate_count": len(candidates),
                "candidate_node_types": [item.node_type for item in candidates],
                "selected_node_types": list(stage_selection.selected_node_types),
                "pm_fit_score": stage_selection.pm_fit_score,
                "rerank_confidence": stage_selection.rerank_confidence,
                "top_margin": stage_selection.top_margin,
                "gate_passed": gate_passed,
            }
        )

        if gate_passed and stage_selection.selected_node_types:
            return stage_selection

        best_selection = stage_selection
        if selection_output.missing_information:
            prior_missing = _safe_list(prior_missing + selection_output.missing_information)
        else:
            prior_missing = _safe_list(prior_missing + ["Need stronger explicit node evidence for this stage objective."])

    if best_selection is None:
        best_selection = PMStageSelection(
            stage_id=stage.id,
            selected_node_types=[],
            selected_nodes=[],
            rationale="No explicit API-doc evidence was found for this stage.",
            pm_fit_score=0.0,
            rerank_confidence=None,
            top_margin=None,
            gate_passed=False,
            passes_used=max_passes,
            missing_information=["No explicit node evidence retrieved for this stage."],
            search_history=list(search_history),
        )
    return best_selection


def _build_clarification_question(stage: PMStagePlan, selection: PMStageSelection) -> str:
    gaps = "; ".join(selection.missing_information[:2]) if selection.missing_information else "insufficient node evidence"
    return _compact(
        (
            f"To continue planning stage '{stage.name}', clarify this: {gaps}. "
            "Please specify integrations, trigger type, and expected outputs for this stage."
        ),
        max_chars=260,
    )


def _record_clarification_answer(
    clarification_state: PMClarificationState,
    user_query: str,
) -> PMClarificationState:
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
        turns.append(PMClarificationTurn(question=question, answer=answer))
    return clarification_state.model_copy(
        update={
            "pending_questions": pending,
            "turns": turns,
        }
    )


def _derive_required_nodes_from_stage_selections(
    selections: List[PMStageSelection],
) -> List[NodeRequirement]:
    output: List[NodeRequirement] = []
    seen = set()
    for selection in selections:
        for candidate in selection.selected_nodes:
            if candidate.node_type in seen:
                continue
            seen.add(candidate.node_type)
            evidence_conf = min(1.0, max(0.0, float(candidate.pm_fit_score or 0.0)))
            blended = _blended_confidence(evidence_conf, candidate.rerank_confidence)
            output.append(
                NodeRequirement(
                    node_type=candidate.node_type,
                    display_name=candidate.display_name,
                    why_required=_compact(
                        f"Stage {selection.stage_id}: {selection.rationale or candidate.capability_summary}",
                        max_chars=260,
                    ),
                    evidence_chunk_ids=list(candidate.evidence_chunk_ids[:8]),
                    evidence_refs=list(candidate.evidence_refs[:6]),
                    evidence_confidence=round(evidence_conf, 2),
                    rerank_confidence=(
                        round(candidate.rerank_confidence, 2)
                        if isinstance(candidate.rerank_confidence, float)
                        else None
                    ),
                    blended_confidence=round(blended, 2),
                    usage_mode=candidate.usage_mode,
                    usable_as_tool=None,
                    has_main_input=None,
                    input_connection_types=[],
                )
            )
    return output


def _derive_proposed_nodes(
    *,
    stage_plan: List[PMStagePlan],
    selections: List[PMStageSelection],
) -> List[ProposedNode]:
    output: List[ProposedNode] = []
    by_stage = {item.stage_id: item for item in selections}
    idx = 0
    for stage in stage_plan:
        selection = by_stage.get(stage.id)
        if selection is None:
            continue
        stage_selected = selection.selected_nodes or []
        if not stage_selected and selection.selected_node_types:
            stage_selected = [
                PMNodeCandidate(
                    node_type=node_type,
                    capability_summary=selection.rationale,
                )
                for node_type in selection.selected_node_types
            ]
        for candidate in stage_selected:
            idx += 1
            output.append(
                ProposedNode(
                    node_id=f"pn_{idx}",
                    node_type=candidate.node_type,
                    stage_id=stage.id,
                    purpose=_compact(
                        f"{stage.objective} | {selection.rationale or candidate.capability_summary}",
                        max_chars=260,
                    ),
                    depends_on=[f"pn_{idx - 1}"] if idx > 1 else [],
                    expected_inputs=list(stage.expected_inputs),
                    expected_outputs=list(stage.expected_outputs),
                    usage_mode=candidate.usage_mode,
                )
            )
    return output


def _derive_architecture_plan(
    *,
    use_case: UseCase,
    stage_plan: List[PMStagePlan],
    selections: List[PMStageSelection],
    missing_information: List[str],
) -> ArchitecturePlan:
    stage_models = [
        ArchitectureStage(
            id=stage.id,
            name=stage.name,
            purpose=stage.objective,
            required_capabilities=list(stage.success_criteria),
            expected_inputs=list(stage.expected_inputs),
            expected_outputs=list(stage.expected_outputs),
            dependencies=list(stage.dependencies),
            notes=None,
        )
        for stage in stage_plan
    ]

    data_flow: List[ArchitectureDataFlowItem] = []
    for stage in stage_plan:
        for dependency in stage.dependencies:
            data_flow.append(
                ArchitectureDataFlowItem(
                    source_stage_id=dependency,
                    target_stage_id=stage.id,
                    data_items=list(stage.expected_inputs) or ["stage_input"],
                    notes="Derived from PM stage dependency.",
                )
            )

    required_nodes = _derive_required_nodes_from_stage_selections(selections)
    notes = [
        _compact(
            f"Stage {item.stage_id}: {item.rationale}",
            max_chars=280,
        )
        for item in selections
        if item.rationale
    ]
    return ArchitecturePlan(
        use_case_id=use_case.id,
        title=use_case.title,
        business_objective=_compact(use_case.business_problem, max_chars=260),
        desired_outcome=_compact(use_case.desired_outcome, max_chars=260),
        workflow_summary=_compact(
            f"Stage-first PM plan with {len(stage_plan)} stages and {len(required_nodes)} evidence-backed nodes.",
            max_chars=320,
        ),
        stages=stage_models,
        data_flow=data_flow,
        assumptions=[
            "Node selection is constrained to explicit API-doc evidence.",
            "Engineer stage will configure parameters and credentials.",
        ],
        missing_information=_safe_list(missing_information),
        implementation_notes_for_engineer=_safe_list(notes),
        required_nodes=required_nodes,
    )


def build_architecture_plan(
    *,
    selected_use_case: UseCase,
    required_nodes: List[NodeRequirement],
    node_summaries: Dict[str, NodeFunctionalSummary],
    model: Optional[str],
    request_id: Optional[str],
) -> Tuple[Optional[ArchitecturePlan], Optional[str]]:
    _ = node_summaries, model, request_id
    if not required_nodes:
        return None, None

    stages = [
        PMStagePlan(
            id="stage_1",
            name="Workflow Stage",
            objective="Implement core use-case flow.",
            expected_inputs=["Input payload"],
            expected_outputs=["Processed result"],
            success_criteria=["Business objective covered"],
            dependencies=[],
        )
    ]
    selection = PMStageSelection(
        stage_id="stage_1",
        selected_node_types=[node.node_type for node in required_nodes],
        selected_nodes=[
            PMNodeCandidate(
                node_type=node.node_type,
                display_name=node.display_name,
                capability_summary=node.why_required,
                limitations=[],
                usage_mode=node.usage_mode,
                evidence_chunk_ids=list(node.evidence_chunk_ids),
                evidence_refs=list(node.evidence_refs),
                rerank_confidence=node.rerank_confidence,
                pm_fit_score=float(node.blended_confidence or node.evidence_confidence or 0.0),
            )
            for node in required_nodes
        ],
        rationale="Derived from PM stage bridge.",
        pm_fit_score=0.8,
        rerank_confidence=None,
        top_margin=None,
        gate_passed=True,
        passes_used=1,
        missing_information=[],
        search_history=[],
    )
    plan = _derive_architecture_plan(
        use_case=selected_use_case,
        stage_plan=stages,
        selections=[selection],
        missing_information=[],
    )
    return plan, "Architecture plan bridge generated from selected nodes."


def _planning_summary(
    *,
    use_case: UseCase,
    stage_plan: List[PMStagePlan],
    selections: List[PMStageSelection],
    pm_status: PMStatus,
) -> str:
    selected_nodes = sum(len(item.selected_node_types) for item in selections)
    return _compact(
        (
            f"PM status={pm_status.value}; use_case={use_case.id}; "
            f"stages={len(stage_plan)}; selected_nodes={selected_nodes}."
        ),
        max_chars=260,
    )


def product_manager_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    model, request_id = _runtime_context(state)
    routing_signals = list(state.get("routing_signals") or [])
    if "entered_product_manager_agent" not in routing_signals:
        routing_signals.append("entered_product_manager_agent")

    missing_user_inputs = list(state.get("missing_user_inputs") or [])
    entry_intent = state.get("entry_intent")

    selected_use_case = _normalize_use_case(state.get("selected_use_case"))
    if selected_use_case is None and entry_intent == EntryIntent.workflow_build_request:
        selected_use_case = _derive_use_case_from_direct_build_request(state.get("user_query") or "")
        if selected_use_case is not None:
            routing_signals.append("pm_use_case_derived_from_direct_build_request")

    if selected_use_case is None:
        message = "A selected use case is required before PM planning can continue."
        if entry_intent == EntryIntent.workflow_build_request:
            message = "Please provide a clearer workflow build request so PM planning can continue."
        missing_user_inputs = _safe_list(missing_user_inputs + [message])
        routing_signals.append("pm_missing_selected_use_case")
        workflow_context = WorkflowContext(
            use_case_id="unknown",
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            notes=["product_manager_requires_selected_use_case"],
        )
        return {
            "current_stage": "product_manager_agent",
            "pm_status": PMStatus.pm_failed_no_solution,
            "pm_stage_plan": [],
            "pm_stage_selections": [],
            "pm_stage_progress": PMProgressState(total_stages=0),
            "pm_clarification_state": PMClarificationState(
                attempts_used=0,
                max_attempts=_pm_int("PM_MAX_USER_CLARIFICATIONS", 2),
                pending_questions=[],
                turns=[],
            ),
            "pm_stage_search_history": [],
            "pm_reasoning_trace_full": [],
            "architecture_plan": None,
            "workflow_context": workflow_context,
            "planning_summary": None,
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    pm_status = _normalize_pm_status(state.get("pm_status"))
    stage_plan = _normalize_model_list(state.get("pm_stage_plan"), PMStagePlan)
    stage_selections = _normalize_model_list(state.get("pm_stage_selections"), PMStageSelection)
    stage_progress = _normalize_model(state.get("pm_stage_progress"), PMProgressState)
    clarification_state = _normalize_model(state.get("pm_clarification_state"), PMClarificationState)
    stage_search_history = _normalize_model_list(state.get("pm_stage_search_history"), PMStageSearchState)
    reasoning_trace = list(state.get("pm_reasoning_trace_full") or [])

    if stage_progress is None:
        stage_progress = PMProgressState(total_stages=0)
    if clarification_state is None:
        clarification_state = PMClarificationState(
            attempts_used=0,
            max_attempts=_pm_int("PM_MAX_USER_CLARIFICATIONS", 2),
            pending_questions=[],
            turns=[],
        )

    if pm_status == PMStatus.pm_blocked_waiting_user and clarification_state.pending_questions:
        user_query = _compact(str(state.get("user_query") or ""), max_chars=320)
        if user_query:
            clarification_state = _record_clarification_answer(clarification_state, user_query)
            routing_signals.append("pm_clarification_answer_received")

    if not stage_plan:
        try:
            stage_plan = _plan_stages_with_structured_output(
                use_case=selected_use_case,
                model=model,
                request_id=request_id,
            )
        except Exception as exc:
            logger.warning("pm stage planning fallback to heuristic: %s", str(exc))
            stage_plan = _heuristic_stage_plan(selected_use_case)
        stage_progress = stage_progress.model_copy(update={"total_stages": len(stage_plan)})

    stage_selection_by_id: Dict[str, PMStageSelection] = {item.stage_id: item for item in stage_selections}
    completed_stage_ids = set(stage_progress.completed_stage_ids)
    blocked_stage_ids = set(stage_progress.blocked_stage_ids)
    trace_events: List[Dict[str, Any]] = []

    for stage in stage_plan:
        existing_selection = stage_selection_by_id.get(stage.id)
        if existing_selection and existing_selection.gate_passed:
            completed_stage_ids.add(stage.id)
            blocked_stage_ids.discard(stage.id)
            continue

        stage_progress = stage_progress.model_copy(update={"current_stage_id": stage.id})
        selection = _run_stage_selection_passes(
            use_case=selected_use_case,
            stage=stage,
            model=model,
            request_id=request_id,
            clarification_state=clarification_state,
            trace_events=trace_events,
        )

        stage_selection_by_id[stage.id] = selection
        stage_search_history.extend(selection.search_history)

        passes_by_stage = dict(stage_progress.passes_by_stage)
        passes_by_stage[stage.id] = max(passes_by_stage.get(stage.id, 0), selection.passes_used)
        stage_progress = stage_progress.model_copy(update={"passes_by_stage": passes_by_stage})

        if selection.gate_passed and selection.selected_node_types:
            completed_stage_ids.add(stage.id)
            blocked_stage_ids.discard(stage.id)
            continue

        question = _build_clarification_question(stage, selection)
        if clarification_state.attempts_used < clarification_state.max_attempts:
            clarification_state = clarification_state.model_copy(
                update={
                    "attempts_used": clarification_state.attempts_used + 1,
                    "pending_questions": clarification_state.pending_questions + [question],
                    "turns": clarification_state.turns + [
                        PMClarificationTurn(stage_id=stage.id, question=question, answer=None)
                    ],
                }
            )
            missing_user_inputs = _safe_list(missing_user_inputs + [question])
            blocked_stage_ids.add(stage.id)
            routing_signals.append("pm_blocked_waiting_user")
            pm_status = PMStatus.pm_blocked_waiting_user
            break

        blocked_stage_ids.add(stage.id)
        missing_user_inputs = _safe_list(
            missing_user_inputs
            + [
                f"PM could not find a confident stage solution after {clarification_state.max_attempts} clarifications for '{stage.name}'."
            ]
        )
        routing_signals.append("pm_failed_no_solution")
        pm_status = PMStatus.pm_failed_no_solution
        break

    if pm_status in (PMStatus.pm_blocked_waiting_user, PMStatus.pm_failed_no_solution):
        if len(completed_stage_ids) == len(stage_plan):
            pm_status = PMStatus.pm_completed
            blocked_stage_ids.clear()
    if pm_status is None:
        pm_status = PMStatus.pm_completed

    stage_progress = stage_progress.model_copy(
        update={
            "completed_stage_ids": sorted(completed_stage_ids),
            "blocked_stage_ids": sorted(blocked_stage_ids),
            "total_stages": len(stage_plan),
        }
    )

    stage_selections = [stage_selection_by_id[stage.id] for stage in stage_plan if stage.id in stage_selection_by_id]
    reasoning_trace.extend(trace_events)

    emit_trace_event(
        trace_logger,
        event="pm_stage_run",
        request_id=request_id,
        stage="multi_agent.product_manager",
        payload={
            "use_case_id": selected_use_case.id,
            "pm_status": pm_status.value,
            "stages_total": len(stage_plan),
            "stages_completed": len(stage_progress.completed_stage_ids),
            "stages_blocked": len(stage_progress.blocked_stage_ids),
            "clarifications_used": clarification_state.attempts_used,
            "llm_calls_trace_items": len(trace_events),
        },
    )

    if pm_status in (PMStatus.pm_blocked_waiting_user, PMStatus.pm_failed_no_solution):
        partial_plan = _derive_architecture_plan(
            use_case=selected_use_case,
            stage_plan=stage_plan,
            selections=[item for item in stage_selections if item.gate_passed],
            missing_information=missing_user_inputs,
        )
        proposed_nodes = _derive_proposed_nodes(
            stage_plan=stage_plan,
            selections=[item for item in stage_selections if item.gate_passed],
        )
        workflow_context = WorkflowContext(
            use_case_id=selected_use_case.id,
            planning_ready=False,
            handoff_target=None,
            required_node_types=[node.node_type for node in partial_plan.required_nodes],
            unresolved_inputs=missing_user_inputs,
            notes=[
                f"pm_status={pm_status.value}",
                f"completed_stages={len(stage_progress.completed_stage_ids)}",
            ],
        )
        return {
            "current_stage": "product_manager_agent",
            "pm_status": pm_status,
            "pm_stage_plan": stage_plan,
            "pm_stage_selections": stage_selections,
            "pm_stage_progress": stage_progress,
            "pm_clarification_state": clarification_state,
            "pm_stage_search_history": stage_search_history,
            "pm_reasoning_trace_full": reasoning_trace,
            "architecture_plan": partial_plan,
            "workflow_context": workflow_context,
            "planning_summary": _planning_summary(
                use_case=selected_use_case,
                stage_plan=stage_plan,
                selections=stage_selections,
                pm_status=pm_status,
            ),
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": proposed_nodes,
            "required_credentials": [],
        }

    architecture_plan = _derive_architecture_plan(
        use_case=selected_use_case,
        stage_plan=stage_plan,
        selections=stage_selections,
        missing_information=missing_user_inputs,
    )
    proposed_nodes = _derive_proposed_nodes(stage_plan=stage_plan, selections=stage_selections)

    workflow_context = WorkflowContext(
        use_case_id=selected_use_case.id,
        planning_ready=True,
        handoff_target=AgentStage.engineer_agent,
        required_node_types=[node.node_type for node in architecture_plan.required_nodes],
        unresolved_inputs=missing_user_inputs,
        notes=[
            f"pm_status={pm_status.value}",
            f"stages={len(stage_plan)}",
            f"required_nodes={len(architecture_plan.required_nodes)}",
        ],
    )
    if "handoff_ready_engineer" not in routing_signals:
        routing_signals.append("handoff_ready_engineer")

    return {
        "current_stage": "product_manager_agent",
        "pm_status": PMStatus.pm_completed,
        "pm_stage_plan": stage_plan,
        "pm_stage_selections": stage_selections,
        "pm_stage_progress": stage_progress,
        "pm_clarification_state": clarification_state,
        "pm_stage_search_history": stage_search_history,
        "pm_reasoning_trace_full": reasoning_trace,
        "architecture_plan": architecture_plan,
        "workflow_context": workflow_context,
        "planning_summary": _planning_summary(
            use_case=selected_use_case,
            stage_plan=stage_plan,
            selections=stage_selections,
            pm_status=PMStatus.pm_completed,
        ),
        "missing_user_inputs": missing_user_inputs,
        "routing_signals": routing_signals,
        "target_stage": AgentStage.engineer_agent,
        "proposed_nodes": proposed_nodes,
        "required_credentials": [],
    }
