from __future__ import annotations

import html
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    ProposedNode,
    UseCase,
    WorkflowContext,
)
from ...llm import get_langchain_chat_model, resolve_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event, emit_trace_event
from ...token_budget import estimate_messages_tokens
from ...workflow.retriever import retrieve_docs
from ..multi_agent_state import MultiAgentGraphState

logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")

_MAX_REQUIRED_NODES = 12
_NODE_TYPE_LINE_RE = re.compile(r"(?im)^Node Type:\s*(.+?)\s*$")
_USABLE_AS_TOOL_LINE_RE = re.compile(r"(?im)^Usable As Tool:\s*(true|false)\s*$")
_INPUTS_LINE_RE = re.compile(r"(?im)^Inputs:\s*(.+?)\s*$")
_BLEND_EVIDENCE_ALPHA = 0.75
_BLEND_RERANK_BETA = 0.25
_MAX_PM_SUMMARY_SNIPPETS = 3
_PM_MAX_SNIPPET_CHARS = 280
_PM_MAX_PROMPT_TOKENS_FALLBACK = 2500
_PM_PLAN_TEMPERATURE = 0.1


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
    sources: List[str] = Field(default_factory=list)


class NodeEvidenceCompactionItem(BaseModel):
    node_type: str
    purpose: str = ""
    capabilities: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    usage_constraints: List[str] = Field(default_factory=list)
    data_io_hints: List[str] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)


class NodeEvidenceCompactionOutput(BaseModel):
    compacted_nodes: List[NodeEvidenceCompactionItem] = Field(default_factory=list)


class NodeEvidenceSummaryItem(BaseModel):
    node_type: str
    purpose: str
    capabilities: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    usage_constraints: List[str] = Field(default_factory=list)
    data_io_hints: List[str] = Field(default_factory=list)


class NodeEvidenceSummaryOutput(BaseModel):
    summaries: List[NodeEvidenceSummaryItem] = Field(default_factory=list)


class ArchitecturePlanDraft(BaseModel):
    workflow_summary: str
    stages: List[ArchitectureStage] = Field(default_factory=list)
    data_flow: List[ArchitectureDataFlowItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    implementation_notes_for_engineer: List[str] = Field(default_factory=list)
    planning_summary: Optional[str] = None


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
    source_counts: Dict[str, int] = field(default_factory=dict)
    docs_page_keys: List[str] = field(default_factory=list)

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
        source: str,
        docs_page_key: Optional[str],
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
        if source:
            self.source_counts[source] = self.source_counts.get(source, 0) + 1
        if docs_page_key and docs_page_key not in self.docs_page_keys:
            self.docs_page_keys.append(docs_page_key)
        if display_name and not self.display_name:
            self.display_name = display_name


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
    # If a JSON fragment leaks into the reference, keep only the left side.
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


def _safe_list(values: List[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = _sanitize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _use_case_query(use_case: UseCase) -> str:
    return (
        f"Use case title: {use_case.title}\n"
        f"Business problem: {use_case.business_problem}\n"
        f"Desired outcome: {use_case.desired_outcome}\n"
        f"Expected business value: {use_case.expected_value}\n"
        "Goal: retrieve n8n API/docs evidence for required workflow nodes."
    )


def retrieve_pm_api_docs(
    use_case: UseCase,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = _use_case_query(use_case)
    return retrieve_docs(query, request_id=request_id)


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
    lowered = lowered.replace(" ", "")
    lowered = lowered.replace("-", "_")
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


def _chunk_source_label(chunk: Dict[str, Any]) -> str:
    context_kind = str(chunk.get("context_kind") or "").strip().lower()
    if context_kind == "linked_def":
        return "linked_def"
    rerank_score = _normalize_rerank_score(chunk.get("rerank_score"))
    if rerank_score is not None:
        return "docs_reranked"
    return "docs"


def _chunk_doc_page_key(chunk: Dict[str, Any]) -> str:
    context_kind = str(chunk.get("context_kind") or "").strip().lower()
    if context_kind == "linked_def":
        explicit = str(chunk.get("link_doc_page_key") or "").strip()
        if explicit:
            return explicit
    metadata = _chunk_metadata(chunk)
    page_key, _, _ = derive_doc_page_key(metadata, row_url=str(chunk.get("url") or ""))
    return str(page_key or "").strip()


def _docs_rerank_scores_by_page(chunks: List[Dict[str, Any]]) -> Dict[str, List[float]]:
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
    source = _chunk_source_label(chunk)
    if source != "linked_def":
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
        constraints.append(f"Observed input connection types: {', '.join(accumulator.input_connection_types[:4])}.")
    return _safe_list(constraints)


def _build_node_functional_summary(
    *,
    node_type: str,
    display_name: Optional[str],
    accumulator: _NodeEvidenceAccumulator,
    usage_mode: str,
) -> NodeFunctionalSummary:
    snippets: List[str] = []
    for text in accumulator.all_texts[:_MAX_PM_SUMMARY_SNIPPETS]:
        compacted = _compact(_sanitize_text(text), max_chars=_PM_MAX_SNIPPET_CHARS)
        if compacted:
            snippets.append(compacted)

    purpose = ""
    for text in accumulator.all_texts:
        units = _split_text_units(text)
        if units:
            purpose = units[0]
            break
    if not purpose:
        label = display_name or node_type
        purpose = f"Use '{label}' to implement one required capability of the selected use case."

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
            "note",
            "credential",
            "auth",
            "permission",
        ),
    )

    data_io_hints: List[str] = []
    if accumulator.input_connection_types:
        data_io_hints.append(f"Input types: {', '.join(accumulator.input_connection_types[:4])}.")
    io_units = _extract_signal_items(
        accumulator.all_texts,
        keywords=("input", "output", "payload", "data", "item"),
        max_items=3,
        max_chars=170,
    )
    data_io_hints.extend(io_units)
    data_io_hints = _safe_list(data_io_hints)

    sources = sorted(accumulator.source_counts.keys())
    return NodeFunctionalSummary(
        node_type=node_type,
        display_name=display_name,
        purpose=_compact(purpose, max_chars=220),
        capabilities=_safe_list(capabilities[:4]),
        limitations=_safe_list(limitations[:3]),
        usage_constraints=_usage_constraints_from_accumulator(accumulator, usage_mode),
        data_io_hints=data_io_hints[:4],
        evidence_refs=list(accumulator.refs[:6]),
        evidence_snippets=snippets,
        sources=sources,
    )


def _build_why_required(summary: NodeFunctionalSummary) -> str:
    parts = [summary.purpose]
    if summary.capabilities:
        parts.append(f"Capabilities: {', '.join(summary.capabilities[:2])}.")
    if summary.usage_constraints:
        parts.append(summary.usage_constraints[0])
    return _compact(" ".join(parts), max_chars=260)


def _apply_summaries_to_required_nodes(
    required_nodes: List[NodeRequirement],
    node_summaries: Dict[str, NodeFunctionalSummary],
) -> List[NodeRequirement]:
    output: List[NodeRequirement] = []
    for node in required_nodes:
        summary = node_summaries.get(node.node_type)
        if summary is None:
            output.append(node)
            continue
        output.append(
            node.model_copy(
                update={
                    "why_required": _build_why_required(summary),
                    "display_name": summary.display_name or node.display_name,
                }
            )
        )
    return output


def _normalize_rerank_score(raw_score: Any) -> Optional[float]:
    if raw_score is None:
        return None
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    if 0.0 <= score <= 1.0:
        return score
    # Reranker models can return logits; map them to [0,1].
    if score > 50:
        return 1.0
    if score < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-score))


def _blended_confidence(evidence_confidence: float, rerank_confidence: Optional[float]) -> float:
    evidence = min(1.0, max(0.0, float(evidence_confidence)))
    if rerank_confidence is None:
        return evidence
    rerank = min(1.0, max(0.0, float(rerank_confidence)))
    blended = (_BLEND_EVIDENCE_ALPHA * evidence) + (_BLEND_RERANK_BETA * rerank)
    return min(1.0, max(0.0, blended))


def _extract_required_nodes_with_summaries(
    chunks: List[Dict[str, Any]]
) -> Tuple[List[NodeRequirement], Dict[str, NodeFunctionalSummary], Dict[str, List[str]]]:
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
        source = _chunk_source_label(chunk)
        page_key = _chunk_doc_page_key(chunk)
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
            source=source,
            docs_page_key=page_key,
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
    raw_evidence_by_type: Dict[str, List[str]] = {}
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
        has_ai_input = item.ai_input_count > 0
        usage_mode = _classify_usage_mode(
            usable_as_tool=usable_as_tool,
            has_main_input=has_main_input,
            has_ai_input=has_ai_input,
        )
        node_summary = _build_node_functional_summary(
            node_type=item.node_type,
            display_name=item.display_name,
            accumulator=item,
            usage_mode=usage_mode,
        )
        node_summaries[item.node_type] = node_summary
        raw_evidence_by_type[item.node_type] = list(item.all_texts)
        requirements.append(
            NodeRequirement(
                node_type=item.node_type,
                display_name=item.display_name,
                why_required=_build_why_required(node_summary),
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
    return requirements, node_summaries, raw_evidence_by_type


def extract_required_nodes_from_docs(chunks: List[Dict[str, Any]]) -> List[NodeRequirement]:
    requirements, _, _ = _extract_required_nodes_with_summaries(chunks)
    return requirements


def _render_node_summary_line(
    node: NodeRequirement,
    summary: Optional[NodeFunctionalSummary],
) -> str:
    if summary is None:
        return (
            f"- {node.node_type} | display={node.display_name or '-'} "
            f"| evidence_confidence={node.evidence_confidence} "
            f"| rerank_confidence={node.rerank_confidence if node.rerank_confidence is not None else 'n/a'} "
            f"| blended_confidence={node.blended_confidence if node.blended_confidence is not None else 'n/a'} "
            f"| usage_mode={node.usage_mode}"
        )
    return (
        f"- {node.node_type} | display={summary.display_name or node.display_name or '-'} "
        f"| evidence_confidence={node.evidence_confidence} "
        f"| rerank_confidence={node.rerank_confidence if node.rerank_confidence is not None else 'n/a'} "
        f"| blended_confidence={node.blended_confidence if node.blended_confidence is not None else 'n/a'} "
        f"| usage_mode={node.usage_mode} "
        f"| purpose={summary.purpose} "
        f"| capabilities={'; '.join(summary.capabilities[:3]) if summary.capabilities else '-'} "
        f"| limitations={'; '.join(summary.limitations[:2]) if summary.limitations else '-'} "
        f"| usage_constraints={'; '.join(summary.usage_constraints[:2]) if summary.usage_constraints else '-'} "
        f"| data_io_hints={'; '.join(summary.data_io_hints[:2]) if summary.data_io_hints else '-'} "
        f"| evidence_refs={'; '.join(summary.evidence_refs[:2]) if summary.evidence_refs else '-'}"
    )


def _plan_prompts(
    use_case: UseCase,
    required_nodes: List[NodeRequirement],
    node_summaries: Optional[Dict[str, NodeFunctionalSummary]] = None,
) -> Tuple[str, str]:
    allowed_node_types = [node.node_type for node in required_nodes]
    node_lines = [
        _render_node_summary_line(node, (node_summaries or {}).get(node.node_type))
        for node in required_nodes
    ]
    system_prompt = (
        "You are a product manager for automation architecture planning. "
        "Produce implementation-oriented planning output, not workflow JSON."
    )
    user_prompt = (
        "Create a planning-level architecture for a future engineer implementation.\n"
        "Rules:\n"
        "- Do not generate workflow JSON.\n"
        "- Do not set node parameters or credential values.\n"
        "- Keep the plan one level above implementation details.\n"
        "- Stages must be concrete and implementation-ready.\n"
        "- Record assumptions and missing information instead of guessing.\n"
        "- Use the provided functional summaries and evidence snippets to justify node inclusion.\n"
        "- Keep architecture planning-level; engineer will configure params/credentials later.\n"
        "- Treat nodes with usage_mode=tool_only as AI-tool sub-nodes; do not use them as standard action steps.\n"
        "- You may reference only these node types as required nodes, and order them logically. Do not need to use all of them. You must deeply think which nodes fit the best for the selected use case: "
        f"{', '.join(allowed_node_types)}.\n\n"
        "Selected use case:\n"
        f"- id: {use_case.id}\n"
        f"- title: {use_case.title}\n"
        f"- business_problem: {use_case.business_problem}\n"
        f"- desired_outcome: {use_case.desired_outcome}\n"
        f"- expected_value: {use_case.expected_value}\n\n"
        "Retrieved node evidence:\n"
        f"{chr(10).join(node_lines)}\n\n"
        "Return structured output only."
    )
    return system_prompt, user_prompt


def _estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return estimate_messages_tokens(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )


def _plan_with_structured_output(
    *,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    request_id: Optional[str],
) -> ArchitecturePlanDraft:
    chat_model = get_langchain_chat_model(model=model, temperature=_PM_PLAN_TEMPERATURE)
    if chat_model is None:
        raise RuntimeError("langchain chat model unavailable")

    resolved_model = resolve_model(model)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.product_manager.structured",
        model=resolved_model,
        messages=prompt_messages,
        estimated_tokens=_estimate_prompt_tokens(system_prompt, user_prompt),
        params={"temperature": _PM_PLAN_TEMPERATURE},
    )

    structured_llm = chat_model.with_structured_output(ArchitecturePlanDraft)
    started = time.perf_counter()
    output = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    serialized = output.model_dump(exclude_none=True) if isinstance(output, ArchitecturePlanDraft) else output
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.product_manager.structured",
        model=resolved_model,
        latency_ms=latency_ms,
        content=json.dumps(serialized, ensure_ascii=False),
        usage=None,
        extra={"temperature": _PM_PLAN_TEMPERATURE},
    )
    if isinstance(output, ArchitecturePlanDraft):
        return output
    return ArchitecturePlanDraft.model_validate(output)


def _deterministic_compact_summary(summary: NodeFunctionalSummary) -> NodeFunctionalSummary:
    return NodeFunctionalSummary(
        node_type=summary.node_type,
        display_name=summary.display_name,
        purpose=_compact(summary.purpose, max_chars=150),
        capabilities=[_compact(item, max_chars=110) for item in summary.capabilities[:2]],
        limitations=[_compact(item, max_chars=110) for item in summary.limitations[:1]],
        usage_constraints=[_compact(item, max_chars=120) for item in summary.usage_constraints[:1]],
        data_io_hints=[_compact(item, max_chars=110) for item in summary.data_io_hints[:2]],
        evidence_refs=list(summary.evidence_refs[:2]),
        evidence_snippets=[_compact(item, max_chars=150) for item in summary.evidence_snippets[:1]],
        sources=list(summary.sources),
    )


def _compact_node_evidence_with_structured_output(
    *,
    use_case: UseCase,
    required_nodes: List[NodeRequirement],
    node_summaries: Dict[str, NodeFunctionalSummary],
    model: Optional[str],
    request_id: Optional[str],
) -> Dict[str, NodeFunctionalSummary]:
    chat_model = get_langchain_chat_model(model=model, temperature=0.0)
    if chat_model is None:
        raise RuntimeError("langchain chat model unavailable")

    compactable_payload = [
        {
            "node_type": item.node_type,
            "display_name": item.display_name,
            "purpose": item.purpose,
            "capabilities": list(item.capabilities),
            "limitations": list(item.limitations),
            "usage_constraints": list(item.usage_constraints),
            "data_io_hints": list(item.data_io_hints),
            "evidence_snippets": list(item.evidence_snippets),
        }
        for item in node_summaries.values()
    ]
    compactable_json = json.dumps(compactable_payload, ensure_ascii=False)

    system_prompt = (
        "You compact node evidence summaries for a product manager planning prompt. "
        "Preserve the most relevant functional meaning. Do not invent facts."
    )
    user_prompt = (
        "Compact these node summaries while preserving business-relevant capabilities and constraints.\n"
        "Rules:\n"
        "- Keep one record per node_type.\n"
        "- Keep purpose concise and factual.\n"
        "- Keep only the most relevant capabilities/limitations for architecture decisions.\n"
        "- Keep usage constraints and data-flow hints if present.\n"
        "- Do not remove key constraints that could cause wrong node selection.\n"
        "- Do not output workflow JSON.\n\n"
        f"Use case title: {use_case.title}\n"
        f"Desired outcome: {use_case.desired_outcome}\n"
        f"Allowed node types: {', '.join(node.node_type for node in required_nodes)}\n\n"
        f"Node summaries JSON:\n{compactable_json}"
    )
    resolved_model = resolve_model(model)
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.product_manager.compaction",
        model=resolved_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        estimated_tokens=_estimate_prompt_tokens(system_prompt, user_prompt),
        params={"temperature": 0.0},
    )

    structured_llm = chat_model.with_structured_output(NodeEvidenceCompactionOutput)
    started = time.perf_counter()
    output = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    serialized = (
        output.model_dump(exclude_none=True) if isinstance(output, NodeEvidenceCompactionOutput) else output
    )
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.product_manager.compaction",
        model=resolved_model,
        latency_ms=latency_ms,
        content=json.dumps(serialized, ensure_ascii=False),
        usage=None,
        extra={"temperature": 0.0},
    )

    compacted = (
        output
        if isinstance(output, NodeEvidenceCompactionOutput)
        else NodeEvidenceCompactionOutput.model_validate(output)
    )
    by_type = {item.node_type: item for item in compacted.compacted_nodes}
    result: Dict[str, NodeFunctionalSummary] = {}
    for node_type, summary in node_summaries.items():
        compact_item = by_type.get(node_type)
        if compact_item is None:
            result[node_type] = _deterministic_compact_summary(summary)
            continue
        result[node_type] = NodeFunctionalSummary(
            node_type=node_type,
            display_name=summary.display_name,
            purpose=_compact(compact_item.purpose or summary.purpose, max_chars=170),
            capabilities=_safe_list([_compact(item, max_chars=120) for item in compact_item.capabilities[:3]]),
            limitations=_safe_list([_compact(item, max_chars=120) for item in compact_item.limitations[:2]]),
            usage_constraints=_safe_list(
                [_compact(item, max_chars=140) for item in compact_item.usage_constraints[:2]]
            )
            or list(summary.usage_constraints[:1]),
            data_io_hints=_safe_list([_compact(item, max_chars=120) for item in compact_item.data_io_hints[:2]]),
            evidence_refs=list(summary.evidence_refs[:3]),
            evidence_snippets=_safe_list(
                [_compact(item, max_chars=170) for item in compact_item.evidence_snippets[:2]]
            ),
            sources=list(summary.sources),
        )
    return result


def _summarize_node_evidence_with_structured_output(
    *,
    use_case: UseCase,
    required_nodes: List[NodeRequirement],
    node_summaries: Dict[str, NodeFunctionalSummary],
    raw_evidence_by_type: Dict[str, List[str]],
    model: Optional[str],
    request_id: Optional[str],
) -> Dict[str, NodeFunctionalSummary]:
    # Keep tests and offline runs stable; if no model is passed, fall back to deterministic summaries.
    if not isinstance(model, str) or not model.strip():
        return node_summaries

    chat_model = get_langchain_chat_model(model=model, temperature=0.0)
    if chat_model is None:
        raise RuntimeError("langchain chat model unavailable")

    payload = []
    for node in required_nodes:
        summary = node_summaries.get(node.node_type)
        if summary is None:
            continue
        raw_texts = raw_evidence_by_type.get(node.node_type) or []
        payload.append(
            {
                "node_type": node.node_type,
                "display_name": summary.display_name or node.display_name,
                "usage_mode": node.usage_mode,
                "usable_as_tool": node.usable_as_tool,
                "has_main_input": node.has_main_input,
                "input_connection_types": list(node.input_connection_types),
                "evidence_refs": list(node.evidence_refs[:4]),
                "evidence_texts": [
                    _compact(_sanitize_text(item), max_chars=650) for item in raw_texts[:3] if _sanitize_text(item)
                ]
                or list(summary.evidence_snippets[:2]),
                "fallback_summary": {
                    "purpose": summary.purpose,
                    "capabilities": list(summary.capabilities),
                    "limitations": list(summary.limitations),
                    "usage_constraints": list(summary.usage_constraints),
                    "data_io_hints": list(summary.data_io_hints),
                },
            }
        )

    if not payload:
        return node_summaries

    payload_json = json.dumps(payload, ensure_ascii=False)
    system_prompt = (
        "You are a product manager assistant that summarizes API-doc node evidence. "
        "Extract factual capabilities and limitations from provided evidence only."
    )
    user_prompt = (
        "For each node, produce a concise planning-oriented functional summary.\n"
        "Rules:\n"
        "- Use only the provided evidence_texts and metadata; do not invent facts.\n"
        "- Keep architecture-level language (no parameter values, no credential values).\n"
        "- capability items should describe what the node can do.\n"
        "- limitation items should describe constraints, requirements, or caveats.\n"
        "- usage_constraints must reflect usage_mode/tool-vs-action behavior.\n"
        "- data_io_hints should mention practical input/output semantics when evidenced.\n"
        "- Return structured output only.\n\n"
        f"Use case title: {use_case.title}\n"
        f"Desired outcome: {use_case.desired_outcome}\n\n"
        f"Node evidence JSON:\n{payload_json}"
    )

    resolved_model = resolve_model(model)
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.product_manager.node_summary",
        model=resolved_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        estimated_tokens=_estimate_prompt_tokens(system_prompt, user_prompt),
        params={"temperature": 0.0},
    )
    structured_llm = chat_model.with_structured_output(NodeEvidenceSummaryOutput)
    started = time.perf_counter()
    output = structured_llm.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    serialized = output.model_dump(exclude_none=True) if isinstance(output, NodeEvidenceSummaryOutput) else output
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.product_manager.node_summary",
        model=resolved_model,
        latency_ms=latency_ms,
        content=json.dumps(serialized, ensure_ascii=False),
        usage=None,
        extra={"temperature": 0.0},
    )

    validated = (
        output
        if isinstance(output, NodeEvidenceSummaryOutput)
        else NodeEvidenceSummaryOutput.model_validate(output)
    )
    by_type = {item.node_type: item for item in validated.summaries}
    enriched: Dict[str, NodeFunctionalSummary] = {}
    for node_type, summary in node_summaries.items():
        llm_summary = by_type.get(node_type)
        if llm_summary is None:
            enriched[node_type] = summary
            continue
        usage_constraints = _safe_list(
            [_compact(item, max_chars=150) for item in llm_summary.usage_constraints]
        ) or list(summary.usage_constraints[:2])
        enriched[node_type] = NodeFunctionalSummary(
            node_type=node_type,
            display_name=summary.display_name,
            purpose=_compact(llm_summary.purpose or summary.purpose, max_chars=220),
            capabilities=_safe_list([_compact(item, max_chars=140) for item in llm_summary.capabilities[:4]]),
            limitations=_safe_list([_compact(item, max_chars=140) for item in llm_summary.limitations[:3]]),
            usage_constraints=usage_constraints,
            data_io_hints=_safe_list([_compact(item, max_chars=140) for item in llm_summary.data_io_hints[:4]]),
            evidence_refs=list(summary.evidence_refs),
            evidence_snippets=list(summary.evidence_snippets),
            sources=list(summary.sources),
        )
    return enriched


def _trim_node_summaries_for_budget(
    *,
    use_case: UseCase,
    required_nodes: List[NodeRequirement],
    node_summaries: Dict[str, NodeFunctionalSummary],
    max_prompt_tokens: int,
) -> Tuple[Dict[str, NodeFunctionalSummary], List[str], int]:
    if not node_summaries:
        return {}, [], 0

    kept_nodes = list(required_nodes)
    summaries = dict(node_summaries)
    excluded: List[str] = []

    while kept_nodes:
        selected = {node.node_type: summaries[node.node_type] for node in kept_nodes if node.node_type in summaries}
        system_prompt, user_prompt = _plan_prompts(use_case, kept_nodes, selected)
        estimated = _estimate_prompt_tokens(system_prompt, user_prompt)
        if estimated <= max_prompt_tokens:
            return selected, excluded, estimated
        if len(kept_nodes) == 1:
            only_type = kept_nodes[0].node_type
            selected[only_type] = _deterministic_compact_summary(selected[only_type])
            system_prompt, user_prompt = _plan_prompts(use_case, kept_nodes, selected)
            estimated = _estimate_prompt_tokens(system_prompt, user_prompt)
            return selected, excluded, estimated
        removed = kept_nodes.pop()
        excluded.append(removed.node_type)

    return {}, excluded, 0


def _heuristic_draft(use_case: UseCase, required_nodes: List[NodeRequirement]) -> ArchitecturePlanDraft:
    node_types = [node.node_type for node in required_nodes]
    stages = [
        ArchitectureStage(
            id="stage_intake",
            name="Intake and Trigger",
            purpose="Capture the event or record that starts the business process.",
            required_capabilities=[
                "Receive or detect business event",
                "Normalize the initial payload shape",
            ],
            expected_inputs=["Business event context"],
            expected_outputs=["Normalized payload"],
            dependencies=[],
            notes="Engineer should pick concrete trigger and normalization setup.",
        ),
        ArchitectureStage(
            id="stage_processing",
            name="Business Processing",
            purpose="Apply business rules and transform payload into actionable records.",
            required_capabilities=[
                "Validate required business fields",
                "Apply routing/decision logic",
                "Prepare target-system payload",
            ],
            expected_inputs=["Normalized payload"],
            expected_outputs=["Action-ready payload"],
            dependencies=["stage_intake"],
            notes="Keep logic modular to allow iteration in engineering.",
        ),
        ArchitectureStage(
            id="stage_delivery",
            name="Delivery and Confirmation",
            purpose="Send data to destination systems and capture delivery outcomes.",
            required_capabilities=[
                "Write/update target system",
                "Record success/failure outcomes",
                "Notify stakeholders on failure paths",
            ],
            expected_inputs=["Action-ready payload"],
            expected_outputs=["Delivery status", "Audit trace"],
            dependencies=["stage_processing"],
            notes="Include retry and error-routing strategy in engineer implementation.",
        ),
    ]
    data_flow = [
        ArchitectureDataFlowItem(
            source_stage_id="stage_intake",
            target_stage_id="stage_processing",
            data_items=["normalized payload", "event metadata"],
            notes=None,
        ),
        ArchitectureDataFlowItem(
            source_stage_id="stage_processing",
            target_stage_id="stage_delivery",
            data_items=["action-ready payload", "decision outcome"],
            notes=None,
        ),
    ]
    assumptions = [
        "Business input has a stable event source for automation.",
        "Downstream destination supports the required write/update operation.",
    ]
    if node_types:
        assumptions.append(
            "Engineer should configure only the evidence-backed node set for the first implementation iteration."
        )
    return ArchitecturePlanDraft(
        workflow_summary=(
            f"Implement '{use_case.title}' with a three-stage flow that ingests business events, "
            "applies business processing rules, and delivers outcomes to the destination system."
        ),
        stages=stages,
        data_flow=data_flow,
        assumptions=assumptions,
        missing_information=[],
        implementation_notes_for_engineer=[
            "Configure parameters and credentials in engineering phase only.",
            "Preserve a clear mapping from stage outputs to downstream inputs.",
            "Validate failure handling and operational visibility before production rollout.",
        ],
        planning_summary=(
            f"Architecture planning prepared for use case '{use_case.id}' using "
            f"{len(required_nodes)} evidence-backed node type(s)."
        ),
    )


def _collect_missing_inputs(
    use_case: UseCase,
    required_nodes: List[NodeRequirement],
    *,
    include_missing_evidence_msg: bool = True,
) -> List[str]:
    missing: List[str] = []
    if len(use_case.business_problem.strip()) < 20:
        missing.append("Please provide more detail about the business problem impact and current process.")
    if len(use_case.desired_outcome.strip()) < 15:
        missing.append("Please clarify the desired workflow outcome and success criteria.")
    if include_missing_evidence_msg and not required_nodes:
        missing.append(
            "No explicit node evidence was retrieved from API docs. Please provide a more specific integration context."
        )
    return _safe_list(missing)


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


def _derive_proposed_nodes_from_plan(plan: ArchitecturePlan) -> List[ProposedNode]:
    stage_ids = [stage.id for stage in plan.stages]
    output: List[ProposedNode] = []
    for idx, requirement in enumerate(plan.required_nodes, start=1):
        stage_id = stage_ids[min(idx - 1, len(stage_ids) - 1)] if stage_ids else None
        output.append(
            ProposedNode(
                node_id=f"pn_{idx}",
                node_type=requirement.node_type,
                stage_id=stage_id,
                purpose=requirement.why_required,
                depends_on=[f"pn_{idx - 1}"] if idx > 1 else [],
                usage_mode=requirement.usage_mode,
                usable_as_tool=requirement.usable_as_tool,
                has_main_input=requirement.has_main_input,
                input_connection_types=list(requirement.input_connection_types),
            )
        )
    return output


def _remove_json_like_content(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"\{[\s\S]*\}", "", cleaned)
    return _compact(cleaned, max_chars=240)


def build_architecture_plan(
    *,
    selected_use_case: UseCase,
    required_nodes: List[NodeRequirement],
    node_summaries: Dict[str, NodeFunctionalSummary],
    model: Optional[str],
    request_id: Optional[str],
) -> Tuple[Optional[ArchitecturePlan], Optional[str]]:
    if not required_nodes:
        return None, None

    base_missing = _collect_missing_inputs(selected_use_case, required_nodes)
    max_prompt_tokens = max(
        1,
        int(getattr(settings, "MAX_CONTEXT_TOKENS", _PM_MAX_PROMPT_TOKENS_FALLBACK) or _PM_MAX_PROMPT_TOKENS_FALLBACK),
    )
    active_summaries = dict(node_summaries)
    llm_calls_used = 0
    compacted = False
    excluded_nodes: List[str] = []

    system_prompt, user_prompt = _plan_prompts(selected_use_case, required_nodes, active_summaries)
    initial_tokens = _estimate_prompt_tokens(system_prompt, user_prompt)
    final_tokens = initial_tokens
    compacted_tokens = initial_tokens
    if initial_tokens > max_prompt_tokens:
        compacted = True
        try:
            active_summaries = _compact_node_evidence_with_structured_output(
                use_case=selected_use_case,
                required_nodes=required_nodes,
                node_summaries=active_summaries,
                model=model,
                request_id=request_id,
            )
            llm_calls_used += 1
        except Exception as exc:
            logger.warning("product manager compaction failed, using deterministic compaction: %s", str(exc))
            active_summaries = {
                node_type: _deterministic_compact_summary(summary)
                for node_type, summary in active_summaries.items()
            }

        system_prompt, user_prompt = _plan_prompts(selected_use_case, required_nodes, active_summaries)
        compacted_tokens = _estimate_prompt_tokens(system_prompt, user_prompt)
        if compacted_tokens > max_prompt_tokens:
            trimmed_summaries, excluded_nodes, compacted_tokens = _trim_node_summaries_for_budget(
                use_case=selected_use_case,
                required_nodes=required_nodes,
                node_summaries=active_summaries,
                max_prompt_tokens=max_prompt_tokens,
            )
            if trimmed_summaries:
                active_summaries = trimmed_summaries
                required_nodes = [node for node in required_nodes if node.node_type in trimmed_summaries]
            system_prompt, user_prompt = _plan_prompts(selected_use_case, required_nodes, active_summaries)
            compacted_tokens = _estimate_prompt_tokens(system_prompt, user_prompt)
        final_tokens = compacted_tokens

    emit_trace_event(
        trace_logger,
        event="pm_planning_budget",
        request_id=request_id,
        stage="multi_agent.product_manager.planning",
        payload={
            "max_prompt_tokens": max_prompt_tokens,
            "initial_tokens": initial_tokens,
            "compacted_tokens": compacted_tokens,
            "final_tokens": final_tokens,
            "compacted": compacted,
            "excluded_nodes": excluded_nodes,
            "required_nodes_after_budget": len(required_nodes),
        },
    )

    try:
        draft = _plan_with_structured_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            request_id=request_id,
        )
        llm_calls_used += 1
    except Exception as exc:
        logger.warning("product manager structured planning failed, using fallback: %s", str(exc))
        draft = _heuristic_draft(selected_use_case, required_nodes)

    missing_information = _safe_list(list(draft.missing_information) + base_missing)
    notes = [_remove_json_like_content(item) for item in draft.implementation_notes_for_engineer]
    notes = [item for item in notes if item]
    plan = ArchitecturePlan(
        use_case_id=selected_use_case.id,
        title=selected_use_case.title,
        business_objective=_compact(selected_use_case.business_problem),
        desired_outcome=_compact(selected_use_case.desired_outcome),
        workflow_summary=_remove_json_like_content(draft.workflow_summary),
        stages=list(draft.stages),
        data_flow=list(draft.data_flow),
        assumptions=_safe_list([_compact(item) for item in draft.assumptions]),
        missing_information=missing_information,
        implementation_notes_for_engineer=notes,
        required_nodes=required_nodes,
    )
    planning_summary = draft.planning_summary or (
        f"Prepared architecture plan for use case '{selected_use_case.id}' with "
        f"{len(required_nodes)} evidence-backed nodes."
    )
    emit_trace_event(
        trace_logger,
        event="pm_planning_outcome",
        request_id=request_id,
        stage="multi_agent.product_manager.planning",
        payload={
            "llm_calls_used": llm_calls_used,
            "compacted": compacted,
            "excluded_nodes": excluded_nodes,
            "required_nodes_final": len(required_nodes),
        },
    )
    return plan, _compact(planning_summary, max_chars=260)


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
    if len(query) < 12:
        return None

    title = _compact(query, max_chars=80)
    return UseCase(
        id="direct_build_request",
        title=title,
        business_problem=f"User requested a new workflow to solve: {query}",
        desired_outcome=f"Deliver a workflow architecture for: {query}",
        expected_value="Provide a working first version of the requested automation.",
        feasibility="unknown: implementation details must be defined in engineering.",
        priority_score=70.0,
        why_selected="Direct workflow build request routed to product manager.",
    )


def product_manager_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    model, request_id = _runtime_context(state)
    routing_signals = list(state.get("routing_signals") or [])
    routing_signals.append("entered_product_manager_agent")
    missing_user_inputs = list(state.get("missing_user_inputs") or [])

    entry_intent = state.get("entry_intent")
    selected_use_case = _normalize_use_case(state.get("selected_use_case"))
    if selected_use_case is None and entry_intent == EntryIntent.workflow_build_request:
        selected_use_case = _derive_use_case_from_direct_build_request(state.get("user_query") or "")
        if selected_use_case is not None:
            routing_signals.append("pm_use_case_derived_from_direct_build_request")
            routing_signals.append(f"selected_use_case:{selected_use_case.id}")

    if selected_use_case is None:
        missing_msg = "A selected use case is required before product manager planning can continue."
        if entry_intent == EntryIntent.workflow_build_request:
            missing_msg = "Please provide a clearer workflow build request so planning can continue."
        missing_user_inputs = _safe_list(
            missing_user_inputs + [missing_msg]
        )
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
            "architecture_plan": None,
            "workflow_context": workflow_context,
            "planning_summary": None,
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    docs_chunks = retrieve_pm_api_docs(selected_use_case, request_id=request_id)
    required_nodes, node_summaries, raw_evidence_by_type = _extract_required_nodes_with_summaries(docs_chunks)
    required_nodes_before_usage_filter = len(required_nodes)
    required_nodes, dropped_tool_only_nodes = _filter_required_nodes_for_usage(
        use_case=selected_use_case,
        required_nodes=required_nodes,
    )
    kept_types = {node.node_type for node in required_nodes}
    node_summaries = {
        node_type: summary for node_type, summary in node_summaries.items() if node_type in kept_types
    }
    raw_evidence_by_type = {
        node_type: values for node_type, values in raw_evidence_by_type.items() if node_type in kept_types
    }
    if required_nodes and isinstance(model, str) and model.strip():
        try:
            node_summaries = _summarize_node_evidence_with_structured_output(
                use_case=selected_use_case,
                required_nodes=required_nodes,
                node_summaries=node_summaries,
                raw_evidence_by_type=raw_evidence_by_type,
                model=model,
                request_id=request_id,
            )
        except Exception as exc:
            logger.warning("product manager node evidence summarization failed, using fallback: %s", str(exc))
        required_nodes = _apply_summaries_to_required_nodes(required_nodes, node_summaries)
    emit_trace_event(
        trace_logger,
        event="pm_node_evidence",
        request_id=request_id,
        stage="multi_agent.product_manager.evidence",
        payload={
            "docs_chunks": len(docs_chunks),
            "required_nodes_before_usage_filter": required_nodes_before_usage_filter,
            "required_nodes_after_usage_filter": len(required_nodes),
            "node_types": [node.node_type for node in required_nodes],
            "rerank_sources": {
                node.node_type: (
                    "docs_rerank"
                    if node.rerank_confidence is not None
                    else "none"
                )
                for node in required_nodes
            },
        },
    )
    if dropped_tool_only_nodes:
        routing_signals.append(f"pm_filtered_tool_only_nodes:{len(dropped_tool_only_nodes)}")
        trace_logger.info(
            "product manager node-usage filter: request_id=%s dropped_tool_only=%d kept=%d",
            request_id or "-",
            len(dropped_tool_only_nodes),
            len(required_nodes),
        )
    if not required_nodes:
        if dropped_tool_only_nodes:
            missing_user_inputs = _safe_list(
                missing_user_inputs
                + [
                    (
                        "Retrieved node evidence is tool-only for AI agent tool-calling. "
                        "Confirm if this workflow should run as an AI agent tool-calling design "
                        "or provide action-node constraints."
                    )
                ]
            )
            routing_signals.append("pm_only_tool_nodes_after_usage_filter")
        missing_user_inputs = _safe_list(
            missing_user_inputs
            + _collect_missing_inputs(
                selected_use_case,
                required_nodes,
                include_missing_evidence_msg=not bool(dropped_tool_only_nodes),
            )
        )
        routing_signals.append("pm_no_explicit_node_evidence")
        routing_signals.append("pm_no_actionable_plan")
        workflow_context = WorkflowContext(
            use_case_id=selected_use_case.id,
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            notes=["strict_evidence_policy_blocked_planning"],
        )
        trace_logger.info(
            (
                "product manager planning: request_id=%s use_case=%s docs=%d required_nodes=0 "
                "actionable=false dropped_tool_only=%d"
            ),
            request_id or "-",
            selected_use_case.id,
            len(docs_chunks),
            len(dropped_tool_only_nodes),
        )
        return {
            "current_stage": "product_manager_agent",
            "architecture_plan": None,
            "workflow_context": workflow_context,
            "planning_summary": "No actionable plan generated: explicit node evidence was not found.",
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    architecture_plan, planning_summary = build_architecture_plan(
        selected_use_case=selected_use_case,
        required_nodes=required_nodes,
        node_summaries=node_summaries,
        model=model,
        request_id=request_id,
    )
    if architecture_plan is None:
        missing_user_inputs = _safe_list(
            missing_user_inputs + ["Unable to produce architecture plan from current evidence."]
        )
        routing_signals.append("pm_no_actionable_plan")
        workflow_context = WorkflowContext(
            use_case_id=selected_use_case.id,
            planning_ready=False,
            handoff_target=None,
            required_node_types=[node.node_type for node in required_nodes],
            unresolved_inputs=missing_user_inputs,
            notes=["architecture_plan_generation_failed"],
        )
        return {
            "current_stage": "product_manager_agent",
            "architecture_plan": None,
            "workflow_context": workflow_context,
            "planning_summary": planning_summary,
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    missing_user_inputs = _safe_list(
        missing_user_inputs + list(architecture_plan.missing_information)
    )
    planning_ready = bool(architecture_plan.stages and required_nodes)
    target_stage = AgentStage.engineer_agent if planning_ready else None
    proposed_nodes = _derive_proposed_nodes_from_plan(architecture_plan) if planning_ready else []
    required_credentials = []
    if planning_ready:
        routing_signals.append("handoff_ready_engineer")
    else:
        routing_signals.append("pm_no_actionable_plan")
    workflow_context = WorkflowContext(
        use_case_id=selected_use_case.id,
        planning_ready=planning_ready,
        handoff_target=target_stage,
        required_node_types=[node.node_type for node in required_nodes],
        unresolved_inputs=missing_user_inputs,
        notes=[
            f"retrieved_doc_chunks={len(docs_chunks)}",
            f"required_nodes={len(required_nodes)}",
        ],
    )

    trace_logger.info(
        (
            "product manager planning: request_id=%s use_case=%s docs=%d required_nodes=%d "
            "planning_ready=%s dropped_tool_only=%d"
        ),
        request_id or "-",
        selected_use_case.id,
        len(docs_chunks),
        len(required_nodes),
        planning_ready,
        len(dropped_tool_only_nodes),
    )

    return {
        "current_stage": "product_manager_agent",
        "architecture_plan": architecture_plan,
        "workflow_context": workflow_context,
        "planning_summary": planning_summary,
        "missing_user_inputs": missing_user_inputs,
        "routing_signals": routing_signals,
        "target_stage": target_stage,
        "proposed_nodes": proposed_nodes,
        "required_credentials": required_credentials,
    }
