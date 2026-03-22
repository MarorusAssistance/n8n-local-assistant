from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ...config import settings
from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    DecisionSlot,
    EntryIntent,
    PMClarificationState,
    PMClarificationTurn,
    PMProgressState,
    PMStagePlan,
    PMStatus,
    StageKind,
    UseCase,
    WorkflowContext,
)
from ...llm import get_langchain_chat_model, invoke_openai_structured_output
from ...observability import emit_llm_output_event, emit_llm_prompt_event, emit_trace_event
from ...token_budget import estimate_messages_tokens
from ..multi_agent_state import MultiAgentGraphState
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


class _AbstractPlanningOutput(BaseModel):
    workflow_summary: str = ""
    stages: List[ArchitectureStage] = Field(default_factory=list)
    data_flow: List[ArchitectureDataFlowItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    handoff_notes: List[str] = Field(default_factory=list)
    explicit_user_operations: List[str] = Field(default_factory=list)
    covered_operations: List[str] = Field(default_factory=list)
    planning_ready: bool = True


_OPERATION_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "capture": (
        "receive",
        "receiving",
        "incoming",
        "inbound",
        "capture",
        "listen",
        "watch",
        "monitor",
        "consume",
        "trigger",
        "recibir",
        "captur",
        "consum",
        "escuch",
        "vigila",
        "monitoriza",
    ),
    "fetch": (
        "fetch",
        "read",
        "retrieve",
        "load",
        "get",
        "pull",
        "download",
        "leer",
        "obtener",
        "recuper",
        "descarg",
        "consult",
    ),
    "transform": (
        "transform",
        "normalize",
        "parse",
        "extract",
        "enrich",
        "clean",
        "format",
        "modify",
        "convert",
        "transform",
        "normaliz",
        "parse",
        "extra",
        "enriqu",
        "limpi",
        "convier",
    ),
    "classify": (
        "classify",
        "classification",
        "categorize",
        "categorise",
        "label",
        "triage",
        "rank",
        "score",
        "prioritize",
        "prioritise",
        "decide",
        "clasific",
        "categoriz",
        "etiquet",
        "prioriz",
        "triag",
        "decid",
    ),
    "route": (
        "route",
        "branch",
        "split",
        "dispatch",
        "forward",
        "move",
        "enrutar",
        "ramific",
        "deriv",
        "redirig",
        "mover",
        "distrib",
    ),
    "store": (
        "store",
        "persist",
        "save",
        "write",
        "append",
        "record",
        "register",
        "guardar",
        "almacen",
        "persist",
        "escri",
        "registra",
        "anad",
        "añad",
    ),
    "notify": (
        "notify",
        "alert",
        "message",
        "post",
        "publish",
        "send",
        "reply",
        "notific",
        "alert",
        "avis",
        "mensaje",
        "public",
        "envi",
        "responder",
    ),
    "sync": (
        "sync",
        "synchroniz",
        "mirror",
        "replicate",
        "sincroniz",
        "replic",
    ),
}

_OPERATION_DISPLAY_NAMES: Dict[str, str] = {
    "capture": "capture/ingest",
    "fetch": "fetch/read",
    "transform": "transform/process",
    "classify": "classify/decide",
    "route": "route/branch",
    "store": "store/persist",
    "notify": "notify/output",
    "sync": "sync",
}

_ANALYSIS_ONLY_HINTS: Tuple[str, ...] = (
    "only",
    "just",
    "solo",
    "solamente",
    "simplemente",
    "únicamente",
    "unicamente",
    "report",
    "reporta",
    "reportar",
    "show",
    "display",
    "return",
    "output",
    "expose",
    "mostrar",
    "muestra",
    "devolver",
    "devuelve",
    "inform",
    "listar",
)

_TERMINAL_EFFECT_OPERATIONS = {"store", "notify", "sync"}
_AMBIGUOUS_TERMINAL_OPERATIONS = {"classify", "route", "transform"}
_STAGE_KIND_BY_OPERATION: Dict[str, StageKind] = {
    "capture": StageKind.trigger_intake,
    "fetch": StageKind.fetch_read,
    "transform": StageKind.transform_process,
    "classify": StageKind.classify_decision,
    "route": StageKind.route_branch,
    "store": StageKind.persist_store,
    "notify": StageKind.notify_output,
    "sync": StageKind.apply_update_source,
}
_PRIMARY_OPERATION_BY_STAGE_KIND: Dict[StageKind, str] = {
    StageKind.trigger_intake: "capture",
    StageKind.fetch_read: "fetch",
    StageKind.transform_process: "transform",
    StageKind.classify_decision: "classify",
    StageKind.route_branch: "route",
    StageKind.apply_update_source: "sync",
    StageKind.persist_store: "store",
    StageKind.notify_output: "notify",
}
_NON_OPERABLE_UI_HINTS: Tuple[str, ...] = (
    "filter",
    "filtered",
    "visible",
    "view",
    "see",
    "browse",
    "inspect",
    "mostrar",
    "ver",
    "visible",
    "filtrar",
)
_OPERABLE_APPLY_HINTS: Tuple[str, ...] = (
    "label",
    "tag",
    "folder",
    "move",
    "archive",
    "update",
    "mark",
    "etiquet",
    "mover",
    "archiv",
    "actualiz",
    "marcar",
)
_NON_QUESTION_MISSING_INFO_PREFIXES: Tuple[str, ...] = (
    "none",
    "none.",
    "ninguna",
    "ninguna.",
    "ninguno",
    "ninguno.",
    "n/a",
    "na",
)
_QUESTION_LEAD_HINTS: Tuple[str, ...] = (
    "what ",
    "which ",
    "how ",
    "where ",
    "when ",
    "should ",
    "do you ",
    "can you ",
    "please ",
    "provide ",
    "confirm ",
    "que ",
    "qué ",
    "como ",
    "cómo ",
    "donde ",
    "dónde ",
    "cuando ",
    "cuándo ",
    "cual ",
    "cuál ",
    "indica ",
    "confirma ",
)
_QUESTION_SIGNAL_HINTS: Tuple[str, ...] = (
    "what should happen",
    "which concrete",
    "what concrete",
    "how should",
    "where should",
    "provide the",
    "confirm the",
    "please restate",
    "restate the workflow goal",
    "restate the full workflow",
    "que quieres hacer",
    "que debe pasar",
    "que sistema",
    "que origen",
    "que metodo",
    "que método",
    "indica el valor",
    "confirma el valor",
)


def _pm_int(name: str, default: int) -> int:
    raw = getattr(settings, name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _sanitize_text(text: Any) -> str:
    raw = html.unescape(str(text or ""))
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def _compact(text: Any, max_chars: int = 260) -> str:
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


def _safe_list(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = _sanitize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _looks_like_user_question(text: Any) -> bool:
    normalized = _sanitize_text(text)
    lowered = normalized.lower()
    if not normalized or lowered in _NON_QUESTION_MISSING_INFO_PREFIXES:
        return False
    if lowered.startswith("none.") or lowered.startswith("none ") or lowered.startswith("ninguna "):
        return False
    if "?" in normalized:
        return True
    if any(lowered.startswith(prefix) for prefix in _QUESTION_LEAD_HINTS):
        return True
    return any(signal in lowered for signal in _QUESTION_SIGNAL_HINTS)


def _sanitize_missing_information_items(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = _sanitize_text(value)
        if not item or not _looks_like_user_question(item) or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _stage_trace_summary(stages: Sequence[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for stage in stages:
        purpose = getattr(stage, "purpose", None) or getattr(stage, "objective", "")
        output.append(
            {
                "id": getattr(stage, "id", None),
                "name": getattr(stage, "name", None),
                "stage_kind": getattr(getattr(stage, "stage_kind", None), "value", getattr(stage, "stage_kind", None)),
                "business_effect": _compact(getattr(stage, "business_effect", "") or "", max_chars=120),
                "target_entity": getattr(stage, "target_entity", None),
                "user_visible_goal": _compact(getattr(stage, "user_visible_goal", "") or "", max_chars=120),
                "unresolved_decisions": _safe_list(getattr(stage, "unresolved_decisions", []) or []),
                "purpose": _compact(purpose, max_chars=180),
                "required_capabilities": _safe_list(getattr(stage, "required_capabilities", []) or [])[:5],
                "expected_inputs": _safe_list(getattr(stage, "expected_inputs", []) or [])[:5],
                "expected_outputs": _safe_list(getattr(stage, "expected_outputs", []) or [])[:5],
                "dependencies": _safe_list(getattr(stage, "dependencies", []) or []),
                "success_criteria": _safe_list(getattr(stage, "success_criteria", []) or [])[:4],
            }
        )
    return output


def _normalize_operation_name(value: Any) -> Optional[str]:
    normalized = _sanitize_text(value).lower().strip()
    if not normalized:
        return None
    if normalized in _OPERATION_PATTERNS:
        return normalized
    for operation, patterns in _OPERATION_PATTERNS.items():
        if normalized == operation:
            return operation
        if any(_contains_pattern(normalized, pattern) for pattern in patterns):
            return operation
    return None


def _normalize_operations(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        normalized = _normalize_operation_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _detect_operations(*texts: Any) -> List[str]:
    combined = " ".join(_sanitize_text(text).lower() for text in texts if _sanitize_text(text))
    output: List[str] = []
    for operation, patterns in _OPERATION_PATTERNS.items():
        if any(_contains_pattern(combined, pattern) for pattern in patterns):
            output.append(operation)
    return _normalize_operations(output)


def _infer_stage_operations(stage: ArchitectureStage) -> List[str]:
    if stage.stage_kind is not None:
        primary = _PRIMARY_OPERATION_BY_STAGE_KIND.get(stage.stage_kind)
        return [primary] if primary else []
    return _detect_operations(
        stage.name,
        stage.purpose,
        " ".join(stage.required_capabilities),
        " ".join(stage.expected_inputs),
        " ".join(stage.expected_outputs),
        " ".join(stage.success_criteria),
        stage.notes or "",
    )


def _infer_stage_kind(
    *,
    stage: ArchitectureStage,
    stage_operations: Sequence[str],
    is_sink: bool,
) -> StageKind:
    combined = _sanitize_text(
        " ".join(
            [
                stage.name,
                stage.purpose,
                " ".join(stage.required_capabilities),
                " ".join(stage.expected_inputs),
                " ".join(stage.expected_outputs),
                " ".join(stage.success_criteria),
                stage.notes or "",
            ]
        )
    ).lower()
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["classify"]):
        return StageKind.classify_decision
    if any(_contains_pattern(combined, hint) for hint in _OPERABLE_APPLY_HINTS):
        return StageKind.apply_update_source
    if is_sink and any(_contains_pattern(combined, hint) for hint in _NON_OPERABLE_UI_HINTS):
        return StageKind.apply_update_source
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["store"]):
        return StageKind.persist_store
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["notify"]):
        return StageKind.notify_output
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["route"]):
        return StageKind.route_branch
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["fetch"]):
        return StageKind.fetch_read
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["capture"]):
        return StageKind.trigger_intake
    for operation in stage_operations:
        kind = _STAGE_KIND_BY_OPERATION.get(operation)
        if kind is not None:
            return kind
    return StageKind.transform_process


def _infer_target_entity(stage: ArchitectureStage) -> Optional[str]:
    combined = _sanitize_text(
        " ".join(
            [
                stage.name,
                stage.purpose,
                " ".join(stage.expected_inputs),
                " ".join(stage.expected_outputs),
            ]
        )
    ).lower()
    for entity in (
        "gmail",
        "email",
        "correo",
        "slack",
        "notion",
        "sheet",
        "database",
        "ticket",
        "message",
    ):
        if entity in combined:
            return entity
    return None


def _stage_looks_non_operable(stage: ArchitectureStage) -> bool:
    combined = _sanitize_text(
        " ".join(
            [
                stage.name,
                stage.purpose,
                " ".join(stage.success_criteria),
                stage.notes or "",
            ]
        )
    ).lower()
    if not any(_contains_pattern(combined, hint) for hint in _NON_OPERABLE_UI_HINTS):
        return False
    if any(_contains_pattern(combined, hint) for hint in _OPERABLE_APPLY_HINTS):
        return False
    return True


def _mixed_stage_role_issue(
    *,
    stage: ArchitectureStage,
    explicit_operations: Sequence[str],
) -> Optional[str]:
    operations = set(_infer_stage_operations(stage))
    combined = _sanitize_text(
        " ".join(
            [
                stage.name,
                stage.purpose,
                " ".join(stage.required_capabilities),
                " ".join(stage.expected_inputs),
                " ".join(stage.expected_outputs),
                " ".join(stage.success_criteria),
                stage.notes or "",
            ]
        )
    ).lower()
    has_intake = (
        stage.stage_kind in {StageKind.trigger_intake, StageKind.fetch_read}
        or bool(operations.intersection({"capture", "fetch"}))
    )
    has_processing = (
        stage.stage_kind in {StageKind.classify_decision, StageKind.route_branch}
        or bool(operations.intersection({"classify", "route"}))
    )
    has_effect = (
        stage.stage_kind in {StageKind.apply_update_source, StageKind.persist_store, StageKind.notify_output}
        or bool(operations.intersection(_TERMINAL_EFFECT_OPERATIONS))
        or any(_contains_pattern(combined, hint) for hint in _OPERABLE_APPLY_HINTS)
    )
    if has_intake and has_effect:
        return (
            f"Stage '{stage.id}' ({stage.name}) mixes intake/fetch work with downstream outcome application. "
            "Split it into separate stages so one stage captures the source item and a later stage applies, stores, or notifies the result."
        )
    if len(explicit_operations) >= 2 and has_intake and has_processing:
        return (
            f"Stage '{stage.id}' ({stage.name}) mixes intake/fetch work with downstream processing or classification. "
            "Split it into separate stages with one dominant operable action each."
        )
    return None


def _enrich_stages(plan_output: _AbstractPlanningOutput) -> _AbstractPlanningOutput:
    sinks = {stage.id for stage in _sink_stages(plan_output.stages, plan_output.data_flow)}
    enriched_stages: List[ArchitectureStage] = []
    for stage in plan_output.stages:
        stage_operations = _infer_stage_operations(stage)
        stage_kind = stage.stage_kind or _infer_stage_kind(
            stage=stage,
            stage_operations=stage_operations,
            is_sink=stage.id in sinks,
        )
        business_effect = (
            _compact(stage.business_effect, max_chars=220)
            if stage.business_effect
            else _compact(stage.purpose or "Execute this workflow step.", max_chars=220)
        )
        target_entity = stage.target_entity or _infer_target_entity(stage)
        user_visible_goal = stage.user_visible_goal or _compact(
            " ".join(stage.success_criteria) or stage.name,
            max_chars=220,
        )
        unresolved_decisions = _safe_list(stage.unresolved_decisions)
        enriched_stages.append(
            stage.model_copy(
                update={
                    "stage_kind": stage_kind,
                    "business_effect": business_effect,
                    "target_entity": target_entity,
                    "user_visible_goal": user_visible_goal,
                    "unresolved_decisions": unresolved_decisions,
                }
            )
        )
    return plan_output.model_copy(update={"stages": enriched_stages})


def _sink_stages(
    stages: Sequence[ArchitectureStage],
    flows: Sequence[ArchitectureDataFlowItem],
) -> List[ArchitectureStage]:
    if not stages:
        return []
    outgoing = {
        _sanitize_text(item.source_stage_id)
        for item in flows
        if _sanitize_text(item.source_stage_id)
    }
    sinks = [stage for stage in stages if stage.id not in outgoing]
    return sinks or [stages[-1]]


def _request_explicitly_allows_terminal_analysis(*texts: Any) -> bool:
    combined = " ".join(_sanitize_text(text).lower() for text in texts if _sanitize_text(text))
    if not combined:
        return False
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["store"]):
        return False
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["notify"]):
        return False
    if any(_contains_pattern(combined, pattern) for pattern in _OPERATION_PATTERNS["sync"]):
        return False
    has_analysis_hint = any(_contains_pattern(combined, hint) for hint in _ANALYSIS_ONLY_HINTS)
    has_terminal_analysis = any(
        _contains_pattern(combined, pattern)
        for pattern in (
            "classify",
            "classification",
            "analyze",
            "analyse",
            "analiza",
            "clasifica",
            "report",
            "return",
            "output",
            "show",
            "display",
            "mostrar",
            "devolver",
        )
    )
    return has_analysis_hint and has_terminal_analysis


def _terminal_outcome_question(
    *,
    sink_stage: ArchitectureStage,
    terminal_operations: Sequence[str],
) -> str:
    if any(operation == "classify" for operation in terminal_operations):
        return (
            f"The workflow currently ends at stage '{sink_stage.name}' with a classification result. "
            "What should happen with that classification next: apply it to the source item, save it somewhere, "
            "notify someone, or route it to a downstream action?"
        )
    if any(operation == "route" for operation in terminal_operations):
        return (
            f"The workflow currently ends at stage '{sink_stage.name}' with a routing decision. "
            "What concrete downstream action should happen after that routing result?"
        )
    return (
        f"The workflow currently ends at stage '{sink_stage.name}' without a clear business outcome. "
        "What should the workflow do with that result next?"
    )


def _question_to_decision_slot(
    *,
    question: str,
    stage_id: Optional[str],
    user_query: str,
    use_case: UseCase,
) -> DecisionSlot:
    lowered_question = _sanitize_text(question).lower()
    if "workflow goal" in lowered_question or "abstract workflow plan" in lowered_question:
        slot_key = "workflow_goal"
    else:
        slot_key = infer_decision_slot_key(question, stage_name=stage_id or "")
    return build_decision_slot(
        slot_key=slot_key,
        owner_agent=AgentStage.product_manager_agent,
        question_text=question,
        stage_id=stage_id,
        question_intent=slot_key,
        user_query=user_query,
        context_texts=[use_case.title, use_case.business_problem, use_case.desired_outcome],
    )


def _missing_terminal_outcome_issue(
    *,
    use_case: UseCase,
    user_query: str,
    plan_output: _AbstractPlanningOutput,
) -> Optional[str]:
    sinks = _sink_stages(plan_output.stages, plan_output.data_flow)
    if not sinks:
        return None

    allowed_terminal_analysis = _request_explicitly_allows_terminal_analysis(
        user_query,
        use_case.title,
        use_case.business_problem,
        use_case.desired_outcome,
        plan_output.workflow_summary,
    )
    if allowed_terminal_analysis:
        return None

    for sink_stage in sinks:
        sink_operations = _infer_stage_operations(sink_stage)
        if not sink_operations:
            continue
        if set(sink_operations).intersection(_TERMINAL_EFFECT_OPERATIONS):
            return None
        if set(sink_operations).intersection(_AMBIGUOUS_TERMINAL_OPERATIONS):
            return _terminal_outcome_question(
                sink_stage=sink_stage,
                terminal_operations=sink_operations,
            )
    return None


def _validation_question_from_issues(issues: Sequence[str]) -> str:
    if not issues:
        return (
            "I could not produce a coherent abstract workflow plan that covers the full request. "
            "Please restate the workflow goal including trigger, processing, and outcome."
        )
    first_issue = _sanitize_text(issues[0])
    lowered_issue = first_issue.lower()
    if any(
        token in lowered_issue
        for token in (
            "unknown target stage",
            "unknown source stage",
            "depends on unknown stage",
            "data flow references",
            "abstract plan returned no stages",
            "does not materialize real stages",
            "fewer than 2 stages",
        )
    ):
        return (
            "I could not reconcile the latest clarification with the workflow goal. "
            "Please restate the full workflow goal including trigger, processing, and final outcome."
        )
    if first_issue.lower().startswith("the workflow currently ends at stage"):
        return _compact(first_issue, max_chars=220)
    return _compact(
        (
            "I could not produce a coherent abstract workflow plan that covers the full request. "
            f"Current issue: {first_issue} Please restate the workflow goal including trigger, processing, and outcome."
        ),
        max_chars=220,
    )


def _validate_abstract_plan_output(
    *,
    use_case: UseCase,
    user_query: str,
    plan_output: _AbstractPlanningOutput,
) -> Tuple[_AbstractPlanningOutput, List[str]]:
    plan_output = _enrich_stages(plan_output)
    explicit_operations = _normalize_operations(plan_output.explicit_user_operations)
    if not explicit_operations:
        explicit_operations = _detect_operations(
            user_query,
            use_case.title,
            use_case.business_problem,
            use_case.desired_outcome,
        )

    covered_operations = _normalize_operations(plan_output.covered_operations)
    stage_covered_operations: List[str] = []
    for stage in plan_output.stages:
        stage_covered_operations.extend(_infer_stage_operations(stage))
    covered_operations = _normalize_operations(list(covered_operations) + stage_covered_operations)

    normalized_output = plan_output.model_copy(
        update={
            "explicit_user_operations": explicit_operations,
            "covered_operations": covered_operations,
        }
    )

    issues: List[str] = []
    if not normalized_output.stages:
        issues.append("The abstract plan returned no stages.")
        return normalized_output, issues

    stage_ids = {stage.id for stage in normalized_output.stages}
    for stage in normalized_output.stages:
        for dependency in stage.dependencies:
            if dependency and dependency not in stage_ids:
                issues.append(
                    f"Stage '{stage.id}' depends on unknown stage '{dependency}'."
                )
    for flow in normalized_output.data_flow:
        if flow.source_stage_id not in stage_ids:
            issues.append(
                f"Data flow references unknown source stage '{flow.source_stage_id}'."
            )
        if flow.target_stage_id not in stage_ids:
            issues.append(
                f"Data flow references unknown target stage '{flow.target_stage_id}'."
            )

    if len(explicit_operations) >= 2 and len(normalized_output.stages) < 2:
        issues.append(
            "The request contains multiple explicit workflow operations, but the abstract plan defines fewer than 2 stages."
        )

    uncovered_operations = [
        _OPERATION_DISPLAY_NAMES.get(operation, operation)
        for operation in explicit_operations
        if operation not in covered_operations
    ]
    if uncovered_operations:
        issues.append(
            "The abstract plan does not materialize real stages for these explicit operations: "
            + ", ".join(uncovered_operations)
            + "."
        )

    for stage in normalized_output.stages:
        mixed_role_issue = _mixed_stage_role_issue(
            stage=stage,
            explicit_operations=explicit_operations,
        )
        if mixed_role_issue:
            issues.append(mixed_role_issue)
            break

    sink_stages = _sink_stages(normalized_output.stages, normalized_output.data_flow)
    for sink_stage in sink_stages:
        if not _request_explicitly_allows_terminal_analysis(
            user_query,
            use_case.title,
            use_case.business_problem,
            use_case.desired_outcome,
            normalized_output.workflow_summary,
        ) and _stage_looks_non_operable(sink_stage):
            issues.append(
                _compact(
                    (
                        f"The workflow currently ends at stage '{sink_stage.name}' with a user-visible outcome instead of an operable action. "
                        "What concrete action should the workflow perform with that result: apply it to the source item, save it, notify someone, or route it downstream?"
                    ),
                    max_chars=220,
                )
            )
            break

    terminal_outcome_issue = _missing_terminal_outcome_issue(
        use_case=use_case,
        user_query=user_query,
        plan_output=normalized_output,
    )
    if terminal_outcome_issue:
        issues.append(terminal_outcome_issue)

    return normalized_output, _safe_list(issues)


def _repair_abstract_plan_with_structured_output(
    *,
    use_case: UseCase,
    request_context_query: str,
    clarification_state: PMClarificationState,
    invalid_output: _AbstractPlanningOutput,
    validation_issues: Sequence[str],
    model: Optional[str],
    request_id: Optional[str],
) -> _AbstractPlanningOutput:
    system_prompt = (
        "You are product_manager_agent repairing an abstract workflow plan for an n8n assistant. "
        "Return a corrected abstract plan that fully covers the user request stage by stage. "
        "Do not select nodes, credentials, parameters, endpoints, or workflow JSON. "
        "Fix semantic coverage and stage/data_flow coherence."
    )
    user_prompt = (
        "Repair this invalid abstract workflow plan.\n\n"
        f"{_problem_statement(use_case=use_case, request_context_query=request_context_query, clarification_state=clarification_state)}\n\n"
        f"Invalid plan:\n{invalid_output.model_dump_json(exclude_none=True)}\n\n"
        "Validation issues that must be fixed:\n"
        + "\n".join(f"- {issue}" for issue in validation_issues)
        + "\n\nRules:\n"
        "- Keep the plan abstract.\n"
        "- Every explicit operation requested by the user must be covered by one or more real stages.\n"
        "- Each stage must keep one dominant operable action.\n"
        "- Do not mix source intake with downstream classification, storage, notification, or source mutation in the same stage.\n"
        "- data_flow and stage dependencies may reference only existing stage ids.\n"
        "- If the request contains more than one explicit operation, return at least 2 stages.\n"
        "- Preserve source system, trigger style, processing mode, and final outcome.\n"
        "- If no clarification is needed, missing_information must be an empty array.\n"
        "- Never put confirmations, summaries, or values like 'None' inside missing_information.\n"
        "- If the workflow would end only in a classification, decision, routing outcome, or transformation result, either add the missing final business outcome stage when it is explicit in the request, or set planning_ready=false and ask what should happen with that result.\n"
        "- Do not invent review or approval stages unless the user explicitly asked for them.\n"
        "- If the user explicitly asked for a silent workflow with no notifications, do not ask again about notifications.\n"
        "- Write user-facing clarification questions in the same language as the user's request.\n"
        "- Set planning_ready=true only if the repaired plan is coherent enough to hand off to architect_agent.\n"
    )
    output = _invoke_structured_output(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=_AbstractPlanningOutput,
        model=model,
        request_id=request_id,
        temperature=0.0,
        stage="multi_agent.product_manager.abstract_plan_repair",
    )
    if not output.stages:
        raise RuntimeError("PM abstract plan repair returned no stages")
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


def _normalize_use_case(value: Any) -> Optional[UseCase]:
    if isinstance(value, UseCase):
        return value
    if isinstance(value, dict):
        try:
            return UseCase.model_validate(value)
        except Exception:
            return None
    return None


def _request_context_from_use_case(use_case: UseCase) -> str:
    business_problem = _sanitize_text(use_case.business_problem)
    desired_outcome = _sanitize_text(use_case.desired_outcome)
    prefixes = (
        "User requested a new workflow for:",
        "Deliver an abstract workflow plan that satisfies:",
    )
    for value in (business_problem, desired_outcome):
        for prefix in prefixes:
            if value.startswith(prefix):
                extracted = _sanitize_text(value[len(prefix) :])
                if extracted:
                    return extracted
    return _sanitize_text(use_case.title) or business_problem or desired_outcome


def _request_context_query(
    *,
    state: MultiAgentGraphState,
    use_case: Optional[UseCase],
    current_user_query: str,
) -> str:
    explicit = _sanitize_text(state.get("request_context_query"))
    if explicit:
        return _compact(explicit, max_chars=500)
    if use_case is not None:
        derived = _request_context_from_use_case(use_case)
        if derived:
            return _compact(derived, max_chars=500)
    return _compact(current_user_query, max_chars=500)


def _clarification_answer_texts(
    clarification_state: Optional[PMClarificationState],
) -> List[str]:
    if clarification_state is None:
        return []
    values: List[str] = []
    for slot in clarification_state.resolved_slots:
        if slot.answer:
            values.append(slot.answer)
    for turn in clarification_state.turns:
        if turn.answer:
            values.append(turn.answer)
    return _safe_list(values)


def _extract_priority_labels(*texts: Any) -> List[str]:
    variants = {
        "low": {"low", "bajo"},
        "medium": {"medium", "medio"},
        "high": {"high", "alto"},
        "critical": {"critical", "critico", "crítico"},
    }
    output: List[str] = []
    seen = set()
    for text in texts:
        tokens = re.findall(r"[a-záéíóúñü]+", _sanitize_text(text).lower())
        for token in tokens:
            for canonical, allowed in variants.items():
                if token not in allowed or canonical in seen:
                    continue
                seen.add(canonical)
                output.append(token)
                break
    return output


def _extract_fallback_label(*texts: Any) -> Optional[str]:
    for text in texts:
        sanitized = _sanitize_text(text)
        if not sanitized:
            continue
        match = re.search(r"[\"'“”]?([A-Za-z][A-Za-z0-9 _-]{1,40})[\"'“”]?", sanitized)
        if not match:
            continue
        candidate = _sanitize_text(match.group(1))
        if candidate.lower() == "review" or "review" in candidate.lower():
            return "Review"
    combined = " ".join(_sanitize_text(text).lower() for text in texts if _sanitize_text(text))
    if "review" in combined:
        return "Review"
    return None


def _enrich_architecture_plan_with_constraints(
    *,
    plan: ArchitecturePlan,
    request_context_query: str,
    clarification_state: Optional[PMClarificationState],
) -> ArchitecturePlan:
    answer_texts = _clarification_answer_texts(clarification_state)
    combined_texts = [request_context_query, *answer_texts]
    combined = " ".join(_sanitize_text(text).lower() for text in combined_texts if _sanitize_text(text))
    priority_labels = _extract_priority_labels(*combined_texts)
    fallback_label = _extract_fallback_label(*combined_texts)
    source_is_gmail = "gmail" in combined
    semantic_classification = any(
        hint in combined
        for hint in ("semantic", "semantica", "semántica", "context", "contexto", "subject", "asunto", "body", "cuerpo")
    )
    per_item_trigger = any(
        hint in combined
        for hint in ("cada correo", "cada email", "every email", "cada mensaje", "new email", "nuevo correo", "en cuanto llegue", "siempre que se reciba")
    )
    visible_in_gmail = source_is_gmail and any(
        hint in combined
        for hint in ("visible", "gmail ui", "filtrar", "filter", "label", "tag", "etiquet")
    )
    apply_back_to_source = any(
        hint in combined
        for hint in (
            "same email",
            "same gmail",
            "same message",
            "same source",
            "apply it to the source",
            "apply it back",
            "mismo correo",
            "mismo email",
            "mismo mensaje",
            "mismo gmail",
            "mismo origen",
            "aplica el resultado al mismo",
            "anade la etiqueta al correo",
            "anade un tag al correo",
            "añade la etiqueta al correo",
            "añade un tag al correo",
        )
    )
    sink_stage_ids = {stage.id for stage in _sink_stages(plan.stages, plan.data_flow)}

    shared_notes: List[str] = []
    if priority_labels:
        shared_notes.append(
            "Urgency levels must remain exactly: " + ", ".join(priority_labels) + "."
        )
    if fallback_label:
        shared_notes.append(
            f"If the workflow cannot determine a confident urgency level, use fallback label '{fallback_label}'."
        )
    if semantic_classification:
        shared_notes.append(
            "Urgency must be inferred semantically from the email subject and body, not only from static metadata."
        )
    if visible_in_gmail:
        shared_notes.append(
            "Any source-side update must remain visible in Gmail UI so the user can filter emails later."
        )
    if per_item_trigger:
        shared_notes.append(
            "The workflow should run for each newly received email, not as a manual or batch-only process."
        )

    enriched_stages: List[ArchitectureStage] = []
    for stage in plan.stages:
        stage_kind = stage.stage_kind
        if (
            stage.id in sink_stage_ids
            and stage_kind in {StageKind.persist_store, StageKind.notify_output, StageKind.transform_process}
            and (apply_back_to_source or visible_in_gmail)
        ):
            stage_kind = StageKind.apply_update_source
        notes = _safe_list([stage.notes or ""])
        if stage_kind == StageKind.trigger_intake and per_item_trigger:
            notes.append("Preserve one execution per newly received email.")
        if stage_kind == StageKind.classify_decision:
            if priority_labels:
                notes.append("Return exactly one urgency level from: " + ", ".join(priority_labels) + ".")
            if semantic_classification:
                notes.append("Classify urgency from the semantic meaning of subject and body.")
            if fallback_label:
                notes.append(f"If classification is uncertain, use fallback label '{fallback_label}'.")
        if stage_kind == StageKind.apply_update_source:
            if source_is_gmail:
                notes.append("Apply the result back onto the same Gmail message.")
            if visible_in_gmail:
                notes.append("The applied label must be visible in Gmail UI for later filtering.")
            if priority_labels:
                notes.append("Applied label values should mirror the classification levels: " + ", ".join(priority_labels) + ".")
            if fallback_label:
                notes.append(f"Use fallback label '{fallback_label}' when no confident urgency level is available.")
        enriched_stages.append(
            stage.model_copy(
                update={
                    "stage_kind": stage_kind,
                    "notes": "\n".join(_safe_list(notes)) or None,
                    "target_entity": stage.target_entity or ("gmail" if source_is_gmail else stage.target_entity),
                }
            )
        )

    implementation_notes = _safe_list(list(plan.implementation_notes_for_engineer) + shared_notes)
    return plan.model_copy(
        update={
            "stages": enriched_stages,
            "implementation_notes_for_engineer": implementation_notes,
        }
    )


def _derive_use_case_from_direct_build_request(user_query: str) -> Optional[UseCase]:
    query = _compact(user_query, max_chars=140)
    if not query:
        return None
    return UseCase(
        id="direct_build_request",
        title=_compact(query, max_chars=80),
        business_problem=f"User requested a new workflow for: {query}",
        desired_outcome=f"Deliver an abstract workflow plan that satisfies: {query}",
        expected_value="Provide a high-quality workflow plan ready for architectural grounding.",
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


def _record_clarification_answer(
    clarification_state: PMClarificationState,
    user_query: str,
) -> PMClarificationState:
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
    resolved_slot_keys = {slot.slot_key for slot in newly_resolved}
    if not pending_slots:
        for question in pending:
            for idx in range(len(turns) - 1, -1, -1):
                if turns[idx].question == question and not turns[idx].answer:
                    turns[idx] = turns[idx].model_copy(update={"answer": answer})
                    break
            else:
                turns.append(PMClarificationTurn(question=question, answer=answer))
    for slot in newly_resolved:
        for idx in range(len(turns) - 1, -1, -1):
            if turns[idx].slot_key == slot.slot_key and not turns[idx].answer:
                turns[idx] = turns[idx].model_copy(update={"answer": answer})
                break
    if pending_slots and not newly_resolved and len(pending_slots) == 1:
        slot = pending_slots[0].model_copy(update={"answer": answer, "answer_status": "resolved"})
        newly_resolved = [slot]
        remaining_slots = []
        resolved_slot_keys = {slot.slot_key}
    return clarification_state.model_copy(
        update={
            "pending_questions": pending_slot_questions(remaining_slots) if pending_slots else [],
            "pending_slots": remaining_slots,
            "resolved_slots": merge_decision_slots(resolved_slots, newly_resolved),
            "turns": turns,
        }
    )


def _rephrase_pending_pm_questions(
    clarification_state: PMClarificationState,
    *,
    user_query: str,
    use_case: UseCase,
) -> PMClarificationState:
    pending_slots = list(clarification_state.pending_slots)
    localized_pending = localize_question_list(
        clarification_state.pending_questions,
        user_query=user_query,
        context_texts=[
            use_case.title,
            use_case.business_problem,
            use_case.desired_outcome,
            use_case.expected_value,
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
                        use_case.title,
                        use_case.business_problem,
                        use_case.desired_outcome,
                        use_case.expected_value,
                    ],
                )
            }
        )
        for slot in pending_slots
    ]
    return clarification_state.model_copy(
        update={"pending_questions": localized_pending, "pending_slots": localized_slots, "turns": turns}
    )


def _problem_statement(
    *,
    use_case: UseCase,
    request_context_query: str,
    clarification_state: PMClarificationState,
) -> str:
    clarification_lines = [
        f"Question: {turn.question}\nAnswer: {turn.answer}"
        for turn in clarification_state.turns
        if turn.answer
    ]
    slot_lines = [
        f"Decision slot [{slot.slot_key}] answer: {slot.answer}"
        for slot in clarification_state.resolved_slots
        if slot.answer
    ]
    clarification_text = "\n\n".join((clarification_lines + slot_lines)[:6])
    return (
        f"Original workflow request: {_compact(request_context_query, max_chars=500)}\n"
        f"Use case title: {use_case.title}\n"
        f"Business problem: {use_case.business_problem}\n"
        f"Desired outcome: {use_case.desired_outcome}\n"
        f"Expected business value: {use_case.expected_value}\n"
        f"Clarifications already provided:\n{clarification_text or '-'}"
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
            params={"temperature": temperature, "structured": True, "provider": "openai_json_schema"},
        )
        response = invoke_openai_structured_output(
            messages=messages,
            model=model,
            output_model=output_model,
            temperature=temperature,
        )
        emit_llm_output_event(
            trace_logger,
            request_id=request_id,
            stage=stage,
            model=model,
            latency_ms=None,
            content=response.model_dump_json(exclude_none=True) if hasattr(response, "model_dump_json") else str(response),
            usage=None,
            extra={"structured": True, "provider": "openai_json_schema"},
        )
        return response

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


def _resolved_slot_answer_map(clarification_state: Optional[PMClarificationState]) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    if clarification_state is None:
        return answers
    for slot in clarification_state.resolved_slots:
        answer = _sanitize_text(slot.answer)
        if not answer:
            continue
        answers[slot.slot_key] = answer
    return answers


def _heuristic_apply_mode_label(
    *,
    result_mode_text: str,
    combined_context: str,
) -> Tuple[Optional[str], Optional[str]]:
    text = " ".join([result_mode_text, combined_context]).lower()
    if any(token in text for token in ("label", "labels", "tag", "tags", "etiquet", "filter", "filtrar", "visible")):
        return "apply_update_source", "label"
    if any(token in text for token in ("move", "mover", "folder", "carpeta", "archive", "archiv")):
        return "apply_update_source", "move"
    if any(token in text for token in ("save", "store", "persist", "guardar", "almacen", "sheet", "sheets", "database", "table")):
        return "persist_store", "store"
    if any(token in text for token in ("notify", "notification", "alert", "notific", "avis")):
        return "notify_output", "notify"
    if any(token in text for token in ("apply", "aplicar", "source item", "elemento origen", "gmail", "correo")):
        return "apply_update_source", "apply"
    return None, None


def _heuristic_abstract_plan(
    use_case: UseCase,
    *,
    request_context_query: str = "",
    clarification_state: Optional[PMClarificationState] = None,
) -> _AbstractPlanningOutput:
    resolved_answers = _resolved_slot_answer_map(clarification_state)
    combined_context = " ".join(
        [
            _sanitize_text(use_case.title),
            _sanitize_text(use_case.business_problem),
            _sanitize_text(use_case.desired_outcome),
            _sanitize_text(request_context_query),
            *[_sanitize_text(value) for value in resolved_answers.values()],
        ]
    ).lower()
    explicit_operations = _detect_operations(
        request_context_query,
        use_case.title,
        use_case.business_problem,
        use_case.desired_outcome,
    )
    result_mode_answer = resolved_answers.get("result_application_mode", "")
    source_answer = resolved_answers.get("source_system", "")
    classification_answer = resolved_answers.get("classification_method", "")
    target_system = "gmail" if "gmail" in f"{combined_context} {source_answer.lower()}" else "email"
    wants_email = any(token in combined_context for token in ("email", "emails", "correo", "correos", "gmail", "imap", "outlook"))
    wants_ai = any(
        token in f"{combined_context} {classification_answer.lower()}"
        for token in (" ai ", " ia ", "llm", "openai", "ollama", "semantic", "semant", "modelo", "model")
    )
    apply_kind, apply_mode = _heuristic_apply_mode_label(
        result_mode_text=result_mode_answer,
        combined_context=combined_context,
    )
    allows_terminal_analysis = _request_explicitly_allows_terminal_analysis(
        request_context_query,
        use_case.title,
        use_case.business_problem,
        use_case.desired_outcome,
        result_mode_answer,
    )

    if wants_email or {"capture", "classify"}.intersection(explicit_operations):
        stage_intake_name = "Detect New Gmail Emails" if target_system == "gmail" else "Receive Incoming Emails"
        stage_intake_purpose = (
            "Start the workflow for each new Gmail email as it arrives and expose the raw message payload."
            if target_system == "gmail"
            else "Start the workflow for each new incoming email and expose the raw message payload."
        )
        stages: List[ArchitectureStage] = [
            ArchitectureStage(
                id="stage_intake",
                name=stage_intake_name,
                purpose=stage_intake_purpose,
                stage_kind=StageKind.trigger_intake,
                business_effect="Capture the incoming email event and normalize the working payload.",
                target_entity="gmail_message" if target_system == "gmail" else "email_message",
                user_visible_goal="The workflow reacts automatically whenever a new email arrives.",
                required_capabilities=["Trigger on each incoming email", "Normalize subject and body for downstream processing"],
                expected_inputs=["Incoming email event"],
                expected_outputs=["Normalized email payload with subject, body, sender, and message identifiers"],
                dependencies=[],
                success_criteria=["Each new incoming email starts the workflow once without duplicates."],
            ),
            ArchitectureStage(
                id="stage_processing",
                name="Classify Email Urgency with AI" if wants_ai else "Classify Email Urgency",
                purpose=(
                    "Interpret the subject and body semantically with an AI model and assign one urgency level."
                    if wants_ai
                    else "Determine the urgency level from the subject and body using the requested business logic."
                ),
                stage_kind=StageKind.classify_decision,
                business_effect="Produce exactly one urgency level for each incoming email.",
                target_entity="gmail_message" if target_system == "gmail" else "email_message",
                user_visible_goal="Each email receives a clear urgency decision.",
                required_capabilities=[
                    "Read the normalized email payload",
                    "Assign an urgency level such as low, medium, high, or critical",
                ],
                expected_inputs=["Normalized email payload with subject and body"],
                expected_outputs=["Urgency classification result"],
                dependencies=["stage_intake"],
                success_criteria=["Each incoming email receives exactly one urgency level."],
                notes="If the classification is uncertain, a later stage may apply a fallback label such as Review.",
            ),
        ]
        data_flow = [
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_processing",
                data_items=["normalized email payload"],
                notes="The captured email is sent to the classification stage.",
            )
        ]
        assumptions = [
            "Architect agent should ground the trigger against the actual email source selected by the user or inferred from context.",
        ]
        missing_information: List[str] = []
        handoff_notes = [
            "Architect agent should preserve the trigger -> classify -> apply sequence without mixing responsibilities across stages."
        ]
        covered_operations = ["capture", "classify"]

        if apply_kind == "apply_update_source":
            apply_name = (
                "Apply Urgency Label in Gmail"
                if target_system == "gmail"
                else "Apply Urgency Result to Source Email"
            )
            apply_purpose = (
                "Apply the resulting urgency level back onto the same Gmail message as a visible label so the user can filter it later."
                if apply_mode == "label" and target_system == "gmail"
                else "Apply the resulting workflow outcome back onto the source email item."
            )
            stages.append(
                ArchitectureStage(
                    id="stage_apply",
                    name=apply_name,
                    purpose=apply_purpose,
                    stage_kind=StageKind.apply_update_source,
                    business_effect="The source email is updated with the workflow result.",
                    target_entity="gmail_message" if target_system == "gmail" else "email_message",
                    user_visible_goal=(
                        "The urgency label is visible in Gmail so the user can filter emails later."
                        if target_system == "gmail"
                        else "The email system shows the applied urgency result."
                    ),
                    required_capabilities=["Update the original email item", "Apply a visible urgency label or equivalent source-side marker"],
                    expected_inputs=["Urgency classification result", "Source email identifiers"],
                    expected_outputs=["Source email updated with urgency label"],
                    dependencies=["stage_processing"],
                    success_criteria=["The original email item is updated exactly once with the chosen urgency label."],
                    notes="If no urgency level can be determined confidently, the applied fallback label should be Review.",
                )
            )
            data_flow.append(
                ArchitectureDataFlowItem(
                    source_stage_id="stage_processing",
                    target_stage_id="stage_apply",
                    data_items=["urgency classification result", "source email identifiers"],
                    notes="The classification result is applied back to the original email item.",
                )
            )
            covered_operations.extend(["sync", "route"])
        elif apply_kind == "persist_store":
            stages.append(
                ArchitectureStage(
                    id="stage_apply",
                    name="Persist Urgency Result",
                    purpose="Persist the resulting urgency classification in the selected downstream storage location.",
                    stage_kind=StageKind.persist_store,
                    business_effect="The classification result is stored for later access.",
                    target_entity="classification_record",
                    user_visible_goal="The workflow preserves the urgency result outside the classifier.",
                    required_capabilities=["Persist the classification result", "Associate the result with the source email"],
                    expected_inputs=["Urgency classification result", "Source email identifiers"],
                    expected_outputs=["Stored urgency record"],
                    dependencies=["stage_processing"],
                    success_criteria=["The urgency result is stored exactly once."],
                )
            )
            data_flow.append(
                ArchitectureDataFlowItem(
                    source_stage_id="stage_processing",
                    target_stage_id="stage_apply",
                    data_items=["urgency classification result", "source email identifiers"],
                    notes="The classification result is stored downstream.",
                )
            )
            covered_operations.append("store")
        elif apply_kind == "notify_output":
            stages.append(
                ArchitectureStage(
                    id="stage_apply",
                    name="Notify on Classified Email",
                    purpose="Notify the configured recipient or system about the urgency result.",
                    stage_kind=StageKind.notify_output,
                    business_effect="The urgency result is communicated downstream.",
                    target_entity="notification",
                    user_visible_goal="Relevant recipients receive the urgency result.",
                    required_capabilities=["Send the urgency result to a downstream recipient or channel"],
                    expected_inputs=["Urgency classification result", "Source email identifiers"],
                    expected_outputs=["Notification delivered"],
                    dependencies=["stage_processing"],
                    success_criteria=["The notification is emitted once for each relevant email."],
                )
            )
            data_flow.append(
                ArchitectureDataFlowItem(
                    source_stage_id="stage_processing",
                    target_stage_id="stage_apply",
                    data_items=["urgency classification result", "source email identifiers"],
                    notes="The classification result is communicated downstream.",
                )
            )
            covered_operations.append("notify")
        elif not allows_terminal_analysis:
            missing_information.append(
                _terminal_outcome_question(
                    sink_stage=stages[-1],
                    terminal_operations=["classify"],
                )
            )

        planning_ready = not missing_information
        summary_parts = [
            "Abstract workflow plan for incoming email urgency classification",
            "with automatic intake",
            "semantic classification" if wants_ai else "urgency classification",
        ]
        if apply_kind == "apply_update_source":
            summary_parts.append("and source-side label application")
        elif apply_kind == "persist_store":
            summary_parts.append("and persistence of the urgency result")
        elif apply_kind == "notify_output":
            summary_parts.append("and downstream notification")
        elif not planning_ready:
            summary_parts.append("awaiting clarification for the final outcome")
        return _AbstractPlanningOutput(
            workflow_summary=_compact(" ".join(summary_parts) + ".", max_chars=320),
            stages=stages,
            data_flow=data_flow,
            assumptions=assumptions,
            missing_information=missing_information,
            handoff_notes=handoff_notes,
            explicit_user_operations=_normalize_operations(
                list(explicit_operations or ["capture", "classify"])
                + ([apply_kind == "apply_update_source" and "sync" or ""] if apply_kind else [])
                + ([apply_kind == "persist_store" and "store" or ""] if apply_kind else [])
                + ([apply_kind == "notify_output" and "notify" or ""] if apply_kind else [])
            ),
            covered_operations=covered_operations,
            planning_ready=planning_ready,
        )

    processing_label = "Decisioning"
    processing_purpose = "Apply the core business logic to classify, decide, or transform the payload."
    if any(token in combined_context for token in ("heuristic", "rule-based", "rule based", "deterministic")):
        processing_label = "Rule-Based Decisioning"
        processing_purpose = "Apply the explicit heuristic or rule-based business logic requested by the user."
    stages = [
        ArchitectureStage(
            id="stage_intake",
            name="Intake",
            purpose="Capture the incoming event or source item and normalize the payload.",
            required_capabilities=["Receive the input", "Normalize the working payload"],
            expected_inputs=["Incoming trigger or source item"],
            expected_outputs=["Normalized workflow payload"],
            dependencies=[],
            success_criteria=["The workflow receives the relevant item reliably."],
        ),
        ArchitectureStage(
            id="stage_processing",
            name=processing_label,
            purpose=processing_purpose,
            required_capabilities=["Interpret the payload", "Apply workflow business rules"],
            expected_inputs=["Normalized workflow payload"],
            expected_outputs=["Decision result or enriched payload"],
            dependencies=["stage_intake"],
            success_criteria=["The workflow produces a clear business decision or enrichment."],
        ),
    ]
    missing_information = []
    planning_ready = True
    if not allows_terminal_analysis:
        missing_information.append(
            _terminal_outcome_question(
                sink_stage=stages[-1],
                terminal_operations=["transform"],
            )
        )
        planning_ready = False
    return _AbstractPlanningOutput(
        workflow_summary=_compact(
            f"Abstract workflow plan for '{use_case.title}' with intake and decisioning stages.",
            max_chars=320,
        ),
        stages=stages,
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_processing",
                data_items=["normalized workflow payload"],
                notes="Normalized payload feeds the business logic stage.",
            ),
        ],
        assumptions=["The trigger/source for the workflow can be identified during later architecture work."],
        missing_information=missing_information,
        handoff_notes=[
            "Architect agent should ground each stage into node candidates without changing the abstract stage intent."
        ],
        explicit_user_operations=explicit_operations,
        covered_operations=["capture", "transform"],
        planning_ready=planning_ready,
    )


def _plan_abstract_workflow_with_structured_output(
    *,
    use_case: UseCase,
    request_context_query: str,
    clarification_state: PMClarificationState,
    model: Optional[str],
    request_id: Optional[str],
) -> _AbstractPlanningOutput:
    system_prompt = (
        "You are product_manager_agent for an n8n workflow assistant. "
        "Your role is abstract workflow planning only. "
        "Design how the workflow should behave stage by stage, but do not select node types, "
        "do not mention n8n nodes, do not mention credentials, do not mention parameter names, "
        "do not mention API endpoints, and do not generate workflow JSON. "
        "You are preparing a clean handoff for a future architect_agent that will later ground the plan into nodes. "
        "Preserve the user's exact semantics instead of broadening the request."
    )
    user_prompt = (
        "Create an abstract workflow plan from this request context.\n\n"
        f"{_problem_statement(use_case=use_case, request_context_query=request_context_query, clarification_state=clarification_state)}\n\n"
        "Rules:\n"
        "- Output planning-level stages only.\n"
        "- If the request contains more than one explicit workflow operation, return at least 2 stages.\n"
        "- Otherwise return the minimum number of stages that still covers the full request coherently.\n"
        "- Each stage must define purpose, required_capabilities, expected_inputs, expected_outputs, dependencies, and success_criteria.\n"
        "- Each stage must have one dominant operable action that the workflow can actually execute.\n"
        "- Stages must describe behavior, not implementation details.\n"
        "- Do not mix source intake with downstream classification, storage, notification, or source mutation in the same stage.\n"
        "- User-visible outcomes like filtering, visibility, browsing, or 'seeing items in Gmail' belong in user_visible_goal or success_criteria, not as the primary action of a stage.\n"
        "- Preserve the exact source system, trigger style, processing mode, and final outcome named by the user.\n"
        "- Determine whether the workflow ends in a useful business effect, such as applying, saving, notifying, routing to a concrete downstream action, or intentionally exposing the result to the user.\n"
        "- If the workflow would otherwise end only in a classification, decision, routing outcome, or transformation result, ask what should happen with that result unless the user explicitly wants analysis or reporting only.\n"
        "- If the user said heuristic, rule-based, deterministic, manual, inbound, outgoing, receive, or send, keep that distinction explicit in the plan.\n"
        "- Distinguish receiving email from sending email; they are not interchangeable.\n"
        "- explicit_user_operations must list the canonical workflow operations explicitly requested by the user.\n"
        "- covered_operations must list the canonical workflow operations materially covered by the returned stages.\n"
        "- Do not leave requested operations covered only in workflow_summary, assumptions, data_flow, or handoff_notes; they must appear in real stages.\n"
        "- Do not invent validation, feedback, manual review, refinement, or approval stages unless the user explicitly asked for them.\n"
        "- Never output concrete node types or workflow JSON.\n"
        "- If the user explicitly named systems like Gmail, Slack, or Google Sheets, treat them as business/system context, not node choices.\n"
        "- Preserve concrete business constraints mentioned by the user, including label values, fallback values, source-system visibility requirements, trigger cadence, and whether classification should be semantic/AI-driven.\n"
        "- Carry those concrete constraints into the relevant stages and handoff_notes so architect_agent and engineer_agent can implement them later without re-asking the same thing.\n"
        "- If no clarification is needed, missing_information must be an empty array. Never place statements like 'None' or summary text there.\n"
        "- If the user already made the workflow silent or said there should be no notifications, do not ask again about notifications.\n"
        "- If the user already defined that the result should be applied back to Gmail or made visible there for filtering, do not ask again for downstream action.\n"
        "- Set planning_ready=true only if the abstract workflow plan is coherent enough to hand off to architect_agent.\n"
        "- If essential planning information is missing, set planning_ready=false and list the missing_information as concise user-facing clarification questions.\n"
        "- Write missing_information questions in the same language as the user's request.\n"
        "- handoff_notes must explain what architect_agent should preserve when grounding this plan into nodes.\n"
    )
    output = _invoke_structured_output(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=_AbstractPlanningOutput,
        model=model,
        request_id=request_id,
        temperature=0.2,
        stage="multi_agent.product_manager.abstract_plan",
    )
    if not output.stages:
        raise RuntimeError("PM abstract planning returned no stages")
    return output


def _build_architecture_plan(
    *,
    use_case: UseCase,
    plan_output: _AbstractPlanningOutput,
    request_context_query: str,
    clarification_state: Optional[PMClarificationState],
) -> ArchitecturePlan:
    enriched_output = _enrich_stages(plan_output)
    plan = ArchitecturePlan(
        use_case_id=use_case.id,
        title=use_case.title,
        business_objective=_compact(use_case.business_problem, max_chars=260),
        desired_outcome=_compact(use_case.desired_outcome, max_chars=260),
        workflow_summary=_compact(enriched_output.workflow_summary, max_chars=320),
        stages=list(enriched_output.stages),
        data_flow=list(enriched_output.data_flow),
        assumptions=_safe_list(enriched_output.assumptions),
        missing_information=_safe_list(enriched_output.missing_information),
        implementation_notes_for_engineer=_safe_list(enriched_output.handoff_notes),
        required_nodes=[],
    )
    return _enrich_architecture_plan_with_constraints(
        plan=plan,
        request_context_query=request_context_query,
        clarification_state=clarification_state,
    )


def _build_decision_slots_for_plan(
    *,
    use_case: UseCase,
    user_query: str,
    plan_output: _AbstractPlanningOutput,
) -> List[DecisionSlot]:
    slots: List[DecisionSlot] = []
    for question in _safe_list(plan_output.missing_information):
        stage_id = None
        lowered_question = _sanitize_text(question).lower()
        for stage in plan_output.stages:
            if stage.name and _sanitize_text(stage.name).lower() in lowered_question:
                stage_id = stage.id
                break
        slots.append(
            _question_to_decision_slot(
                question=question,
                stage_id=stage_id,
                user_query=user_query,
                use_case=use_case,
            )
        )
    return slots


def _derive_pm_stage_plan(plan: ArchitecturePlan) -> List[PMStagePlan]:
    return [
        PMStagePlan(
            id=stage.id,
            name=stage.name,
            objective=stage.purpose,
            expected_inputs=list(stage.expected_inputs),
            expected_outputs=list(stage.expected_outputs),
            success_criteria=list(stage.success_criteria),
            dependencies=list(stage.dependencies),
        )
        for stage in plan.stages
    ]


def _build_stage_progress(stage_plan: List[PMStagePlan], *, completed: bool) -> PMProgressState:
    stage_ids = [stage.id for stage in stage_plan]
    return PMProgressState(
        total_stages=len(stage_plan),
        current_stage_id=None if completed else (stage_ids[0] if stage_ids else None),
        completed_stage_ids=stage_ids if completed else [],
        blocked_stage_ids=[] if completed else stage_ids[:1],
        passes_by_stage={},
    )


def _planning_summary(
    *,
    use_case: UseCase,
    stage_plan: List[PMStagePlan],
    pm_status: PMStatus,
) -> str:
    return _compact(
        (
            f"PM status={pm_status.value}; use_case={use_case.id}; "
            f"abstract_stages={len(stage_plan)}; handoff_target=architect_agent."
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
    user_query = str(state.get("user_query") or "")
    seed_request_context_query = _request_context_query(
        state=state,
        use_case=None,
        current_user_query=user_query,
    )

    selected_use_case = _normalize_use_case(state.get("selected_use_case"))
    if selected_use_case is None and entry_intent == EntryIntent.workflow_build_request:
        selected_use_case = _derive_use_case_from_direct_build_request(seed_request_context_query)
        if selected_use_case is not None:
            routing_signals.append("pm_use_case_derived_from_direct_build_request")
    request_context_query = _request_context_query(
        state=state,
        use_case=selected_use_case,
        current_user_query=user_query,
    )

    if selected_use_case is None:
        message = "A selected use case or direct workflow build request is required before PM planning can continue."
        missing_user_inputs = _safe_list(missing_user_inputs + [message])
        routing_signals.append("pm_missing_selected_use_case")
        workflow_context = WorkflowContext(
            use_case_id="unknown",
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            clarification_owner=AgentStage.product_manager_agent,
            clarification_reason="planning_gap",
            last_block_cause="planning_gap",
            notes=["product_manager_requires_selected_use_case", "abstract_plan_only"],
        )
        return {
            "current_stage": "product_manager_agent",
            "request_context_query": seed_request_context_query,
            "selected_use_case": None,
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
            "pending_decision_slots": [],
            "resolved_decision_slots": [],
            "clarification_owner": AgentStage.product_manager_agent,
            "clarification_reason": "planning_gap",
            "last_block_cause": "planning_gap",
            "planning_summary": None,
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    pm_status = _normalize_pm_status(state.get("pm_status"))
    clarification_state = _normalize_model(state.get("pm_clarification_state"), PMClarificationState)
    if clarification_state is None:
        clarification_state = PMClarificationState(
            attempts_used=0,
            max_attempts=_pm_int("PM_MAX_USER_CLARIFICATIONS", 2),
            pending_questions=[],
            turns=[],
        )

    if (
        pm_status == PMStatus.pm_blocked_waiting_user
        and (clarification_state.pending_questions or clarification_state.pending_slots)
        and user_query
    ):
        if is_question_rephrase_request(user_query):
            clarification_state = _rephrase_pending_pm_questions(
                clarification_state,
                user_query=user_query,
                use_case=selected_use_case,
            )
            localized_questions = list(clarification_state.pending_questions)
            workflow_context = WorkflowContext(
                use_case_id=selected_use_case.id,
                planning_ready=False,
                handoff_target=None,
                required_node_types=[],
                unresolved_inputs=localized_questions,
                pending_decision_slots=list(clarification_state.pending_slots),
                resolved_decision_slots=list(clarification_state.resolved_slots),
                clarification_owner=AgentStage.product_manager_agent,
                clarification_reason="planning_gap",
                last_block_cause="planning_gap",
                notes=["pm_status=pm_blocked_waiting_user", "abstract_plan_only", "pm_question_rephrased"],
            )
            routing_signals.append("pm_question_rephrased")
            return {
                "current_stage": "product_manager_agent",
                "request_context_query": request_context_query,
                "selected_use_case": selected_use_case,
                "pm_status": PMStatus.pm_blocked_waiting_user,
                "pm_stage_plan": state.get("pm_stage_plan") or [],
                "pm_stage_selections": state.get("pm_stage_selections") or [],
                "pm_stage_progress": state.get("pm_stage_progress") or PMProgressState(total_stages=0),
                "pm_clarification_state": clarification_state,
                "pm_stage_search_history": state.get("pm_stage_search_history") or [],
                "pm_reasoning_trace_full": state.get("pm_reasoning_trace_full") or [],
                "architecture_plan": _normalize_model(state.get("architecture_plan"), ArchitecturePlan),
                "workflow_context": workflow_context,
                "pending_decision_slots": list(clarification_state.pending_slots),
                "resolved_decision_slots": list(clarification_state.resolved_slots),
                "clarification_owner": AgentStage.product_manager_agent,
                "clarification_reason": "planning_gap",
                "last_block_cause": "planning_gap",
                "planning_summary": state.get("planning_summary"),
                "missing_user_inputs": localized_questions,
                "routing_signals": routing_signals,
                "target_stage": None,
                "proposed_nodes": [],
                "required_credentials": [],
            }
        clarification_state = _record_clarification_answer(clarification_state, user_query)
        if "pm_clarification_answer_received" not in routing_signals:
            routing_signals.append("pm_clarification_answer_received")
        emit_trace_event(
            trace_logger,
            event="clarification_slot_resolved",
            request_id=request_id,
            stage="multi_agent.product_manager",
            payload={
                "owner_agent": AgentStage.product_manager_agent.value,
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

    planning_validation_issues: List[str] = []
    repair_attempted = False
    try:
        plan_output = _plan_abstract_workflow_with_structured_output(
            use_case=selected_use_case,
            request_context_query=request_context_query,
            clarification_state=clarification_state,
            model=model,
            request_id=request_id,
        )
        plan_output, planning_validation_issues = _validate_abstract_plan_output(
            use_case=selected_use_case,
            user_query=request_context_query,
            plan_output=plan_output,
        )
        if planning_validation_issues:
            repair_attempted = True
            plan_output = _repair_abstract_plan_with_structured_output(
                use_case=selected_use_case,
                request_context_query=request_context_query,
                clarification_state=clarification_state,
                invalid_output=plan_output,
                validation_issues=planning_validation_issues,
                model=model,
                request_id=request_id,
            )
            plan_output, planning_validation_issues = _validate_abstract_plan_output(
                use_case=selected_use_case,
                user_query=request_context_query,
                plan_output=plan_output,
            )
    except Exception as exc:
        logger.warning("pm abstract planning fallback triggered: %s", str(exc))
        plan_output = _heuristic_abstract_plan(
            selected_use_case,
            request_context_query=request_context_query,
            clarification_state=clarification_state,
        )
        plan_output, planning_validation_issues = _validate_abstract_plan_output(
            use_case=selected_use_case,
            user_query=request_context_query,
            plan_output=plan_output,
        )

    if planning_validation_issues:
        clarification_message = localize_question_text(
            _validation_question_from_issues(planning_validation_issues),
            user_query=request_context_query,
            context_texts=[
                selected_use_case.title,
                selected_use_case.business_problem,
                selected_use_case.desired_outcome,
            ],
        )
        plan_output = plan_output.model_copy(
            update={
                "planning_ready": False,
                "missing_information": _sanitize_missing_information_items(
                    list(plan_output.missing_information) + [clarification_message]
                ),
            }
        )

    sanitized_missing_information = _sanitize_missing_information_items(plan_output.missing_information)
    localized_missing_information = localize_question_list(
        sanitized_missing_information,
        user_query=request_context_query,
        context_texts=[
            selected_use_case.title,
            selected_use_case.business_problem,
            selected_use_case.desired_outcome,
        ],
    )
    localized_missing_information = _sanitize_missing_information_items(localized_missing_information)
    effective_planning_ready = bool(plan_output.stages) and not planning_validation_issues and not localized_missing_information
    if (
        localized_missing_information != list(plan_output.missing_information)
        or bool(plan_output.planning_ready) != effective_planning_ready
    ):
        plan_output = plan_output.model_copy(
            update={
                "missing_information": localized_missing_information,
                "planning_ready": effective_planning_ready,
            }
        )

    architecture_plan = _build_architecture_plan(
        use_case=selected_use_case,
        plan_output=plan_output,
        request_context_query=request_context_query,
        clarification_state=clarification_state,
    )
    stage_plan = _derive_pm_stage_plan(architecture_plan)
    clarification_questions = _safe_list(plan_output.missing_information)
    clarification_slots = _build_decision_slots_for_plan(
        use_case=selected_use_case,
        user_query=request_context_query,
        plan_output=plan_output,
    )

    reasoning_trace = [
        {
            "planning_ready": bool(plan_output.planning_ready),
            "stage_count": len(stage_plan),
            "missing_information_count": len(clarification_questions),
            "explicit_user_operations": list(plan_output.explicit_user_operations),
            "covered_operations": list(plan_output.covered_operations),
            "validation_issues": list(planning_validation_issues),
            "repair_attempted": repair_attempted,
            "pending_decision_slots": [
                {
                    "slot_key": slot.slot_key,
                    "stage_id": slot.stage_id,
                    "question_intent": slot.question_intent,
                }
                for slot in clarification_slots
            ],
        }
    ]

    if not stage_plan:
        pm_status = PMStatus.pm_failed_no_solution
        missing_user_inputs = _safe_list(
            missing_user_inputs + ["PM could not derive any abstract workflow stages from the request."]
        )
        routing_signals.append("pm_failed_no_solution")
        workflow_context = WorkflowContext(
            use_case_id=selected_use_case.id,
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            pending_decision_slots=[],
            resolved_decision_slots=list(clarification_state.resolved_slots),
            clarification_owner=AgentStage.product_manager_agent,
            clarification_reason="planning_gap",
            last_block_cause="planning_gap",
            notes=["pm_failed_no_solution", "abstract_plan_only"],
        )
        return {
            "current_stage": "product_manager_agent",
            "request_context_query": request_context_query,
            "selected_use_case": selected_use_case,
            "pm_status": pm_status,
            "pm_stage_plan": [],
            "pm_stage_selections": [],
            "pm_stage_progress": PMProgressState(total_stages=0),
            "pm_clarification_state": clarification_state,
            "pm_stage_search_history": [],
            "pm_reasoning_trace_full": reasoning_trace,
            "architecture_plan": None,
            "workflow_context": workflow_context,
            "pending_decision_slots": [],
            "resolved_decision_slots": list(clarification_state.resolved_slots),
            "clarification_owner": AgentStage.product_manager_agent,
            "clarification_reason": "planning_gap",
            "last_block_cause": "planning_gap",
            "planning_summary": None,
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    if clarification_questions or not plan_output.planning_ready:
        if clarification_state.attempts_used < clarification_state.max_attempts:
            turns = list(clarification_state.turns)
            for slot in clarification_slots:
                if any(
                    (turn.slot_key == slot.slot_key and (turn.stage_id or "") == (slot.stage_id or ""))
                    for turn in turns
                ):
                    continue
                turns.append(
                    PMClarificationTurn(
                        stage_id=slot.stage_id,
                        slot_key=slot.slot_key,
                        question=slot.question_text,
                        answer=None,
                    )
                )
            clarification_state = clarification_state.model_copy(
                update={
                    "attempts_used": clarification_state.attempts_used + 1,
                    "pending_questions": pending_slot_questions(clarification_slots) or clarification_questions,
                    "pending_slots": clarification_slots,
                    "turns": turns,
                }
            )
            pm_status = PMStatus.pm_blocked_waiting_user
            missing_user_inputs = _safe_list(
                missing_user_inputs + (pending_slot_questions(clarification_slots) or clarification_questions)
            )
            if "pm_blocked_waiting_user" not in routing_signals:
                routing_signals.append("pm_blocked_waiting_user")
        else:
            pm_status = PMStatus.pm_failed_no_solution
            missing_user_inputs = _safe_list(
                missing_user_inputs + (pending_slot_questions(clarification_slots) or clarification_questions)
            )
            if "pm_failed_no_solution" not in routing_signals:
                routing_signals.append("pm_failed_no_solution")

        workflow_context = WorkflowContext(
            use_case_id=selected_use_case.id,
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            pending_decision_slots=list(clarification_state.pending_slots),
            resolved_decision_slots=list(clarification_state.resolved_slots),
            clarification_owner=AgentStage.product_manager_agent,
            clarification_reason="planning_gap",
            last_block_cause="planning_gap",
            notes=[
                f"pm_status={pm_status.value}",
                "abstract_plan_only",
                "waiting_for_clarification",
            ],
        )
        stage_progress = _build_stage_progress(stage_plan, completed=False)
        emit_trace_event(
            trace_logger,
            event="pm_abstract_plan",
            request_id=request_id,
            stage="multi_agent.product_manager",
            payload={
                "pm_status": pm_status.value,
                "use_case_id": selected_use_case.id,
                "title": architecture_plan.title,
                "workflow_summary": architecture_plan.workflow_summary,
                "stage_count": len(stage_plan),
                "stages": _stage_trace_summary(stage_plan),
                "missing_information": _safe_list(architecture_plan.missing_information),
                "assumptions": _safe_list(architecture_plan.assumptions)[:6],
                "clarification_attempts": clarification_state.attempts_used,
                "planning_ready": False,
                "handoff_target": None,
                "pending_decision_slots": [
                    {
                        "slot_key": slot.slot_key,
                        "stage_id": slot.stage_id,
                        "question_text": slot.question_text,
                    }
                    for slot in clarification_state.pending_slots
                ],
                "resolved_decision_slots": [
                    {
                        "slot_key": slot.slot_key,
                        "stage_id": slot.stage_id,
                    }
                    for slot in clarification_state.resolved_slots
                ],
                "explicit_user_operations": list(plan_output.explicit_user_operations),
                "covered_operations": list(plan_output.covered_operations),
                "validation_issues": list(planning_validation_issues),
                "repair_attempted": repair_attempted,
            },
        )
        emit_trace_event(
            trace_logger,
            event="clarification_slot_requested",
            request_id=request_id,
            stage="multi_agent.product_manager",
            payload={
                "owner_agent": AgentStage.product_manager_agent.value,
                "pending_slots": [
                    {
                        "slot_key": slot.slot_key,
                        "stage_id": slot.stage_id,
                        "question_text": slot.question_text,
                    }
                    for slot in clarification_state.pending_slots
                ],
            },
        )
        return {
            "current_stage": "product_manager_agent",
            "request_context_query": request_context_query,
            "selected_use_case": selected_use_case,
            "pm_status": pm_status,
            "pm_stage_plan": stage_plan,
            "pm_stage_selections": [],
            "pm_stage_progress": stage_progress,
            "pm_clarification_state": clarification_state,
            "pm_stage_search_history": [],
            "pm_reasoning_trace_full": reasoning_trace,
            "architecture_plan": architecture_plan,
            "workflow_context": workflow_context,
            "pending_decision_slots": list(clarification_state.pending_slots),
            "resolved_decision_slots": list(clarification_state.resolved_slots),
            "clarification_owner": AgentStage.product_manager_agent,
            "clarification_reason": "planning_gap",
            "last_block_cause": "planning_gap",
            "planning_summary": _planning_summary(
                use_case=selected_use_case,
                stage_plan=stage_plan,
                pm_status=pm_status,
            ),
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    pm_status = PMStatus.pm_completed
    if "handoff_ready_architect" not in routing_signals:
        routing_signals.append("handoff_ready_architect")

    workflow_context = WorkflowContext(
        use_case_id=selected_use_case.id,
        planning_ready=True,
        handoff_target=AgentStage.architect_agent,
        required_node_types=[],
        unresolved_inputs=[],
        pending_decision_slots=[],
        resolved_decision_slots=list(clarification_state.resolved_slots),
        clarification_owner=None,
        clarification_reason=None,
        last_block_cause=None,
        notes=[
            f"pm_status={pm_status.value}",
            "abstract_plan_only",
            "handoff_target=architect_agent",
        ],
    )
    stage_progress = _build_stage_progress(stage_plan, completed=True)

    emit_trace_event(
        trace_logger,
        event="pm_abstract_plan",
        request_id=request_id,
        stage="multi_agent.product_manager",
        payload={
            "pm_status": pm_status.value,
            "use_case_id": selected_use_case.id,
            "title": architecture_plan.title,
            "workflow_summary": architecture_plan.workflow_summary,
            "stage_count": len(stage_plan),
            "stages": _stage_trace_summary(stage_plan),
            "missing_information": _safe_list(architecture_plan.missing_information),
            "assumptions": _safe_list(architecture_plan.assumptions)[:6],
            "clarification_attempts": clarification_state.attempts_used,
            "planning_ready": True,
            "handoff_target": AgentStage.architect_agent.value,
            "resolved_decision_slots": [
                {
                    "slot_key": slot.slot_key,
                    "stage_id": slot.stage_id,
                }
                for slot in clarification_state.resolved_slots
            ],
            "explicit_user_operations": list(plan_output.explicit_user_operations),
            "covered_operations": list(plan_output.covered_operations),
            "validation_issues": list(planning_validation_issues),
            "repair_attempted": repair_attempted,
        },
    )

    return {
        "current_stage": "product_manager_agent",
        "request_context_query": request_context_query,
        "selected_use_case": selected_use_case,
        "pm_status": pm_status,
        "pm_stage_plan": stage_plan,
        "pm_stage_selections": [],
        "pm_stage_progress": stage_progress,
        "pm_clarification_state": clarification_state,
        "pm_stage_search_history": [],
        "pm_reasoning_trace_full": reasoning_trace,
        "architecture_plan": architecture_plan,
        "workflow_context": workflow_context,
        "pending_decision_slots": [],
        "resolved_decision_slots": list(clarification_state.resolved_slots),
        "clarification_owner": None,
        "clarification_reason": None,
        "last_block_cause": None,
        "planning_summary": _planning_summary(
            use_case=selected_use_case,
            stage_plan=stage_plan,
            pm_status=pm_status,
        ),
        "missing_user_inputs": [],
        "routing_signals": routing_signals,
        "target_stage": None,
        "proposed_nodes": [],
        "required_credentials": [],
    }
