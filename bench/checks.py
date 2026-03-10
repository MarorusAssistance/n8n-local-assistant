from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .models import CaseSpec, RequirementSpec
from .scoring import compute_compliance_score
from .trace import TraceEntry, extract_retrieval_views


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    payload: Optional[Dict[str, Any]]
    extracted_text: str
    failures: List[str]


@dataclass(frozen=True)
class CheckCounts:
    passed: int
    total: int
    failures: List[str]

    @property
    def ratio(self) -> float:
        if self.total <= 0:
            return 1.0
        return max(0.0, min(1.0, self.passed / self.total))


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TECHNICAL_ARTIFACTS_RE = re.compile(
    r"\b(n8n-nodes-|credential[s]?|api[_\-\s]?key|oauth|token\s*=)\b",
    re.IGNORECASE,
)
_WORKFLOW_JSON_KEYS = {"nodes", "connections", "final_workflow_json"}
_INTENT_TO_STAGE = {
    "business_discovery_conversation": "commercial_agent",
    "workflow_build_request": "product_manager_agent",
    "workflow_edit_request": "engineer_agent",
    "workflow_fix_request": "qa_agent",
    "information_request": "consultant_agent",
    "unknown": None,
}


def run_checks(
    response_text: str,
    *,
    case: CaseSpec,
    trace_entries: Optional[Sequence[TraceEntry]] = None,
) -> Dict[str, Any]:
    parse_result = parse_json_object(response_text)
    section_failures: Dict[str, List[str]] = {
        "json_parse_ok": list(parse_result.failures),
    }

    if not parse_result.ok or not parse_result.payload:
        score, breakdown = compute_compliance_score(
            parse_ok=False,
            section_ratios={},
            section_failures=section_failures,
        )
        return {
            "parse_ok": False,
            "payload": None,
            "extracted_json": parse_result.extracted_text,
            "compliance_score": score,
            "breakdown": breakdown,
            "failures": parse_result.failures,
            "trace": {
                "json_parse_ok": {"failures": parse_result.failures},
            },
        }

    payload = parse_result.payload
    contract_counts = _check_multi_agent_contract_schema(payload)
    router_counts = _check_router_routing_graph(payload)
    commercial_counts = _check_commercial_selection(payload)
    product_manager_counts = _check_product_manager_planning(payload)
    safety_counts = _check_planning_safety(payload)
    requirement_counts = _check_case_requirements(case.requirements, payload)
    limits_counts = _check_case_limits(case, payload)
    trace_counts = _check_retrieval_trace(
        payload,
        trace_entries or [],
        case_stage=case.stage,
    )

    router_counts = _merge_counts(router_counts, contract_counts)
    router_counts = _merge_counts(
        router_counts,
        _filter_requirement_counts(requirement_counts, section="router_routing_graph"),
    )
    commercial_counts = _merge_counts(
        commercial_counts,
        _filter_requirement_counts(requirement_counts, section="commercial_selection"),
    )
    product_manager_counts = _merge_counts(
        product_manager_counts,
        _filter_requirement_counts(requirement_counts, section="product_manager_planning"),
    )
    safety_counts = _merge_counts(
        safety_counts,
        _filter_requirement_counts(requirement_counts, section="planning_safety_guardrails"),
    )
    router_counts = _merge_counts(
        router_counts,
        _filter_limit_counts(limits_counts, section="router_routing_graph"),
    )
    commercial_counts = _merge_counts(
        commercial_counts,
        _filter_limit_counts(limits_counts, section="commercial_selection"),
    )
    product_manager_counts = _merge_counts(
        product_manager_counts,
        _filter_limit_counts(limits_counts, section="product_manager_planning"),
    )
    safety_counts = _merge_counts(
        safety_counts,
        _filter_limit_counts(limits_counts, section="planning_safety_guardrails"),
    )

    section_ratios = {
        "router_routing_graph": router_counts.ratio,
        "commercial_selection": commercial_counts.ratio,
        "product_manager_planning": product_manager_counts.ratio,
        "planning_safety_guardrails": safety_counts.ratio,
        "retrieval_trace_checks": trace_counts.ratio,
    }
    section_failures.update(
        {
            "router_routing_graph": router_counts.failures,
            "commercial_selection": commercial_counts.failures,
            "product_manager_planning": product_manager_counts.failures,
            "planning_safety_guardrails": safety_counts.failures,
            "retrieval_trace_checks": trace_counts.failures,
        }
    )

    score, breakdown = compute_compliance_score(
        parse_ok=True,
        section_ratios=section_ratios,
        section_failures=section_failures,
    )

    failures = (
        contract_counts.failures
        + router_counts.failures
        + commercial_counts.failures
        + product_manager_counts.failures
        + safety_counts.failures
        + trace_counts.failures
    )
    return {
        "parse_ok": True,
        "payload": payload,
        "extracted_json": parse_result.extracted_text,
        "compliance_score": score,
        "breakdown": breakdown,
        "failures": failures,
        "trace": {
            "json_parse_ok": {"failures": []},
            "multi_agent_contract_schema": _counts_to_trace(contract_counts),
            "router_routing_graph": _counts_to_trace(router_counts),
            "commercial_selection": _counts_to_trace(commercial_counts),
            "product_manager_planning": _counts_to_trace(product_manager_counts),
            "planning_safety_guardrails": _counts_to_trace(safety_counts),
            "retrieval_trace_checks": _counts_to_trace(trace_counts),
        },
    }


def parse_json_object(response_text: str) -> ParseResult:
    text = (response_text or "").strip()
    if not text:
        return ParseResult(
            ok=False,
            payload=None,
            extracted_text="",
            failures=["assistant response is empty"],
        )

    candidates: List[str] = [text]
    for match in _FENCED_JSON_RE.finditer(text):
        value = (match.group(1) or "").strip()
        if value:
            candidates.append(value)
    candidates.extend(_scan_braced_json_candidates(text))

    seen = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    parse_errors: List[str] = []
    for candidate in unique_candidates:
        try:
            payload = json.loads(candidate)
        except Exception as exc:
            parse_errors.append(f"failed candidate parse: {str(exc)}")
            continue
        if not isinstance(payload, dict):
            parse_errors.append("parsed JSON is not an object")
            continue
        return ParseResult(ok=True, payload=payload, extracted_text=candidate, failures=[])

    if not parse_errors:
        parse_errors.append("no JSON object could be extracted from response")
    return ParseResult(
        ok=False,
        payload=None,
        extracted_text="",
        failures=parse_errors[:6],
    )


def _scan_braced_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for start_idx, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : idx + 1].strip()
                    if candidate:
                        candidates.append(candidate)
                    break
            if depth < 0:
                break
    return candidates


def _check_multi_agent_contract_schema(payload: Mapping[str, Any]) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    checks = [
        ("entry_intent", lambda value: isinstance(value, str)),
        ("target_stage", lambda value: isinstance(value, str) or value is None),
        ("confidence", lambda value: isinstance(value, (int, float))),
        ("routing_signals", lambda value: isinstance(value, list)),
        ("current_stage", lambda value: isinstance(value, str) or value is None),
        ("missing_user_inputs", lambda value: isinstance(value, list)),
        ("status", lambda value: isinstance(value, str)),
    ]
    for key, validator in checks:
        total += 1
        value = payload.get(key)
        if validator(value):
            passed += 1
        else:
            failures.append(f"contract field '{key}' has invalid type")

    routing_signals = payload.get("routing_signals")
    if isinstance(routing_signals, list):
        for idx, item in enumerate(routing_signals):
            total += 1
            if isinstance(item, str):
                passed += 1
            else:
                failures.append(f"routing_signals[{idx}] must be string")

    missing_inputs = payload.get("missing_user_inputs")
    if isinstance(missing_inputs, list):
        for idx, item in enumerate(missing_inputs):
            total += 1
            if isinstance(item, str):
                passed += 1
            else:
                failures.append(f"missing_user_inputs[{idx}] must be string")

    return CheckCounts(passed=passed, total=max(1, total), failures=failures)


def _check_router_routing_graph(payload: Mapping[str, Any]) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    intent = str(payload.get("entry_intent") or "")
    target_stage = payload.get("target_stage")
    current_stage = payload.get("current_stage")
    routing_signals = payload.get("routing_signals") if isinstance(payload.get("routing_signals"), list) else []

    total += 1
    expected_stage = _INTENT_TO_STAGE.get(intent)
    if _routing_consistency_ok(
        intent=intent,
        current_stage=current_stage,
        target_stage=target_stage,
        expected_stage=expected_stage,
    ):
        passed += 1
    else:
        failures.append(
            "intent/route mismatch: "
            f"intent='{intent}' current_stage='{current_stage}' "
            f"expected_entry_stage='{expected_stage}' target_stage='{target_stage}'"
        )

    total += 1
    if intent != "unknown" or target_stage is None:
        passed += 1
    else:
        failures.append("unknown intent must not force target_stage")

    if "fix_signals_detected" in routing_signals and "edit_signals_detected" in routing_signals:
        total += 1
        if intent == "workflow_fix_request":
            passed += 1
        else:
            failures.append("fix signals and edit signals present but intent is not workflow_fix_request")

    total += 1
    if current_stage is None or isinstance(current_stage, str):
        passed += 1
    else:
        failures.append("current_stage must be string or null")

    return CheckCounts(passed=passed, total=max(1, total), failures=failures)


def _routing_consistency_ok(
    *,
    intent: str,
    current_stage: Any,
    target_stage: Any,
    expected_stage: Any,
) -> bool:
    if intent == "unknown":
        return target_stage is None

    if intent in {"workflow_edit_request", "workflow_fix_request", "information_request"}:
        return target_stage == expected_stage

    if intent == "workflow_build_request":
        if current_stage == "product_manager_agent" and target_stage in {
            "product_manager_agent",
            "engineer_agent",
            None,
        }:
            return True
        return target_stage == expected_stage

    if intent == "business_discovery_conversation":
        if current_stage == "commercial_agent":
            return target_stage in {"commercial_agent", "product_manager_agent", None}
        if current_stage == "product_manager_agent":
            return target_stage in {"product_manager_agent", "engineer_agent", None}
        return target_stage == expected_stage

    return target_stage == expected_stage


def _check_commercial_selection(payload: Mapping[str, Any]) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    discovered = payload.get("discovered_use_cases")
    selected = payload.get("selected_use_case")
    alternatives = payload.get("alternative_use_cases")
    selection_reason = payload.get("selection_reason")

    if not isinstance(discovered, list):
        discovered = []
    if not isinstance(alternatives, list):
        alternatives = []

    if discovered:
        total += 1
        if isinstance(selected, Mapping):
            passed += 1
        else:
            failures.append("discovered_use_cases present but selected_use_case is missing")

    if isinstance(selected, Mapping):
        selected_id = str(selected.get("id") or "")
        alt_ids = {
            str(item.get("id") or "")
            for item in alternatives
            if isinstance(item, Mapping)
        }
        total += 1
        if selected_id and selected_id not in alt_ids:
            passed += 1
        else:
            failures.append("selected_use_case must not appear in alternative_use_cases")

        if alternatives:
            selected_score = selected.get("priority_score")
            alt_scores = [
                item.get("priority_score")
                for item in alternatives
                if isinstance(item, Mapping)
            ]
            if isinstance(selected_score, (int, float)) and all(
                isinstance(score, (int, float)) for score in alt_scores
            ):
                total += 1
                if float(selected_score) >= max(float(score) for score in alt_scores):
                    passed += 1
                else:
                    failures.append("selected_use_case priority_score is below alternatives")

    if selected is None:
        total += 1
        if alternatives == [] and isinstance(selection_reason, str) and selection_reason.strip():
            passed += 1
        else:
            failures.append("when selected_use_case is null, alternatives must be empty and selection_reason explicit")

    text_fields: List[str] = []
    for item in discovered:
        if not isinstance(item, Mapping):
            continue
        text_fields.extend(
            str(item.get(key) or "")
            for key in ("title", "business_problem", "desired_outcome")
        )
    if isinstance(selected, Mapping):
        text_fields.extend(
            str(selected.get(key) or "")
            for key in ("title", "business_problem", "desired_outcome")
        )

    if text_fields:
        total += 1
        combined = " ".join(text_fields)
        if not _TECHNICAL_ARTIFACTS_RE.search(combined):
            passed += 1
        else:
            failures.append("commercial output contains technical implementation artifacts")

    return CheckCounts(passed=passed, total=max(1, total), failures=failures)


def _check_product_manager_planning(payload: Mapping[str, Any]) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    architecture_plan = payload.get("architecture_plan")
    workflow_context = payload.get("workflow_context")
    target_stage = payload.get("target_stage")

    planning_ready = None
    handoff_target = None
    if isinstance(workflow_context, Mapping):
        planning_ready = workflow_context.get("planning_ready")
        handoff_target = workflow_context.get("handoff_target")

    if isinstance(architecture_plan, Mapping):
        required_nodes = architecture_plan.get("required_nodes")
        total += 1
        if isinstance(required_nodes, list):
            passed += 1
        else:
            failures.append("architecture_plan.required_nodes must be a list")
            required_nodes = []

        for idx, node in enumerate(required_nodes):
            if not isinstance(node, Mapping):
                total += 1
                failures.append(f"required_nodes[{idx}] must be object")
                continue
            total += 1
            if isinstance(node.get("node_type"), str) and str(node.get("node_type")).strip():
                passed += 1
            else:
                failures.append(f"required_nodes[{idx}] missing node_type")

            total += 1
            chunk_ids = node.get("evidence_chunk_ids")
            refs = node.get("evidence_refs")
            if (
                isinstance(chunk_ids, list)
                and isinstance(refs, list)
                and bool(chunk_ids or refs)
            ):
                passed += 1
            else:
                failures.append(f"required_nodes[{idx}] missing evidence references")

    if planning_ready is True:
        total += 1
        if target_stage == "engineer_agent":
            passed += 1
        else:
            failures.append("planning_ready=true requires target_stage=engineer_agent")

        total += 1
        if handoff_target == "engineer_agent":
            passed += 1
        else:
            failures.append("planning_ready=true requires workflow_context.handoff_target=engineer_agent")

        total += 1
        if isinstance(architecture_plan, Mapping):
            passed += 1
        else:
            failures.append("planning_ready=true requires architecture_plan object")

    if planning_ready is False:
        total += 1
        if target_stage in (None, "product_manager_agent", "commercial_agent"):
            passed += 1
        else:
            failures.append("planning_ready=false should not handoff directly to engineer")

    return CheckCounts(passed=passed, total=max(1, total), failures=failures)


def _check_planning_safety(payload: Mapping[str, Any]) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    for forbidden in sorted(_WORKFLOW_JSON_KEYS):
        total += 1
        if forbidden not in payload:
            passed += 1
        else:
            failures.append(f"forbidden top-level workflow key found: {forbidden}")

    architecture_plan = payload.get("architecture_plan")
    if isinstance(architecture_plan, Mapping):
        for forbidden in sorted(_WORKFLOW_JSON_KEYS):
            total += 1
            if forbidden not in architecture_plan:
                passed += 1
            else:
                failures.append(f"forbidden architecture_plan key found: {forbidden}")

    return CheckCounts(passed=passed, total=max(1, total), failures=failures)


def _check_retrieval_trace(
    payload: Mapping[str, Any],
    entries: Sequence[TraceEntry],
    *,
    case_stage: str,
) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    if case_stage != "product_manager":
        return CheckCounts(passed=1, total=1, failures=[])

    retrieval_events = [entry for entry in entries if entry.event_type == "retrieval_final"]
    total += 1
    if len(retrieval_events) == 1:
        passed += 1
    else:
        failures.append(f"retrieval_final count expected=1 got={len(retrieval_events)}")

    views = extract_retrieval_views(entries)
    pre_count = (
        len(views.get("pre_docs", []))
        + len(views.get("pre_linked_node", []))
        + len(views.get("pre_linked_credential", []))
    )
    post_count = (
        len(views.get("post_docs", []))
        + len(views.get("post_linked_node", []))
        + len(views.get("post_linked_credential", []))
    )

    total += 1
    if pre_count > 0:
        passed += 1
    else:
        failures.append("retrieval trace missing pre-rerank chunks")

    total += 1
    if post_count > 0:
        passed += 1
    else:
        failures.append("retrieval trace missing post-rerank chunks")

    architecture_plan = payload.get("architecture_plan")
    if isinstance(architecture_plan, Mapping):
        required_nodes = architecture_plan.get("required_nodes")
        if isinstance(required_nodes, list) and required_nodes:
            total += 1
            if post_count > 0:
                passed += 1
            else:
                failures.append(
                    "required_nodes present but no retrieval evidence chunks in trace"
                )

    return CheckCounts(passed=passed, total=max(1, total), failures=failures)


def _check_case_limits(
    case: CaseSpec,
    payload: Mapping[str, Any],
) -> Dict[str, CheckCounts]:
    counts: Dict[str, List[str]] = {
        "router_routing_graph": [],
        "commercial_selection": [],
        "product_manager_planning": [],
        "planning_safety_guardrails": [],
    }
    totals: Dict[str, int] = {key: 0 for key in counts.keys()}
    passes: Dict[str, int] = {key: 0 for key in counts.keys()}

    limits = case.limits
    if limits and limits.max_missing_user_inputs is not None:
        totals["product_manager_planning"] += 1
        max_allowed = int(limits.max_missing_user_inputs)
        missing_items = payload.get("missing_user_inputs")
        actual = len(missing_items) if isinstance(missing_items, list) else 0
        if actual <= max_allowed:
            passes["product_manager_planning"] += 1
        else:
            counts["product_manager_planning"].append(
                "max_missing_user_inputs exceeded: "
                f"expected <= {max_allowed}, got {actual}"
            )

    output: Dict[str, CheckCounts] = {}
    for section in counts.keys():
        output[section] = CheckCounts(
            passed=passes[section],
            total=max(0, totals[section]),
            failures=counts[section],
        )
    return output


def _check_case_requirements(
    requirements: Sequence[RequirementSpec],
    payload: Mapping[str, Any],
) -> Dict[str, CheckCounts]:
    counts: Dict[str, List[str]] = {
        "router_routing_graph": [],
        "commercial_selection": [],
        "product_manager_planning": [],
        "planning_safety_guardrails": [],
    }
    totals: Dict[str, int] = {key: 0 for key in counts.keys()}
    passes: Dict[str, int] = {key: 0 for key in counts.keys()}

    for requirement in requirements:
        section = _section_for_requirement(requirement)
        totals[section] += 1
        ok, failure = _eval_requirement(requirement, payload)
        if ok:
            passes[section] += 1
        elif failure:
            counts[section].append(failure)

    output: Dict[str, CheckCounts] = {}
    for section in counts.keys():
        output[section] = CheckCounts(
            passed=passes[section],
            total=max(0, totals[section]),
            failures=counts[section],
        )
    return output


def _eval_requirement(
    requirement: RequirementSpec,
    payload: Mapping[str, Any],
) -> tuple[bool, Optional[str]]:
    req_type = requirement.type
    value = requirement.value

    if req_type in {"must_equal_field", "must_not_equal_field"}:
        path = str(value.get("path"))
        expected = value.get("equals") if req_type == "must_equal_field" else value.get("not_equals")
        resolved = _resolve_path(payload, path)
        if req_type == "must_equal_field":
            ok = resolved.exists and resolved.value == expected
            if not ok:
                return False, f"{req_type} failed for path '{path}': expected={expected!r} got={resolved.value!r}"
            return True, None
        ok = (not resolved.exists) or resolved.value != expected
        if not ok:
            return False, f"{req_type} failed for path '{path}': disallowed value {expected!r}"
        return True, None

    if req_type in {"must_be_null_field", "must_not_be_null_field"}:
        path = str(value.get("path"))
        resolved = _resolve_path(payload, path)
        if req_type == "must_be_null_field":
            ok = resolved.exists and resolved.value is None
            if not ok:
                return False, f"{req_type} failed for path '{path}'"
            return True, None
        ok = resolved.exists and resolved.value is not None
        if not ok:
            return False, f"{req_type} failed for path '{path}'"
        return True, None

    if req_type in {"must_include_routing_signal", "must_not_include_routing_signal"}:
        signal = str(value)
        signals = payload.get("routing_signals")
        items = signals if isinstance(signals, list) else []
        contains = signal in items
        if req_type == "must_include_routing_signal":
            if not contains:
                return False, f"routing_signals missing expected value '{signal}'"
            return True, None
        if contains:
            return False, f"routing_signals contains forbidden value '{signal}'"
        return True, None

    if req_type == "must_have_required_nodes_min":
        expected_min = int(value)
        architecture_plan = payload.get("architecture_plan")
        required_nodes = architecture_plan.get("required_nodes") if isinstance(architecture_plan, Mapping) else []
        actual = len(required_nodes) if isinstance(required_nodes, list) else 0
        if actual < expected_min:
            return False, f"required_nodes count too low: expected >= {expected_min}, got {actual}"
        return True, None

    if req_type == "must_have_required_nodes_with_evidence":
        expected_flag = bool(value)
        architecture_plan = payload.get("architecture_plan")
        required_nodes = architecture_plan.get("required_nodes") if isinstance(architecture_plan, Mapping) else []
        has_evidence = True
        if not isinstance(required_nodes, list) or not required_nodes:
            has_evidence = False
        else:
            for node in required_nodes:
                if not isinstance(node, Mapping):
                    has_evidence = False
                    break
                evidence_chunk_ids = node.get("evidence_chunk_ids")
                evidence_refs = node.get("evidence_refs")
                if not isinstance(evidence_chunk_ids, list) or not isinstance(evidence_refs, list):
                    has_evidence = False
                    break
                if not evidence_chunk_ids and not evidence_refs:
                    has_evidence = False
                    break
        if has_evidence != expected_flag:
            return False, f"must_have_required_nodes_with_evidence failed (expected {expected_flag})"
        return True, None

    if req_type == "must_not_include_workflow_json_keys":
        forbidden = [str(item).strip() for item in value]
        found = [key for key in forbidden if key in payload]
        architecture_plan = payload.get("architecture_plan")
        if isinstance(architecture_plan, Mapping):
            found.extend(
                f"architecture_plan.{key}"
                for key in forbidden
                if key in architecture_plan
            )
        if found:
            return False, f"found forbidden workflow keys: {', '.join(found)}"
        return True, None

    if req_type == "must_have_planning_ready":
        expected = bool(value)
        context = payload.get("workflow_context")
        got = context.get("planning_ready") if isinstance(context, Mapping) else None
        if got is not expected:
            return False, f"workflow_context.planning_ready expected {expected}, got {got!r}"
        return True, None

    if req_type == "must_have_handoff_target":
        expected = value
        context = payload.get("workflow_context")
        got = context.get("handoff_target") if isinstance(context, Mapping) else None
        if got != expected:
            return False, f"workflow_context.handoff_target expected {expected!r}, got {got!r}"
        return True, None

    return False, f"unsupported requirement type: {req_type}"


def _section_for_requirement(requirement: RequirementSpec) -> str:
    req_type = requirement.type
    if req_type in {
        "must_include_routing_signal",
        "must_not_include_routing_signal",
    }:
        return "router_routing_graph"
    if req_type in {
        "must_have_required_nodes_min",
        "must_have_required_nodes_with_evidence",
        "must_have_planning_ready",
        "must_have_handoff_target",
    }:
        return "product_manager_planning"
    if req_type == "must_not_include_workflow_json_keys":
        return "planning_safety_guardrails"

    value = requirement.value
    path = str(value.get("path") or "") if isinstance(value, Mapping) else ""
    if path.startswith("architecture_plan.") or path.startswith("workflow_context."):
        return "product_manager_planning"
    if path.startswith("selected_use_case") or path.startswith("discovered_use_cases") or path.startswith("alternative_use_cases"):
        return "commercial_selection"
    return "router_routing_graph"


@dataclass(frozen=True)
class _PathResult:
    exists: bool
    value: Any


def _resolve_path(payload: Mapping[str, Any], path: str) -> _PathResult:
    current: Any = payload
    tokens = _parse_path(path)
    for token in tokens:
        if isinstance(token, str):
            if not isinstance(current, Mapping) or token not in current:
                return _PathResult(False, None)
            current = current[token]
            continue
        if not isinstance(token, int):
            return _PathResult(False, None)
        if not isinstance(current, list) or token < 0 or token >= len(current):
            return _PathResult(False, None)
        current = current[token]
    return _PathResult(True, current)


def _parse_path(path: str) -> List[Any]:
    if not path:
        return []
    out: List[Any] = []
    chunks = path.split(".")
    for chunk in chunks:
        match = re.match(r"^([A-Za-z0-9_-]+)(\[\d+\])*$", chunk)
        if not match:
            out.append(chunk)
            continue
        key = match.group(1)
        out.append(key)
        indexes = re.findall(r"\[(\d+)\]", chunk)
        out.extend(int(index) for index in indexes)
    return out


def _merge_counts(base: CheckCounts, extra: CheckCounts) -> CheckCounts:
    return CheckCounts(
        passed=base.passed + extra.passed,
        total=base.total + extra.total,
        failures=[*base.failures, *extra.failures],
    )


def _filter_requirement_counts(
    all_counts: Mapping[str, CheckCounts],
    *,
    section: str,
) -> CheckCounts:
    counts = all_counts.get(section)
    if counts is None:
        return CheckCounts(passed=0, total=0, failures=[])
    return counts


def _filter_limit_counts(
    all_counts: Mapping[str, CheckCounts],
    *,
    section: str,
) -> CheckCounts:
    counts = all_counts.get(section)
    if counts is None:
        return CheckCounts(passed=0, total=0, failures=[])
    return counts


def _counts_to_trace(counts: CheckCounts) -> Dict[str, Any]:
    return {
        "passed": counts.passed,
        "total": counts.total,
        "ratio": counts.ratio,
        "failures": counts.failures,
    }
