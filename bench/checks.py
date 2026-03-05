from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .models import CaseSpec, RequirementSpec
from .scoring import compute_compliance_score


@dataclass(frozen=True)
class Catalog:
    node_types: Set[str]
    required_credentials_by_node_type: Dict[str, Set[str]]
    credential_types: Set[str]
    supported_nodes_by_credential: Dict[str, Set[str]]


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    workflow: Optional[Dict[str, Any]]
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
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def load_catalog(nodes_path: Path, credentials_path: Path) -> Catalog:
    nodes_raw = json.loads(nodes_path.read_text(encoding="utf-8"))
    credentials_raw = json.loads(credentials_path.read_text(encoding="utf-8"))
    if not isinstance(nodes_raw, list):
        raise ValueError(f"Expected list in {nodes_path}")
    if not isinstance(credentials_raw, list):
        raise ValueError(f"Expected list in {credentials_path}")
    return build_catalog_from_payload(nodes_raw, credentials_raw)


def build_catalog_from_payload(
    nodes_payload: Sequence[Mapping[str, Any]],
    credentials_payload: Sequence[Mapping[str, Any]],
) -> Catalog:
    node_types: Set[str] = set()
    required_credentials_by_node_type: Dict[str, Set[str]] = {}

    for node in nodes_payload:
        node_type = str(node.get("name") or "").strip()
        if not node_type:
            continue
        node_types.add(node_type)
        required: Set[str] = set()
        raw_creds = node.get("credentials")
        if isinstance(raw_creds, list):
            for cred in raw_creds:
                if not isinstance(cred, Mapping):
                    continue
                cred_name = str(cred.get("name") or "").strip()
                if not cred_name:
                    continue
                if bool(cred.get("required")):
                    required.add(cred_name)
        required_credentials_by_node_type[node_type] = required

    credential_types: Set[str] = set()
    supported_nodes_by_credential: Dict[str, Set[str]] = {}
    for credential in credentials_payload:
        cred_type = str(credential.get("name") or "").strip()
        if not cred_type:
            continue
        credential_types.add(cred_type)
        supported = set()
        raw_supported = credential.get("supportedNodes")
        if isinstance(raw_supported, list):
            for node_type in raw_supported:
                text = str(node_type or "").strip()
                if text:
                    supported.add(text)
        supported_nodes_by_credential[cred_type] = supported

    return Catalog(
        node_types=node_types,
        required_credentials_by_node_type=required_credentials_by_node_type,
        credential_types=credential_types,
        supported_nodes_by_credential=supported_nodes_by_credential,
    )


def run_checks(
    response_text: str,
    *,
    case: CaseSpec,
    catalog: Catalog,
) -> Dict[str, Any]:
    parse_result = parse_workflow_json(response_text)
    section_failures: Dict[str, List[str]] = {
        "json_parse_ok": list(parse_result.failures),
    }

    if not parse_result.ok or not parse_result.workflow:
        score, breakdown = compute_compliance_score(
            parse_ok=False,
            section_ratios={},
            section_failures=section_failures,
        )
        return {
            "parse_ok": False,
            "workflow": None,
            "extracted_json": parse_result.extracted_text,
            "compliance_score": score,
            "breakdown": breakdown,
            "failures": parse_result.failures,
            "trace": {
                "json_parse_ok": {
                    "failures": parse_result.failures,
                }
            },
        }

    workflow = parse_result.workflow
    schema_counts, names_to_types, edges = _check_workflow_min_schema(workflow)
    node_type_counts = _check_node_types_exist(workflow, catalog)
    creds_shape_counts = _check_credentials_shape_and_existence(workflow, catalog)
    creds_compat_counts = _check_credential_compatibility(workflow, catalog)
    req_counts = _check_case_requirements(case.requirements, workflow, names_to_types, edges)
    limits_counts = _check_case_limits(case, workflow)

    req_limits_total = req_counts.total + limits_counts.total
    req_limits_passed = req_counts.passed + limits_counts.passed
    req_limits_failures = [*req_counts.failures, *limits_counts.failures]
    req_limits_counts = CheckCounts(
        passed=req_limits_passed,
        total=req_limits_total,
        failures=req_limits_failures,
    )

    section_ratios = {
        "workflow_min_schema": schema_counts.ratio,
        "node_types_exist": node_type_counts.ratio,
        "credentials_shape_and_existence": creds_shape_counts.ratio,
        "credential_compatibility": creds_compat_counts.ratio,
        "requirements_and_limits": req_limits_counts.ratio,
    }
    section_failures.update(
        {
            "workflow_min_schema": schema_counts.failures,
            "node_types_exist": node_type_counts.failures,
            "credentials_shape_and_existence": creds_shape_counts.failures,
            "credential_compatibility": creds_compat_counts.failures,
            "requirements_and_limits": req_limits_counts.failures,
        }
    )

    score, breakdown = compute_compliance_score(
        parse_ok=True,
        section_ratios=section_ratios,
        section_failures=section_failures,
    )
    failures = [
        *schema_counts.failures,
        *node_type_counts.failures,
        *creds_shape_counts.failures,
        *creds_compat_counts.failures,
        *req_limits_counts.failures,
    ]
    return {
        "parse_ok": True,
        "workflow": workflow,
        "extracted_json": parse_result.extracted_text,
        "compliance_score": score,
        "breakdown": breakdown,
        "failures": failures,
        "trace": {
            "json_parse_ok": {"failures": []},
            "workflow_min_schema": _counts_to_trace(schema_counts),
            "node_types_exist": _counts_to_trace(node_type_counts),
            "credentials_shape_and_existence": _counts_to_trace(creds_shape_counts),
            "credential_compatibility": _counts_to_trace(creds_compat_counts),
            "requirements": _counts_to_trace(req_counts),
            "limits": _counts_to_trace(limits_counts),
            "requirements_and_limits": _counts_to_trace(req_limits_counts),
        },
    }


def parse_workflow_json(response_text: str) -> ParseResult:
    text = (response_text or "").strip()
    if not text:
        return ParseResult(
            ok=False,
            workflow=None,
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
    unique_candidates = []
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
        return ParseResult(ok=True, workflow=payload, extracted_text=candidate, failures=[])

    if not parse_errors:
        parse_errors.append("no JSON object could be extracted from response")
    return ParseResult(
        ok=False,
        workflow=None,
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


def _check_workflow_min_schema(
    workflow: Mapping[str, Any]
) -> Tuple[CheckCounts, Dict[str, str], List[Tuple[str, str]]]:
    failures: List[str] = []
    passed = 0
    total = 0

    nodes_raw = workflow.get("nodes")
    total += 1
    if isinstance(nodes_raw, list):
        passed += 1
    else:
        failures.append("workflow.nodes must be a list")
        nodes_raw = []

    connections_raw = workflow.get("connections")
    total += 1
    if isinstance(connections_raw, dict):
        passed += 1
    else:
        failures.append("workflow.connections must be an object")
        connections_raw = {}

    node_names: Set[str] = set()
    names_to_types: Dict[str, str] = {}
    for idx, node in enumerate(nodes_raw):
        if not isinstance(node, Mapping):
            total += 1
            failures.append(f"node[{idx}] must be an object")
            continue
        name = str(node.get("name") or "").strip()
        node_type = str(node.get("type") or "").strip()

        total += 1
        if name:
            passed += 1
        else:
            failures.append(f"node[{idx}] missing name")

        total += 1
        if node_type:
            passed += 1
        else:
            failures.append(f"node[{idx}] missing type")

        if name:
            total += 1
            if name in node_names:
                failures.append(f"duplicated node name: {name}")
            else:
                passed += 1
                node_names.add(name)
                names_to_types[name] = node_type

        if "position" in node:
            total += 1
            position = node.get("position")
            if _valid_position(position):
                passed += 1
            else:
                failures.append(f"node[{idx}] has invalid position")

    edges: List[Tuple[str, str]] = []
    for source_name, outputs in connections_raw.items():
        source_text = str(source_name)
        total += 1
        if source_text in node_names:
            passed += 1
        else:
            failures.append(f"connection source node not found: {source_text}")

        total += 1
        if isinstance(outputs, Mapping):
            passed += 1
        else:
            failures.append(f"connections[{source_text}] must be an object")
            continue

        for _, output_lists in outputs.items():
            total += 1
            if isinstance(output_lists, list):
                passed += 1
            else:
                failures.append(f"connections[{source_text}] output must be a list")
                continue

            for output_list in output_lists:
                total += 1
                if isinstance(output_list, list):
                    passed += 1
                else:
                    failures.append(f"connections[{source_text}] nested output must be a list")
                    continue

                for connection in output_list:
                    total += 1
                    if isinstance(connection, Mapping):
                        passed += 1
                    else:
                        failures.append(
                            f"connections[{source_text}] contains non-object connection"
                        )
                        continue
                    target_name = str(connection.get("node") or "").strip()
                    total += 1
                    if target_name and target_name in node_names:
                        passed += 1
                        edges.append((source_text, target_name))
                    else:
                        failures.append(
                            f"connection target node not found from {source_text}: {target_name or '<empty>'}"
                        )

    return CheckCounts(passed=passed, total=total, failures=failures), names_to_types, edges


def _check_node_types_exist(workflow: Mapping[str, Any], catalog: Catalog) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return CheckCounts(
            passed=0,
            total=1,
            failures=["cannot validate node types: workflow.nodes is empty or invalid"],
        )

    for idx, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        total += 1
        node_type = str(node.get("type") or "").strip()
        if not node_type:
            failures.append(f"node[{idx}] missing type for catalog validation")
            continue
        if node_type in catalog.node_types:
            passed += 1
        else:
            failures.append(f"unknown node type: {node_type}")

    if total == 0:
        total = 1
    return CheckCounts(passed=passed, total=total, failures=failures)


def _check_credentials_shape_and_existence(
    workflow: Mapping[str, Any], catalog: Catalog
) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return CheckCounts(
            passed=0,
            total=1,
            failures=["cannot validate credentials: workflow.nodes is invalid"],
        )

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_name = str(node.get("name") or "<unknown>")
        credentials = node.get("credentials")
        if credentials is None:
            continue

        total += 1
        if isinstance(credentials, Mapping):
            passed += 1
        else:
            failures.append(f"node '{node_name}' credentials must be an object")
            continue

        for credential_type, credential_value in credentials.items():
            credential_key = str(credential_type or "").strip()
            total += 1
            if credential_key:
                passed += 1
            else:
                failures.append(f"node '{node_name}' has empty credential key")

            total += 1
            if credential_key in catalog.credential_types:
                passed += 1
            else:
                failures.append(
                    f"node '{node_name}' references unknown credential type: {credential_key or '<empty>'}"
                )

            total += 1
            if isinstance(credential_value, Mapping):
                passed += 1
            else:
                failures.append(
                    f"node '{node_name}' credential '{credential_key}' must be an object"
                )

    if total == 0:
        return CheckCounts(passed=1, total=1, failures=[])
    return CheckCounts(passed=passed, total=total, failures=failures)


def _check_credential_compatibility(workflow: Mapping[str, Any], catalog: Catalog) -> CheckCounts:
    failures: List[str] = []
    passed = 0
    total = 0

    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return CheckCounts(
            passed=0,
            total=1,
            failures=["cannot validate credential compatibility: workflow.nodes is invalid"],
        )

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_name = str(node.get("name") or "<unknown>")
        node_type = str(node.get("type") or "").strip()
        credentials = node.get("credentials")
        credential_keys: Set[str] = set()
        if isinstance(credentials, Mapping):
            credential_keys = {
                str(credential_type or "").strip()
                for credential_type in credentials.keys()
                if str(credential_type or "").strip()
            }

        required_credentials = catalog.required_credentials_by_node_type.get(node_type, set())
        for required in sorted(required_credentials):
            total += 1
            if required in credential_keys:
                passed += 1
            else:
                failures.append(
                    f"node '{node_name}' ({node_type}) is missing required credential: {required}"
                )

        for credential_type in sorted(credential_keys):
            supported_nodes = catalog.supported_nodes_by_credential.get(credential_type, set())
            if not supported_nodes:
                continue
            total += 1
            if node_type in supported_nodes:
                passed += 1
            else:
                failures.append(
                    f"credential '{credential_type}' is not compatible with node type '{node_type}'"
                )

    if total == 0:
        return CheckCounts(passed=1, total=1, failures=[])
    return CheckCounts(passed=passed, total=total, failures=failures)


def _check_case_requirements(
    requirements: Sequence[RequirementSpec],
    workflow: Mapping[str, Any],
    names_to_types: Mapping[str, str],
    edges: Sequence[Tuple[str, str]],
) -> CheckCounts:
    if not requirements:
        return CheckCounts(passed=1, total=1, failures=[])

    failures: List[str] = []
    passed = 0
    total = 0
    nodes = workflow.get("nodes")
    nodes_list = nodes if isinstance(nodes, list) else []

    for idx, requirement in enumerate(requirements, start=1):
        total += 1
        ok, message = _evaluate_requirement(
            requirement=requirement,
            nodes=nodes_list,
            names_to_types=names_to_types,
            edges=edges,
        )
        if ok:
            passed += 1
            continue
        failures.append(f"requirement[{idx}] {requirement.type}: {message}")

    return CheckCounts(passed=passed, total=total, failures=failures)


def _evaluate_requirement(
    *,
    requirement: RequirementSpec,
    nodes: Sequence[Any],
    names_to_types: Mapping[str, str],
    edges: Sequence[Tuple[str, str]],
) -> Tuple[bool, str]:
    if requirement.type == "must_include_node_type":
        expected_type = str(requirement.value).strip()
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            if str(node.get("type") or "").strip() == expected_type:
                return True, ""
        return False, f"node type '{expected_type}' was not found"

    if requirement.type == "must_include_keyword_in_node_params":
        value = requirement.value if isinstance(requirement.value, Mapping) else {}
        key = str(value.get("key") or "").strip()
        expected_equals = value.get("equals", None)
        expected_contains = value.get("contains", None)
        found_values: List[Any] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            params = node.get("parameters")
            if isinstance(params, Mapping):
                found_values.extend(_collect_values_for_key(params, key))

        if not found_values:
            return False, f"parameter key '{key}' was not found in any node parameters"
        if "equals" in value:
            if any(item == expected_equals for item in found_values):
                return True, ""
            return False, f"key '{key}' did not match equals={expected_equals!r}"
        if "contains" in value:
            expected_text = str(expected_contains)
            if any(expected_text in str(item) for item in found_values):
                return True, ""
            return False, f"key '{key}' did not contain {expected_text!r}"
        return True, ""

    if requirement.type == "must_have_schedule_daily_at":
        target_time = str(requirement.value)
        if _has_daily_schedule_at(nodes, target_time):
            return True, ""
        return False, f"no schedule/cron node found with daily time {target_time}"

    if requirement.type == "must_have_connection":
        value = requirement.value if isinstance(requirement.value, Mapping) else {}
        from_name = str(value.get("from") or "").strip()
        to_name = str(value.get("to") or "").strip()
        from_type = str(value.get("from_type") or "").strip()
        to_type = str(value.get("to_type") or "").strip()
        for source_name, target_name in edges:
            source_type = names_to_types.get(source_name, "")
            target_type = names_to_types.get(target_name, "")
            if from_name and to_name:
                if source_name == from_name and target_name == to_name:
                    return True, ""
            if from_type and to_type:
                if source_type == from_type and target_type == to_type:
                    return True, ""
        if from_name and to_name:
            return False, f"connection {from_name} -> {to_name} was not found"
        return False, f"connection {from_type} -> {to_type} was not found"

    return False, "unsupported requirement type"


def _check_case_limits(case: CaseSpec, workflow: Mapping[str, Any]) -> CheckCounts:
    limits = case.limits
    if limits is None or limits.max_nodes is None:
        return CheckCounts(passed=1, total=1, failures=[])

    nodes = workflow.get("nodes")
    node_count = len(nodes) if isinstance(nodes, list) else 0
    if node_count <= limits.max_nodes:
        return CheckCounts(passed=1, total=1, failures=[])
    return CheckCounts(
        passed=0,
        total=1,
        failures=[f"max_nodes exceeded: {node_count} > {limits.max_nodes}"],
    )


def _collect_values_for_key(payload: Mapping[str, Any], key: str) -> List[Any]:
    values: List[Any] = []

    def _walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for k, v in item.items():
                if k == key:
                    values.append(v)
                _walk(v)
            return
        if isinstance(item, list):
            for child in item:
                _walk(child)

    _walk(payload)
    return values


def _has_daily_schedule_at(nodes: Sequence[Any], target_time: str) -> bool:
    normalized_target = _normalize_time_text(target_time)
    if not normalized_target:
        return False

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_type = str(node.get("type") or "").lower()
        if "schedule" not in node_type and "cron" not in node_type:
            continue
        params = node.get("parameters")
        for candidate in _extract_time_candidates(params):
            if candidate == normalized_target:
                return True
    return False


def _extract_time_candidates(value: Any) -> Set[str]:
    result: Set[str] = set()

    def _walk(item: Any) -> None:
        if isinstance(item, Mapping):
            hour_candidates = _extract_hour_values(item)
            minute_candidates = _extract_minute_values(item)
            for hour in hour_candidates:
                for minute in minute_candidates:
                    text = _normalize_hour_minute(hour, minute)
                    if text:
                        result.add(text)

            for key in ("cronExpression", "cron", "expression", "customCron"):
                raw = item.get(key)
                if isinstance(raw, str):
                    for candidate in _times_from_text(raw):
                        result.add(candidate)

            for child in item.values():
                _walk(child)
            return

        if isinstance(item, list):
            for child in item:
                _walk(child)
            return

        if isinstance(item, str):
            for candidate in _times_from_text(item):
                result.add(candidate)

    _walk(value)
    return result


def _times_from_text(text: str) -> Set[str]:
    result: Set[str] = set()
    for match in _TIME_RE.finditer(text):
        hour = int(match.group(1))
        minute = int(match.group(2))
        normalized = _normalize_hour_minute(hour, minute)
        if normalized:
            result.add(normalized)

    cron_parts = [part for part in text.strip().split() if part]
    if len(cron_parts) >= 5:
        minute = cron_parts[0]
        hour = cron_parts[1]
        if minute.isdigit() and hour.isdigit():
            normalized = _normalize_hour_minute(int(hour), int(minute))
            if normalized:
                result.add(normalized)
    return result


def _extract_hour_values(item: Mapping[str, Any]) -> List[Any]:
    values: List[Any] = []
    for key in ("hour", "hours", "triggerAtHour"):
        if key in item:
            values.append(item[key])
    return values


def _extract_minute_values(item: Mapping[str, Any]) -> List[Any]:
    values: List[Any] = []
    for key in ("minute", "minutes", "triggerAtMinute"):
        if key in item:
            values.append(item[key])
    return values


def _normalize_hour_minute(hour_value: Any, minute_value: Any) -> Optional[str]:
    try:
        hour = int(str(hour_value).strip())
        minute = int(str(minute_value).strip())
    except Exception:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _normalize_time_text(value: str) -> Optional[str]:
    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        return None
    return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"


def _valid_position(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        float(value[0])
        float(value[1])
        return True
    except Exception:
        return False


def _counts_to_trace(counts: CheckCounts) -> Dict[str, Any]:
    return {
        "passed": counts.passed,
        "total": counts.total,
        "ratio": round(counts.ratio, 6),
        "failures": counts.failures,
    }
