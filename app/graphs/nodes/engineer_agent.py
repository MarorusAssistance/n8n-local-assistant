from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitecturePlan,
    BlockedNode,
    EntryIntent,
    ImplementationQueueItem,
    ImplementationStatus,
    ImplementedNode,
    MissingUserInput,
    ProposedNode,
    RequiredCredential,
    VariableDefinition,
    WorkflowDraft,
    WorkflowDraftConnection,
    WorkflowDraftNode,
    WorkflowVersion,
    WorkflowContext,
)
from ...workflow.n8n_client import N8NClient, N8NClientError
from ..multi_agent_state import MultiAgentGraphState

logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")

_N8N_WORKFLOW_READ_ONLY_FIELDS = {
    "active",
    "id",
    "createdAt",
    "updatedAt",
    "versionId",
}

_NODE_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "n8n-nodes-base.webhook": ["path"],
    "n8n-nodes-base.httpRequest": ["url"],
    "n8n-nodes-base.googleSheets": ["sheetName"],
    "n8n-nodes-base.postgres": ["operation"],
    "n8n-nodes-base.mysql": ["operation"],
    "n8n-nodes-base.slack": ["channel", "text"],
    "n8n-nodes-base.if": ["conditions"],
    "n8n-nodes-base.switch": ["rules"],
    "n8n-nodes-base.set": ["values"],
}

_DEFAULT_PARAM_VALUES: Dict[str, str] = {
    "path": "incoming-event",
    "httpMethod": "POST",
    "method": "POST",
    "operation": "append",
}


def _short_type(node_type: str) -> str:
    value = str(node_type or "").strip()
    if not value:
        return "node"
    return value.split(".")[-1]


def _safe_text(value: Any, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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


def _normalize_model(value: Any, cls: Any) -> Optional[Any]:
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        try:
            return cls.model_validate(value)
        except Exception:
            return None
    return None


def _normalize_list(values: Any, cls: Any) -> List[Any]:
    if not isinstance(values, list):
        return []
    normalized: List[Any] = []
    for value in values:
        model = _normalize_model(value, cls)
        if model is not None:
            normalized.append(model)
    return normalized


def _normalize_status(value: Any) -> Optional[ImplementationStatus]:
    if isinstance(value, ImplementationStatus):
        return value
    if isinstance(value, str):
        try:
            return ImplementationStatus(value)
        except ValueError:
            return None
    return None


def _build_stage_dependency_map(plan: ArchitecturePlan) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for item in plan.data_flow:
        target = item.target_stage_id
        source = item.source_stage_id
        if not target or not source:
            continue
        current = output.setdefault(target, [])
        if source not in current:
            current.append(source)
    return output


def _derive_proposed_nodes(
    *,
    plan: ArchitecturePlan,
    workflow_context: Optional[WorkflowContext],
    provided_nodes: List[ProposedNode],
) -> List[ProposedNode]:
    if provided_nodes:
        return provided_nodes

    stage_ids = [stage.id for stage in plan.stages]
    proposed: List[ProposedNode] = []
    for idx, requirement in enumerate(plan.required_nodes, start=1):
        stage_id = None
        if stage_ids:
            stage_id = stage_ids[min(idx - 1, len(stage_ids) - 1)]
        proposed.append(
            ProposedNode(
                node_id=f"pn_{idx}",
                node_type=requirement.node_type,
                stage_id=stage_id,
                purpose=requirement.why_required or f"Implement node '{requirement.node_type}'.",
                usage_mode=requirement.usage_mode,
                usable_as_tool=requirement.usable_as_tool,
                has_main_input=requirement.has_main_input,
                input_connection_types=list(requirement.input_connection_types),
            )
        )

    if proposed:
        return proposed

    required_types = list(workflow_context.required_node_types) if workflow_context else []
    for idx, node_type in enumerate(required_types, start=1):
        proposed.append(
            ProposedNode(
                node_id=f"pn_{idx}",
                node_type=node_type,
                purpose=f"Derived from workflow context required node type '{node_type}'.",
            )
        )
    return proposed


def _derive_required_credentials(
    *,
    provided_credentials: List[RequiredCredential],
) -> List[RequiredCredential]:
    return provided_credentials


def _build_queue(
    *,
    plan: ArchitecturePlan,
    proposed_nodes: List[ProposedNode],
    existing_queue: List[ImplementationQueueItem],
) -> List[ImplementationQueueItem]:
    if existing_queue:
        return existing_queue

    stage_dependencies = _build_stage_dependency_map(plan)
    by_stage: Dict[str, List[ProposedNode]] = {}
    for item in proposed_nodes:
        if not item.stage_id:
            continue
        by_stage.setdefault(item.stage_id, []).append(item)

    queue: List[ImplementationQueueItem] = []
    for idx, node in enumerate(proposed_nodes):
        dependencies = list(node.depends_on)
        if not dependencies and node.stage_id:
            for source_stage in stage_dependencies.get(node.stage_id, []):
                source_nodes = by_stage.get(source_stage, [])
                if source_nodes:
                    dependencies.append(source_nodes[0].node_id)
        if not dependencies and idx > 0:
            dependencies.append(proposed_nodes[idx - 1].node_id)
        queue.append(
            ImplementationQueueItem(
                queue_id=node.node_id,
                node_type=node.node_type,
                stage_id=node.stage_id,
                purpose=node.purpose,
                dependencies=dependencies,
                expected_inputs=list(node.expected_inputs),
                expected_outputs=list(node.expected_outputs),
                status="pending",
            )
        )
    return queue


def _required_params(node_type: str) -> List[str]:
    if node_type in _NODE_REQUIRED_PARAMS:
        return list(_NODE_REQUIRED_PARAMS[node_type])
    return []


def _infer_param_value(
    *,
    param_name: str,
    queue_item: ImplementationQueueItem,
    plan: ArchitecturePlan,
) -> Optional[str]:
    if param_name in _DEFAULT_PARAM_VALUES:
        return _DEFAULT_PARAM_VALUES[param_name]
    if param_name == "values":
        return "mapped_fields"
    if param_name == "conditions":
        return "business_conditions"
    if param_name == "rules":
        return "routing_rules"
    if param_name == "sheetName" and "sheet" in plan.title.lower():
        return "Sheet1"
    if param_name == "url":
        return None
    _ = queue_item
    return None


def _extract_json_pairs(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return out
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        if value is None:
            continue
        out[str(key)] = str(value)
    return out


def _extract_user_supplied_values(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not text:
        return out

    out.update(_extract_json_pairs(text))
    for match in re.finditer(r"([A-Za-z0-9_.:\-]+)\s*(?:=|:)\s*([^\n,;]+)", text):
        key = str(match.group(1) or "").strip()
        value = str(match.group(2) or "").strip().strip("\"'")
        if not key or not value:
            continue
        out[key] = value
    return out


def _lookup_user_value(input_key: str, provided_values: Dict[str, str]) -> Optional[str]:
    if input_key in provided_values:
        return provided_values[input_key]
    lowered = input_key.lower()
    for key, value in provided_values.items():
        if key.lower() == lowered:
            return value
    suffix = input_key.split(":")[-1]
    if suffix in provided_values:
        return provided_values[suffix]
    for key, value in provided_values.items():
        if key.lower() == suffix.lower():
            return value
    return None


def _missing_input(
    *,
    category: str,
    queue_id: str,
    key_name: str,
    reason: str,
) -> MissingUserInput:
    input_key = f"{category}:{queue_id}:{key_name}"
    if category == "credential":
        question = (
            f"Provide credential reference for '{key_name}' to configure node '{queue_id}'."
        )
    elif category == "mapping":
        question = (
            f"Provide mapping definition for '{key_name}' required by node '{queue_id}'."
        )
    elif category == "handoff":
        question = reason
    else:
        question = f"Provide value for '{key_name}' required by node '{queue_id}'."
    return MissingUserInput(
        input_id=input_key,
        input_key=input_key,
        missing_item=key_name,
        reason=reason,
        blocking_node_id=queue_id,
        category=category,
        question=_safe_text(question, max_chars=240),
    )


def _merge_missing_details(
    existing: List[MissingUserInput],
    additions: Iterable[MissingUserInput],
) -> List[MissingUserInput]:
    by_id: Dict[str, MissingUserInput] = {item.input_id: item for item in existing}
    for item in additions:
        by_id[item.input_id] = item
    return list(by_id.values())


def _queue_dependencies_met(
    item: ImplementationQueueItem,
    queue_by_id: Dict[str, ImplementationQueueItem],
) -> bool:
    for dep in item.dependencies:
        dep_item = queue_by_id.get(dep)
        if dep_item is None:
            return False
        if dep_item.status != "implemented":
            return False
    return True


def _normalize_workflow_draft(
    value: Any,
    *,
    plan: ArchitecturePlan,
) -> WorkflowDraft:
    model = _normalize_model(value, WorkflowDraft)
    if model is not None:
        return model
    return WorkflowDraft(
        name=plan.title or "Workflow Draft",
        use_case_id=plan.use_case_id,
        summary=plan.workflow_summary,
        nodes=[],
        connections=[],
        metadata={},
    )


def _upsert_draft_node(draft: WorkflowDraft, node: WorkflowDraftNode) -> None:
    for idx, current in enumerate(draft.nodes):
        if current.node_id == node.node_id:
            draft.nodes[idx] = node
            return
    draft.nodes.append(node)


def _append_version(
    versions: List[WorkflowVersion],
    draft: WorkflowDraft,
    *,
    reason: str,
) -> int:
    next_version = (versions[-1].version + 1) if versions else 1
    versions.append(
        WorkflowVersion(
            version=next_version,
            reason=reason,
            workflow_draft=draft.model_copy(deep=True),
            implemented_node_count=len(draft.nodes),
        )
    )
    return next_version


def _queue_node_id_map(draft: WorkflowDraft) -> Dict[str, str]:
    value = draft.metadata.get("queue_node_map")
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    return {}


def _set_queue_node_id_map(draft: WorkflowDraft, mapping: Dict[str, str]) -> None:
    draft.metadata["queue_node_map"] = {str(key): str(value) for key, value in mapping.items()}


def _resolved_inputs(draft: WorkflowDraft) -> Dict[str, str]:
    value = draft.metadata.get("resolved_inputs")
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    return {}


def _set_resolved_inputs(draft: WorkflowDraft, values: Dict[str, str]) -> None:
    draft.metadata["resolved_inputs"] = {str(key): str(value) for key, value in values.items()}


def _update_variable_registry(
    registry: List[VariableDefinition],
    *,
    queue_item: ImplementationQueueItem,
    node_id: str,
    queue: List[ImplementationQueueItem],
) -> List[VariableDefinition]:
    by_key: Dict[Tuple[str, str], VariableDefinition] = {
        (item.name, item.origin_node_id): item for item in registry
    }
    downstream = [
        candidate.queue_id
        for candidate in queue
        if queue_item.queue_id in candidate.dependencies
    ]
    outputs = list(queue_item.expected_outputs) or [f"{_short_type(queue_item.node_type)}_output"]
    for output_name in outputs:
        variable_name = output_name if "." in output_name else f"{queue_item.queue_id}.{output_name}"
        definition = VariableDefinition(
            name=variable_name,
            origin_node_id=node_id,
            destination_node_ids=downstream,
            semantic_meaning=f"Output from {queue_item.queue_id} for downstream workflow steps.",
            expected_format="unknown",
            mapping_notes=(
                f"Mapped from node type '{queue_item.node_type}' to dependent nodes."
                if downstream
                else "Terminal output in current draft."
            ),
        )
        by_key[(definition.name, definition.origin_node_id)] = definition
    return list(by_key.values())


def _build_connections(
    queue: List[ImplementationQueueItem],
    queue_node_ids: Dict[str, str],
) -> List[WorkflowDraftConnection]:
    output: List[WorkflowDraftConnection] = []
    seen: set[Tuple[str, str]] = set()

    implemented = [item for item in queue if item.status == "implemented"]
    for idx, item in enumerate(implemented):
        target_node_id = queue_node_ids.get(item.queue_id)
        if not target_node_id:
            continue
        sources = list(item.dependencies)
        if not sources and idx > 0:
            sources = [implemented[idx - 1].queue_id]
        for source_queue_id in sources:
            source_node_id = queue_node_ids.get(source_queue_id)
            if not source_node_id:
                continue
            key = (source_node_id, target_node_id)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                WorkflowDraftConnection(
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                )
            )
    return output


def _draft_to_final_workflow_json(draft: WorkflowDraft) -> Dict[str, Any]:
    node_name_by_id = {node.node_id: node.name for node in draft.nodes}
    nodes_payload: List[Dict[str, Any]] = []
    for idx, node in enumerate(draft.nodes):
        parameters: Dict[str, Any] = {}
        parameters.update(node.parameters_inferred)
        parameters.update(node.parameters_known)
        node_payload: Dict[str, Any] = {
            "id": node.node_id,
            "name": node.name,
            "type": node.node_type,
            "typeVersion": 1,
            "position": node.position or [240 * idx, 300],
            "parameters": parameters,
        }
        if node.credential_refs:
            node_payload["credentials"] = {
                name: {"id": ref, "name": ref}
                for name, ref in node.credential_refs.items()
            }
        nodes_payload.append(node_payload)

    connections_payload: Dict[str, Any] = {}
    for connection in draft.connections:
        source_name = node_name_by_id.get(connection.source_node_id)
        target_name = node_name_by_id.get(connection.target_node_id)
        if not source_name or not target_name:
            continue
        entry = connections_payload.setdefault(source_name, {"main": [[]]})
        main_outputs = entry.get("main")
        if not isinstance(main_outputs, list) or not main_outputs:
            entry["main"] = [[]]
            main_outputs = entry["main"]
        output_bucket = main_outputs[0]
        if not isinstance(output_bucket, list):
            main_outputs[0] = []
            output_bucket = main_outputs[0]
        output_bucket.append({"node": target_name, "type": "main", "index": 0})

    return {
        "name": draft.name,
        "active": False,
        "settings": {},
        "nodes": nodes_payload,
        "connections": connections_payload,
    }


def _merge_engineer_missing_inputs(
    existing: List[MissingUserInput],
    *,
    provided_values: Dict[str, str],
) -> Tuple[List[MissingUserInput], Dict[str, str]]:
    resolved: Dict[str, str] = {}
    remaining: List[MissingUserInput] = []
    for item in existing:
        value = _lookup_user_value(item.input_key, provided_values)
        if value:
            resolved[item.input_key] = value
            continue
        remaining.append(item)
    return remaining, resolved


def _single_freeform_answer(text: str) -> str:
    value = _safe_text(text, max_chars=280)
    if ":" in value and "=" in value:
        return ""
    return value


def _runtime_flag(state: MultiAgentGraphState, key: str, default: bool = False) -> bool:
    runtime_context = state.get("runtime_context")
    if not isinstance(runtime_context, dict):
        runtime_context = state.get("workflow_context")
    if not isinstance(runtime_context, dict):
        return default
    value = runtime_context.get(key)
    if isinstance(value, bool):
        return value
    return default


def _persist_fields(
    *,
    active_workflow_id: Optional[str],
    active_workflow_name: Optional[str],
    active_workflow_url: Optional[str],
    workflow_persisted: bool,
    workflow_persist_action: Optional[str],
    workflow_api_sync_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "active_workflow_id": active_workflow_id,
        "active_workflow_name": active_workflow_name,
        "active_workflow_url": active_workflow_url,
        "workflow_persisted": workflow_persisted,
        "workflow_persist_action": workflow_persist_action,
        "workflow_api_sync_result": workflow_api_sync_result,
    }


def _default_workflow_url(base_url: str, workflow_id: Optional[str]) -> Optional[str]:
    if not workflow_id:
        return None
    return f"{str(base_url or '').rstrip('/')}/workflow/{workflow_id}"


def _persist_workflow_candidate(
    *,
    state: MultiAgentGraphState,
    final_workflow_json: Dict[str, Any],
    workflow_name: str,
    request_id: Optional[str],
) -> Dict[str, Any]:
    current_id_raw = state.get("active_workflow_id")
    current_name_raw = state.get("active_workflow_name")
    current_url_raw = state.get("active_workflow_url")
    current_id = str(current_id_raw).strip() if isinstance(current_id_raw, str) else None
    current_name = str(current_name_raw).strip() if isinstance(current_name_raw, str) else None
    current_url = str(current_url_raw).strip() if isinstance(current_url_raw, str) else None
    workflow_name_value = str(workflow_name or "").strip() or current_name

    if not _runtime_flag(state, "persist_to_n8n", default=False):
        return _persist_fields(
            active_workflow_id=current_id,
            active_workflow_name=workflow_name_value,
            active_workflow_url=current_url,
            workflow_persisted=False,
            workflow_persist_action="skipped",
            workflow_api_sync_result={
                "ok": False,
                "skipped": True,
                "reason": "persist_to_n8n disabled",
            },
        )

    client = N8NClient()
    payload = {
        key: value
        for key, value in dict(final_workflow_json).items()
        if key not in _N8N_WORKFLOW_READ_ONLY_FIELDS
    }
    if not isinstance(payload.get("settings"), dict):
        payload["settings"] = {}
    if workflow_name_value and not payload.get("name"):
        payload["name"] = workflow_name_value

    action = "updated" if current_id else "created"
    try:
        if current_id:
            response = client.update_workflow(current_id, payload)
        else:
            response = client.create_workflow(payload)
    except N8NClientError as exc:
        logger.warning(
            "engineer workflow persistence failed: request_id=%s action=%s status=%s error=%s",
            request_id or "-",
            action,
            exc.status_code,
            str(exc),
        )
        return _persist_fields(
            active_workflow_id=current_id,
            active_workflow_name=workflow_name_value,
            active_workflow_url=current_url,
            workflow_persisted=False,
            workflow_persist_action=f"{action}_failed",
            workflow_api_sync_result={
                "ok": False,
                "action": action,
                "error": str(exc),
                "status_code": exc.status_code,
            },
        )

    response_data = response.get("data") if isinstance(response.get("data"), dict) else {}
    persisted_id = (
        str(response.get("id")).strip()
        if isinstance(response.get("id"), (str, int))
        else (
            str(response_data.get("id")).strip()
            if isinstance(response_data.get("id"), (str, int))
            else current_id
        )
    )
    persisted_name = (
        str(response.get("name")).strip()
        if isinstance(response.get("name"), str) and str(response.get("name")).strip()
        else (
            str(response_data.get("name")).strip()
            if isinstance(response_data.get("name"), str) and str(response_data.get("name")).strip()
            else workflow_name_value
        )
    )
    persisted_url = (
        str(response.get("url")).strip()
        if isinstance(response.get("url"), str) and str(response.get("url")).strip()
        else (
            str(response_data.get("url")).strip()
            if isinstance(response_data.get("url"), str) and str(response_data.get("url")).strip()
            else _default_workflow_url(client.base_url, persisted_id)
        )
    )
    return _persist_fields(
        active_workflow_id=persisted_id,
        active_workflow_name=persisted_name,
        active_workflow_url=persisted_url,
        workflow_persisted=True,
        workflow_persist_action=action,
        workflow_api_sync_result={
            "ok": True,
            "action": action,
            "id": persisted_id,
        },
    )


def engineer_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    model, request_id = _runtime_context(state)
    _ = model
    entry_intent = state.get("entry_intent")
    routing_signals = list(state.get("routing_signals") or [])
    if "entered_engineer_agent" not in routing_signals:
        routing_signals.append("entered_engineer_agent")
    engineer_notes = list(state.get("engineer_notes") or [])

    architecture_plan = _normalize_model(state.get("architecture_plan"), ArchitecturePlan)
    workflow_context = _normalize_model(state.get("workflow_context"), WorkflowContext)
    provided_nodes = _normalize_list(state.get("proposed_nodes"), ProposedNode)
    provided_credentials = _normalize_list(state.get("required_credentials"), RequiredCredential)
    queue = _normalize_list(state.get("node_implementation_queue"), ImplementationQueueItem)
    implemented_nodes = _normalize_list(state.get("implemented_nodes"), ImplementedNode)
    blocked_nodes = _normalize_list(state.get("blocked_nodes"), BlockedNode)
    variable_registry = _normalize_list(state.get("variable_registry"), VariableDefinition)
    missing_details = _normalize_list(state.get("missing_user_input_details"), MissingUserInput)
    workflow_versions = _normalize_list(state.get("workflow_versions"), WorkflowVersion)
    missing_user_inputs = list(state.get("missing_user_inputs") or [])
    implementation_status = _normalize_status(state.get("implementation_status"))
    final_workflow_json = (
        dict(state.get("final_workflow_json"))
        if isinstance(state.get("final_workflow_json"), dict)
        else {}
    )
    active_workflow_id = (
        str(state.get("active_workflow_id")).strip()
        if isinstance(state.get("active_workflow_id"), str) and str(state.get("active_workflow_id")).strip()
        else None
    )
    active_workflow_name = (
        str(state.get("active_workflow_name")).strip()
        if isinstance(state.get("active_workflow_name"), str) and str(state.get("active_workflow_name")).strip()
        else None
    )
    active_workflow_url = (
        str(state.get("active_workflow_url")).strip()
        if isinstance(state.get("active_workflow_url"), str) and str(state.get("active_workflow_url")).strip()
        else None
    )
    workflow_persisted = bool(state.get("workflow_persisted", False))
    workflow_persist_action = (
        str(state.get("workflow_persist_action")).strip()
        if isinstance(state.get("workflow_persist_action"), str) and str(state.get("workflow_persist_action")).strip()
        else None
    )
    workflow_api_sync_result = (
        dict(state.get("workflow_api_sync_result"))
        if isinstance(state.get("workflow_api_sync_result"), dict)
        else {}
    )
    persist_payload = _persist_fields(
        active_workflow_id=active_workflow_id,
        active_workflow_name=active_workflow_name,
        active_workflow_url=active_workflow_url,
        workflow_persisted=workflow_persisted,
        workflow_persist_action=workflow_persist_action,
        workflow_api_sync_result=workflow_api_sync_result,
    )

    resume_requested = bool(state.get("resume_requested"))
    user_query = str(state.get("user_query") or "").strip()

    if architecture_plan is None:
        if entry_intent == EntryIntent.workflow_edit_request:
            handoff_missing = _missing_input(
                category="handoff",
                queue_id="engineer_handoff",
                key_name="architecture_plan",
                reason=(
                    "Missing product-manager handoff. Please provide architecture plan, "
                    "proposed nodes, and required credentials to continue implementation."
                ),
            )
            missing_details = _merge_missing_details(missing_details, [handoff_missing])
            missing_user_inputs = [item.question for item in missing_details]
            implementation_status = ImplementationStatus.blocked_waiting_user
            routing_signals.append("engineer_blocked_waiting_user")
            engineer_notes.append("Engineer blocked: PM handoff missing for edit request.")
            return {
                "current_stage": "engineer_agent",
                "target_stage": None,
                "implementation_status": implementation_status,
                "missing_user_inputs": missing_user_inputs,
                "missing_user_input_details": missing_details,
                "routing_signals": routing_signals,
                "engineer_notes": engineer_notes,
                "final_workflow_json": {},
                "workflow_draft": None,
                "workflow_versions": workflow_versions,
                "node_implementation_queue": queue,
                "implemented_nodes": implemented_nodes,
                "blocked_nodes": blocked_nodes,
                "variable_registry": variable_registry,
                "proposed_nodes": provided_nodes,
                "required_credentials": provided_credentials,
                "resume_requested": False,
                **persist_payload,
            }

        implementation_status = ImplementationStatus.failed
        engineer_notes.append("Engineer failed: architecture_plan is required.")
        return {
            "current_stage": "engineer_agent",
            "target_stage": None,
            "implementation_status": implementation_status,
            "missing_user_inputs": missing_user_inputs,
            "missing_user_input_details": missing_details,
            "routing_signals": routing_signals,
            "engineer_notes": engineer_notes,
            "final_workflow_json": {},
            "workflow_draft": None,
            "workflow_versions": workflow_versions,
            "node_implementation_queue": queue,
            "implemented_nodes": implemented_nodes,
            "blocked_nodes": blocked_nodes,
            "variable_registry": variable_registry,
            "proposed_nodes": provided_nodes,
            "required_credentials": provided_credentials,
            "resume_requested": False,
            **persist_payload,
        }

    proposed_nodes = _derive_proposed_nodes(
        plan=architecture_plan,
        workflow_context=workflow_context,
        provided_nodes=provided_nodes,
    )
    required_credentials = _derive_required_credentials(
        provided_credentials=provided_credentials,
    )

    if not proposed_nodes:
        missing_details = _merge_missing_details(
            missing_details,
            [
                _missing_input(
                    category="handoff",
                    queue_id="engineer_handoff",
                    key_name="proposed_nodes",
                    reason=(
                        "No proposed nodes were provided by handoff. "
                        "Please define at least one required node to implement."
                    ),
                )
            ],
        )
        missing_user_inputs = [item.question for item in missing_details]
        implementation_status = ImplementationStatus.blocked_waiting_user
        routing_signals.append("engineer_blocked_waiting_user")
        engineer_notes.append("Engineer blocked: no proposed nodes available.")
        return {
            "current_stage": "engineer_agent",
            "target_stage": None,
            "implementation_status": implementation_status,
            "missing_user_inputs": missing_user_inputs,
            "missing_user_input_details": missing_details,
            "routing_signals": routing_signals,
            "engineer_notes": engineer_notes,
            "final_workflow_json": {},
            "workflow_draft": None,
            "workflow_versions": workflow_versions,
            "node_implementation_queue": queue,
            "implemented_nodes": implemented_nodes,
            "blocked_nodes": blocked_nodes,
            "variable_registry": variable_registry,
            "proposed_nodes": proposed_nodes,
            "required_credentials": required_credentials,
            "resume_requested": False,
            **persist_payload,
        }

    queue = _build_queue(
        plan=architecture_plan,
        proposed_nodes=proposed_nodes,
        existing_queue=queue,
    )
    queue_by_id = {item.queue_id: item for item in queue}

    for implemented in implemented_nodes:
        queue_item = queue_by_id.get(implemented.queue_id)
        if queue_item and queue_item.status != "implemented":
            queue_item.status = "implemented"
    for blocked in blocked_nodes:
        queue_item = queue_by_id.get(blocked.queue_id)
        if queue_item and queue_item.status == "pending":
            queue_item.status = "blocked"

    workflow_draft = _normalize_workflow_draft(
        state.get("workflow_draft"),
        plan=architecture_plan,
    )
    if not workflow_versions:
        _append_version(workflow_versions, workflow_draft, reason="initialized engineer workflow draft")

    provided_values = _extract_user_supplied_values(user_query)
    remaining_missing, resolved_from_input = _merge_engineer_missing_inputs(
        missing_details,
        provided_values=provided_values,
    )
    missing_details = remaining_missing

    resolved_inputs = _resolved_inputs(workflow_draft)
    resolved_inputs.update(resolved_from_input)
    if resume_requested and not resolved_from_input and len(missing_details) == 1:
        freeform = _single_freeform_answer(user_query)
        if freeform:
            fallback_key = missing_details[0].input_key
            resolved_inputs[fallback_key] = freeform
            missing_details = []
            engineer_notes.append(
                f"Applied freeform user response to '{fallback_key}' during resume."
            )
    _set_resolved_inputs(workflow_draft, resolved_inputs)

    unresolved_missing_ids = {item.input_id for item in missing_details}
    active_blocked_nodes: List[BlockedNode] = []
    for blocked in blocked_nodes:
        active_ids = [item_id for item_id in blocked.missing_input_ids if item_id in unresolved_missing_ids]
        queue_item = queue_by_id.get(blocked.queue_id)
        if active_ids:
            active_blocked_nodes.append(
                BlockedNode(
                    queue_id=blocked.queue_id,
                    node_type=blocked.node_type,
                    reason=blocked.reason,
                    missing_input_ids=active_ids,
                )
            )
            if queue_item is not None and queue_item.status != "implemented":
                queue_item.status = "blocked"
            continue
        if queue_item is not None and queue_item.status == "blocked":
            queue_item.status = "pending"
    blocked_nodes = active_blocked_nodes
    missing_user_inputs = [item.question for item in missing_details]

    queue_node_ids = _queue_node_id_map(workflow_draft)
    implementation_status = implementation_status or ImplementationStatus.ready
    if queue:
        implementation_status = ImplementationStatus.in_progress

    while True:
        pending_items = [item for item in queue if item.status == "pending"]
        if not pending_items:
            break

        progressed = False
        for queue_item in pending_items:
            if not _queue_dependencies_met(queue_item, queue_by_id):
                continue

            unresolved_inputs: List[MissingUserInput] = []
            known_params: Dict[str, Any] = {}
            inferred_params: Dict[str, Any] = {}
            credential_refs: Dict[str, str] = {}

            for param_name in _required_params(queue_item.node_type):
                input_key = f"parameter:{queue_item.queue_id}:{param_name}"
                value = _lookup_user_value(input_key, resolved_inputs)
                if value:
                    known_params[param_name] = value
                    continue
                inferred = _infer_param_value(
                    param_name=param_name,
                    queue_item=queue_item,
                    plan=architecture_plan,
                )
                if inferred is not None:
                    inferred_params[param_name] = inferred
                    continue
                unresolved_inputs.append(
                    _missing_input(
                        category="parameter",
                        queue_id=queue_item.queue_id,
                        key_name=param_name,
                        reason=(
                            f"Parameter '{param_name}' is required for node '{queue_item.node_type}' "
                            "and could not be inferred safely."
                        ),
                    )
                )

            credential_requirements = [
                item
                for item in required_credentials
                if (item.required_for == queue_item.queue_id) or (item.node_type == queue_item.node_type)
            ]
            for credential in credential_requirements:
                input_key = f"credential:{queue_item.queue_id}:{credential.credential_name}"
                value = (
                    _lookup_user_value(input_key, resolved_inputs)
                    or _lookup_user_value(credential.credential_key, resolved_inputs)
                    or _lookup_user_value(credential.credential_name, resolved_inputs)
                )
                if value:
                    credential_refs[credential.credential_name] = value
                    continue
                unresolved_inputs.append(
                    _missing_input(
                        category="credential",
                        queue_id=queue_item.queue_id,
                        key_name=credential.credential_name,
                        reason=(
                            f"Credential reference for '{credential.credential_name}' is required "
                            f"to configure node '{queue_item.node_type}'."
                        ),
                    )
                )

            if unresolved_inputs:
                queue_item.status = "blocked"
                blocked_nodes = [
                    item for item in blocked_nodes if item.queue_id != queue_item.queue_id
                ]
                blocked_nodes.append(
                    BlockedNode(
                        queue_id=queue_item.queue_id,
                        node_type=queue_item.node_type,
                        reason="Node implementation blocked by missing critical inputs.",
                        missing_input_ids=[item.input_id for item in unresolved_inputs],
                    )
                )
                missing_details = _merge_missing_details(missing_details, unresolved_inputs)
                missing_user_inputs = [item.question for item in missing_details]
                implementation_status = ImplementationStatus.blocked_waiting_user
                routing_signals.append("engineer_blocked_waiting_user")
                engineer_notes.append(
                    f"Blocked node '{queue_item.queue_id}' ({queue_item.node_type}) waiting for user input."
                )
                _append_version(
                    workflow_versions,
                    workflow_draft,
                    reason=f"blocked node {queue_item.queue_id} awaiting required inputs",
                )
                if workflow_context is not None:
                    workflow_context.unresolved_inputs = list(missing_user_inputs)
                    workflow_context.handoff_target = None
                trace_logger.info(
                    "engineer iteration blocked: request_id=%s node=%s missing=%d",
                    request_id or "-",
                    queue_item.queue_id,
                    len(unresolved_inputs),
                )
                return {
                    "current_stage": "engineer_agent",
                    "target_stage": None,
                    "implementation_status": implementation_status,
                    "workflow_draft": workflow_draft,
                    "workflow_versions": workflow_versions,
                    "node_implementation_queue": queue,
                    "implemented_nodes": implemented_nodes,
                    "blocked_nodes": blocked_nodes,
                    "variable_registry": variable_registry,
                    "missing_user_inputs": missing_user_inputs,
                    "missing_user_input_details": missing_details,
                    "routing_signals": routing_signals,
                    "engineer_notes": engineer_notes,
                    "final_workflow_json": {},
                    "proposed_nodes": proposed_nodes,
                    "required_credentials": required_credentials,
                    "workflow_context": workflow_context,
                    "resume_requested": False,
                    **persist_payload,
                }

            node_id = queue_node_ids.get(queue_item.queue_id) or queue_item.queue_id
            queue_node_ids[queue_item.queue_id] = node_id
            _set_queue_node_id_map(workflow_draft, queue_node_ids)
            position = [240 * max(0, len(workflow_draft.nodes)), 300]
            draft_node = WorkflowDraftNode(
                node_id=node_id,
                name=f"{_short_type(queue_item.node_type)}_{len(queue_node_ids)}",
                node_type=queue_item.node_type,
                purpose=queue_item.purpose,
                stage_id=queue_item.stage_id,
                parameters_known=known_params,
                parameters_inferred=inferred_params,
                parameters_unresolved=[],
                credential_refs=credential_refs,
                expected_inputs=list(queue_item.expected_inputs),
                expected_outputs=list(queue_item.expected_outputs)
                or [f"{_short_type(queue_item.node_type)}_output"],
                dependencies=list(queue_item.dependencies),
                position=position,
                notes=["implemented by engineer_agent iterative step"],
            )
            _upsert_draft_node(workflow_draft, draft_node)

            queue_item.status = "implemented"
            blocked_nodes = [item for item in blocked_nodes if item.queue_id != queue_item.queue_id]
            version = _append_version(
                workflow_versions,
                workflow_draft,
                reason=f"implemented node {queue_item.queue_id} ({queue_item.node_type})",
            )
            implemented_nodes.append(
                ImplementedNode(
                    queue_id=queue_item.queue_id,
                    node_id=node_id,
                    node_type=queue_item.node_type,
                    purpose=queue_item.purpose,
                    version=version,
                )
            )
            variable_registry = _update_variable_registry(
                variable_registry,
                queue_item=queue_item,
                node_id=node_id,
                queue=queue,
            )
            progressed = True

        if not progressed:
            implementation_status = ImplementationStatus.failed
            routing_signals.append("engineer_dependency_resolution_failed")
            engineer_notes.append(
                "Engineer failed due to unresolved queue dependencies without actionable user input."
            )
            break

    remaining_queue_items = [item for item in queue if item.status != "implemented"]
    if remaining_queue_items:
        if missing_details:
            implementation_status = ImplementationStatus.blocked_waiting_user
            routing_signals.append("engineer_blocked_waiting_user")
            engineer_notes.append(
                "Engineer paused: unresolved inputs remain for blocked nodes."
            )
        else:
            implementation_status = ImplementationStatus.failed
            routing_signals.append("engineer_incomplete_queue_failed")
            engineer_notes.append(
                "Engineer failed: queue contains unresolved items without explicit missing inputs."
            )

    if implementation_status == ImplementationStatus.failed:
        if workflow_context is not None:
            workflow_context.handoff_target = None
        return {
            "current_stage": "engineer_agent",
            "target_stage": None,
            "implementation_status": implementation_status,
            "workflow_draft": workflow_draft,
            "workflow_versions": workflow_versions,
            "node_implementation_queue": queue,
            "implemented_nodes": implemented_nodes,
            "blocked_nodes": blocked_nodes,
            "variable_registry": variable_registry,
            "missing_user_inputs": missing_user_inputs,
            "missing_user_input_details": missing_details,
            "routing_signals": routing_signals,
            "engineer_notes": engineer_notes,
            "final_workflow_json": final_workflow_json,
            "proposed_nodes": proposed_nodes,
            "required_credentials": required_credentials,
            "workflow_context": workflow_context,
            "resume_requested": False,
            **persist_payload,
        }

    if implementation_status == ImplementationStatus.blocked_waiting_user:
        if workflow_context is not None:
            workflow_context.handoff_target = None
            workflow_context.unresolved_inputs = list(missing_user_inputs)
        return {
            "current_stage": "engineer_agent",
            "target_stage": None,
            "implementation_status": implementation_status,
            "workflow_draft": workflow_draft,
            "workflow_versions": workflow_versions,
            "node_implementation_queue": queue,
            "implemented_nodes": implemented_nodes,
            "blocked_nodes": blocked_nodes,
            "variable_registry": variable_registry,
            "missing_user_inputs": missing_user_inputs,
            "missing_user_input_details": missing_details,
            "routing_signals": routing_signals,
            "engineer_notes": engineer_notes,
            "final_workflow_json": {},
            "proposed_nodes": proposed_nodes,
            "required_credentials": required_credentials,
            "workflow_context": workflow_context,
            "resume_requested": False,
            **persist_payload,
        }

    workflow_draft.connections = _build_connections(queue, queue_node_ids)
    final_workflow_json = _draft_to_final_workflow_json(workflow_draft)
    _append_version(workflow_versions, workflow_draft, reason="connected workflow graph and produced final candidate")

    persist_payload = _persist_workflow_candidate(
        state=state,
        final_workflow_json=final_workflow_json,
        workflow_name=workflow_draft.name,
        request_id=request_id,
    )
    if persist_payload.get("workflow_persisted"):
        action = str(persist_payload.get("workflow_persist_action") or "").strip() or "persisted"
        routing_signals.append(f"workflow_persist_{action}")
    elif persist_payload.get("workflow_persist_action") == "skipped":
        routing_signals.append("workflow_persist_skipped")
    else:
        routing_signals.append("workflow_persist_failed")

    implementation_status = ImplementationStatus.completed
    missing_details = []
    missing_user_inputs = []
    blocked_nodes = []
    if workflow_context is None:
        workflow_context = WorkflowContext(
            use_case_id=architecture_plan.use_case_id,
            planning_ready=True,
            handoff_target=AgentStage.qa_agent,
            required_node_types=[node.node_type for node in architecture_plan.required_nodes],
            unresolved_inputs=[],
            notes=[],
        )
    workflow_context.unresolved_inputs = []
    workflow_context.handoff_target = AgentStage.qa_agent
    if "handoff_ready_qa" not in routing_signals:
        routing_signals.append("handoff_ready_qa")
    engineer_notes.append("Engineer completed iterative workflow construction. QA handoff is ready.")

    trace_logger.info(
        "engineer completed: request_id=%s nodes=%d versions=%d",
        request_id or "-",
        len(workflow_draft.nodes),
        len(workflow_versions),
    )

    return {
        "current_stage": "engineer_agent",
        "target_stage": AgentStage.qa_agent,
        "implementation_status": implementation_status,
        "workflow_draft": workflow_draft,
        "workflow_versions": workflow_versions,
        "final_workflow_json": final_workflow_json,
        "node_implementation_queue": queue,
        "implemented_nodes": implemented_nodes,
        "blocked_nodes": blocked_nodes,
        "variable_registry": variable_registry,
        "missing_user_inputs": missing_user_inputs,
        "missing_user_input_details": missing_details,
        "routing_signals": routing_signals,
        "engineer_notes": engineer_notes,
        "proposed_nodes": proposed_nodes,
        "required_credentials": required_credentials,
        "workflow_context": workflow_context,
        "resume_requested": False,
        **persist_payload,
    }

