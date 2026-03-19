from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from ...config import settings
from ...db import query_definition_chunks_by_entity
from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    BlockedNode,
    EntryIntent,
    ImplementationQueueItem,
    ImplementationStatus,
    ImplementedNode,
    MissingUserInput,
    NodeRequirement,
    ProposedNode,
    RequiredCredential,
    VariableDefinition,
    WorkflowDraft,
    WorkflowDraftConnection,
    WorkflowDraftNode,
    WorkflowVersion,
    WorkflowContext,
)
from ...llm import get_langchain_chat_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event, emit_trace_event
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
_NODE_DEFINITION_SOURCE = settings.LINKED_DEFS_NODES_SOURCE or "n8n-nodes"
_CREDENTIAL_DEFINITION_SOURCE = settings.LINKED_DEFS_CREDENTIALS_SOURCE or "n8n-credentials"


class DeveloperParameterDefinition(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: str = ""
    field_type: Optional[str] = None
    required: bool = False
    default_value: Any = None
    scope: Optional[str] = None
    options_preview: List[str] = Field(default_factory=list)


class DeveloperCredentialDefinition(BaseModel):
    credential_type: str
    display_name: Optional[str] = None
    supported_nodes: List[str] = Field(default_factory=list)
    field_names: List[str] = Field(default_factory=list)
    summary: str = ""
    source_refs: List[str] = Field(default_factory=list)


class DeveloperNodeDefinition(BaseModel):
    node_type: str
    display_name: Optional[str] = None
    type_version: int = Field(default=1, ge=1)
    summary: str = ""
    parameter_schema: List[DeveloperParameterDefinition] = Field(default_factory=list)
    credential_types_required: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    raw_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class DeveloperMissingInputDecision(BaseModel):
    key_name: str
    category: str = "parameter"
    reason: str
    question: str


class DeveloperVariableOutput(BaseModel):
    name: str
    semantic_meaning: str
    expected_format: Optional[str] = None
    destination_queue_ids: List[str] = Field(default_factory=list)
    mapping_notes: Optional[str] = None


class NodeImplementationDecision(BaseModel):
    parameters_known: Dict[str, Any] = Field(default_factory=dict)
    parameters_inferred: Dict[str, Any] = Field(default_factory=dict)
    parameters_unresolved: List[str] = Field(default_factory=list)
    credential_refs: Dict[str, str] = Field(default_factory=dict)
    missing_inputs: List[DeveloperMissingInputDecision] = Field(default_factory=list)
    variable_outputs: List[DeveloperVariableOutput] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    can_apply: bool = False


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


def _safe_string_list(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _coerce_positive_int(value: Any, default: int = 1) -> int:
    candidates: List[Any] = []
    if isinstance(value, (list, tuple)):
        candidates.extend(list(value))
    else:
        candidates.append(value)

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, (int, float)):
            try:
                coerced = int(candidate)
            except (TypeError, ValueError, OverflowError):
                continue
            if coerced >= 1:
                return coerced
            continue
        text = str(candidate).strip()
        if not text:
            continue
        try:
            coerced = int(float(text))
        except (TypeError, ValueError, OverflowError):
            continue
        if coerced >= 1:
            return coerced
    return max(1, int(default or 1))


def _queue_item_trace_summary(queue_item: ImplementationQueueItem) -> Dict[str, Any]:
    return {
        "queue_id": queue_item.queue_id,
        "node_type": queue_item.node_type,
        "stage_id": queue_item.stage_id,
        "status": queue_item.status,
        "dependencies": list(queue_item.dependencies),
        "expected_inputs": list(queue_item.expected_inputs),
        "expected_outputs": list(queue_item.expected_outputs),
        "purpose": _safe_text(queue_item.purpose, max_chars=220),
    }


def _node_definition_trace_summary(node_definition: Optional[DeveloperNodeDefinition]) -> Dict[str, Any]:
    if node_definition is None:
        return {}
    return {
        "node_type": node_definition.node_type,
        "display_name": node_definition.display_name,
        "type_version": node_definition.type_version,
        "summary": _safe_text(node_definition.summary, max_chars=220),
        "parameter_names": [item.name for item in node_definition.parameter_schema],
        "credential_types_required": list(node_definition.credential_types_required),
        "source_refs": list(node_definition.source_refs[:3]),
    }


def _decision_trace_summary(decision: NodeImplementationDecision) -> Dict[str, Any]:
    return {
        "can_apply": decision.can_apply,
        "parameters_known_keys": sorted(decision.parameters_known.keys()),
        "parameters_inferred_keys": sorted(decision.parameters_inferred.keys()),
        "parameters_unresolved": list(decision.parameters_unresolved),
        "credential_ref_keys": sorted(decision.credential_refs.keys()),
        "missing_inputs": [
            {
                "key_name": item.key_name,
                "category": item.category,
                "reason": _safe_text(item.reason, max_chars=180),
                "question": _safe_text(item.question, max_chars=180),
            }
            for item in decision.missing_inputs
        ],
        "variable_outputs": [
            {
                "name": item.name,
                "semantic_meaning": _safe_text(item.semantic_meaning, max_chars=160),
                "expected_format": item.expected_format,
                "destination_queue_ids": list(item.destination_queue_ids),
            }
            for item in decision.variable_outputs
        ],
        "notes": list(decision.notes),
    }


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
    workflow_draft: Optional[WorkflowDraft],
) -> List[ProposedNode]:
    if provided_nodes:
        return provided_nodes

    if workflow_draft is not None and workflow_draft.nodes:
        return [
            ProposedNode(
                node_id=node.node_id,
                node_type=node.node_type,
                stage_id=node.stage_id,
                purpose=node.purpose or f"Configure existing node '{node.name}'.",
                depends_on=list(node.dependencies),
                expected_inputs=list(node.expected_inputs),
                expected_outputs=list(node.expected_outputs),
            )
            for node in workflow_draft.nodes
        ]

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
                    dependencies.extend(item.node_id for item in source_nodes)
        if not dependencies and idx > 0:
            dependencies.append(proposed_nodes[idx - 1].node_id)
        queue.append(
            ImplementationQueueItem(
                queue_id=node.node_id,
                node_type=node.node_type,
                stage_id=node.stage_id,
                purpose=node.purpose,
                dependencies=_safe_string_list(dependencies),
                expected_inputs=list(node.expected_inputs),
                expected_outputs=list(node.expected_outputs),
                status="pending",
            )
        )
    return queue


def _raw_json_from_row(row: Dict[str, Any]) -> Any:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and "raw_json" in metadata:
        return metadata.get("raw_json")
    return None


def _append_parameter_definition(
    by_name: Dict[str, DeveloperParameterDefinition],
    definition: DeveloperParameterDefinition,
) -> None:
    existing = by_name.get(definition.name)
    if existing is None:
        by_name[definition.name] = definition
        return
    if not existing.display_name and definition.display_name:
        existing.display_name = definition.display_name
    if not existing.description and definition.description:
        existing.description = definition.description
    if not existing.field_type and definition.field_type:
        existing.field_type = definition.field_type
    if not existing.required and definition.required:
        existing.required = True
    if existing.default_value is None and definition.default_value is not None:
        existing.default_value = definition.default_value
    if not existing.scope and definition.scope:
        existing.scope = definition.scope
    if definition.options_preview:
        existing.options_preview = _safe_string_list(
            list(existing.options_preview) + list(definition.options_preview)
        )


def _parameter_defs_from_raw(raw_json: Any, *, scope: Optional[str]) -> List[DeveloperParameterDefinition]:
    if isinstance(raw_json, dict):
        properties = raw_json.get("properties")
        if isinstance(properties, list):
            output: List[DeveloperParameterDefinition] = []
            for prop in properties:
                if not isinstance(prop, dict):
                    continue
                name = str(prop.get("name") or "").strip()
                if not name:
                    continue
                options: List[str] = []
                raw_options = prop.get("options")
                if isinstance(raw_options, list):
                    for option in raw_options[:10]:
                        if isinstance(option, dict):
                            value = option.get("value") or option.get("name")
                        else:
                            value = option
                        text = str(value or "").strip()
                        if text:
                            options.append(text)
                output.append(
                    DeveloperParameterDefinition(
                        name=name,
                        display_name=str(prop.get("displayName") or "").strip() or None,
                        description=_safe_text(prop.get("description") or "", max_chars=240),
                        field_type=str(prop.get("type") or "").strip() or None,
                        required=bool(prop.get("required", False)),
                        default_value=prop.get("default"),
                        scope=scope,
                        options_preview=_safe_string_list(options),
                    )
                )
            return output
    if isinstance(raw_json, list):
        output = []
        for item in raw_json:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            output.append(
                DeveloperParameterDefinition(
                    name=name,
                    display_name=str(item.get("displayName") or "").strip() or None,
                    description=_safe_text(item.get("description") or "", max_chars=240),
                    field_type=str(item.get("type") or "").strip() or None,
                    required=bool(item.get("required", False)),
                    default_value=item.get("default"),
                    scope=scope,
                )
            )
        return output
    return []


def get_node_definition(node_type: str) -> Optional[DeveloperNodeDefinition]:
    node_type_value = str(node_type or "").strip()
    if not node_type_value:
        return None
    rows = query_definition_chunks_by_entity(
        entity_key="nodeType",
        entity_id=node_type_value,
        source_value=_NODE_DEFINITION_SOURCE,
    )
    if not rows:
        rows = query_definition_chunks_by_entity(
            entity_key="nodeType",
            entity_id=node_type_value,
            source_value=None,
        )
    if not rows:
        return None

    overview = next(
        (
            row
            for row in rows
            if str((row.get("metadata") or {}).get("kind") or row.get("__meta_kind") or "") == "NODE_OVERVIEW"
        ),
        rows[0],
    )
    overview_meta = overview.get("metadata") if isinstance(overview.get("metadata"), dict) else {}
    display_name = str(overview_meta.get("displayName") or overview.get("title") or "").strip() or None
    source_refs = _safe_string_list(row.get("url") for row in rows if row.get("url"))
    summary_parts = []
    by_name: Dict[str, DeveloperParameterDefinition] = {}
    credential_types: List[str] = []

    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_json = _raw_json_from_row(row)
        scope = str(row.get("section") or metadata.get("section") or "").strip() or None
        for item in _parameter_defs_from_raw(raw_json, scope=scope):
            _append_parameter_definition(by_name, item)
        raw_param_names = metadata.get("param_names")
        if isinstance(raw_param_names, list):
            for name in raw_param_names:
                text = str(name or "").strip()
                if text:
                    _append_parameter_definition(
                        by_name,
                        DeveloperParameterDefinition(name=text, scope=scope),
                    )
        raw_credential_types = metadata.get("credentialTypes_required")
        if isinstance(raw_credential_types, list):
            credential_types.extend(str(item).strip() for item in raw_credential_types if str(item).strip())
        text = str(row.get("text") or "").strip()
        if text and len(summary_parts) < 3:
            summary_parts.append(_safe_text(text, max_chars=240))

    version = overview_meta.get("version") or 1
    return DeveloperNodeDefinition(
        node_type=node_type_value,
        display_name=display_name,
        type_version=_coerce_positive_int(version, default=1),
        summary="\n".join(summary_parts),
        parameter_schema=list(by_name.values()),
        credential_types_required=_safe_string_list(credential_types),
        source_refs=source_refs,
        raw_chunks=rows,
    )


def get_node_parameter_schema(node_type: str) -> List[DeveloperParameterDefinition]:
    definition = get_node_definition(node_type)
    if definition is None:
        return []
    return list(definition.parameter_schema)


def get_credential_definition(credential_type: str) -> Optional[DeveloperCredentialDefinition]:
    credential_type_value = str(credential_type or "").strip()
    if not credential_type_value:
        return None
    rows = query_definition_chunks_by_entity(
        entity_key="credentialType",
        entity_id=credential_type_value,
        source_value=_CREDENTIAL_DEFINITION_SOURCE,
    )
    if not rows:
        rows = query_definition_chunks_by_entity(
            entity_key="credentialType",
            entity_id=credential_type_value,
            source_value=None,
        )
    if not rows:
        return None
    overview = next(
        (
            row
            for row in rows
            if str((row.get("metadata") or {}).get("kind") or row.get("__meta_kind") or "") == "CRED_OVERVIEW"
        ),
        rows[0],
    )
    overview_meta = overview.get("metadata") if isinstance(overview.get("metadata"), dict) else {}
    display_name = str(overview_meta.get("displayName") or overview.get("title") or "").strip() or None
    supported_nodes = overview_meta.get("supportedNodes") if isinstance(overview_meta.get("supportedNodes"), list) else []
    field_names: List[str] = []
    summary_parts: List[str] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_field_names = metadata.get("field_names")
        if isinstance(raw_field_names, list):
            field_names.extend(str(item).strip() for item in raw_field_names if str(item).strip())
        text = str(row.get("text") or "").strip()
        if text and len(summary_parts) < 3:
            summary_parts.append(_safe_text(text, max_chars=220))
    return DeveloperCredentialDefinition(
        credential_type=credential_type_value,
        display_name=display_name,
        supported_nodes=_safe_string_list(supported_nodes),
        field_names=_safe_string_list(field_names),
        summary="\n".join(summary_parts),
        source_refs=_safe_string_list(row.get("url") for row in rows if row.get("url")),
    )


def get_node_credential_requirements(node_type: str) -> List[DeveloperCredentialDefinition]:
    definition = get_node_definition(node_type)
    if definition is None:
        return []
    output: List[DeveloperCredentialDefinition] = []
    for credential_type in definition.credential_types_required:
        credential_definition = get_credential_definition(credential_type)
        if credential_definition is not None:
            output.append(credential_definition)
        else:
            output.append(
                DeveloperCredentialDefinition(
                    credential_type=credential_type,
                    display_name=credential_type,
                )
            )
    return output


def get_active_workflow(workflow_id: str) -> Dict[str, Any]:
    return N8NClient().get_workflow(workflow_id)


def _dependencies_from_workflow_connections(
    workflow: Dict[str, Any],
    name_to_id: Dict[str, str],
) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {node_id: [] for node_id in name_to_id.values()}
    connections = workflow.get("connections") if isinstance(workflow.get("connections"), dict) else {}
    for source_name, source_payload in connections.items():
        source_id = name_to_id.get(str(source_name))
        if not source_id or not isinstance(source_payload, dict):
            continue
        main = source_payload.get("main")
        if not isinstance(main, list):
            continue
        for bucket in main:
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                if not isinstance(item, dict):
                    continue
                target_name = str(item.get("node") or "").strip()
                target_id = name_to_id.get(target_name)
                if not target_id:
                    continue
                current = output.setdefault(target_id, [])
                if source_id not in current:
                    current.append(source_id)
    return output


def _draft_from_active_workflow_payload(
    workflow: Dict[str, Any],
    *,
    workflow_id: str,
) -> Tuple[WorkflowDraft, List[ProposedNode], ArchitecturePlan, WorkflowContext]:
    workflow_name = str(workflow.get("name") or f"Workflow {workflow_id}").strip()
    nodes_raw = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []
    name_to_id: Dict[str, str] = {}
    draft_nodes: List[WorkflowDraftNode] = []

    for idx, raw_node in enumerate(nodes_raw, start=1):
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node.get("id") or f"wf_{idx}").strip()
        node_name = str(raw_node.get("name") or node_id).strip() or node_id
        node_type = str(raw_node.get("type") or "").strip()
        type_version = raw_node.get("typeVersion") or 1
        position = raw_node.get("position") if isinstance(raw_node.get("position"), list) else [260 * idx, 300]
        parameters = raw_node.get("parameters") if isinstance(raw_node.get("parameters"), dict) else {}
        credential_refs: Dict[str, str] = {}
        raw_credentials = raw_node.get("credentials")
        if isinstance(raw_credentials, dict):
            for key, value in raw_credentials.items():
                if isinstance(value, dict):
                    reference = str(value.get("id") or value.get("name") or "").strip()
                else:
                    reference = str(value or "").strip()
                if reference:
                    credential_refs[str(key)] = reference
        name_to_id[node_name] = node_id
        draft_nodes.append(
            WorkflowDraftNode(
                node_id=node_id,
                name=node_name,
                node_type=node_type,
                type_version=max(1, int(type_version or 1)),
                purpose=f"Existing workflow node '{node_name}'.",
                stage_id=f"stage_{idx}",
                parameters_known=dict(parameters),
                parameters_inferred={},
                parameters_unresolved=[],
                credential_refs=credential_refs,
                expected_inputs=[],
                expected_outputs=[],
                dependencies=[],
                position=[int(position[0]), int(position[1])] if len(position) >= 2 else [260 * idx, 300],
                notes=["bootstrapped_from_active_workflow"],
            )
        )

    dependencies = _dependencies_from_workflow_connections(workflow, name_to_id)
    for node in draft_nodes:
        node.dependencies = list(dependencies.get(node.node_id, []))

    proposed_nodes = [
        ProposedNode(
            node_id=node.node_id,
            node_type=node.node_type,
            stage_id=node.stage_id,
            purpose=node.purpose,
            depends_on=list(node.dependencies),
            expected_inputs=[],
            expected_outputs=[],
        )
        for node in draft_nodes
    ]
    stages: List[ArchitectureStage] = []
    data_flow: List[ArchitectureDataFlowItem] = []
    required_nodes: List[NodeRequirement] = []
    for idx, node in enumerate(draft_nodes, start=1):
        stage_id = node.stage_id or f"stage_{idx}"
        stage_dependencies = [
            item.stage_id
            for item in draft_nodes
            if item.node_id in node.dependencies and item.stage_id
        ]
        stages.append(
            ArchitectureStage(
                id=stage_id,
                name=node.name,
                purpose=node.purpose,
                required_capabilities=[f"Configure node type {node.node_type}"],
                expected_inputs=[],
                expected_outputs=[],
                dependencies=stage_dependencies,
                success_criteria=[f"Node '{node.name}' is configured correctly."],
            )
        )
        for source_stage_id in stage_dependencies:
            data_flow.append(
                ArchitectureDataFlowItem(
                    source_stage_id=source_stage_id,
                    target_stage_id=stage_id,
                    data_items=[],
                )
            )
        required_nodes.append(
            NodeRequirement(
                node_type=node.node_type,
                why_required=f"Existing workflow node '{node.name}' is part of the active workflow.",
                evidence_chunk_ids=[],
                evidence_refs=[],
                evidence_confidence=0.0,
            )
        )

    draft = WorkflowDraft(
        name=workflow_name,
        use_case_id=workflow_id,
        summary=f"Bootstrapped from active workflow {workflow_id}.",
        nodes=draft_nodes,
        connections=[
            WorkflowDraftConnection(source_node_id=source_id, target_node_id=target_id)
            for target_id, source_ids in dependencies.items()
            for source_id in source_ids
        ],
        metadata={"bootstrapped_from_active_workflow": True},
    )
    plan = ArchitecturePlan(
        use_case_id=workflow_id,
        title=workflow_name,
        business_objective=f"Edit existing workflow '{workflow_name}'.",
        desired_outcome="Update the existing workflow safely.",
        workflow_summary=f"Existing workflow '{workflow_name}' loaded from n8n for iterative configuration.",
        stages=stages,
        data_flow=data_flow,
        assumptions=[],
        missing_information=[],
        implementation_notes_for_engineer=["Bootstrapped from an existing workflow fetched from n8n."],
        required_nodes=required_nodes,
    )
    workflow_context = WorkflowContext(
        use_case_id=workflow_id,
        planning_ready=True,
        handoff_target=AgentStage.engineer_agent,
        required_node_types=[node.node_type for node in draft_nodes],
        unresolved_inputs=[],
        notes=["bootstrapped_from_active_workflow"],
    )
    return draft, proposed_nodes, plan, workflow_context


def _invoke_structured_output(
    *,
    system_prompt: str,
    user_prompt: str,
    output_model: Any,
    model: Optional[str],
    request_id: Optional[str],
    stage: str,
    temperature: float = 0.0,
) -> Any:
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError("No model configured for engineer structured output")
    llm = get_langchain_chat_model(model=model, temperature=temperature)
    if llm is None:
        raise RuntimeError("LangChain chat model is unavailable")
    structured = llm.with_structured_output(output_model)
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


def _decision_prompt_payload(
    *,
    user_query: str,
    architecture_plan: ArchitecturePlan,
    queue_item: ImplementationQueueItem,
    current_node: WorkflowDraftNode,
    node_definition: DeveloperNodeDefinition,
    parameter_schema: List[DeveloperParameterDefinition],
    credential_requirements: List[DeveloperCredentialDefinition],
    upstream_variables: List[VariableDefinition],
    resolved_inputs: Dict[str, str],
    downstream_queue_ids: List[str],
) -> str:
    return json.dumps(
        {
            "user_query": user_query,
            "workflow_name": architecture_plan.title,
            "workflow_summary": architecture_plan.workflow_summary,
            "queue_item": queue_item.model_dump(mode="json"),
            "current_node": current_node.model_dump(mode="json"),
            "node_definition": node_definition.model_dump(mode="json"),
            "parameter_schema": [item.model_dump(mode="json") for item in parameter_schema],
            "credential_requirements": [item.model_dump(mode="json") for item in credential_requirements],
            "upstream_variables": [item.model_dump(mode="json") for item in upstream_variables],
            "resolved_inputs": resolved_inputs,
            "downstream_queue_ids": downstream_queue_ids,
            "rules": {
                "do_not_invent_secrets": True,
                "do_not_invent_ids_or_urls": True,
                "preserve_existing_configuration": True,
                "prefer_blocking_over_guessing": True,
            },
        },
        ensure_ascii=True,
    )


def _decide_node_implementation_with_structured_output(
    *,
    user_query: str,
    architecture_plan: ArchitecturePlan,
    queue_item: ImplementationQueueItem,
    current_node: WorkflowDraftNode,
    node_definition: DeveloperNodeDefinition,
    parameter_schema: List[DeveloperParameterDefinition],
    credential_requirements: List[DeveloperCredentialDefinition],
    upstream_variables: List[VariableDefinition],
    resolved_inputs: Dict[str, str],
    downstream_queue_ids: List[str],
    model: Optional[str],
    request_id: Optional[str],
) -> NodeImplementationDecision:
    system_prompt = (
        "You are the developer agent for an n8n workflow. Configure one node at a time using only the provided structured context. "
        "Do not redesign the workflow. Preserve ids, names, positions, and connections. "
        "Keep real user-confirmed values in parameters_known. Use parameters_inferred only for safe, non-sensitive values supported by the schema and workflow context. "
        "Never invent secrets, credential ids, endpoint URLs, resource identifiers, or business rules. "
        "If critical information is missing, set can_apply=false and return explicit missing_inputs."
    )
    response = _invoke_structured_output(
        system_prompt=system_prompt,
        user_prompt=_decision_prompt_payload(
            user_query=user_query,
            architecture_plan=architecture_plan,
            queue_item=queue_item,
            current_node=current_node,
            node_definition=node_definition,
            parameter_schema=parameter_schema,
            credential_requirements=credential_requirements,
            upstream_variables=upstream_variables,
            resolved_inputs=resolved_inputs,
            downstream_queue_ids=downstream_queue_ids,
        ),
        output_model=NodeImplementationDecision,
        model=model,
        request_id=request_id,
        stage="multi_agent.engineer.node_decision",
        temperature=0.0,
    )
    if isinstance(response, NodeImplementationDecision):
        return response
    return NodeImplementationDecision.model_validate(response)


def _fallback_node_implementation_decision(
    *,
    queue_item: ImplementationQueueItem,
    current_node: WorkflowDraftNode,
    parameter_schema: List[DeveloperParameterDefinition],
    credential_requirements: List[DeveloperCredentialDefinition],
    resolved_inputs: Dict[str, str],
    downstream_queue_ids: List[str],
) -> NodeImplementationDecision:
    known = dict(current_node.parameters_known)
    inferred = dict(current_node.parameters_inferred)
    unresolved: List[str] = []
    missing_inputs: List[DeveloperMissingInputDecision] = []
    credential_refs = dict(current_node.credential_refs)

    for param in parameter_schema:
        input_key = f"parameter:{queue_item.queue_id}:{param.name}"
        supplied = _lookup_user_value(input_key, resolved_inputs)
        if supplied is not None:
            known[param.name] = supplied
            continue
        if param.name in known or param.name in inferred:
            continue
        if param.required:
            unresolved.append(param.name)
            missing_inputs.append(
                DeveloperMissingInputDecision(
                    key_name=param.name,
                    category="parameter",
                    reason=f"Parameter '{param.name}' is required by the indexed node definition and is still unknown.",
                    question=f"Provide value for parameter '{param.name}' required by node '{queue_item.queue_id}'.",
                )
            )

    for credential in credential_requirements:
        reference_key = credential.display_name or credential.credential_type
        supplied = (
            _lookup_user_value(f"credential:{queue_item.queue_id}:{reference_key}", resolved_inputs)
            or _lookup_user_value(credential.credential_type, resolved_inputs)
            or _lookup_user_value(reference_key, resolved_inputs)
        )
        if supplied:
            credential_refs[reference_key] = supplied
            continue
        if reference_key in credential_refs:
            continue
        missing_inputs.append(
            DeveloperMissingInputDecision(
                key_name=reference_key,
                category="credential",
                reason=f"Credential reference for '{reference_key}' is required by the indexed node definition.",
                question=f"Provide the credential reference to use for '{reference_key}' in node '{queue_item.queue_id}'.",
            )
        )

    outputs = [
        DeveloperVariableOutput(
            name=(item if "." in item else f"{queue_item.queue_id}.{item}"),
            semantic_meaning=f"Output of {queue_item.queue_id} for downstream workflow steps.",
            expected_format="unknown",
            destination_queue_ids=list(downstream_queue_ids),
            mapping_notes="Fallback developer decision based on workflow queue context.",
        )
        for item in (list(queue_item.expected_outputs) or [f"{_short_type(queue_item.node_type)}_output"])
    ]
    return NodeImplementationDecision(
        parameters_known=known,
        parameters_inferred=inferred,
        parameters_unresolved=_safe_string_list(unresolved),
        credential_refs=credential_refs,
        missing_inputs=missing_inputs,
        variable_outputs=outputs,
        notes=["fallback_structured_decision"],
        can_apply=not missing_inputs,
    )


def _required_credentials_from_definitions(proposed_nodes: List[ProposedNode]) -> List[RequiredCredential]:
    output: List[RequiredCredential] = []
    seen = set()
    for node in proposed_nodes:
        for credential in get_node_credential_requirements(node.node_type):
            key = (node.node_id, credential.credential_type)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                RequiredCredential(
                    credential_key=credential.credential_type,
                    node_type=node.node_type,
                    credential_name=credential.display_name or credential.credential_type,
                    required_for=node.node_id,
                    source="developer_lookup",
                )
            )
    return output

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
    question: Optional[str] = None,
) -> MissingUserInput:
    input_key = f"{category}:{queue_id}:{key_name}"
    if not question and category == "credential":
        question = (
            f"Provide credential reference for '{key_name}' to configure node '{queue_id}'."
        )
    elif not question and category == "mapping":
        question = (
            f"Provide mapping definition for '{key_name}' required by node '{queue_id}'."
        )
    elif not question and category == "handoff":
        question = reason
    elif not question:
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


def _find_draft_node(draft: WorkflowDraft, queue_id: str) -> Optional[WorkflowDraftNode]:
    for node in draft.nodes:
        if node.node_id == queue_id:
            return node
    mapping = _queue_node_id_map(draft)
    mapped_id = mapping.get(queue_id)
    if not mapped_id:
        return None
    for node in draft.nodes:
        if node.node_id == mapped_id:
            return node
    return None


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


def _ensure_queue_node_id_map(
    draft: WorkflowDraft,
    queue: List[ImplementationQueueItem],
) -> Dict[str, str]:
    mapping = _queue_node_id_map(draft)
    changed = False
    for item in queue:
        if item.queue_id in mapping:
            continue
        if any(node.node_id == item.queue_id for node in draft.nodes):
            mapping[item.queue_id] = item.queue_id
            changed = True
    if changed:
        _set_queue_node_id_map(draft, mapping)
    return mapping


def _resolved_inputs(draft: WorkflowDraft) -> Dict[str, str]:
    value = draft.metadata.get("resolved_inputs")
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    return {}


def _set_resolved_inputs(draft: WorkflowDraft, values: Dict[str, str]) -> None:
    draft.metadata["resolved_inputs"] = {str(key): str(value) for key, value in values.items()}


def _missing_details_from_decision(
    *,
    queue_item: ImplementationQueueItem,
    decision: NodeImplementationDecision,
) -> List[MissingUserInput]:
    output: List[MissingUserInput] = []
    for item in decision.missing_inputs:
        output.append(
            _missing_input(
                category=item.category,
                queue_id=queue_item.queue_id,
                key_name=item.key_name,
                reason=item.reason,
                question=item.question,
            )
        )
    return output


def _node_context_for_queue_item(
    *,
    draft: WorkflowDraft,
    queue_item: ImplementationQueueItem,
    queue_node_ids: Dict[str, str],
    node_definition: DeveloperNodeDefinition,
) -> WorkflowDraftNode:
    existing = _find_draft_node(draft, queue_item.queue_id)
    if existing is not None:
        updated = existing.model_copy(deep=True)
        if not updated.stage_id:
            updated.stage_id = queue_item.stage_id
        if not updated.purpose:
            updated.purpose = queue_item.purpose
        if not updated.expected_inputs:
            updated.expected_inputs = list(queue_item.expected_inputs)
        if not updated.expected_outputs:
            updated.expected_outputs = list(queue_item.expected_outputs)
        if not updated.dependencies:
            updated.dependencies = list(queue_item.dependencies)
        if getattr(updated, "type_version", 1) < 1:
            updated.type_version = node_definition.type_version
        return updated

    node_id = queue_node_ids.get(queue_item.queue_id) or queue_item.queue_id
    queue_node_ids[queue_item.queue_id] = node_id
    return WorkflowDraftNode(
        node_id=node_id,
        name=f"{_short_type(queue_item.node_type)}_{len(queue_node_ids)}",
        node_type=queue_item.node_type,
        type_version=node_definition.type_version,
        purpose=queue_item.purpose,
        stage_id=queue_item.stage_id,
        parameters_known={},
        parameters_inferred={},
        parameters_unresolved=[],
        credential_refs={},
        expected_inputs=list(queue_item.expected_inputs),
        expected_outputs=list(queue_item.expected_outputs),
        dependencies=list(queue_item.dependencies),
        position=[240 * max(0, len(draft.nodes)), 300],
        notes=["created_by_engineer_agent"],
    )


def _update_variable_registry(
    registry: List[VariableDefinition],
    *,
    queue_item: ImplementationQueueItem,
    node_id: str,
    queue: List[ImplementationQueueItem],
    decision: NodeImplementationDecision,
) -> List[VariableDefinition]:
    by_key: Dict[Tuple[str, str], VariableDefinition] = {
        (item.name, item.origin_node_id): item for item in registry
    }
    downstream_default = [
        candidate.queue_id
        for candidate in queue
        if queue_item.queue_id in candidate.dependencies
    ]
    outputs = list(decision.variable_outputs)
    if not outputs:
        outputs = [
            DeveloperVariableOutput(
                name=(output_name if "." in output_name else f"{queue_item.queue_id}.{output_name}"),
                semantic_meaning=f"Output from {queue_item.queue_id} for downstream workflow steps.",
                expected_format="unknown",
                destination_queue_ids=list(downstream_default),
                mapping_notes=(
                    f"Mapped from node type '{queue_item.node_type}' to dependent nodes."
                    if downstream_default
                    else "Terminal output in current draft."
                ),
            )
            for output_name in (list(queue_item.expected_outputs) or [f"{_short_type(queue_item.node_type)}_output"])
        ]
    for output in outputs:
        variable_name = output.name if "." in output.name else f"{queue_item.queue_id}.{output.name}"
        definition = VariableDefinition(
            name=variable_name,
            origin_node_id=node_id,
            destination_node_ids=list(output.destination_queue_ids or downstream_default),
            semantic_meaning=output.semantic_meaning,
            expected_format=output.expected_format,
            mapping_notes=output.mapping_notes,
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
            "typeVersion": max(1, int(getattr(node, "type_version", 1) or 1)),
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
        emit_trace_event(
            trace_logger,
            event="workflow_persist_result",
            request_id=request_id,
            stage="multi_agent.workflow_persist",
            payload={
                "workflow_name": workflow_name_value,
                "action": "skipped",
                "reason": "persist_to_n8n disabled",
                "node_count": len(final_workflow_json.get("nodes") or []),
                "connection_count": len(final_workflow_json.get("connections") or {}),
                "active_workflow_id": current_id,
            },
        )
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
        emit_trace_event(
            trace_logger,
            event="workflow_persist_result",
            request_id=request_id,
            stage="multi_agent.workflow_persist",
            payload={
                "workflow_name": workflow_name_value,
                "action": action,
                "ok": False,
                "status_code": exc.status_code,
                "error": str(exc),
                "node_count": len(payload.get("nodes") or []),
                "connection_keys": sorted((payload.get("connections") or {}).keys())
                if isinstance(payload.get("connections"), dict)
                else [],
                "active_workflow_id": current_id,
            },
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
    emit_trace_event(
        trace_logger,
        event="workflow_persist_result",
        request_id=request_id,
        stage="multi_agent.workflow_persist",
        payload={
            "workflow_name": persisted_name,
            "action": action,
            "ok": True,
            "active_workflow_id": persisted_id,
            "active_workflow_url": persisted_url,
            "node_count": len(payload.get("nodes") or []),
            "connection_keys": sorted((payload.get("connections") or {}).keys())
            if isinstance(payload.get("connections"), dict)
            else [],
        },
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

    workflow_draft = _normalize_model(state.get("workflow_draft"), WorkflowDraft)

    if architecture_plan is None and entry_intent == EntryIntent.workflow_edit_request and active_workflow_id:
        try:
            workflow_payload = get_active_workflow(active_workflow_id)
            workflow_draft, boot_nodes, architecture_plan, workflow_context = _draft_from_active_workflow_payload(
                workflow_payload,
                workflow_id=active_workflow_id,
            )
            if not provided_nodes:
                provided_nodes = boot_nodes
            if not active_workflow_name:
                active_workflow_name = workflow_draft.name
            if not active_workflow_url:
                active_workflow_url = _default_workflow_url(N8NClient().base_url, active_workflow_id)
            persist_payload = _persist_fields(
                active_workflow_id=active_workflow_id,
                active_workflow_name=active_workflow_name,
                active_workflow_url=active_workflow_url,
                workflow_persisted=workflow_persisted,
                workflow_persist_action=workflow_persist_action,
                workflow_api_sync_result=workflow_api_sync_result,
            )
            if "engineer_bootstrapped_from_active_workflow" not in routing_signals:
                routing_signals.append("engineer_bootstrapped_from_active_workflow")
            emit_trace_event(
                trace_logger,
                event="engineer_bootstrap_active_workflow",
                request_id=request_id,
                stage="multi_agent.engineer",
                payload={
                    "active_workflow_id": active_workflow_id,
                    "workflow_name": workflow_draft.name,
                    "node_count": len(workflow_draft.nodes),
                    "connection_count": len(workflow_draft.connections),
                    "proposed_nodes": [
                        {
                            "node_id": item.node_id,
                            "node_type": item.node_type,
                            "stage_id": item.stage_id,
                        }
                        for item in boot_nodes
                    ],
                },
            )
        except N8NClientError as exc:
            missing_details = _merge_missing_details(
                missing_details,
                [
                    _missing_input(
                        category="handoff",
                        queue_id="engineer_handoff",
                        key_name="active_workflow",
                        reason=f"The active workflow could not be loaded from n8n: {exc}",
                    )
                ],
            )
            emit_trace_event(
                trace_logger,
                event="engineer_bootstrap_active_workflow_failed",
                request_id=request_id,
                stage="multi_agent.engineer",
                payload={
                    "active_workflow_id": active_workflow_id,
                    "error": str(exc),
                },
            )

    if architecture_plan is None:
        if entry_intent == EntryIntent.workflow_edit_request:
            handoff_missing = _missing_input(
                category="handoff",
                queue_id="engineer_handoff",
                key_name="active_workflow_id",
                reason=(
                    "Missing workflow handoff. Provide an active workflow id or architect handoff to continue implementation."
                ),
            )
            missing_details = _merge_missing_details(missing_details, [handoff_missing])
            missing_user_inputs = [item.question for item in missing_details]
            implementation_status = ImplementationStatus.blocked_waiting_user
            routing_signals.append("engineer_blocked_waiting_user")
            engineer_notes.append("Engineer blocked: workflow edit request missing active workflow context.")
            emit_trace_event(
                trace_logger,
                event="engineer_result",
                request_id=request_id,
                stage="multi_agent.engineer",
                payload={
                    "status": implementation_status.value,
                    "reason": "missing_active_workflow_context",
                    "missing_user_inputs": list(missing_user_inputs),
                },
            )
            return {
                "current_stage": "engineer_agent",
                "target_stage": None,
                "implementation_status": implementation_status,
                "missing_user_inputs": missing_user_inputs,
                "missing_user_input_details": missing_details,
                "routing_signals": routing_signals,
                "engineer_notes": engineer_notes,
                "final_workflow_json": {},
                "workflow_draft": workflow_draft,
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
        emit_trace_event(
            trace_logger,
            event="engineer_result",
            request_id=request_id,
            stage="multi_agent.engineer",
            payload={
                "status": implementation_status.value,
                "reason": "missing_architecture_plan",
            },
        )
        return {
            "current_stage": "engineer_agent",
            "target_stage": None,
            "implementation_status": implementation_status,
            "missing_user_inputs": missing_user_inputs,
            "missing_user_input_details": missing_details,
            "routing_signals": routing_signals,
            "engineer_notes": engineer_notes,
            "final_workflow_json": {},
            "workflow_draft": workflow_draft,
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

    workflow_draft = _normalize_workflow_draft(
        workflow_draft,
        plan=architecture_plan,
    )
    proposed_nodes = _derive_proposed_nodes(
        plan=architecture_plan,
        workflow_context=workflow_context,
        provided_nodes=provided_nodes,
        workflow_draft=workflow_draft,
    )
    required_credentials = provided_credentials or _required_credentials_from_definitions(
        proposed_nodes
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
        emit_trace_event(
            trace_logger,
            event="engineer_result",
            request_id=request_id,
            stage="multi_agent.engineer",
            payload={
                "status": implementation_status.value,
                "reason": "no_proposed_nodes",
                "missing_user_inputs": list(missing_user_inputs),
            },
        )
        return {
            "current_stage": "engineer_agent",
            "target_stage": None,
            "implementation_status": implementation_status,
            "missing_user_inputs": missing_user_inputs,
            "missing_user_input_details": missing_details,
            "routing_signals": routing_signals,
            "engineer_notes": engineer_notes,
            "final_workflow_json": {},
            "workflow_draft": workflow_draft,
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
    emit_trace_event(
        trace_logger,
        event="engineer_handoff_loaded",
        request_id=request_id,
        stage="multi_agent.engineer",
        payload={
            "entry_intent": entry_intent.value if isinstance(entry_intent, EntryIntent) else str(entry_intent or ""),
            "workflow_name": workflow_draft.name,
            "active_workflow_id": persist_payload.get("active_workflow_id"),
            "proposed_node_count": len(proposed_nodes),
            "required_credential_count": len(required_credentials),
            "queue_count": len(queue),
            "proposed_nodes": [
                {
                    "node_id": item.node_id,
                    "node_type": item.node_type,
                    "stage_id": item.stage_id,
                    "depends_on": list(item.depends_on),
                }
                for item in proposed_nodes
            ],
            "queue": [_queue_item_trace_summary(item) for item in queue],
        },
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

    queue_node_ids = _ensure_queue_node_id_map(workflow_draft, queue)
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

            node_definition = get_node_definition(queue_item.node_type)
            emit_trace_event(
                trace_logger,
                event="engineer_node_iteration_start",
                request_id=request_id,
                stage="multi_agent.engineer.node_iteration",
                payload={
                    "queue_item": _queue_item_trace_summary(queue_item),
                    "resolved_input_keys": sorted(resolved_inputs.keys())[:20],
                    "current_blocked_nodes": [item.queue_id for item in blocked_nodes],
                    "node_definition": _node_definition_trace_summary(node_definition),
                },
            )
            if node_definition is None:
                unresolved_inputs = [
                    _missing_input(
                        category="dependency",
                        queue_id=queue_item.queue_id,
                        key_name=queue_item.node_type,
                        reason=(
                            f"Indexed node definition for '{queue_item.node_type}' is unavailable, so the node cannot be configured safely."
                        ),
                        question=(
                            f"The node definition for '{queue_item.node_type}' is not available in the local index. Choose another node or refresh the definitions index."
                        ),
                    )
                ]
                queue_item.status = "blocked"
                blocked_nodes = [item for item in blocked_nodes if item.queue_id != queue_item.queue_id]
                blocked_nodes.append(
                    BlockedNode(
                        queue_id=queue_item.queue_id,
                        node_type=queue_item.node_type,
                        reason="Node configuration blocked because no indexed definition is available.",
                        missing_input_ids=[item.input_id for item in unresolved_inputs],
                    )
                )
                missing_details = _merge_missing_details(missing_details, unresolved_inputs)
                missing_user_inputs = [item.question for item in missing_details]
                implementation_status = ImplementationStatus.blocked_waiting_user
                routing_signals.append("engineer_blocked_waiting_user")
                engineer_notes.append(
                    f"Blocked node '{queue_item.queue_id}' because no indexed definition was found for '{queue_item.node_type}'."
                )
                emit_trace_event(
                    trace_logger,
                    event="engineer_node_blocked",
                    request_id=request_id,
                    stage="multi_agent.engineer.node_iteration",
                    payload={
                        "queue_item": _queue_item_trace_summary(queue_item),
                        "reason": "missing_indexed_node_definition",
                        "missing_user_inputs": [item.question for item in unresolved_inputs],
                    },
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

            parameter_schema = get_node_parameter_schema(queue_item.node_type)
            credential_requirements = get_node_credential_requirements(queue_item.node_type)
            current_node = _node_context_for_queue_item(
                draft=workflow_draft,
                queue_item=queue_item,
                queue_node_ids=queue_node_ids,
                node_definition=node_definition,
            )
            downstream_queue_ids = [
                candidate.queue_id
                for candidate in queue
                if queue_item.queue_id in candidate.dependencies
            ]
            upstream_variables = [
                item
                for item in variable_registry
                if queue_item.queue_id in item.destination_node_ids
            ]

            try:
                decision = _decide_node_implementation_with_structured_output(
                    user_query=user_query,
                    architecture_plan=architecture_plan,
                    queue_item=queue_item,
                    current_node=current_node,
                    node_definition=node_definition,
                    parameter_schema=parameter_schema,
                    credential_requirements=credential_requirements,
                    upstream_variables=upstream_variables,
                    resolved_inputs=resolved_inputs,
                    downstream_queue_ids=downstream_queue_ids,
                    model=model,
                    request_id=request_id,
                )
            except Exception as exc:
                logger.warning(
                    "engineer structured decision fallback: request_id=%s node=%s error=%s",
                    request_id or "-",
                    queue_item.queue_id,
                    str(exc),
                )
                decision = _fallback_node_implementation_decision(
                    queue_item=queue_item,
                    current_node=current_node,
                    parameter_schema=parameter_schema,
                    credential_requirements=credential_requirements,
                    resolved_inputs=resolved_inputs,
                    downstream_queue_ids=downstream_queue_ids,
                )
            emit_trace_event(
                trace_logger,
                event="engineer_node_decision",
                request_id=request_id,
                stage="multi_agent.engineer.node_decision",
                payload={
                    "queue_item": _queue_item_trace_summary(queue_item),
                    "upstream_variables": [
                        {
                            "name": item.name,
                            "origin_node_id": item.origin_node_id,
                            "destination_node_ids": list(item.destination_node_ids),
                        }
                        for item in upstream_variables
                    ],
                    "downstream_queue_ids": list(downstream_queue_ids),
                    "parameter_schema_names": [item.name for item in parameter_schema],
                    "credential_requirement_types": [
                        item.credential_type for item in credential_requirements
                    ],
                    "decision": _decision_trace_summary(decision),
                },
            )

            unresolved_inputs = _missing_details_from_decision(
                queue_item=queue_item,
                decision=decision,
            )
            if unresolved_inputs or not decision.can_apply:
                current_node.parameters_known = dict(decision.parameters_known)
                current_node.parameters_inferred = dict(decision.parameters_inferred)
                current_node.parameters_unresolved = _safe_string_list(decision.parameters_unresolved)
                current_node.credential_refs = dict(decision.credential_refs)
                current_node.notes = _safe_string_list(
                    list(current_node.notes) + list(decision.notes) + ["blocked_by_missing_inputs"]
                )
                _upsert_draft_node(workflow_draft, current_node)
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
                emit_trace_event(
                    trace_logger,
                    event="engineer_node_blocked",
                    request_id=request_id,
                    stage="multi_agent.engineer.node_iteration",
                    payload={
                        "queue_item": _queue_item_trace_summary(queue_item),
                        "decision": _decision_trace_summary(decision),
                        "missing_details": [
                            {
                                "input_id": item.input_id,
                                "input_key": item.input_key,
                                "category": item.category,
                                "question": item.question,
                            }
                            for item in unresolved_inputs
                        ],
                    },
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
            current_node.parameters_known = dict(decision.parameters_known)
            current_node.parameters_inferred = dict(decision.parameters_inferred)
            current_node.parameters_unresolved = []
            current_node.credential_refs = dict(decision.credential_refs)
            current_node.expected_inputs = list(queue_item.expected_inputs or current_node.expected_inputs)
            current_node.expected_outputs = (
                [item.name for item in decision.variable_outputs]
                or list(queue_item.expected_outputs or current_node.expected_outputs)
            )
            current_node.dependencies = list(queue_item.dependencies)
            current_node.type_version = max(1, int(node_definition.type_version or current_node.type_version or 1))
            current_node.notes = _safe_string_list(
                list(current_node.notes) + list(decision.notes) + ["implemented_by_engineer_agent"]
            )
            _upsert_draft_node(workflow_draft, current_node)
            queue_node_ids[queue_item.queue_id] = current_node.node_id
            _set_queue_node_id_map(workflow_draft, queue_node_ids)

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
                    node_id=current_node.node_id,
                    node_type=queue_item.node_type,
                    purpose=queue_item.purpose,
                    version=version,
                )
            )
            variable_registry = _update_variable_registry(
                variable_registry,
                queue_item=queue_item,
                node_id=current_node.node_id,
                queue=queue,
                decision=decision,
            )
            progressed = True

            incremental_json = _draft_to_final_workflow_json(workflow_draft)
            persist_payload = _persist_workflow_candidate(
                state={**state, **persist_payload},
                final_workflow_json=incremental_json,
                workflow_name=workflow_draft.name,
                request_id=request_id,
            )
            emit_trace_event(
                trace_logger,
                event="engineer_node_implemented",
                request_id=request_id,
                stage="multi_agent.engineer.node_iteration",
                payload={
                    "queue_item": _queue_item_trace_summary(queue_item),
                    "decision": _decision_trace_summary(decision),
                    "version_count": len(workflow_versions),
                    "active_workflow_id": persist_payload.get("active_workflow_id"),
                    "persist_action": persist_payload.get("workflow_persist_action"),
                },
            )

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
        emit_trace_event(
            trace_logger,
            event="engineer_result",
            request_id=request_id,
            stage="multi_agent.engineer",
            payload={
                "status": implementation_status.value,
                "remaining_queue": [_queue_item_trace_summary(item) for item in queue if item.status != "implemented"],
                "blocked_nodes": [item.queue_id for item in blocked_nodes],
                "missing_user_inputs": list(missing_user_inputs),
            },
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
        emit_trace_event(
            trace_logger,
            event="engineer_result",
            request_id=request_id,
            stage="multi_agent.engineer",
            payload={
                "status": implementation_status.value,
                "blocked_nodes": [item.queue_id for item in blocked_nodes],
                "missing_user_inputs": list(missing_user_inputs),
            },
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

    if not workflow_draft.connections:
        workflow_draft.connections = _build_connections(queue, queue_node_ids)
    final_workflow_json = _draft_to_final_workflow_json(workflow_draft)
    _append_version(workflow_versions, workflow_draft, reason="connected workflow graph and produced final candidate")

    persist_payload = _persist_workflow_candidate(
        state={**state, **persist_payload},
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
            required_node_types=[node.node_type for node in proposed_nodes],
            unresolved_inputs=[],
            notes=[],
        )
    workflow_context.unresolved_inputs = []
    workflow_context.handoff_target = AgentStage.qa_agent
    workflow_context.required_node_types = [node.node_type for node in proposed_nodes]
    if "handoff_ready_qa" not in routing_signals:
        routing_signals.append("handoff_ready_qa")
    engineer_notes.append("Engineer completed iterative workflow construction. QA handoff is ready.")

    emit_trace_event(
        trace_logger,
        event="engineer_result",
        request_id=request_id,
        stage="multi_agent.engineer",
        payload={
            "status": implementation_status.value,
            "node_count": len(workflow_draft.nodes),
            "version_count": len(workflow_versions),
            "implemented_nodes": [
                {
                    "queue_id": item.queue_id,
                    "node_id": item.node_id,
                    "node_type": item.node_type,
                    "version": item.version,
                }
                for item in implemented_nodes
            ],
            "variable_registry": [
                {
                    "name": item.name,
                    "origin_node_id": item.origin_node_id,
                    "destination_node_ids": list(item.destination_node_ids),
                    "semantic_meaning": item.semantic_meaning,
                }
                for item in variable_registry
            ],
            "active_workflow_id": persist_payload.get("active_workflow_id"),
            "persist_action": persist_payload.get("workflow_persist_action"),
            "target_stage": AgentStage.qa_agent.value,
        },
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

