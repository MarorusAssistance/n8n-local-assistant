from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import json
import re

from .indexing import fnv1a32


COMPLEX_TYPES = {"collection", "fixedCollection"}
DEFAULT_OPTIONS_PREVIEW = 10
DEFAULT_CREDENTIAL_FIELDS_PER_CHUNK = 40


@dataclass(frozen=True)
class Scope:
    resource: Optional[str]
    operation: Optional[str]


@dataclass
class BuildMetrics:
    nodes_processed: int = 0
    credentials_processed: int = 0
    scopes_detected: int = 0
    chunks_by_kind: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def add_kind(self, kind: str) -> None:
        self.chunks_by_kind[kind] = self.chunks_by_kind.get(kind, 0) + 1

    def merge(self, other: "BuildMetrics") -> None:
        self.nodes_processed += other.nodes_processed
        self.credentials_processed += other.credentials_processed
        self.scopes_detected += other.scopes_detected
        for kind, count in other.chunks_by_kind.items():
            self.chunks_by_kind[kind] = self.chunks_by_kind.get(kind, 0) + count
        self.warnings.extend(other.warnings)


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in data if isinstance(item, dict)]


def build_nodes_chunks(
    nodes: Sequence[Mapping[str, Any]],
    *,
    source: str = "n8n-nodes",
    include_raw_json: bool = False,
    options_preview_limit: int = DEFAULT_OPTIONS_PREVIEW,
) -> Tuple[List[Dict[str, Any]], BuildMetrics]:
    chunks: List[Dict[str, Any]] = []
    metrics = BuildMetrics()
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_chunks, node_metrics = _build_single_node_chunks(
            node,
            source=source,
            include_raw_json=include_raw_json,
            options_preview_limit=options_preview_limit,
        )
        chunks.extend(node_chunks)
        metrics.merge(node_metrics)
    return chunks, metrics


def build_credentials_chunks(
    credentials: Sequence[Mapping[str, Any]],
    *,
    source: str = "n8n-credentials",
    include_raw_json: bool = False,
    options_preview_limit: int = DEFAULT_OPTIONS_PREVIEW,
    fields_per_chunk: int = DEFAULT_CREDENTIAL_FIELDS_PER_CHUNK,
) -> Tuple[List[Dict[str, Any]], BuildMetrics]:
    if fields_per_chunk <= 0:
        fields_per_chunk = DEFAULT_CREDENTIAL_FIELDS_PER_CHUNK
    chunks: List[Dict[str, Any]] = []
    metrics = BuildMetrics()
    for credential in credentials:
        if not isinstance(credential, Mapping):
            continue
        cred_chunks, cred_metrics = _build_single_credential_chunks(
            credential,
            source=source,
            include_raw_json=include_raw_json,
            options_preview_limit=options_preview_limit,
            fields_per_chunk=fields_per_chunk,
        )
        chunks.extend(cred_chunks)
        metrics.merge(cred_metrics)
    return chunks, metrics


def _build_single_node_chunks(
    node: Mapping[str, Any],
    *,
    source: str,
    include_raw_json: bool,
    options_preview_limit: int,
) -> Tuple[List[Dict[str, Any]], BuildMetrics]:
    metrics = BuildMetrics(nodes_processed=1)
    node_type = _as_text(node.get("name")) or "unknown-node"
    display_name = _as_text(node.get("displayName")) or node_type
    version = node.get("version")
    group = _as_text_list(node.get("group"))
    description = _as_text(node.get("description"))
    usable_as_tool = bool(node.get("usableAsTool"))
    inputs = _as_text_list(node.get("inputs"))
    outputs = _as_text_list(node.get("outputs"))
    doc_urls = _extract_node_doc_urls(node)
    credential_types = _extract_node_credential_types(node.get("credentials"))

    properties = node.get("properties")
    if not isinstance(properties, list):
        properties = []
        metrics.warnings.append(f"node:{node_type}: properties missing")
    elif not properties:
        metrics.warnings.append(f"node:{node_type}: properties empty")

    resource_prop, resources = _detect_resource_property(properties)
    scopes, operation_info, scope_conditions = _detect_scopes(properties, resources)
    metrics.scopes_detected += len(scopes)

    has_scope_signals = _has_scope_signals(properties)
    has_scoped_axes = any(scope.resource is not None or scope.operation is not None for scope in scopes)
    has_resource_axis = any(scope.resource is not None for scope in scopes)
    has_operation_axis = any(scope.operation is not None for scope in scopes)
    has_resource_operation_scopes = any(scope.resource is not None and scope.operation is not None for scope in scopes)
    if has_scope_signals:
        if not has_scoped_axes:
            metrics.warnings.append(
                f"node:{node_type}: no resource/operation scopes detected"
            )
        elif not has_resource_operation_scopes:
            if has_operation_axis and not has_resource_axis:
                metrics.warnings.append(
                    f"node:{node_type}: operation-only scopes detected (reduced retrieval precision possible)"
                )
            elif has_resource_axis and not has_operation_axis:
                metrics.warnings.append(
                    f"node:{node_type}: resource-only scopes detected (reduced retrieval precision possible)"
                )
            else:
                metrics.warnings.append(
                    f"node:{node_type}: partial resource/operation scope pairing detected (reduced retrieval precision possible)"
                )

    params_by_scope: Dict[Scope, List[Mapping[str, Any]]] = {}
    global_params: List[Mapping[str, Any]] = []
    complex_scoped: Dict[Tuple[Scope, str], List[Mapping[str, Any]]] = {}
    complex_global: Dict[str, List[Mapping[str, Any]]] = {}

    for prop in properties:
        if not isinstance(prop, Mapping):
            continue
        if _is_resource_property(prop) or _is_operation_property(prop):
            continue
        prop_scopes = _resolve_property_scopes(prop, scopes)
        if _is_complex_property(prop):
            root_name = _as_text(prop.get("name")) or _as_text(prop.get("displayName")) or "unnamed"
            if prop_scopes:
                for scope in prop_scopes:
                    complex_scoped.setdefault((scope, root_name), []).append(prop)
            else:
                complex_global.setdefault(root_name, []).append(prop)
            continue
        if prop_scopes:
            for scope in prop_scopes:
                params_by_scope.setdefault(scope, []).append(prop)
        else:
            global_params.append(prop)

    chunks: List[Dict[str, Any]] = []
    base_url = doc_urls[0] if doc_urls else f"n8n://node/{node_type}"
    base_meta: Dict[str, Any] = {
        "nodeType": node_type,
        "displayName": display_name,
        "version": version,
        "group": group,
        "credentialTypes_required": credential_types,
        "doc_urls": doc_urls,
        "title": display_name,
        "url": base_url,
        "source": source,
    }

    overview_meta = dict(base_meta)
    overview_meta.update(
        {"kind": "NODE_OVERVIEW", "resource": None, "operation": None, "param_names": [], "section": "Overview"}
    )
    chunks.append(
        _build_chunk(
            chunk_id=_build_chunk_id(node_type, "NODE_OVERVIEW"),
            content=_render_node_overview(
                node_type=node_type,
                display_name=display_name,
                description=description,
                group=group,
                version=version,
                inputs=inputs,
                outputs=outputs,
                usable_as_tool=usable_as_tool,
                credential_types=credential_types,
                doc_urls=doc_urls,
            ),
            metadata=overview_meta,
            source=source,
            kind="NODE_OVERVIEW",
            raw_json=node if include_raw_json else None,
        )
    )
    metrics.add_kind("NODE_OVERVIEW")

    if resource_prop is not None and resources:
        resource_meta = dict(base_meta)
        resource_meta.update(
            {
                "kind": "NODE_RESOURCE",
                "resource": None,
                "operation": None,
                "param_names": [],
                "resource_names": resources,
                "section": "Resources",
            }
        )
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(node_type, "NODE_RESOURCE"),
                content=_render_node_resource(
                    node_type=node_type,
                    display_name=display_name,
                    resources=resources,
                ),
                metadata=resource_meta,
                source=source,
                kind="NODE_RESOURCE",
                raw_json=resource_prop if include_raw_json else None,
            )
        )
        metrics.add_kind("NODE_RESOURCE")

    for scope in scopes:
        op_key = (scope.resource, scope.operation)
        op_info = operation_info.get(op_key, {})
        op_conditions = scope_conditions.get(op_key, {})

        resource_operation_meta = dict(base_meta)
        resource_operation_meta.update(
            {
                "kind": "NODE_RESOURCE_OPERATION",
                "resource": scope.resource,
                "operation": scope.operation,
                "param_names": [],
                "section": _scope_section(scope),
            }
        )
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(
                    node_type,
                    "NODE_RESOURCE_OPERATION",
                    *_scope_id_parts(scope),
                ),
                content=_render_node_resource_operation(
                    node_type=node_type,
                    display_name=display_name,
                    resource=scope.resource,
                    operation=scope.operation,
                    operation_name=_as_text(op_info.get("name")),
                    action=_as_text(op_info.get("action")),
                    description=_as_text(op_info.get("description")),
                    operation_conditions=op_conditions,
                ),
                metadata=resource_operation_meta,
                source=source,
                kind="NODE_RESOURCE_OPERATION",
                raw_json=op_info if include_raw_json else None,
            )
        )
        metrics.add_kind("NODE_RESOURCE_OPERATION")

        scoped_params = params_by_scope.get(scope, [])
        scoped_params_meta = dict(base_meta)
        scoped_params_meta.update(
            {
                "kind": "NODE_PARAMS_FOR_RESOURCE_OPERATION",
                "resource": scope.resource,
                "operation": scope.operation,
                "param_names": _collect_param_names(scoped_params),
                "section": _scope_section(scope, suffix="params"),
            }
        )
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(
                    node_type,
                    "NODE_PARAMS_FOR_RESOURCE_OPERATION",
                    *_scope_id_parts(scope),
                ),
                content=_render_node_params_for_scope(
                    node_type=node_type,
                    display_name=display_name,
                    resource=scope.resource,
                    operation=scope.operation,
                    params=scoped_params,
                    options_preview_limit=options_preview_limit,
                ),
                metadata=scoped_params_meta,
                source=source,
                kind="NODE_PARAMS_FOR_RESOURCE_OPERATION",
                raw_json={"properties": scoped_params} if include_raw_json else None,
            )
        )
        metrics.add_kind("NODE_PARAMS_FOR_RESOURCE_OPERATION")

    if global_params:
        global_meta = dict(base_meta)
        global_meta.update(
            {
                "kind": "NODE_GLOBAL_PARAMS",
                "resource": None,
                "operation": None,
                "param_names": _collect_param_names(global_params),
                "section": "Global params",
            }
        )
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(node_type, "NODE_GLOBAL_PARAMS"),
                content=_render_node_global_params(
                    node_type=node_type,
                    display_name=display_name,
                    params=global_params,
                    options_preview_limit=options_preview_limit,
                ),
                metadata=global_meta,
                source=source,
                kind="NODE_GLOBAL_PARAMS",
                raw_json={"properties": global_params} if include_raw_json else None,
            )
        )
        metrics.add_kind("NODE_GLOBAL_PARAMS")

    for (scope, root_name), root_props in complex_scoped.items():
        entries: List[Dict[str, Any]] = []
        for root_prop in root_props:
            entries.extend(_flatten_complex_field(root_prop))
        if not entries:
            metrics.warnings.append(
                f"node:{node_type}: complex field '{root_name}' has no entries for {scope.resource}/{scope.operation}"
            )
            continue
        complex_meta = dict(base_meta)
        complex_meta.update(
            {
                "kind": "NODE_COMPLEX_FIELD",
                "resource": scope.resource,
                "operation": scope.operation,
                "group_path": root_name,
                "param_names": [entry["path"] for entry in entries],
                "section": _scope_section(scope, suffix=f"complex:{root_name}"),
            }
        )
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(
                    node_type,
                    "NODE_COMPLEX_FIELD",
                    *_scope_id_parts(scope),
                    root_name,
                ),
                content=_render_node_complex_field(
                    node_type=node_type,
                    display_name=display_name,
                    root_name=root_name,
                    resource=scope.resource,
                    operation=scope.operation,
                    entries=entries,
                    options_preview_limit=options_preview_limit,
                ),
                metadata=complex_meta,
                source=source,
                kind="NODE_COMPLEX_FIELD",
                raw_json=root_props if include_raw_json else None,
            )
        )
        metrics.add_kind("NODE_COMPLEX_FIELD")

    for root_name, root_props in complex_global.items():
        entries: List[Dict[str, Any]] = []
        for root_prop in root_props:
            entries.extend(_flatten_complex_field(root_prop))
        if not entries:
            metrics.warnings.append(f"node:{node_type}: complex field '{root_name}' has no entries")
            continue
        complex_meta = dict(base_meta)
        complex_meta.update(
            {
                "kind": "NODE_COMPLEX_FIELD",
                "resource": None,
                "operation": None,
                "group_path": root_name,
                "param_names": [entry["path"] for entry in entries],
                "section": f"Global complex:{root_name}",
            }
        )
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(node_type, "NODE_COMPLEX_FIELD", root_name, "global"),
                content=_render_node_complex_field(
                    node_type=node_type,
                    display_name=display_name,
                    root_name=root_name,
                    resource=None,
                    operation=None,
                    entries=entries,
                    options_preview_limit=options_preview_limit,
                ),
                metadata=complex_meta,
                source=source,
                kind="NODE_COMPLEX_FIELD",
                raw_json=root_props if include_raw_json else None,
            )
        )
        metrics.add_kind("NODE_COMPLEX_FIELD")

    return chunks, metrics


def _build_single_credential_chunks(
    credential: Mapping[str, Any],
    *,
    source: str,
    include_raw_json: bool,
    options_preview_limit: int,
    fields_per_chunk: int,
) -> Tuple[List[Dict[str, Any]], BuildMetrics]:
    metrics = BuildMetrics(credentials_processed=1)
    cred_type = _as_text(credential.get("name")) or "unknown-credential"
    display_name = _as_text(credential.get("displayName")) or cred_type
    documentation_key = _as_text(credential.get("documentationUrl"))
    supported_nodes = _as_text_list(credential.get("supportedNodes"))
    if not supported_nodes:
        metrics.warnings.append(f"credential:{cred_type}: supportedNodes empty")
    test_request = _extract_credential_test_request(credential.get("test"))
    icon_url = credential.get("iconUrl")
    icon_value = icon_url if isinstance(icon_url, (str, dict, list)) else None

    fields = credential.get("properties")
    if not isinstance(fields, list):
        fields = []
        metrics.warnings.append(f"credential:{cred_type}: properties missing")
    elif not fields:
        metrics.warnings.append(f"credential:{cred_type}: properties empty")

    doc_url = (
        documentation_key
        if documentation_key and documentation_key.startswith("http")
        else f"n8n://credential/{cred_type}"
    )
    base_meta: Dict[str, Any] = {
        "credentialType": cred_type,
        "displayName": display_name,
        "supportedNodes": supported_nodes,
        "test_request": test_request,
        "documentationKey": documentation_key,
        "title": display_name,
        "url": doc_url,
        "source": source,
    }
    if icon_value is not None:
        base_meta["iconUrl"] = icon_value

    chunks: List[Dict[str, Any]] = []
    overview_meta = dict(base_meta)
    overview_meta.update({"kind": "CRED_OVERVIEW", "field_names": [], "section": "Overview"})
    chunks.append(
        _build_chunk(
            chunk_id=_build_chunk_id(cred_type, "CRED_OVERVIEW"),
            content=_render_credential_overview(
                credential_type=cred_type,
                display_name=display_name,
                documentation_key=documentation_key,
                supported_nodes=supported_nodes,
                icon_url=icon_value,
                test_request=test_request,
            ),
            metadata=overview_meta,
            source=source,
            kind="CRED_OVERVIEW",
            raw_json=credential if include_raw_json else None,
        )
    )
    metrics.add_kind("CRED_OVERVIEW")

    field_groups = _split_in_groups(fields, fields_per_chunk)
    for index, field_group in enumerate(field_groups, start=1):
        fields_meta = dict(base_meta)
        fields_meta.update(
            {
                "kind": "CRED_FIELDS",
                "field_names": _collect_param_names(field_group),
                "section": f"Fields {index}/{len(field_groups)}",
            }
        )
        field_id_parts = [cred_type, "CRED_FIELDS"]
        if len(field_groups) > 1:
            field_id_parts.append(f"part-{index}")
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(*field_id_parts),
                content=_render_credential_fields(
                    credential_type=cred_type,
                    display_name=display_name,
                    fields=field_group,
                    options_preview_limit=options_preview_limit,
                    chunk_index=index,
                    total_chunks=len(field_groups),
                ),
                metadata=fields_meta,
                source=source,
                kind="CRED_FIELDS",
                raw_json={"properties": field_group} if include_raw_json else None,
            )
        )
        metrics.add_kind("CRED_FIELDS")

    for field_prop in fields:
        if not isinstance(field_prop, Mapping):
            continue
        if not _should_build_credential_field_detail(
            field_prop,
            options_preview_limit=options_preview_limit,
        ):
            continue
        field_name = _as_text(field_prop.get("name")) or "unnamed"
        detail_meta = dict(base_meta)
        detail_meta.update(
            {
                "kind": "CRED_FIELD_DETAIL",
                "field_names": [field_name],
                "section": f"Field detail: {field_name}",
            }
        )
        chunks.append(
            _build_chunk(
                chunk_id=_build_chunk_id(cred_type, "CRED_FIELD_DETAIL", field_name),
                content=_render_credential_field_detail(
                    credential_type=cred_type,
                    display_name=display_name,
                    field_prop=field_prop,
                    options_preview_limit=max(options_preview_limit * 3, 20),
                ),
                metadata=detail_meta,
                source=source,
                kind="CRED_FIELD_DETAIL",
                raw_json=field_prop if include_raw_json else None,
            )
        )
        metrics.add_kind("CRED_FIELD_DETAIL")

    return chunks, metrics


def _detect_resource_property(
    properties: Sequence[Any],
) -> Tuple[Optional[Mapping[str, Any]], List[str]]:
    for prop in properties:
        if isinstance(prop, Mapping) and _is_resource_property(prop):
            return prop, _extract_option_values(prop)
    return None, []


def _detect_scopes(
    properties: Sequence[Any],
    resources: Sequence[str],
) -> Tuple[
    List[Scope],
    Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]],
    Dict[Tuple[Optional[str], Optional[str]], Dict[str, List[Any]]],
]:
    resource_list = list(resources)
    resource_ops: Dict[str, List[str]] = {resource: [] for resource in resource_list}
    operation_info: Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]] = {}
    scope_conditions: Dict[Tuple[Optional[str], Optional[str]], Dict[str, List[Any]]] = {}
    global_operations: List[str] = []
    unknown_resources: List[str] = []

    for prop in properties:
        if not isinstance(prop, Mapping) or not _is_operation_property(prop):
            continue
        show = _normalize_show_map(_extract_show_map(prop))
        op_values = _extract_option_values(prop)
        resources_for_op = _as_text_list(show.get("resource"))
        if resources_for_op:
            for resource in resources_for_op:
                if resource not in resource_ops:
                    resource_ops[resource] = []
                    _append_unique(unknown_resources, resource)
                for operation in op_values:
                    _append_unique(resource_ops[resource], operation)
                    operation_info[(resource, operation)] = _extract_operation_detail(prop, operation)
                    scope_conditions[(resource, operation)] = show
            continue
        for operation in op_values:
            _append_unique(global_operations, operation)
            if resource_list:
                for resource in resource_list:
                    _append_unique(resource_ops[resource], operation)
                    operation_info[(resource, operation)] = _extract_operation_detail(prop, operation)
                    scope_conditions[(resource, operation)] = show
            else:
                operation_info[(None, operation)] = _extract_operation_detail(prop, operation)
                scope_conditions[(None, operation)] = show

    if not resource_list and unknown_resources:
        resource_list.extend(unknown_resources)

    scopes: List[Scope] = []
    if resource_list:
        for resource in resource_list:
            operations = resource_ops.get(resource, [])
            if operations:
                for operation in operations:
                    scopes.append(Scope(resource=resource, operation=operation))
            else:
                scopes.append(Scope(resource=resource, operation=None))
    elif global_operations:
        for operation in global_operations:
            scopes.append(Scope(resource=None, operation=operation))

    return _unique_scopes(scopes), operation_info, scope_conditions


def _resolve_property_scopes(
    prop: Mapping[str, Any],
    known_scopes: Sequence[Scope],
) -> List[Scope]:
    show = _normalize_show_map(_extract_show_map(prop))
    resource_filter = _as_text_list(show.get("resource"))
    operation_filter = _as_text_list(show.get("operation"))

    if not resource_filter and not operation_filter:
        return []

    matches: List[Scope] = []
    for scope in known_scopes:
        if resource_filter and scope.resource not in resource_filter:
            continue
        if operation_filter and scope.operation not in operation_filter:
            continue
        matches.append(scope)
    if matches:
        return _unique_scopes(matches)

    fallback: List[Scope] = []
    if resource_filter and operation_filter:
        for resource in resource_filter:
            for operation in operation_filter:
                fallback.append(Scope(resource=resource, operation=operation))
    elif resource_filter:
        for resource in resource_filter:
            fallback.append(Scope(resource=resource, operation=None))
    else:
        for operation in operation_filter:
            fallback.append(Scope(resource=None, operation=operation))
    return _unique_scopes(fallback)


def _flatten_complex_field(root_property: Mapping[str, Any]) -> List[Dict[str, Any]]:
    root_name = _as_text(root_property.get("name")) or _as_text(root_property.get("displayName")) or "unnamed"
    root_show = _normalize_show_map(_extract_show_map(root_property))
    output: List[Dict[str, Any]] = []
    _flatten_complex_recursive(
        prop=root_property,
        current_path=root_name,
        inherited_show=root_show,
        output=output,
    )
    return output


def _flatten_complex_recursive(
    *,
    prop: Mapping[str, Any],
    current_path: str,
    inherited_show: Dict[str, List[Any]],
    output: List[Dict[str, Any]],
) -> None:
    prop_type = _as_text(prop.get("type"))
    local_show = _normalize_show_map(_extract_show_map(prop))
    merged_show = _merge_show_maps(inherited_show, local_show)

    if prop_type == "collection":
        options = prop.get("options")
        if not isinstance(options, list):
            return
        for child in options:
            if not isinstance(child, Mapping):
                continue
            child_name = _as_text(child.get("name")) or _as_text(child.get("displayName")) or "unnamed"
            child_path = f"{current_path}.{child_name}"
            child_type = _as_text(child.get("type"))
            child_show = _merge_show_maps(merged_show, _normalize_show_map(_extract_show_map(child)))
            if child_type in COMPLEX_TYPES:
                _flatten_complex_recursive(
                    prop=child,
                    current_path=child_path,
                    inherited_show=child_show,
                    output=output,
                )
            else:
                output.append(_build_flat_field_entry(child, child_path, child_show))
        return

    if prop_type == "fixedCollection":
        options = prop.get("options")
        if not isinstance(options, list):
            return
        for group in options:
            if not isinstance(group, Mapping):
                continue
            group_name = _as_text(group.get("name")) or _as_text(group.get("displayName")) or "unnamed"
            group_show = _merge_show_maps(merged_show, _normalize_show_map(_extract_show_map(group)))
            values = group.get("values")
            if not isinstance(values, list):
                continue
            for child in values:
                if not isinstance(child, Mapping):
                    continue
                child_name = _as_text(child.get("name")) or _as_text(child.get("displayName")) or "unnamed"
                child_path = f"{current_path}.{group_name}.{child_name}"
                child_type = _as_text(child.get("type"))
                child_show = _merge_show_maps(group_show, _normalize_show_map(_extract_show_map(child)))
                if child_type in COMPLEX_TYPES:
                    _flatten_complex_recursive(
                        prop=child,
                        current_path=child_path,
                        inherited_show=child_show,
                        output=output,
                    )
                else:
                    output.append(_build_flat_field_entry(child, child_path, child_show))
        return


def _build_flat_field_entry(
    prop: Mapping[str, Any],
    path: str,
    show: Dict[str, List[Any]],
) -> Dict[str, Any]:
    return {
        "path": path,
        "name": _as_text(prop.get("name")) or path,
        "displayName": _as_text(prop.get("displayName")),
        "type": _as_text(prop.get("type")),
        "required": bool(prop.get("required")),
        "default": prop.get("default"),
        "description": _as_text(prop.get("description")),
        "options": prop.get("options") if isinstance(prop.get("options"), list) else None,
        "show": show,
        "secret": _is_secret_field(prop),
    }


def _extract_operation_detail(prop: Mapping[str, Any], operation_value: str) -> Dict[str, Any]:
    options = prop.get("options")
    if not isinstance(options, list):
        return {"value": operation_value}
    for option in options:
        if not isinstance(option, Mapping):
            continue
        if _as_text(option.get("value")) != operation_value:
            continue
        return {
            "value": operation_value,
            "name": _as_text(option.get("name")),
            "action": _as_text(option.get("action")),
            "description": _as_text(option.get("description")),
        }
    return {"value": operation_value}


def _extract_node_doc_urls(node: Mapping[str, Any]) -> List[str]:
    codex = node.get("codex")
    if not isinstance(codex, Mapping):
        return []
    resources = codex.get("resources")
    if not isinstance(resources, Mapping):
        return []
    urls: List[str] = []
    for key in ("primaryDocumentation", "credentialDocumentation"):
        value = resources.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            url = _as_text(item.get("url"))
            if url:
                _append_unique(urls, url)
    return urls


def _extract_node_credential_types(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    names: List[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _as_text(item.get("name"))
        if name:
            _append_unique(names, name)
    return names


def _extract_credential_test_request(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    request = value.get("request")
    if not isinstance(request, Mapping):
        return None
    payload: Dict[str, Any] = {}
    for key in ("baseURL", "url", "method"):
        if key in request:
            payload[key] = request.get(key)
    if "qs" in request:
        payload["qs"] = request.get("qs")
    return payload or None


def _build_chunk(
    *,
    chunk_id: str,
    content: str,
    metadata: MutableMapping[str, Any],
    source: str,
    kind: str,
    raw_json: Optional[Any] = None,
) -> Dict[str, Any]:
    metadata["id"] = chunk_id
    metadata["kind"] = kind
    metadata["source"] = source
    metadata["contentHash"] = fnv1a32(content)
    if raw_json is not None:
        metadata["rawHash"] = fnv1a32(_canonical_json(raw_json))
        metadata["raw_json"] = raw_json
    else:
        metadata["rawHash"] = None

    chunk: Dict[str, Any] = {
        "id": chunk_id,
        "content": content,
        "metadata": dict(metadata),
        "source": source,
        "kind": kind,
    }
    if raw_json is not None:
        chunk["raw_json"] = raw_json
    return chunk


def _build_chunk_id(*parts: Any) -> str:
    encoded_parts = [_sanitize_id_part(part) for part in parts if part is not None]
    return "|".join(encoded_parts)


def _scope_id_parts(scope: Scope) -> Tuple[str, str]:
    return (
        f"resource={_scope_id_value(scope.resource)}",
        f"operation={_scope_id_value(scope.operation)}",
    )


def _scope_id_value(value: Optional[str]) -> str:
    text = _as_text(value)
    if not text:
        return "__none__"
    return text


def _scope_value(value: Optional[str]) -> str:
    text = _as_text(value)
    if not text:
        return "-"
    return text


def _scope_label(resource: Optional[str], operation: Optional[str]) -> str:
    if resource is None and operation is None:
        return "global"
    return f"resource={_scope_value(resource)}, operation={_scope_value(operation)}"


def _scope_section(scope: Scope, suffix: Optional[str] = None) -> str:
    base = _scope_label(scope.resource, scope.operation)
    if suffix:
        return f"{base} {suffix}"
    return base


def _sanitize_id_part(value: Any) -> str:
    text = _as_text(value) or "na"
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._:-]", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text or "na"


def _should_build_credential_field_detail(
    field_prop: Mapping[str, Any],
    *,
    options_preview_limit: int,
) -> bool:
    options = field_prop.get("options")
    if isinstance(options, list) and len(options) > max(options_preview_limit, 10):
        return True
    prop_type = _as_text(field_prop.get("type"))
    if prop_type in COMPLEX_TYPES:
        return True
    description = (
        (_as_text(field_prop.get("name")) or "")
        + " "
        + (_as_text(field_prop.get("displayName")) or "")
        + " "
        + (_as_text(field_prop.get("description")) or "")
    ).lower()
    if any(token in description for token in ("allow", "domain", "scope", "oauth", "token")):
        return True
    return bool(_extract_show_map(field_prop))


def _is_secret_field(prop: Mapping[str, Any]) -> bool:
    type_options = prop.get("typeOptions")
    return bool(isinstance(type_options, Mapping) and type_options.get("password"))


def _is_resource_property(prop: Mapping[str, Any]) -> bool:
    return _as_text(prop.get("name")) == "resource" and _as_text(prop.get("type")) == "options"


def _is_operation_property(prop: Mapping[str, Any]) -> bool:
    return _as_text(prop.get("name")) == "operation" and _as_text(prop.get("type")) == "options"


def _is_complex_property(prop: Mapping[str, Any]) -> bool:
    return _as_text(prop.get("type")) in COMPLEX_TYPES


def _has_scope_signals(properties: Sequence[Any]) -> bool:
    for prop in properties:
        if not isinstance(prop, Mapping):
            continue
        if _is_resource_property(prop) or _is_operation_property(prop):
            return True
        show = _normalize_show_map(_extract_show_map(prop))
        if "resource" in show or "operation" in show:
            return True
    return False


def _extract_option_values(prop: Mapping[str, Any]) -> List[str]:
    options = prop.get("options")
    if not isinstance(options, list):
        return []
    values: List[str] = []
    for option in options:
        if not isinstance(option, Mapping):
            continue
        value = _as_text(option.get("value"))
        if value:
            _append_unique(values, value)
    return values


def _extract_show_map(prop: Any) -> Dict[str, Any]:
    if not isinstance(prop, Mapping):
        return {}
    display_options = prop.get("displayOptions")
    if not isinstance(display_options, Mapping):
        return {}
    show = display_options.get("show")
    if not isinstance(show, Mapping):
        return {}
    return dict(show)


def _normalize_show_map(show: Any) -> Dict[str, List[Any]]:
    if not isinstance(show, Mapping):
        return {}
    normalized: Dict[str, List[Any]] = {}
    for key in sorted(show.keys()):
        values = show.get(key)
        if isinstance(values, list):
            normalized[key] = list(values)
        else:
            normalized[key] = [values]
    return normalized


def _merge_show_maps(
    first: Mapping[str, List[Any]],
    second: Mapping[str, List[Any]],
) -> Dict[str, List[Any]]:
    merged: Dict[str, List[Any]] = {}
    for key in sorted(set(list(first.keys()) + list(second.keys()))):
        merged_values: List[Any] = []
        for source in (first, second):
            for value in source.get(key, []):
                if value not in merged_values:
                    merged_values.append(value)
        merged[key] = merged_values
    return merged


def _render_node_overview(
    *,
    node_type: str,
    display_name: str,
    description: str,
    group: Sequence[str],
    version: Any,
    inputs: Sequence[str],
    outputs: Sequence[str],
    usable_as_tool: bool,
    credential_types: Sequence[str],
    doc_urls: Sequence[str],
) -> str:
    return "\n".join(
        [
            "Kind: NODE_OVERVIEW",
            f"Node Type: {node_type}",
            f"Display Name: {display_name}",
            f"Description: {description or '-'}",
            f"Group: {_render_list(group)}",
            f"Version: {_value_text(version)}",
            f"Inputs: {_render_list(inputs)}",
            f"Outputs: {_render_list(outputs)}",
            f"Usable As Tool: {_value_text(usable_as_tool)}",
            f"Required Credentials: {_render_list(credential_types)}",
            f"Documentation URLs: {_render_list(doc_urls)}",
        ]
    )


def _render_node_resource(
    *,
    node_type: str,
    display_name: str,
    resources: Sequence[str],
) -> str:
    lines = ["Kind: NODE_RESOURCE", f"Node Type: {node_type}", f"Display Name: {display_name}", "Resources:"]
    lines.extend(f"- {resource}" for resource in resources)
    return "\n".join(lines)


def _render_node_resource_operation(
    *,
    node_type: str,
    display_name: str,
    resource: Optional[str],
    operation: Optional[str],
    operation_name: str,
    action: str,
    description: str,
    operation_conditions: Dict[str, List[Any]],
) -> str:
    lines = [
        "Kind: NODE_RESOURCE_OPERATION",
        f"Node Type: {node_type}",
        f"Display Name: {display_name}",
        f"Resource: {_scope_value(resource)}",
        f"Operation: {_scope_value(operation)}",
    ]
    if operation_name:
        lines.append(f"Operation Label: {operation_name}")
    if action:
        lines.append(f"Action: {action}")
    if description:
        lines.append(f"Description: {description}")
    shown = _render_shown_when(operation_conditions)
    if shown:
        lines.append(shown)
    return "\n".join(lines)


def _render_node_params_for_scope(
    *,
    node_type: str,
    display_name: str,
    resource: Optional[str],
    operation: Optional[str],
    params: Sequence[Mapping[str, Any]],
    options_preview_limit: int,
) -> str:
    lines = [
        "Kind: NODE_PARAMS_FOR_RESOURCE_OPERATION",
        f"Node Type: {node_type}",
        f"Display Name: {display_name}",
        f"Scope: {_scope_label(resource, operation)}",
        "Parameters:",
    ]
    if not params:
        lines.append("- (none)")
    else:
        for prop in params:
            lines.append(_render_property_line(prop, options_preview_limit=options_preview_limit))
    return "\n".join(lines)


def _render_node_global_params(
    *,
    node_type: str,
    display_name: str,
    params: Sequence[Mapping[str, Any]],
    options_preview_limit: int,
) -> str:
    lines = [
        "Kind: NODE_GLOBAL_PARAMS",
        f"Node Type: {node_type}",
        f"Display Name: {display_name}",
        "Scope: global",
        "Parameters:",
    ]
    for prop in params:
        lines.append(_render_property_line(prop, options_preview_limit=options_preview_limit))
    return "\n".join(lines)


def _render_node_complex_field(
    *,
    node_type: str,
    display_name: str,
    root_name: str,
    resource: Optional[str],
    operation: Optional[str],
    entries: Sequence[Mapping[str, Any]],
    options_preview_limit: int,
) -> str:
    scope_label = _scope_label(resource, operation)
    lines = [
        "Kind: NODE_COMPLEX_FIELD",
        f"Node Type: {node_type}",
        f"Display Name: {display_name}",
        f"Group Path: {root_name}",
        f"Scope: {scope_label}",
        "Flattened Fields:",
    ]
    for entry in entries:
        lines.append(_render_flat_field_line(entry, options_preview_limit=options_preview_limit))
    return "\n".join(lines)


def _render_credential_overview(
    *,
    credential_type: str,
    display_name: str,
    documentation_key: str,
    supported_nodes: Sequence[str],
    icon_url: Any,
    test_request: Optional[Dict[str, Any]],
) -> str:
    lines = [
        "Kind: CRED_OVERVIEW",
        f"Credential Type: {credential_type}",
        f"Display Name: {display_name}",
        f"Documentation Key: {documentation_key or '-'}",
        f"Supported Nodes: {_render_list(supported_nodes)}",
        f"Icon URL: {_value_text(icon_url)}",
    ]
    if test_request:
        lines.extend(
            [
                "Test Request:",
                f"- baseURL: {_value_text(test_request.get('baseURL'))}",
                f"- url: {_value_text(test_request.get('url'))}",
                f"- method: {_value_text(test_request.get('method'))}",
            ]
        )
    return "\n".join(lines)


def _render_credential_fields(
    *,
    credential_type: str,
    display_name: str,
    fields: Sequence[Mapping[str, Any]],
    options_preview_limit: int,
    chunk_index: int,
    total_chunks: int,
) -> str:
    lines = [
        "Kind: CRED_FIELDS",
        f"Credential Type: {credential_type}",
        f"Display Name: {display_name}",
        f"Chunk: {chunk_index}/{total_chunks}",
        "Fields:",
    ]
    if not fields:
        lines.append("- (none)")
    else:
        for field_prop in fields:
            lines.append(_render_property_line(field_prop, options_preview_limit=options_preview_limit))
    return "\n".join(lines)


def _render_credential_field_detail(
    *,
    credential_type: str,
    display_name: str,
    field_prop: Mapping[str, Any],
    options_preview_limit: int,
) -> str:
    field_name = _as_text(field_prop.get("name")) or "unnamed"
    return "\n".join(
        [
            "Kind: CRED_FIELD_DETAIL",
            f"Credential Type: {credential_type}",
            f"Display Name: {display_name}",
            f"Field: {field_name}",
            _render_property_line(field_prop, options_preview_limit=options_preview_limit),
        ]
    )


def _render_property_line(prop: Mapping[str, Any], *, options_preview_limit: int) -> str:
    name = _as_text(prop.get("name")) or "unnamed"
    display_name = _as_text(prop.get("displayName"))
    prop_type = _as_text(prop.get("type")) or "unknown"
    required = bool(prop.get("required"))
    default_value = prop.get("default")
    description = _as_text(prop.get("description"))
    options = prop.get("options") if isinstance(prop.get("options"), list) else None
    shown = _render_shown_when(_normalize_show_map(_extract_show_map(prop)))
    parts = [
        f"- {name}" if not display_name or display_name == name else f"- {name} ({display_name})",
        f"type={prop_type}",
        f"required={_value_text(required)}",
        f"default={_value_text(default_value)}",
        f"secret={_value_text(_is_secret_field(prop))}",
    ]
    if description:
        parts.append(f"description={description}")
    options_text = _render_options_preview(options, options_preview_limit)
    if options_text:
        parts.append(f"options={options_text}")
    if shown:
        parts.append(shown)
    return " | ".join(parts)


def _render_flat_field_line(entry: Mapping[str, Any], *, options_preview_limit: int) -> str:
    path = _as_text(entry.get("path")) or "unnamed"
    display_name = _as_text(entry.get("displayName"))
    entry_type = _as_text(entry.get("type")) or "unknown"
    description = _as_text(entry.get("description"))
    options = entry.get("options") if isinstance(entry.get("options"), list) else None
    shown = _render_shown_when(_normalize_show_map(entry.get("show")))
    parts = [
        f"- {path}" if not display_name else f"- {path} ({display_name})",
        f"type={entry_type}",
        f"required={_value_text(bool(entry.get('required')))}",
        f"default={_value_text(entry.get('default'))}",
        f"secret={_value_text(bool(entry.get('secret')))}",
    ]
    if description:
        parts.append(f"description={description}")
    options_text = _render_options_preview(options, options_preview_limit)
    if options_text:
        parts.append(f"options={options_text}")
    if shown:
        parts.append(shown)
    return " | ".join(parts)


def _render_shown_when(show: Mapping[str, List[Any]]) -> str:
    if not show:
        return ""
    fragments: List[str] = []
    for key in sorted(show.keys()):
        values = ", ".join(_value_text(value) for value in show.get(key, []))
        fragments.append(f"{key} in [{values}]")
    return "shown when " + " and ".join(fragments)


def _render_options_preview(options: Optional[Sequence[Any]], limit: int) -> str:
    if not options:
        return ""
    if limit <= 0:
        limit = DEFAULT_OPTIONS_PREVIEW
    preview: List[str] = []
    for item in list(options)[:limit]:
        if isinstance(item, Mapping):
            name = _as_text(item.get("name"))
            value = _as_text(item.get("value"))
            if name and value:
                preview.append(f"{name}={value}")
            elif value:
                preview.append(value)
            elif name:
                preview.append(name)
            else:
                preview.append(_value_text(item))
        else:
            preview.append(_value_text(item))
    remaining = len(options) - len(preview)
    rendered = ", ".join(preview)
    if remaining > 0:
        rendered += f", (+{remaining} more)"
    return rendered


def _collect_param_names(items: Sequence[Any]) -> List[str]:
    names: List[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = _as_text(item.get("name"))
        if name:
            _append_unique(names, name)
    return names


def _split_in_groups(items: Sequence[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [list(items)]
    output: List[List[Any]] = []
    current: List[Any] = []
    for item in items:
        current.append(item)
        if len(current) >= size:
            output.append(current)
            current = []
    if current:
        output.append(current)
    if not output:
        output.append([])
    return output


def _append_unique(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _render_list(values: Sequence[Any]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_value_text(value) for value in values) + "]"


def _value_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _as_text_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = _as_text(item)
        if text:
            _append_unique(output, text)
    return output


def _unique_scopes(scopes: Iterable[Scope]) -> List[Scope]:
    output: List[Scope] = []
    seen = set()
    for scope in scopes:
        key = (scope.resource, scope.operation)
        if key in seen:
            continue
        seen.add(key)
        output.append(scope)
    return output


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
