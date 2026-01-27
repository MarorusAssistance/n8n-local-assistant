from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import json

from ..config import settings


PIVOT_TYPE_KEYWORDS = ("if", "merge", "switch", "router")


def _short_type(node_type: str) -> str:
    """Extract the short node type name (suffix after last dot)."""
    if not node_type:
        return "unknown"
    return node_type.split(".")[-1]


def _truncate_text(text: str, max_chars: int) -> str:
    """Trim and normalize whitespace to keep previews concise."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _has_expression(value: Any, depth: int = 0, max_depth: int = 4) -> bool:
    """Detect template expressions like {{ }} in nested parameters."""
    if depth > max_depth:
        return False
    if isinstance(value, str):
        return "{{" in value or "}}" in value
    if isinstance(value, list):
        return any(_has_expression(item, depth + 1, max_depth) for item in value[:12])
    if isinstance(value, dict):
        for _, item in list(value.items())[:16]:
            if _has_expression(item, depth + 1, max_depth):
                return True
    return False


def _preview_parameters(parameters: Any, max_chars: int) -> str:
    """Generate a small JSON preview of scalar parameters for debugging."""
    if not isinstance(parameters, dict) or not parameters:
        return ""

    preview: Dict[str, Any] = {}
    for key, value in list(parameters.items())[:12]:
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
            preview[key] = _truncate_text(text, 120)
            continue
        if isinstance(value, dict):
            # keep only shallow scalar hints to avoid huge payloads
            scalar_hints: Dict[str, Any] = {}
            for sub_key, sub_value in list(value.items())[:8]:
                if isinstance(sub_value, (str, int, float, bool)):
                    scalar_hints[sub_key] = _truncate_text(str(sub_value), 80)
            if scalar_hints:
                preview[key] = scalar_hints
            continue
        if isinstance(value, list):
            if value and isinstance(value[0], (str, int, float, bool)) and len(value) <= 6:
                preview[key] = [
                    _truncate_text(str(item), 60) for item in value[:6]
                ]
            continue

    if not preview:
        keys = list(parameters.keys())[:10]
        return f"keys={keys}"

    text = json.dumps(preview, ensure_ascii=True)
    return _truncate_text(text, max_chars)


@dataclass(frozen=True)
class NodeSummary:
    node_id: str
    name: str
    node_type: str
    short_type: str
    disabled: bool
    has_credentials: bool
    has_expressions: bool
    parameter_keys: Tuple[str, ...]
    parameters_preview: str
    position: Tuple[int, int]

    @property
    def is_pivot(self) -> bool:
        lowered = self.short_type.lower()
        return any(token in lowered for token in PIVOT_TYPE_KEYWORDS)


@dataclass(frozen=True)
class EdgeSummary:
    source_id: str
    target_id: str
    source_output: str
    target_input: str


@dataclass
class WorkflowSummary:
    workflow_id: str
    name: str
    nodes: Dict[str, NodeSummary]
    edges: List[EdgeSummary]
    truncated: bool

    def __post_init__(self) -> None:
        self._outgoing: Dict[str, List[EdgeSummary]] = {}
        self._incoming: Dict[str, List[EdgeSummary]] = {}
        for edge in self.edges:
            self._outgoing.setdefault(edge.source_id, []).append(edge)
            self._incoming.setdefault(edge.target_id, []).append(edge)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def neighbors(self, node_id: str) -> Set[str]:
        neighbors: Set[str] = set()
        for edge in self._outgoing.get(node_id, []):
            neighbors.add(edge.target_id)
        for edge in self._incoming.get(node_id, []):
            neighbors.add(edge.source_id)
        return neighbors

    def outgoing(self, node_id: str) -> List[EdgeSummary]:
        return list(self._outgoing.get(node_id, []))

    def incoming(self, node_id: str) -> List[EdgeSummary]:
        return list(self._incoming.get(node_id, []))


def _to_node_summary(node: Dict[str, Any], max_chars: int) -> NodeSummary:
    """Convert raw n8n node JSON into a compact NodeSummary."""
    node_id = str(node.get("id") or node.get("name") or "unknown")
    name = str(node.get("name") or node_id)
    node_type = str(node.get("type") or "")
    short_type = _short_type(node_type)
    disabled = bool(node.get("disabled") or False)
    credentials = node.get("credentials")
    has_credentials = isinstance(credentials, dict) and bool(credentials)
    parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
    has_expressions = _has_expression(parameters)
    parameter_keys = tuple(list(parameters.keys())[:16])
    parameters_preview = _preview_parameters(parameters, max_chars=max_chars)

    position_raw = node.get("position") if isinstance(node.get("position"), list) else [0, 0]
    try:
        position = (int(position_raw[0]), int(position_raw[1]))
    except Exception:  # pragma: no cover - defensive
        position = (0, 0)

    return NodeSummary(
        node_id=node_id,
        name=name,
        node_type=node_type,
        short_type=short_type,
        disabled=disabled,
        has_credentials=has_credentials,
        has_expressions=has_expressions,
        parameter_keys=parameter_keys,
        parameters_preview=parameters_preview,
        position=position,
    )


def _parse_edges(
    connections: Dict[str, Any],
    name_to_id: Dict[str, str],
    allowed_node_ids: Set[str],
) -> List[EdgeSummary]:
    """Translate n8n connections to compact EdgeSummary entries."""
    edges: List[EdgeSummary] = []
    if not isinstance(connections, dict):
        return edges

    for source_name, outputs in connections.items():
        source_id = name_to_id.get(str(source_name))
        if not source_id or source_id not in allowed_node_ids:
            continue
        if not isinstance(outputs, dict):
            continue

        for output_key, outputs_lists in outputs.items():
            if not isinstance(outputs_lists, list):
                continue
            for output_index, output_list in enumerate(outputs_lists):
                if not isinstance(output_list, list):
                    continue
                for conn in output_list:
                    if not isinstance(conn, dict):
                        continue
                    target_name = str(conn.get("node") or "")
                    target_id = name_to_id.get(target_name)
                    if not target_id or target_id not in allowed_node_ids:
                        continue
                    target_type = str(conn.get("type") or "main")
                    target_index = int(conn.get("index") or 0)
                    source_output = f"{output_key}[{output_index}]"
                    target_input = f"{target_type}[{target_index}]"
                    edges.append(
                        EdgeSummary(
                            source_id=source_id,
                            target_id=target_id,
                            source_output=source_output,
                            target_input=target_input,
                        )
                    )
    return edges


def build_workflow_summary(
    workflow: Dict[str, Any], workflow_id: str, max_nodes: Optional[int] = None
) -> WorkflowSummary:
    """Create a summary with bounded nodes/edges for downstream heuristics."""
    max_nodes = max_nodes or settings.WORKFLOW_SUMMARY_MAX_NODES
    nodes_raw = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []

    truncated = len(nodes_raw) > max_nodes
    nodes_slice = nodes_raw[:max_nodes]

    node_summaries: Dict[str, NodeSummary] = {}
    name_to_id: Dict[str, str] = {}

    for node in nodes_slice:
        if not isinstance(node, dict):
            continue
        summary = _to_node_summary(node, max_chars=settings.WORKFLOW_MICRO_MAX_CHARS // 2)
        node_summaries[summary.node_id] = summary
        name_to_id[summary.name] = summary.node_id

    allowed_node_ids = set(node_summaries.keys())
    edges = _parse_edges(workflow.get("connections") or {}, name_to_id, allowed_node_ids)

    workflow_name = str(workflow.get("name") or workflow_id)

    return WorkflowSummary(
        workflow_id=str(workflow_id),
        name=workflow_name,
        nodes=node_summaries,
        edges=edges,
        truncated=truncated,
    )


def _collect_neighbors(
    summary: WorkflowSummary, seed_ids: Iterable[str], depth: int, limit: int
) -> List[str]:
    """Collect neighbors around seed nodes using BFS up to depth/limit."""
    visited: List[str] = []
    queue: List[Tuple[str, int]] = [(node_id, 0) for node_id in seed_ids if node_id in summary.nodes]
    seen: Set[str] = set(node_id for node_id, _ in queue)

    while queue and len(visited) < limit:
        node_id, dist = queue.pop(0)
        if node_id not in summary.nodes:
            continue
        visited.append(node_id)
        if dist >= depth:
            continue
        for neighbor in summary.neighbors(node_id):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, dist + 1))
            if len(seen) >= limit * 2:
                break

    return visited[:limit]


def micro_context_for_nodes(
    summary: WorkflowSummary,
    focus_node_ids: Iterable[str],
    depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a small context bundle around focus nodes."""
    depth = depth if depth is not None else settings.WORKFLOW_NEIGHBOR_DEPTH
    max_nodes = max_nodes if max_nodes is not None else settings.WORKFLOW_MAX_NODES_NODE_SPECIFIC

    focus_ids = [node_id for node_id in focus_node_ids if node_id in summary.nodes]
    selected_ids = _collect_neighbors(summary, focus_ids, depth=depth, limit=max_nodes)

    nodes_payload: List[Dict[str, Any]] = []
    for node_id in selected_ids:
        node = summary.nodes[node_id]
        nodes_payload.append(
            {
                "id": node.node_id,
                "name": node.name,
                "type": node.node_type,
                "short_type": node.short_type,
                "disabled": node.disabled,
                "has_credentials": node.has_credentials,
                "has_expressions": node.has_expressions,
                "parameter_keys": list(node.parameter_keys),
                "parameters_preview": node.parameters_preview,
            }
        )

    selected_set = set(selected_ids)
    edges_payload: List[Dict[str, str]] = []
    for edge in summary.edges:
        if edge.source_id in selected_set and edge.target_id in selected_set:
            edges_payload.append(
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "source_output": edge.source_output,
                    "target_input": edge.target_input,
                }
            )

    return {
        "workflow_id": summary.workflow_id,
        "workflow_name": summary.name,
        "focus_node_ids": focus_ids,
        "node_ids": selected_ids,
        "nodes": nodes_payload,
        "edges": edges_payload,
        "truncated": summary.truncated or len(selected_ids) >= max_nodes,
    }


def render_summary_for_planner(summary: WorkflowSummary, max_nodes: int = 60) -> str:
    """Render a compact text summary used by the planner."""
    lines: List[str] = [
        f"Workflow: {summary.name} (id={summary.workflow_id})",
        f"Nodes: {summary.node_count} | Edges: {len(summary.edges)}",
    ]
    if summary.truncated:
        lines.append("Note: workflow summary was truncated for safety.")

    lines.append("Nodes (id | name | type | signals):")
    nodes_list = list(summary.nodes.values())[:max_nodes]
    for node in nodes_list:
        signals: List[str] = []
        if node.disabled:
            signals.append("disabled")
        if node.has_credentials:
            signals.append("creds")
        if node.has_expressions:
            signals.append("expr")
        signal_text = ",".join(signals) if signals else "-"
        lines.append(
            f"- {node.node_id} | {node.name} | {node.short_type} | {signal_text}"
        )

    edge_limit = max_nodes * 2
    if summary.edges:
        lines.append("Edges (source -> target):")
        for edge in summary.edges[:edge_limit]:
            lines.append(f"- {edge.source_id} -> {edge.target_id}")
    return "\n".join(lines)


def render_micro_context(micro_context: Dict[str, Any]) -> str:
    """Render micro-context into a compact prompt-friendly string."""
    nodes = micro_context.get("nodes") or []
    edges = micro_context.get("edges") or []
    focus_ids = micro_context.get("focus_node_ids") or []

    lines: List[str] = [
        f"Workflow: {micro_context.get('workflow_name')} (id={micro_context.get('workflow_id')})",
        f"Focus nodes: {focus_ids}",
        f"Context nodes: {micro_context.get('node_ids')}",
        "Nodes detail:",
    ]

    for node in nodes:
        lines.append(
            "- {id} | {name} | {short_type} | creds={creds} expr={expr} disabled={disabled}".format(
                id=node.get("id"),
                name=node.get("name"),
                short_type=node.get("short_type"),
                creds=node.get("has_credentials"),
                expr=node.get("has_expressions"),
                disabled=node.get("disabled"),
            )
        )
        preview = str(node.get("parameters_preview") or "").strip()
        if preview:
            lines.append(f"  params: {preview}")

    if edges:
        lines.append("Edges detail:")
        for edge in edges[:48]:
            lines.append(
                f"- {edge.get('source')} ({edge.get('source_output')}) -> {edge.get('target')} ({edge.get('target_input')})"
            )

    if micro_context.get("truncated"):
        lines.append("Note: micro context truncated to respect limits.")

    text = "\n".join(lines)
    return _truncate_text(text, settings.WORKFLOW_MICRO_MAX_CHARS)
