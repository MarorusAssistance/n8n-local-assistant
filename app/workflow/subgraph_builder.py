from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

from ..config import settings
from .planner import Plan, PlanScope
from .workflow_summary import EdgeSummary, WorkflowSummary


@dataclass
class SubgraphSelection:
    node_ids: List[str]
    edges: List[EdgeSummary]
    truncated: bool
    reason: str


def _shortest_path(
    summary: WorkflowSummary,
    start: str,
    goal: str,
    max_depth: int = 10,
) -> List[str]:
    """Return a BFS path between two nodes, bounded by max_depth."""
    if start == goal:
        return [start]
    if start not in summary.nodes or goal not in summary.nodes:
        return []

    queue: deque[Tuple[str, List[str], int]] = deque([(start, [start], 0)])
    visited: Set[str] = {start}

    while queue:
        node_id, path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in summary.neighbors(node_id):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path, depth + 1))
    return []


def _expand_neighbors(
    summary: WorkflowSummary,
    seeds: Iterable[str],
    max_nodes: int,
) -> List[str]:
    """Grow a neighborhood from seed nodes until max_nodes is reached."""
    seeds = [node_id for node_id in seeds if node_id in summary.nodes]
    if not seeds:
        return []

    selected: List[str] = []
    seen: Set[str] = set(seeds)
    queue: deque[str] = deque(seeds)

    while queue and len(selected) < max_nodes:
        node_id = queue.popleft()
        if node_id not in summary.nodes:
            continue
        if node_id not in selected:
            selected.append(node_id)
        if len(selected) >= max_nodes:
            break
        for neighbor in summary.neighbors(node_id):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
            if len(seen) > max_nodes * 3:
                break
    return selected[:max_nodes]


def _select_edges(summary: WorkflowSummary, node_ids: Sequence[str]) -> List[EdgeSummary]:
    """Filter edges where both endpoints are in the selected node set."""
    node_set = set(node_ids)
    edges: List[EdgeSummary] = []
    for edge in summary.edges:
        if edge.source_id in node_set and edge.target_id in node_set:
            edges.append(edge)
    return edges


def _score_global(summary: WorkflowSummary) -> List[str]:
    """Rank nodes by heuristic importance for global workflow questions."""
    scored: List[Tuple[float, str]] = []
    for node in summary.nodes.values():
        score = 0.0
        short = node.short_type.lower()
        if any(token in short for token in ("trigger", "webhook", "cron")):
            # 1.4: triggers define workflow entrypoints.
            score += 1.4
        if node.has_credentials:
            # 0.9: auth/config issues are frequent root causes.
            score += 0.9
        if node.has_expressions:
            # 0.9: expressions correlate with subtle failures.
            score += 0.9
        if node.is_pivot:
            # 1.1: control-flow nodes (if/merge/switch) affect many branches.
            score += 1.1
        if any(
            token in short
            for token in (
                "http",
                "request",
                "code",
                "postgres",
                "mysql",
                "slack",
                "gmail",
            )
        ):
            # 0.6: common integration nodes are high-value for global reviews.
            score += 0.6
        if node.disabled:
            # 0.2: disabled nodes can hint at misconfiguration.
            score += 0.2
        if score > 0:
            scored.append((score, node.node_id))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [node_id for _, node_id in scored]


def _path_between_targets(summary: WorkflowSummary, targets: Sequence[str], max_nodes: int) -> List[str]:
    """Connect target nodes by shortest paths, limited to max_nodes."""
    if len(targets) < 2:
        return list(targets)

    selected: List[str] = []
    for node_id in targets:
        if node_id not in selected:
            selected.append(node_id)

    pair_targets = list(targets)[:4]
    for index, start in enumerate(pair_targets):
        for goal in pair_targets[index + 1 :]:
            path = _shortest_path(summary, start, goal, max_depth=8)
            for node_id in path:
                if node_id not in selected:
                    selected.append(node_id)
                if len(selected) >= max_nodes:
                    return selected[:max_nodes]
    return selected[:max_nodes]


def build_subgraph(summary: WorkflowSummary, question: str, plan: Plan) -> SubgraphSelection:
    """Select a subgraph based on plan scope (node-specific, subgraph, global)."""
    if plan.scope == PlanScope.node_specific:
        max_nodes = settings.WORKFLOW_MAX_NODES_NODE_SPECIFIC
        node_ids = _expand_neighbors(summary, plan.target_node_ids, max_nodes=max_nodes)
        edges = _select_edges(summary, node_ids)
        truncated = len(node_ids) >= max_nodes
        return SubgraphSelection(
            node_ids=node_ids,
            edges=edges,
            truncated=truncated,
            reason="node_specific",
        )

    if plan.scope == PlanScope.subgraph:
        max_nodes = settings.WORKFLOW_MAX_NODES_SUBGRAPH
        seeds = plan.target_node_ids
        node_ids = _path_between_targets(summary, seeds, max_nodes=max_nodes)
        if len(node_ids) < max_nodes:
            node_ids = _expand_neighbors(summary, node_ids, max_nodes=max_nodes)
        edges = _select_edges(summary, node_ids)
        truncated = len(node_ids) >= max_nodes
        return SubgraphSelection(node_ids=node_ids, edges=edges, truncated=truncated, reason="subgraph")

    if plan.scope == PlanScope.workflow_global:
        max_nodes = settings.WORKFLOW_MAX_NODES_GLOBAL
        ranked = _score_global(summary)
        node_ids = ranked[:max_nodes]
        if not node_ids:
            node_ids = list(summary.nodes.keys())[:max_nodes]
        edges = _select_edges(summary, node_ids)
        truncated = len(summary.nodes) > max_nodes
        return SubgraphSelection(
            node_ids=node_ids,
            edges=edges,
            truncated=truncated,
            reason="workflow_global",
        )

    # no_workflow fallback
    return SubgraphSelection(node_ids=[], edges=[], truncated=False, reason=plan.scope.value)
