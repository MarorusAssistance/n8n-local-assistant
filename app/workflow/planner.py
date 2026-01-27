from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from ..config import settings
from .workflow_summary import NodeSummary, WorkflowSummary


class PlanScope(str, Enum):
    no_workflow = "no_workflow"
    node_specific = "node_specific"
    subgraph = "subgraph"
    workflow_global = "workflow_global"


class Plan(BaseModel):
    scope: PlanScope
    target_node_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    need_clarification: bool = False
    clarifying_question: Optional[str] = None
    candidates: List[Dict[str, object]] = Field(default_factory=list)


GLOBAL_KEYWORDS = (
    "en general",
    "global",
    "todo el flujo",
    "todo el workflow",
    "mi flujo",
    "mi workflow",
    "arquitectura",
    "diseno",
    "mejorar",
    "optimizar",
    "que falla",
    "por que falla",
    "falla mi flujo",
    "overall",
    "entire workflow",
    "whole workflow",
)

TYPE_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "httprequest": ("http", "request", "api", "endpoint", "rest"),
    "webhook": ("webhook", "trigger", "disparador"),
    "merge": ("merge", "unir", "combinar", "join"),
    "if": ("if", "condicion", "condition", "branch"),
    "switch": ("switch", "ruta", "router", "ramas"),
    "code": ("code", "function", "javascript", "js", "python"),
    "set": ("set", "asignar", "map", "mapper"),
    "postgres": ("postgres", "sql", "database", "db", "query"),
}

TRIGGER_KEYWORDS = ("trigger", "webhook", "schedule", "cron")
EXPRESSION_KEYWORDS = ("expresion", "expression", "{{")
CREDENTIAL_KEYWORDS = ("credencial", "credenciales", "credential", "auth", "api", "key")

TOKEN_RE = re.compile(r"[a-zA-Z0-9_\-]+")


@dataclass
class NodeCandidate:
    node_id: str
    score: float
    reasons: List[str]


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumerics for matching."""
    return [token.lower() for token in TOKEN_RE.findall(text.lower())]


def _global_signal(question: str) -> bool:
    """Detect if the question is about the workflow globally."""
    lowered = question.lower()
    return any(keyword in lowered for keyword in GLOBAL_KEYWORDS)


def _similarity(a: str, b: str) -> float:
    """Compute fuzzy similarity between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _match_type(tokens: Sequence[str], node: NodeSummary) -> float:
    """Score how well tokens match a node type and synonyms."""
    short = node.short_type.lower()
    score = 0.0
    for key, synonyms in TYPE_SYNONYMS.items():
        if key not in short:
            continue
        if any(synonym in tokens for synonym in synonyms):
            # 0.7: strong but not decisive signal; name match can still win.
            score += 0.7
    if any(token in short for token in TRIGGER_KEYWORDS) and any(
        token in tokens for token in TRIGGER_KEYWORDS
    ):
        # 0.5: triggers are important, but still weaker than exact name match.
        score += 0.5
    return score


def _match_name(tokens: Sequence[str], question: str, node: NodeSummary) -> float:
    """Score direct name matches and fuzzy similarity against the question."""
    lowered_name = node.name.lower()
    score = 0.0
    name_similarity = _similarity(question, node.name)
    if name_similarity >= 0.92:
        # 1.2: near-exact match; should dominate other signals.
        score += 1.2
    elif name_similarity >= 0.82:
        # 0.9: strong match, but not fully decisive.
        score += 0.9
    elif name_similarity >= 0.72:
        # 0.6: plausible match, can be overridden by other signals.
        score += 0.6

    matched_tokens = 0
    for token in tokens:
        if len(token) < 3:
            continue
        if token in lowered_name:
            matched_tokens += 1
            # 0.35: token-level hints, accumulate but capped by other signals.
            score += 0.35
    if matched_tokens >= 2:
        # 0.25: bonus for multiple token hits.
        score += 0.25
    return score


def _match_signals(tokens: Sequence[str], node: NodeSummary) -> float:
    """Score heuristic signals like creds/expressions/pivots."""
    score = 0.0
    if node.has_expressions and any(token in tokens for token in EXPRESSION_KEYWORDS):
        # 0.45: expressions are a strong clue but not equivalent to a name match.
        score += 0.45
    if node.has_credentials and any(token in tokens for token in CREDENTIAL_KEYWORDS):
        # 0.45: credentials are informative but still secondary.
        score += 0.45
    if node.is_pivot and any(token in tokens for token in ("merge", "if", "switch")):
        # 0.35: pivot hints help when user mentions branching/merge.
        score += 0.35
    return score


def _score_nodes(summary: WorkflowSummary, question: str) -> List[NodeCandidate]:
    """Rank nodes by name/type/signal similarity to the question."""
    tokens = _tokenize(question)
    candidates: List[NodeCandidate] = []
    for node in summary.nodes.values():
        score = 0.0
        reasons: List[str] = []

        name_score = _match_name(tokens, question, node)
        if name_score > 0:
            score += name_score
            reasons.append("name")

        type_score = _match_type(tokens, node)
        if type_score > 0:
            score += type_score
            reasons.append("type")

        signal_score = _match_signals(tokens, node)
        if signal_score > 0:
            score += signal_score
            reasons.append("signals")

        if score <= 0:
            continue
        candidates.append(NodeCandidate(node_id=node.node_id, score=score, reasons=reasons))

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _heuristic_candidates(summary: WorkflowSummary, limit: int = 6) -> List[NodeCandidate]:
    """Fallback candidates based on generic node importance."""
    scored: List[NodeCandidate] = []
    for node in summary.nodes.values():
        score = 0.0
        reasons: List[str] = []
        short = node.short_type.lower()

        if any(token in short for token in TRIGGER_KEYWORDS):
            # 1.0: triggers are often the root cause or starting point.
            score += 1.0
            reasons.append("trigger")
        if node.has_credentials:
            # 0.6: auth/config issues are common.
            score += 0.6
            reasons.append("creds")
        if node.has_expressions:
            # 0.6: expressions tend to break flows frequently.
            score += 0.6
            reasons.append("expr")
        if node.is_pivot:
            # 0.7: control-flow nodes often drive failures.
            score += 0.7
            reasons.append("pivot")
        if any(token in short for token in ("http", "request", "code", "postgres", "mysql")):
            # 0.5: common integration nodes are usually worth inspecting.
            score += 0.5
            reasons.append("common")

        if score <= 0:
            continue
        scored.append(NodeCandidate(node_id=node.node_id, score=score, reasons=reasons))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


def _candidates_payload(
    summary: WorkflowSummary, candidates: Iterable[NodeCandidate], limit: int = 8
) -> List[Dict[str, object]]:
    """Build a compact payload of candidate nodes for debugging/UI."""
    payload: List[Dict[str, object]] = []
    for candidate in list(candidates)[:limit]:
        node = summary.nodes.get(candidate.node_id)
        if not node:
            continue
        payload.append(
            {
                "id": node.node_id,
                "name": node.name,
                "type": node.short_type,
                "score": round(candidate.score, 3),
                "reasons": candidate.reasons,
            }
        )
    return payload


def _select_scope(
    summary: WorkflowSummary,
    question: str,
    candidates: List[NodeCandidate],
) -> Plan:
    """Choose a plan scope based on candidate scores and global signals."""
    global_signal = _global_signal(question)

    moderate_threshold = 0.85
    # 0.85: two decent matches suggests a subgraph rather than a single node.
    moderate_candidates = [cand for cand in candidates if cand.score >= moderate_threshold]
    if len(moderate_candidates) >= 2 and not global_signal:
        target_ids = [cand.node_id for cand in moderate_candidates[:6]]
        return Plan(
            scope=PlanScope.subgraph,
            target_node_ids=target_ids,
            # 0.6 base + 0.06 per node: confidence grows with consistent signals.
            confidence=min(0.88, 0.6 + 0.06 * len(target_ids)),
        )

    strong_threshold = 1.05
    # 1.05: strong match indicates a specific node is likely.
    strong_candidates = [cand for cand in candidates if cand.score >= strong_threshold]

    if global_signal and not strong_candidates:
        # Global phrasing without strong node match -> inspect workflow globally.
        return Plan(scope=PlanScope.workflow_global, confidence=0.78)

    if strong_candidates:
        target_ids = [cand.node_id for cand in strong_candidates[:6]]
        if len(target_ids) <= 2:
            return Plan(
                scope=PlanScope.node_specific,
                target_node_ids=target_ids,
                # Up to 0.92 because a single strong hit is still heuristic.
                confidence=min(0.92, 0.6 + 0.15 * len(target_ids)),
            )
        return Plan(
            scope=PlanScope.subgraph,
            target_node_ids=target_ids,
            # Slightly lower than node-specific, multiple nodes reduce certainty.
            confidence=min(0.9, 0.62 + 0.05 * len(target_ids)),
        )

    if candidates:
        target_ids = [cand.node_id for cand in candidates[:4]]
        scope = PlanScope.subgraph
        # Weak signals: keep confidence low to encourage clarification.
        confidence = 0.55 if not global_signal else 0.6
        return Plan(scope=scope, target_node_ids=target_ids, confidence=confidence)

    heuristics = _heuristic_candidates(summary, limit=6)
    if heuristics:
        target_ids = [cand.node_id for cand in heuristics[:4]]
        scope = PlanScope.workflow_global if global_signal else PlanScope.subgraph
        # Lowest-confidence fallback when no textual match is found.
        return Plan(scope=scope, target_node_ids=target_ids, confidence=0.42)

    return Plan(
        scope=PlanScope.subgraph,
        confidence=0.2,
        need_clarification=True,
        clarifying_question=(
            "No pude acotar la zona del workflow. Puedes nombrar el nodo o la rama."
        ),
    )


def plan_scope(question: str, summary: Optional[WorkflowSummary]) -> Plan:
    """Top-level planner entrypoint that returns scope + candidates.

    Note: selection is heuristic (string/fuzzy matching + node signals). In the
    future, this can be augmented or replaced with embeddings-based ranking.
    """
    if not summary:
        return Plan(scope=PlanScope.no_workflow, confidence=1.0)

    question = (question or "").strip()
    if not question:
        return Plan(
            scope=PlanScope.workflow_global,
            confidence=0.35,
            need_clarification=True,
            clarifying_question="Que parte del workflow quieres revisar?",
        )

    candidates = _score_nodes(summary, question)
    plan = _select_scope(summary, question, candidates)
    plan.candidates = _candidates_payload(
        summary, candidates, limit=settings.WORKFLOW_MAX_NODES_SUBGRAPH
    )
    return plan
