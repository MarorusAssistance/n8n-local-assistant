from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import json

from ..config import settings
from ..rag import build_context_block
from .node_analyzer import NodeFinding
from .planner import Plan
from .subgraph_builder import SubgraphSelection
from .workflow_summary import WorkflowSummary


SYNTH_SYSTEM_PROMPT = (
    "Eres un asistente experto en n8n con enfoque en diagnostico tecnico. "
    "Debes combinar hallazgos por nodo, contexto de workflow recortado y documentacion. "
    "No inventes configuraciones ni detalles de nodos; cuando falte contexto, pide un dato concreto."
)

SYNTH_INSTRUCTIONS = (
    "Responde en espanol con este formato:\n"
    "Diagnostico:\n"
    "Pasos:\n"
    "Alternativas:\n"
    "Reglas:\n"
    "- menciona nodos por nombre\n"
    "- maximo 3-6 pasos\n"
    "- evita pegar JSON largo\n"
    "- no inventes endpoints, nombres de campos ni valores\n"
)


def _truncate(text: str, max_chars: int) -> str:
    """Trim text to a compact single-line snippet with max length."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _render_selected_nodes(summary: WorkflowSummary, subgraph: SubgraphSelection) -> str:
    """Render a compact list of nodes selected for analysis."""
    lines: List[str] = []
    for node_id in subgraph.node_ids[: settings.WORKFLOW_MAX_NODES_SUBGRAPH]:
        node = summary.nodes.get(node_id)
        if not node:
            continue
        lines.append(f"- {node.node_id} | {node.name} | {node.short_type}")
    return "\n".join(lines)


def _render_findings(findings: Sequence[NodeFinding], max_chars: int = 2400) -> str:
    """Serialize node findings into a bounded JSON snippet."""
    payload = [item.model_dump() for item in findings]
    text = json.dumps(payload, ensure_ascii=True)
    return _truncate(text, max_chars=max_chars)


def _docs_block(chunks: Iterable[Dict[str, Any]], max_chars: int = 1800) -> str:
    """Build a compact documentation block for the prompt."""
    block = build_context_block(list(chunks)[: settings.WORKFLOW_DOCS_TOP_K])
    block = block.strip()
    if len(block) <= max_chars:
        return block
    return block[: max_chars - 3].rstrip() + "..."


def build_synthesis_messages(
    question: str,
    plan: Plan,
    summary: WorkflowSummary,
    subgraph: SubgraphSelection,
    findings: Sequence[NodeFinding],
    docs_chunks: Iterable[Dict[str, Any]],
    chat_context: str = "",
) -> List[Dict[str, str]]:
    """Compose system+user messages for the final synthesis response."""
    nodes_block = _render_selected_nodes(summary, subgraph)
    findings_block = _render_findings(findings)
    docs_block = _docs_block(docs_chunks)

    workflow_overview = (
        f"Workflow: {summary.name} (id={summary.workflow_id})\n"
        f"Scope: {plan.scope.value} | Confidence: {plan.confidence:.2f}\n"
        f"Nodes total: {summary.node_count} | Nodes analizados: {len(subgraph.node_ids)}\n"
        f"Subgraph truncated: {subgraph.truncated}\n"
        "Nodos seleccionados:\n"
        f"{nodes_block}\n"
    )
    if summary.truncated:
        workflow_overview += "Nota: el workflow fue truncado para seguridad.\n"

    context_prefix = ""
    if chat_context.strip():
        context_prefix = f"Contexto reciente del chat (recortado):\n{chat_context.strip()}\n\n"

    user_prompt = (
        context_prefix
        + f"Pregunta del usuario:\n{question.strip()}\n\n"
        f"Resumen del workflow (recortado):\n{workflow_overview}\n"
        f"Hallazgos por nodo (JSON recortado):\n{findings_block}\n\n"
        f"Documentacion relevante:\n{docs_block}\n\n"
        f"{SYNTH_INSTRUCTIONS}"
    )

    return [
        {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
