from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import json
import re

from pydantic import BaseModel, Field

from ..config import settings
from ..llm import chat_completion, resolve_model
from ..rag import build_context_block
from .planner import PlanScope
from .workflow_summary import NodeSummary


NODE_ANALYZER_SYSTEM_PROMPT = (
    "Eres un analista tecnico de workflows n8n. "
    "Debes analizar un nodo concreto dentro de un subgrafo recortado y producir "
    "un JSON breve y accionable. No inventes campos que no existen. "
    "Si falta informacion, indicalo explicitamente."
)

NODE_ANALYZER_INSTRUCTIONS = (
    "Responde SOLO con JSON valido con esta forma:\n"
    "{\n"
    "  \"risk_level\": \"low|medium|high|unknown\",\n"
    "  \"findings\": [\"...\"],\n"
    "  \"recommendations\": [\"...\"],\n"
    "  \"confidence\": 0.0\n"
    "}\n"
    "Reglas:\n"
    "- maximo 4 findings y 5 recommendations\n"
    "- frases cortas y especificas a n8n\n"
    "- si falta contexto, dilo como finding y da una recomendacion de inspeccion\n"
    "- no inventes valores de credenciales, endpoints ni nombres de campos"
)


JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


class NodeFinding(BaseModel):
    node_id: str
    node_name: str
    node_type: str
    scope: PlanScope
    risk_level: str = "unknown"
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = 0.0


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from LLM output."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _docs_block(chunks: Iterable[Dict[str, Any]], max_chars: int = 1200) -> str:
    """Build a compact docs block to keep per-node prompts small."""
    chunks_list = list(chunks)[: settings.WORKFLOW_DOCS_TOP_K]
    block = build_context_block(chunks_list)
    block = block.strip()
    if len(block) <= max_chars:
        return block
    return block[: max_chars - 3].rstrip() + "..."


def _fallback_finding(node: NodeSummary, scope: PlanScope) -> NodeFinding:
    """Return a conservative fallback when LLM output is invalid."""
    findings = [
        "No pude validar el nodo con el modelo; revisa configuracion basica.",
    ]
    recommendations = [
        "Abre el nodo y valida credenciales, inputs y expresiones.",
        "Ejecuta el workflow en modo manual y revisa el output del nodo.",
    ]
    return NodeFinding(
        node_id=node.node_id,
        node_name=node.name,
        node_type=node.short_type,
        scope=scope,
        risk_level="unknown",
        findings=findings,
        recommendations=recommendations,
        confidence=0.2,
    )


def analyze_node(
    question: str,
    node: NodeSummary,
    micro_context_text: str,
    docs_chunks: Iterable[Dict[str, Any]],
    scope: PlanScope,
    model: Optional[str] = None,
) -> NodeFinding:
    """Call the LLM to analyze one node and normalize its JSON output."""
    docs_block = _docs_block(docs_chunks)

    prompt = (
        f"Pregunta del usuario:\n{question.strip()}\n\n"
        f"Nodo objetivo:\n- id: {node.node_id}\n- name: {node.name}\n- type: {node.node_type}\n\n"
        f"Micro-contexto del workflow (recortado):\n{micro_context_text}\n\n"
        f"Documentacion relevante:\n{docs_block}\n\n"
        f"Scope: {scope.value}\n\n"
        f"{NODE_ANALYZER_INSTRUCTIONS}"
    )

    resolved_model = resolve_model(model)

    try:
        response = chat_completion(
            [
                {"role": "system", "content": NODE_ANALYZER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=resolved_model,
            temperature=0.1,
        )
        content = (
            response.model_dump()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception:
        return _fallback_finding(node, scope)

    data = _extract_json(content)
    if not data:
        return _fallback_finding(node, scope)

    risk_level = str(data.get("risk_level") or "unknown")
    findings_raw = data.get("findings") if isinstance(data.get("findings"), list) else []
    recommendations_raw = (
        data.get("recommendations") if isinstance(data.get("recommendations"), list) else []
    )
    confidence = float(data.get("confidence") or 0.0)

    findings = [str(item).strip() for item in findings_raw[:4] if str(item).strip()]
    recommendations = [
        str(item).strip() for item in recommendations_raw[:5] if str(item).strip()
    ]

    return NodeFinding(
        node_id=node.node_id,
        node_name=node.name,
        node_type=node.short_type,
        scope=scope,
        risk_level=risk_level,
        findings=findings,
        recommendations=recommendations,
        confidence=max(0.0, min(confidence, 1.0)),
    )
