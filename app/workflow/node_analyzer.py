from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from ..config import settings
from ..llm import chat_completion, get_langchain_chat_model, resolve_model
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
    '  "risk_level": "low|medium|high|unknown",\n'
    '  "findings": ["..."],\n'
    '  "recommendations": ["..."],\n'
    '  "confidence": 0.0\n'
    "}\n"
    "Reglas:\n"
    "- maximo 4 findings y 5 recommendations\n"
    "- frases cortas y especificas a n8n\n"
    "- si falta contexto, dilo como finding y da una recomendacion de inspeccion\n"
    "- no inventes valores de credenciales, endpoints ni nombres de campos"
)


JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
trace_logger = logging.getLogger("n8n-assistant.trace")


class NodeAnalyzerOutput(BaseModel):
    risk_level: str = "unknown"
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = 0.0


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


def _compact_snippet(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact


def _build_prompt(
    question: str,
    node: NodeSummary,
    micro_context_text: str,
    docs_block: str,
    scope: PlanScope,
) -> str:
    return (
        f"Pregunta del usuario:\n{question.strip()}\n\n"
        f"Nodo objetivo:\n- id: {node.node_id}\n- name: {node.name}\n- type: {node.node_type}\n\n"
        f"Micro-contexto del workflow (recortado):\n{micro_context_text}\n\n"
        f"Documentacion relevante:\n{docs_block}\n\n"
        f"Scope: {scope.value}\n\n"
        f"{NODE_ANALYZER_INSTRUCTIONS}"
    )


def _run_structured_output(prompt: str, model: Optional[str]) -> NodeAnalyzerOutput:
    chat_model = get_langchain_chat_model(model=model, temperature=0.1)
    if chat_model is None:
        raise RuntimeError("langchain chat model is unavailable")

    structured_llm = chat_model.with_structured_output(NodeAnalyzerOutput)
    output = structured_llm.invoke(
        [
            ("system", NODE_ANALYZER_SYSTEM_PROMPT),
            ("human", prompt),
        ]
    )
    if isinstance(output, NodeAnalyzerOutput):
        return output
    return NodeAnalyzerOutput.model_validate(output)


def _run_legacy_json(prompt: str, model: Optional[str]) -> NodeAnalyzerOutput:
    resolved_model = resolve_model(model)
    response = chat_completion(
        [
            {"role": "system", "content": NODE_ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=resolved_model,
        temperature=0.1,
    )
    content = response.model_dump().get("choices", [{}])[0].get("message", {}).get("content", "")

    data = _extract_json(content)
    if not data:
        raise ValueError("node analyzer returned invalid json")
    return NodeAnalyzerOutput.model_validate(data)


def _normalize_output(output: NodeAnalyzerOutput) -> NodeAnalyzerOutput:
    findings = [str(item).strip() for item in output.findings[:4] if str(item).strip()]
    recommendations = [
        str(item).strip() for item in output.recommendations[:5] if str(item).strip()
    ]
    return NodeAnalyzerOutput(
        risk_level=str(output.risk_level or "unknown"),
        findings=findings,
        recommendations=recommendations,
        confidence=max(0.0, min(float(output.confidence or 0.0), 1.0)),
    )


def analyze_node(
    question: str,
    node: NodeSummary,
    micro_context_text: str,
    docs_chunks: Iterable[Dict[str, Any]],
    scope: PlanScope,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> NodeFinding:
    """Call the LLM to analyze one node and normalize its JSON output."""
    docs_block = _docs_block(docs_chunks)
    if trace_logger.isEnabledFor(logging.DEBUG):
        trace_logger.debug(
            "node analyzer prompt: id=%s node=%s type=%s scope=%s docs_chars=%d micro_chars=%d",
            request_id or "-",
            node.node_id,
            node.short_type,
            scope.value,
            len(docs_block),
            len(micro_context_text),
        )

    prompt = _build_prompt(question, node, micro_context_text, docs_block, scope)

    output: Optional[NodeAnalyzerOutput] = None
    try:
        output = _run_structured_output(prompt, model)
    except Exception as exc:
        trace_logger.warning(
            "node analyzer structured output failed: id=%s node=%s type=%s error=%s",
            request_id or "-",
            node.node_id,
            node.short_type,
            str(exc),
        )

    if output is None:
        try:
            output = _run_legacy_json(prompt, model)
        except Exception as exc:
            trace_logger.warning(
                "node analyzer legacy fallback failed: id=%s node=%s type=%s error=%s prompt=%s",
                request_id or "-",
                node.node_id,
                node.short_type,
                str(exc),
                _compact_snippet(prompt, max_chars=220),
            )
            return _fallback_finding(node, scope)

    normalized = _normalize_output(output)
    return NodeFinding(
        node_id=node.node_id,
        node_name=node.name,
        node_type=node.short_type,
        scope=scope,
        risk_level=normalized.risk_level,
        findings=normalized.findings,
        recommendations=normalized.recommendations,
        confidence=normalized.confidence,
    )
