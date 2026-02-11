from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from ..config import settings
from ..schemas import DebugWorkflowRequest
from ..workflow.n8n_client import N8NClient, N8NClientError
from ..workflow.node_analyzer import NodeFinding, analyze_node
from ..workflow.planner import Plan, plan_scope
from ..workflow.retriever import retrieve_docs
from ..workflow.subgraph_builder import SubgraphSelection, build_subgraph
from ..workflow.workflow_summary import (
    WorkflowSummary,
    build_workflow_summary,
    micro_context_for_nodes,
    render_micro_context,
)


class WorkflowService:
    """Workflow-aware helpers used by chat and debug endpoints."""

    def __init__(self, n8n_client: N8NClient, logger: logging.Logger) -> None:
        self._n8n_client = n8n_client
        self._logger = logger
        self._trace_logger = logging.getLogger("n8n-assistant.trace")

    def debug_workflow(self, request: DebugWorkflowRequest) -> Dict[str, Any]:
        """Return planner + subgraph + docs selection for debugging."""
        workflow = self.fetch_workflow(request.workflow_id)
        summary = build_workflow_summary(workflow, workflow_id=request.workflow_id)
        plan = plan_scope(request.question, summary)
        subgraph = build_subgraph(summary, request.question, plan)

        node_details: List[Dict[str, Any]] = []
        if request.include_nodes:
            for node_id in subgraph.node_ids:
                node = summary.nodes.get(node_id)
                if not node:
                    continue
                node_details.append(
                    {
                        "id": node.node_id,
                        "name": node.name,
                        "type": node.short_type,
                        "has_credentials": node.has_credentials,
                        "has_expressions": node.has_expressions,
                        "is_pivot": node.is_pivot,
                        "disabled": node.disabled,
                    }
                )

        docs_payload: List[Dict[str, Any]] = []
        if request.include_docs:
            node_types = [item["type"] for item in node_details if item.get("type")]
            node_names = [item["name"] for item in node_details if item.get("name")]
            docs_chunks = retrieve_docs(request.question, node_types=node_types, node_names=node_names)
            for chunk in docs_chunks:
                text = (chunk.get("text") or "").strip()
                docs_payload.append(
                    {
                        "url": chunk.get("url"),
                        "title": chunk.get("title"),
                        "section": chunk.get("section"),
                        "snippet": self._compact_snippet(text, 240),
                    }
                )

        return {
            "workflow_id": request.workflow_id,
            "question": request.question,
            "summary": {
                "node_count": len(summary.nodes),
                "edge_count": len(summary.edges),
                "truncated": summary.truncated,
            },
            "plan": plan.model_dump(),
            "subgraph": {
                "reason": subgraph.reason,
                "truncated": subgraph.truncated,
                "node_ids": subgraph.node_ids,
                "edges": [
                    {
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "source_output": edge.source_output,
                        "target_input": edge.target_input,
                    }
                    for edge in subgraph.edges
                ],
            },
            "nodes": node_details,
            "docs": docs_payload,
            "notes": {
                "retriever": "retrieve_docs ignores node hints for now",
                "docs_top_k": settings.WORKFLOW_DOCS_TOP_K,
            },
        }

    def fetch_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Fetch workflow JSON from n8n and log basic stats."""
        try:
            workflow = self._n8n_client.get_workflow(workflow_id)
        except N8NClientError as exc:
            self._logger.warning(
                "n8n fetch failed: wf=%s status=%s error=%s",
                workflow_id,
                exc.status_code,
                str(exc),
            )
            detail = (
                f"No pude obtener el workflow {workflow_id} desde n8n. "
                "Revisa N8N_BASE_URL, N8N_API_KEY y el endpoint configurado."
            )
            raise HTTPException(status_code=502, detail=detail) from exc

        nodes = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []
        self._logger.info("n8n fetch ok: wf=%s nodes=%d", workflow_id, len(nodes))
        return workflow

    def retrieve_docs_for_subgraph(
        self,
        question: str,
        summary: WorkflowSummary,
        subgraph: SubgraphSelection,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve docs for selected subgraph (currently question-only)."""
        node_types = [
            summary.nodes[node_id].short_type
            for node_id in subgraph.node_ids
            if node_id in summary.nodes
        ]
        node_names = [
            summary.nodes[node_id].name
            for node_id in subgraph.node_ids
            if node_id in summary.nodes
        ]
        self._trace_logger.info(
            "workflow docs retrieval: id=%s q_len=%d nodes=%d top_k=%d",
            request_id or "-",
            len(question),
            len(node_types),
            settings.WORKFLOW_DOCS_TOP_K,
        )
        if self._trace_logger.isEnabledFor(logging.DEBUG):
            self._trace_logger.debug(
                "workflow docs hints: id=%s node_types=%s node_names=%s",
                request_id or "-",
                node_types[:6],
                node_names[:6],
            )
        try:
            return retrieve_docs(
                question,
                node_types=node_types,
                node_names=node_names,
                request_id=request_id,
            )
        except Exception as exc:
            self._logger.exception("workflow docs retrieval failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    "RAG failed during workflow-aware retrieval. "
                    "Revisa EMBEDDING_MODEL y la configuracion de la base."
                ),
            ) from exc

    def analyze_nodes(
        self,
        question: str,
        summary: WorkflowSummary,
        subgraph: SubgraphSelection,
        docs_chunks: List[Dict[str, Any]],
        plan: Plan,
        requested_model: Optional[str],
        request_id: Optional[str] = None,
    ) -> List[NodeFinding]:
        """Run per-node analysis with micro-context for the selected nodes."""
        findings: List[NodeFinding] = []
        start = time.perf_counter()
        self._trace_logger.info(
            "node analysis start: id=%s nodes=%d scope=%s",
            request_id or "-",
            len(subgraph.node_ids),
            plan.scope.value,
        )
        for node_id in subgraph.node_ids[: settings.WORKFLOW_MAX_NODES_GLOBAL]:
            node = summary.nodes.get(node_id)
            if not node:
                continue
            if self._trace_logger.isEnabledFor(logging.DEBUG):
                self._trace_logger.debug(
                    "node analysis target: id=%s node=%s name=%s type=%s",
                    request_id or "-",
                    node.node_id,
                    node.name,
                    node.short_type,
                )
            micro = micro_context_for_nodes(summary, [node_id])
            micro_text = render_micro_context(micro)
            finding = analyze_node(
                question=question,
                node=node,
                micro_context_text=micro_text,
                docs_chunks=docs_chunks,
                scope=plan.scope,
                model=requested_model,
                request_id=request_id,
            )
            findings.append(finding)
            if self._trace_logger.isEnabledFor(logging.DEBUG):
                self._trace_logger.debug(
                    "node analysis result: id=%s node=%s risk=%s conf=%.2f findings=%d recs=%d",
                    request_id or "-",
                    finding.node_id,
                    finding.risk_level,
                    finding.confidence,
                    len(finding.findings),
                    len(finding.recommendations),
                )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._trace_logger.info(
            "node analysis done: id=%s nodes=%d elapsed_ms=%.1f",
            request_id or "-",
            len(findings),
            elapsed_ms,
        )
        return findings

    def clarification_text(self, plan: Plan, summary: WorkflowSummary) -> str:
        """Compose a clarifying question when scope is ambiguous."""
        parts: List[str] = []
        if plan.clarifying_question:
            parts.append(plan.clarifying_question.strip())
        else:
            parts.append("Necesito un poco mas de contexto para acotar el analisis.")

        candidates = plan.candidates[:6]
        if candidates:
            parts.append("Candidatos detectados (elige uno o menciona otro):")
            for cand in candidates:
                parts.append(f"- {cand.get('id')} | {cand.get('name')} | {cand.get('type')}")
        if summary.truncated:
            parts.append("Nota: el workflow es grande y el resumen fue recortado.")
        return "\n".join(parts)

    @staticmethod
    def _compact_snippet(text: str, max_chars: int) -> str:
        """Compact a text snippet for debug payloads."""
        compact = " ".join(text.split())
        if len(compact) > max_chars:
            compact = compact[: max_chars - 3].rstrip() + "..."
        return compact
