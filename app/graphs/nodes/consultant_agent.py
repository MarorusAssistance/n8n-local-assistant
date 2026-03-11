from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ...config import settings
from ...features.reasoning.multi_agent_contracts import (
    ConsultantQueryAnalysis,
    ConsultantResponse,
    ConsultantRetrievalResult,
    ConsultantSource,
    ConsultantToolUsage,
)
from ...llm import get_langchain_chat_model, resolve_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event, emit_trace_event
from ...rag import retrieve_context
from ..multi_agent_state import MultiAgentGraphState

try:  # Optional dependency path.
    from langchain.agents import create_agent
except Exception:  # pragma: no cover - optional dependency fallback
    create_agent = None  # type: ignore[assignment]

try:  # Optional dependency path.
    from langchain_core.tools import tool
except Exception:  # pragma: no cover - optional dependency fallback
    tool = None  # type: ignore[assignment]


logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")

_API_DOCS_SOURCE = "n8n-docs"


class _AnalysisOutput(BaseModel):
    request_type: str = "general_information"
    key_topics: List[str] = Field(default_factory=list)
    needs_active_workflow_context: bool = False
    retrieval_needed: bool = False
    conversation_history_sufficient: bool = True
    source_limited: bool = False
    selected_sources: List[str] = Field(default_factory=list)
    analysis_notes: List[str] = Field(default_factory=list)


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _runtime(state: MultiAgentGraphState) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    raw = state.get("runtime_context")
    if not isinstance(raw, dict):
        return None, None, {}
    model = raw.get("model") if isinstance(raw.get("model"), str) else None
    request_id = raw.get("request_id") if isinstance(raw.get("request_id"), str) else None
    return model, request_id, raw


def _conversation(runtime_context: Dict[str, Any]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    raw = runtime_context.get("conversation_context")
    if not isinstance(raw, list):
        return output
    for item in raw[-12:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant", "system"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        output.append({"role": str(role), "content": _text(content)[:1200]})
    return output


def _active_workflow(state: MultiAgentGraphState, runtime_context: Dict[str, Any]) -> Dict[str, Any]:
    active: Dict[str, Any] = {}
    if isinstance(runtime_context.get("active_workflow"), dict):
        active.update(runtime_context["active_workflow"])
    for field in ("active_workflow_id", "active_workflow_name", "active_workflow_url"):
        value = state.get(field)
        if isinstance(value, str) and value.strip():
            active[field.replace("active_workflow_", "")] = value.strip()
    final_json = state.get("final_workflow_json")
    if isinstance(final_json, dict) and final_json:
        active["workflow_json"] = final_json
    return active


def _to_source(value: Any) -> Optional[ConsultantSource]:
    if isinstance(value, ConsultantSource):
        return value
    if not isinstance(value, str):
        return None
    try:
        return ConsultantSource(value.strip())
    except Exception:
        return None


def _dedupe(values: Iterable[ConsultantSource]) -> List[ConsultantSource]:
    out: List[ConsultantSource] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _heuristic_analysis(query: str, has_conversation: bool, has_active_workflow: bool) -> ConsultantQueryAnalysis:
    lowered = query.lower()
    selected: List[ConsultantSource] = [ConsultantSource.conversation_history]
    retrieval_needed = False
    request_type = "general_information"
    notes: List[str] = []
    if any(token in lowered for token in ("compare", "difference", "vs", "versus")):
        request_type = "comparison"
    elif any(token in lowered for token in ("recommend", "best", "tradeoff", "should")):
        request_type = "recommendation"
    elif any(token in lowered for token in ("explain", "what is", "how does")):
        request_type = "explanation"

    if any(token in lowered for token in ("node", "trigger", "action", "integration")):
        retrieval_needed = True
        selected.append(ConsultantSource.nodes_index)
    if any(token in lowered for token in ("credential", "oauth", "auth", "api key", "token")):
        retrieval_needed = True
        selected.append(ConsultantSource.credentials_index)
    if any(token in lowered for token in ("docs", "documentation", "official")):
        retrieval_needed = True
        selected.append(ConsultantSource.api_docs_index)
    if any(token in lowered for token in ("template", "blueprint", "starter")):
        retrieval_needed = True
        selected.append(ConsultantSource.templates_index)
        notes.append("templates_index may be unavailable")
    if any(token in lowered for token in ("current workflow", "my workflow", "active workflow", "this workflow")):
        request_type = "workflow_context"
        if has_active_workflow:
            selected.append(ConsultantSource.active_workflow)
        else:
            notes.append("active workflow requested but unavailable")
    if retrieval_needed and len(selected) == 1:
        selected.append(ConsultantSource.api_docs_index)
    if not retrieval_needed and has_conversation:
        notes.append("conversation context appears sufficient")

    return ConsultantQueryAnalysis(
        request_type=request_type,  # type: ignore[arg-type]
        key_topics=[],
        needs_active_workflow_context=request_type == "workflow_context",
        retrieval_needed=retrieval_needed,
        conversation_history_sufficient=not retrieval_needed,
        source_limited=ConsultantSource.templates_index in selected and len(selected) <= 2,
        selected_sources=_dedupe(selected),
        analysis_notes=notes[:8],
    )


def analyze_consultant_query(
    *,
    query: str,
    model: Optional[str],
    request_id: Optional[str],
    conversation_context: List[Dict[str, str]],
    active_workflow_available: bool,
) -> ConsultantQueryAnalysis:
    llm = get_langchain_chat_model(model=model, temperature=0.0)
    if llm is None:
        return _heuristic_analysis(query, bool(conversation_context), active_workflow_available)

    system_prompt = (
        "You are consultant query analyzer. Decide whether retrieval is needed and select minimum sources. "
        "Never choose execution-oriented behavior."
    )
    context_lines = [f"- {item['role']}: {item['content']}" for item in conversation_context[-8:]]
    user_prompt = (
        "Analyze this information request.\n"
        "Allowed sources: conversation_history, nodes_index, credentials_index, api_docs_index, templates_index, active_workflow.\n"
        f"Active workflow available: {active_workflow_available}\n"
        f"Query:\n{query}\n\n"
        f"Conversation:\n{chr(10).join(context_lines) if context_lines else '- (none)'}"
    )
    resolved_model = resolve_model(model)
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage="multi_agent.consultant.analysis",
        model=resolved_model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        params={"temperature": 0.0},
    )
    started = time.perf_counter()
    try:
        output = llm.with_structured_output(_AnalysisOutput).invoke(
            [("system", system_prompt), ("human", user_prompt)]
        )
        latency = (time.perf_counter() - started) * 1000.0
        as_dict = output.model_dump(exclude_none=True) if isinstance(output, _AnalysisOutput) else output
        emit_llm_output_event(
            trace_logger,
            request_id=request_id,
            stage="multi_agent.consultant.analysis",
            model=resolved_model,
            latency_ms=latency,
            content=json.dumps(as_dict, ensure_ascii=False),
            usage=None,
            extra={"temperature": 0.0},
        )
        parsed = output if isinstance(output, _AnalysisOutput) else _AnalysisOutput.model_validate(output)
        selected_sources = _dedupe(
            source
            for source in (_to_source(item) for item in parsed.selected_sources)
            if source is not None
        )
        request_type = parsed.request_type
        if request_type not in {"explanation", "comparison", "recommendation", "workflow_context", "general_information", "unknown"}:
            request_type = "general_information"
        return ConsultantQueryAnalysis(
            request_type=request_type,  # type: ignore[arg-type]
            key_topics=[_text(item) for item in parsed.key_topics if _text(item)],
            needs_active_workflow_context=bool(parsed.needs_active_workflow_context),
            retrieval_needed=bool(parsed.retrieval_needed),
            conversation_history_sufficient=bool(parsed.conversation_history_sufficient),
            source_limited=bool(parsed.source_limited),
            selected_sources=selected_sources,
            analysis_notes=[_text(item)[:180] for item in parsed.analysis_notes if _text(item)][:8],
        )
    except Exception as exc:
        logger.warning("consultant analysis fallback to heuristics: %s", str(exc))
        return _heuristic_analysis(query, bool(conversation_context), active_workflow_available)


def _source_filter(source: ConsultantSource) -> Optional[str]:
    if source == ConsultantSource.api_docs_index:
        return _API_DOCS_SOURCE
    if source == ConsultantSource.nodes_index:
        return settings.LINKED_DEFS_NODES_SOURCE
    if source == ConsultantSource.credentials_index:
        return settings.LINKED_DEFS_CREDENTIALS_SOURCE
    return None


def _compact_retrieval(source: ConsultantSource, query: str, chunks: Sequence[Dict[str, Any]]) -> ConsultantRetrievalResult:
    refs: List[str] = []
    snippets: List[str] = []
    ids: List[str] = []
    for chunk in chunks[:8]:
        doc_id = _text(chunk.get("doc_id"))
        if doc_id:
            ids.append(doc_id)
        ref = _text(chunk.get("url")) or _text(chunk.get("title")) or _text(chunk.get("section")) or doc_id
        if ref:
            refs.append(ref)
        snippet = _text(chunk.get("text"))
        if snippet:
            snippets.append(snippet[:180])
    return ConsultantRetrievalResult(
        source=source,
        query=_text(query)[:220],
        result_count=len(chunks),
        chunk_ids=ids,
        references=refs[:8],
        snippets=snippets[:4],
    )


def _extract_answer_text(result: Any) -> str:
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _conversation_only_answer(
    *,
    query: str,
    conversation_context: Sequence[Dict[str, str]],
) -> str:
    if conversation_context:
        return "I can answer this from conversation context. " + query
    return "I can answer this directly. " + query


def _build_tools(
    *,
    query: str,
    request_id: Optional[str],
    selected_sources: Sequence[ConsultantSource],
    active_workflow: Dict[str, Any],
    tools_used: List[ConsultantToolUsage],
    retrieval_results: List[ConsultantRetrievalResult],
) -> List[Any]:
    if tool is None:
        return []
    top_k = max(3, min(settings.TOP_K, 8))

    def _record(
        tool_name: str,
        source: ConsultantSource,
        result: ConsultantRetrievalResult,
        status: str = "ok",
        notes: Optional[str] = None,
    ) -> None:
        retrieval_results.append(result)
        tools_used.append(
            ConsultantToolUsage(
                tool_name=tool_name,
                source=source,
                call_order=len(tools_used) + 1,
                query=result.query,
                result_count=result.result_count,
                status=status,  # type: ignore[arg-type]
                notes=notes,
            )
        )

    @tool("api_docs_index_tool")
    def api_docs_index_tool(q: str) -> str:
        """Retrieve informational chunks from API docs index."""
        chunks = retrieve_context(q, top_k=top_k, request_id=request_id, source_filter=_source_filter(ConsultantSource.api_docs_index))
        result = _compact_retrieval(ConsultantSource.api_docs_index, q, chunks)
        _record("api_docs_index_tool", ConsultantSource.api_docs_index, result)
        return json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False)

    @tool("nodes_index_tool")
    def nodes_index_tool(q: str) -> str:
        """Retrieve node catalog and node capability chunks."""
        chunks = retrieve_context(q, top_k=top_k, request_id=request_id, source_filter=_source_filter(ConsultantSource.nodes_index))
        result = _compact_retrieval(ConsultantSource.nodes_index, q, chunks)
        _record("nodes_index_tool", ConsultantSource.nodes_index, result)
        return json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False)

    @tool("credentials_index_tool")
    def credentials_index_tool(q: str) -> str:
        """Retrieve credential and auth related chunks."""
        chunks = retrieve_context(q, top_k=top_k, request_id=request_id, source_filter=_source_filter(ConsultantSource.credentials_index))
        result = _compact_retrieval(ConsultantSource.credentials_index, q, chunks)
        _record("credentials_index_tool", ConsultantSource.credentials_index, result)
        return json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False)

    @tool("active_workflow_tool")
    def active_workflow_tool(q: str) -> str:
        """Read active workflow context (no mutation)."""
        _ = q
        workflow_json = active_workflow.get("workflow_json")
        node_count = len(workflow_json.get("nodes", [])) if isinstance(workflow_json, dict) and isinstance(workflow_json.get("nodes"), list) else 0
        if not active_workflow and node_count == 0:
            result = ConsultantRetrievalResult(
                source=ConsultantSource.active_workflow,
                query="active_workflow",
                result_count=0,
                unavailable_reason="No active workflow context is available in this request.",
            )
            _record("active_workflow_tool", ConsultantSource.active_workflow, result, status="unavailable", notes=result.unavailable_reason)
            return json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False)
        refs = []
        if active_workflow.get("id"):
            refs.append(f"id={active_workflow['id']}")
        if active_workflow.get("name"):
            refs.append(f"name={active_workflow['name']}")
        if active_workflow.get("url"):
            refs.append(f"url={active_workflow['url']}")
        snippets = [f"node_count={node_count}"] if node_count else []
        result = ConsultantRetrievalResult(
            source=ConsultantSource.active_workflow,
            query="active_workflow",
            result_count=node_count,
            references=refs,
            snippets=snippets,
        )
        _record("active_workflow_tool", ConsultantSource.active_workflow, result)
        return json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False)

    @tool("templates_index_tool")
    def templates_index_tool(q: str) -> str:
        """Future templates index boundary (currently unavailable)."""
        result = ConsultantRetrievalResult(
            source=ConsultantSource.templates_index,
            query=_text(q)[:220],
            result_count=0,
            unavailable_reason="templates_index is not available in the current runtime.",
        )
        _record("templates_index_tool", ConsultantSource.templates_index, result, status="unavailable", notes=result.unavailable_reason)
        return json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False)

    mapping = {
        ConsultantSource.api_docs_index: api_docs_index_tool,
        ConsultantSource.nodes_index: nodes_index_tool,
        ConsultantSource.credentials_index: credentials_index_tool,
        ConsultantSource.active_workflow: active_workflow_tool,
        ConsultantSource.templates_index: templates_index_tool,
    }
    return [mapping[source] for source in selected_sources if source in mapping]


def _direct_retrieval_for_sources(
    *,
    query: str,
    request_id: Optional[str],
    selected_sources: Sequence[ConsultantSource],
    active_workflow: Dict[str, Any],
    tools_used: List[ConsultantToolUsage],
    retrieval_results: List[ConsultantRetrievalResult],
) -> None:
    for source in selected_sources:
        if source == ConsultantSource.templates_index:
            result = ConsultantRetrievalResult(
                source=ConsultantSource.templates_index,
                query=_text(query)[:220],
                result_count=0,
                unavailable_reason="templates_index is not available in the current runtime.",
            )
            retrieval_results.append(result)
            tools_used.append(
                ConsultantToolUsage(
                    tool_name="templates_index_tool",
                    source=ConsultantSource.templates_index,
                    call_order=len(tools_used) + 1,
                    query=result.query,
                    result_count=0,
                    status="unavailable",
                    notes=result.unavailable_reason,
                )
            )
            continue
        if source == ConsultantSource.active_workflow:
            workflow_json = active_workflow.get("workflow_json")
            node_count = (
                len(workflow_json.get("nodes", []))
                if isinstance(workflow_json, dict) and isinstance(workflow_json.get("nodes"), list)
                else 0
            )
            if not active_workflow and node_count == 0:
                result = ConsultantRetrievalResult(
                    source=ConsultantSource.active_workflow,
                    query="active_workflow",
                    result_count=0,
                    unavailable_reason="No active workflow context is available in this request.",
                )
                status = "unavailable"
            else:
                refs: List[str] = []
                if active_workflow.get("id"):
                    refs.append(f"id={active_workflow['id']}")
                if active_workflow.get("name"):
                    refs.append(f"name={active_workflow['name']}")
                if active_workflow.get("url"):
                    refs.append(f"url={active_workflow['url']}")
                result = ConsultantRetrievalResult(
                    source=ConsultantSource.active_workflow,
                    query="active_workflow",
                    result_count=node_count,
                    references=refs,
                    snippets=[f"node_count={node_count}"] if node_count else [],
                )
                status = "ok"
            retrieval_results.append(result)
            tools_used.append(
                ConsultantToolUsage(
                    tool_name="active_workflow_tool",
                    source=ConsultantSource.active_workflow,
                    call_order=len(tools_used) + 1,
                    query=result.query,
                    result_count=result.result_count,
                    status=status,  # type: ignore[arg-type]
                    notes=result.unavailable_reason,
                )
            )
            continue
        if source not in {
            ConsultantSource.nodes_index,
            ConsultantSource.credentials_index,
            ConsultantSource.api_docs_index,
        }:
            continue
        chunks = retrieve_context(
            query,
            top_k=max(3, min(settings.TOP_K, 8)),
            request_id=request_id,
            source_filter=_source_filter(source),
        )
        result = _compact_retrieval(source, query, chunks)
        retrieval_results.append(result)
        tool_name = {
            ConsultantSource.nodes_index: "nodes_index_tool",
            ConsultantSource.credentials_index: "credentials_index_tool",
            ConsultantSource.api_docs_index: "api_docs_index_tool",
        }[source]
        tools_used.append(
            ConsultantToolUsage(
                tool_name=tool_name,
                source=source,
                call_order=len(tools_used) + 1,
                query=result.query,
                result_count=result.result_count,
                status="ok",
            )
        )


def consultant_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    model, request_id, runtime_context = _runtime(state)
    query = _text(state.get("user_query"))
    conversation_context = _conversation(runtime_context)
    active_workflow = _active_workflow(state, runtime_context)
    routing_signals = list(state.get("routing_signals") or [])
    if "entered_consultant_agent" not in routing_signals:
        routing_signals.append("entered_consultant_agent")

    analysis = analyze_consultant_query(
        query=query,
        model=model,
        request_id=request_id,
        conversation_context=conversation_context,
        active_workflow_available=bool(active_workflow),
    )
    selected_sources = _dedupe(analysis.selected_sources or [ConsultantSource.conversation_history])
    if analysis.retrieval_needed and not any(
        source
        in {
            ConsultantSource.api_docs_index,
            ConsultantSource.nodes_index,
            ConsultantSource.credentials_index,
            ConsultantSource.active_workflow,
            ConsultantSource.templates_index,
        }
        for source in selected_sources
    ):
        selected_sources.append(ConsultantSource.api_docs_index)

    tools_used: List[ConsultantToolUsage] = []
    retrieval_results: List[ConsultantRetrievalResult] = []
    notes: List[str] = []

    answer_text = ""
    if analysis.retrieval_needed:
        tools = _build_tools(
            query=query,
            request_id=request_id,
            selected_sources=selected_sources,
            active_workflow=active_workflow,
            tools_used=tools_used,
            retrieval_results=retrieval_results,
        )
        if create_agent is not None and tools:
            llm = get_langchain_chat_model(model=model, temperature=0.1)
            if llm is not None:
                system_prompt = (
                    "You are consultant_agent for n8n. Informational responses only. "
                    "Do not build/edit/fix workflows and do not output workflow JSON."
                )
                user_prompt = f"Question: {query}\\nUse tools only when needed and be source-grounded."
                resolved_model = resolve_model(model)
                emit_llm_prompt_event(
                    trace_logger,
                    request_id=request_id,
                    stage="multi_agent.consultant.tool_agent",
                    model=resolved_model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    params={"temperature": 0.1, "tool_count": len(tools)},
                )
                started = time.perf_counter()
                try:
                    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
                    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
                    answer_text = _extract_answer_text(result)
                    emit_llm_output_event(
                        trace_logger,
                        request_id=request_id,
                        stage="multi_agent.consultant.tool_agent",
                        model=resolved_model,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        content=answer_text,
                        usage=None,
                        extra={"tool_calls_recorded": len(tools_used)},
                    )
                except Exception as exc:
                    notes.append("tool_agent_failed_fallback")
                    logger.warning("consultant tool agent failed: %s", str(exc))
        if not tools_used and tools:
            for selected_tool in tools:
                try:
                    selected_tool.invoke({"q": query})
                except Exception:
                    try:
                        selected_tool.invoke({"query": query})
                    except Exception:
                        continue
        if not tools_used:
            _direct_retrieval_for_sources(
                query=query,
                request_id=request_id,
                selected_sources=selected_sources,
                active_workflow=active_workflow,
                tools_used=tools_used,
                retrieval_results=retrieval_results,
            )
        if not answer_text:
            lines = ["Based on available sources:"] if retrieval_results else ["I could not retrieve additional sources for this request."]
            for result in retrieval_results[:4]:
                if result.unavailable_reason:
                    lines.append(f"- {result.source.value}: unavailable ({result.unavailable_reason})")
                else:
                    snippet = result.snippets[0] if result.snippets else "No concise snippet available."
                    lines.append(f"- {result.source.value}: {snippet}")
            answer_text = "\\n".join(lines)
    else:
        notes.append("conversation_first_no_tools")
        answer_text = _conversation_only_answer(
            query=query,
            conversation_context=conversation_context,
        )

    consultant_response = ConsultantResponse(
        text=answer_text,
        directly_supported=[
            f"{item.source.value}: {item.references[0]}"
            for item in retrieval_results
            if item.references and not item.unavailable_reason
        ][:6],
        inferred_guidance=[],
        uncertainties=[item.unavailable_reason for item in retrieval_results if item.unavailable_reason][:4],
    )
    used_retrieval = bool(tools_used)

    emit_trace_event(
        trace_logger,
        event="consultant_agent_result",
        request_id=request_id,
        stage="multi_agent.consultant",
        payload={
            "retrieval_needed": analysis.retrieval_needed,
            "selected_sources": [item.value for item in selected_sources],
            "tool_calls": [item.tool_name for item in tools_used],
            "tool_call_order": [item.call_order for item in tools_used],
            "used_retrieval": used_retrieval,
            "retrieval_results_count": len(retrieval_results),
            "response_chars": len(answer_text),
        },
    )

    return {
        "current_stage": "consultant_agent",
        "routing_signals": routing_signals,
        "consultant_query_analysis": analysis,
        "consultant_selected_sources": selected_sources,
        "consultant_tools_used": tools_used,
        "consultant_used_retrieval": used_retrieval,
        "consultant_retrieval_results": retrieval_results,
        "consultant_response": consultant_response,
        "consultant_notes": notes,
    }
