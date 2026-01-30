from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..config import settings
from ..db import check_db
from ..llm import chat_completion, list_models, resolve_model
from ..memory import MemoryStore
from ..rag import (
    append_references,
    build_context_block,
    build_system_prompt,
    collect_references,
    extract_last_user_message,
    format_references,
    normalize_messages,
    retrieve_context,
)
from ..schemas import ChatCompletionRequest, DebugWorkflowRequest
from ..workflow.control_parser import parse_control_state
from ..workflow.n8n_client import N8NClient, N8NClientError
from ..workflow.node_analyzer import NodeFinding, analyze_node
from ..workflow.planner import Plan, PlanScope, plan_scope
from ..workflow.retriever import retrieve_docs
from ..workflow.subgraph_builder import SubgraphSelection, build_subgraph
from ..workflow.synthesizer import build_synthesis_messages
from ..workflow.workflow_summary import (
    WorkflowSummary,
    build_workflow_summary,
    micro_context_for_nodes,
    render_micro_context,
)


class ChatService:
    _auto_conversations: Dict[str, tuple[str, float]] = {}
    _dedup_cache: Dict[str, tuple[str, str, float]] = {}

    def __init__(self, memory_store: MemoryStore) -> None:
        """Initialize service dependencies and logger."""
        self._memory_store = memory_store
        self._n8n_client = N8NClient()
        self._logger = logging.getLogger("n8n-assistant")

    def health(self) -> Dict[str, Any]:
        """Return DB and LM Studio health status."""
        db_ok, db_error = check_db()
        lm_models = list_models()
        lm_ok = lm_models is not None

        status = "ok" if db_ok and lm_ok else "degraded"
        payload: Dict[str, Any] = {
            "status": status,
            "db_ok": db_ok,
            "lmstudio_ok": lm_ok,
        }
        if not db_ok:
            payload["db_error"] = db_error or "db check failed"
        if not lm_ok:
            payload["lmstudio_error"] = "lm studio not responding"
        return payload

    def list_models(self) -> Dict[str, Any]:
        """Return model list in OpenAI-compatible shape."""
        data = list_models()
        if data:
            return data

        model_id = settings.LLM_MODEL or "local-model"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    def list_chats(self, limit: int, offset: int) -> Dict[str, Any]:
        """List stored conversations with paging."""
        if not settings.MEMORY_ENABLED:
            raise HTTPException(status_code=400, detail="Memory is disabled")
        items = self._memory_store.list_conversations(limit, offset)
        return {
            "object": "list",
            "data": [
                {
                    "id": item.conversation_id,
                    "title": item.title,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                    "message_count": item.message_count,
                }
                for item in items
            ],
        }

    def get_chat(self, conversation_id: str) -> Dict[str, Any]:
        """Return a full conversation history by id."""
        if not settings.MEMORY_ENABLED:
            raise HTTPException(status_code=400, detail="Memory is disabled")
        conversation = self._memory_store.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {
            "id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "message_count": conversation.message_count,
            "messages": conversation.messages,
        }

    def debug_workflow(self, request: DebugWorkflowRequest) -> Dict[str, Any]:
        """Return planner + subgraph + docs selection for debugging."""
        workflow = self._fetch_workflow(request.workflow_id)
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

    def create_chat_completions(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
        http_response: Optional[Response] = None,
    ) -> Any:
        """Main chat completion handler (docs-only or workflow-aware)."""
        messages = normalize_messages([msg.model_dump() for msg in request.messages])
        raw_user_message = self._extract_last_user_raw(messages)

        conversation_id: Optional[str] = None
        generated_conversation_id = False
        history: List[Dict[str, str]] = []
        if settings.MEMORY_ENABLED:
            conversation_id = self._get_conversation_id(request, http_request)
            if conversation_id and settings.MEMORY_AUTO_CREATE_CONVERSATION_ID:
                self._remember_auto_conversation(http_request, conversation_id)
            if not conversation_id and settings.MEMORY_AUTO_CREATE_CONVERSATION_ID:
                conversation_id, generated_conversation_id = self._get_or_create_auto_conversation_id(
                    http_request
                )
            if conversation_id:
                history = self._memory_store.get_recent_messages(
                    conversation_id, settings.MEMORY_MAX_MESSAGES
                )

        combined_messages = history + messages if history else list(messages)
        control_combined = parse_control_state(combined_messages)
        control_current = parse_control_state(messages)

        include_history = bool(history) and self._count_non_system(messages) <= 1
        messages_for_prompt = (
            control_combined.cleaned_messages if include_history else control_current.cleaned_messages
        )

        user_message = extract_last_user_message(control_combined.cleaned_messages)
        active_workflow_id = control_combined.active_workflow_id

        if not user_message:
            model = resolve_model(request.model)
            if active_workflow_id:
                text = (
                    f"Workflow activo: {active_workflow_id}. "
                    "Listo. Ahora dime que parte quieres revisar (nodo, rama o global)."
                )
            else:
                raise HTTPException(status_code=400, detail="No user message found")

            self._append_memory(conversation_id, raw_user_message, text)
            if request.stream:
                return StreamingResponse(
                    self._stream_simple_text(text, model=model),
                    headers=self._conversation_headers(conversation_id, generated_conversation_id),
                    media_type="text/event-stream",
                )

            response_dict = self._simple_chat_response(text, model=model)
            if http_response is not None:
                self._apply_conversation_headers(
                    http_response, conversation_id, generated_conversation_id
                )
            return response_dict

        self._logger.info(
            "chat_completion request: messages=%d user_len=%d history=%d wf=%s",
            len(messages),
            len(user_message),
            len(history),
            active_workflow_id,
        )

        if not active_workflow_id:
            return self._handle_docs_only(
                request=request,
                messages_for_prompt=messages_for_prompt,
                user_message=user_message,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
            )

        return self._handle_workflow_aware(
            request=request,
            user_message=user_message,
            messages_for_prompt=messages_for_prompt,
            active_workflow_id=active_workflow_id,
            conversation_id=conversation_id,
            generated_conversation_id=generated_conversation_id,
            http_response=http_response,
            raw_user_message=raw_user_message,
        )

    def _handle_docs_only(
        self,
        request: ChatCompletionRequest,
        messages_for_prompt: List[Dict[str, str]],
        user_message: str,
        conversation_id: Optional[str],
        generated_conversation_id: bool,
        http_response: Optional[Response],
        raw_user_message: Optional[str],
    ) -> Any:
        """RAG-only path when no active workflow is selected."""
        try:
            chunks = retrieve_context(user_message)
        except Exception as exc:
            self._logger.exception("rag retrieval failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    "RAG failed. Check EMBEDDING_MODEL, LM Studio embeddings support, "
                    "and database mapping in .env."
                ),
            ) from exc

        context_block = build_context_block(chunks)
        system_prompt = build_system_prompt(context_block)
        llm_messages = [{"role": "system", "content": system_prompt}]
        llm_messages.extend(messages_for_prompt)

        model = resolve_model(request.model)
        params = self._completion_params(request, stream=request.stream)

        refs = collect_references(chunks)
        if request.stream:
            params["stream"] = True
            try:
                stream = chat_completion(llm_messages, model=model, **params)
            except Exception as exc:
                self._logger.exception("lm studio request failed")
                raise HTTPException(
                    status_code=502, detail="LM Studio request failed"
                ) from exc

            def _on_stream_complete(assistant_text: str) -> None:
                self._append_memory(conversation_id, raw_user_message, assistant_text)

            return StreamingResponse(
                self._stream_with_references(stream, refs, model, _on_stream_complete),
                headers=self._conversation_headers(conversation_id, generated_conversation_id),
                media_type="text/event-stream",
            )

        try:
            llm_response = chat_completion(llm_messages, model=model, **params)
        except Exception as exc:
            self._logger.exception("lm studio request failed")
            raise HTTPException(status_code=502, detail="LM Studio request failed") from exc

        response_dict = llm_response.model_dump()
        assistant_text_raw = response_dict["choices"][0]["message"].get("content") or ""
        assistant_text = append_references(assistant_text_raw, refs)

        self._append_memory(conversation_id, raw_user_message, assistant_text_raw)

        response_dict["choices"][0]["message"]["content"] = assistant_text
        response_dict["model"] = model
        if http_response is not None:
            self._apply_conversation_headers(http_response, conversation_id, generated_conversation_id)
        return response_dict

    def _handle_workflow_aware(
        self,
        request: ChatCompletionRequest,
        user_message: str,
        messages_for_prompt: List[Dict[str, str]],
        active_workflow_id: str,
        conversation_id: Optional[str],
        generated_conversation_id: bool,
        http_response: Optional[Response],
        raw_user_message: Optional[str],
    ) -> Any:
        """Workflow-aware path: plan scope, select subgraph, analyze nodes."""
        workflow = self._fetch_workflow(active_workflow_id)
        summary = build_workflow_summary(workflow, workflow_id=active_workflow_id)

        plan = plan_scope(user_message, summary)
        if plan.scope == PlanScope.no_workflow:
            return self._handle_docs_only(
                request=request,
                messages_for_prompt=messages_for_prompt,
                user_message=user_message,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
            )

        self._logger.info(
            "workflow plan: wf=%s scope=%s targets=%s confidence=%.2f truncated=%s",
            active_workflow_id,
            plan.scope.value,
            plan.target_node_ids[:8],
            plan.confidence,
            summary.truncated,
        )

        if plan.need_clarification:
            model = resolve_model(request.model)
            text = self._clarification_text(plan, summary)
            self._append_memory(conversation_id, raw_user_message, text)
            if request.stream:
                return StreamingResponse(
                    self._stream_simple_text(text, model=model),
                    headers=self._conversation_headers(conversation_id, generated_conversation_id),
                    media_type="text/event-stream",
                )
            response_dict = self._simple_chat_response(text, model=model)
            if http_response is not None:
                self._apply_conversation_headers(http_response, conversation_id, generated_conversation_id)
            return response_dict

        subgraph = build_subgraph(summary, user_message, plan)
        docs_chunks = self._retrieve_docs_for_subgraph(user_message, summary, subgraph)
        refs = collect_references(docs_chunks)

        findings = self._analyze_nodes(user_message, summary, subgraph, docs_chunks, plan, request.model)
        chat_context = self._chat_context_block(messages_for_prompt, user_message)
        llm_messages = build_synthesis_messages(
            question=user_message,
            plan=plan,
            summary=summary,
            subgraph=subgraph,
            findings=findings,
            docs_chunks=docs_chunks,
            chat_context=chat_context,
        )

        model = resolve_model(request.model)
        params = self._completion_params(request, stream=request.stream)

        if request.stream:
            params["stream"] = True
            try:
                stream = chat_completion(llm_messages, model=model, **params)
            except Exception as exc:
                self._logger.exception("lm studio request failed")
                raise HTTPException(status_code=502, detail="LM Studio request failed") from exc

            def _on_stream_complete(assistant_text: str) -> None:
                self._append_memory(conversation_id, raw_user_message, assistant_text)

            return StreamingResponse(
                self._stream_with_references(stream, refs, model, _on_stream_complete),
                headers=self._conversation_headers(conversation_id, generated_conversation_id),
                media_type="text/event-stream",
            )

        try:
            llm_response = chat_completion(llm_messages, model=model, **params)
        except Exception as exc:
            self._logger.exception("lm studio request failed")
            raise HTTPException(status_code=502, detail="LM Studio request failed") from exc

        response_dict = llm_response.model_dump()
        assistant_text_raw = response_dict["choices"][0]["message"].get("content") or ""
        assistant_text = append_references(assistant_text_raw, refs)

        self._append_memory(conversation_id, raw_user_message, assistant_text_raw)

        response_dict["choices"][0]["message"]["content"] = assistant_text
        response_dict["model"] = model
        if http_response is not None:
            self._apply_conversation_headers(http_response, conversation_id, generated_conversation_id)
        return response_dict

    def _fetch_workflow(self, workflow_id: str) -> Dict[str, Any]:
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

    def _retrieve_docs_for_subgraph(
        self, question: str, summary: WorkflowSummary, subgraph: SubgraphSelection
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
        try:
            return retrieve_docs(question, node_types=node_types, node_names=node_names)
        except Exception as exc:
            self._logger.exception("workflow docs retrieval failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    "RAG failed during workflow-aware retrieval. "
                    "Revisa EMBEDDING_MODEL y la configuracion de la base."
                ),
            ) from exc

    def _analyze_nodes(
        self,
        question: str,
        summary: WorkflowSummary,
        subgraph: SubgraphSelection,
        docs_chunks: List[Dict[str, Any]],
        plan: Plan,
        requested_model: Optional[str],
    ) -> List[NodeFinding]:
        """Run per-node analysis with micro-context for the selected nodes."""
        findings: List[NodeFinding] = []
        for node_id in subgraph.node_ids[: settings.WORKFLOW_MAX_NODES_GLOBAL]:
            node = summary.nodes.get(node_id)
            if not node:
                continue
            micro = micro_context_for_nodes(summary, [node_id])
            micro_text = render_micro_context(micro)
            finding = analyze_node(
                question=question,
                node=node,
                micro_context_text=micro_text,
                docs_chunks=docs_chunks,
                scope=plan.scope,
                model=requested_model,
            )
            findings.append(finding)
        return findings

    def _clarification_text(self, plan: Plan, summary: WorkflowSummary) -> str:
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
    def _chat_context_block(messages: List[Dict[str, str]], last_user_message: str) -> str:
        """Build a short recent-chat context block for synthesis prompts."""
        if not messages:
            return ""

        trimmed: List[Dict[str, str]] = []
        skipped_last_user = False
        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content")
            if role != "user" or not isinstance(content, str):
                trimmed.append(msg)
                continue
            if not skipped_last_user and content.strip() == last_user_message.strip():
                skipped_last_user = True
                continue
            trimmed.append(msg)

        trimmed.reverse()
        recent = trimmed[-6:]
        lines: List[str] = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            compact = " ".join(content.split())
            if len(compact) > 240:
                compact = compact[:237].rstrip() + "..."
            lines.append(f"- {role}: {compact}")
        return "\n".join(lines)

    @staticmethod
    def _completion_params(request: ChatCompletionRequest, stream: bool) -> Dict[str, Any]:
        """Translate request fields into LM Studio completion params."""
        params: Dict[str, Any] = {}
        params["temperature"] = (
            request.temperature if request.temperature is not None else settings.DEFAULT_TEMPERATURE
        )
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        else:
            params["top_p"] = settings.DEFAULT_TOP_P
        if request.frequency_penalty is not None:
            params["frequency_penalty"] = request.frequency_penalty
        else:
            params["frequency_penalty"] = settings.DEFAULT_FREQUENCY_PENALTY
        if request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty
        else:
            params["presence_penalty"] = settings.DEFAULT_PRESENCE_PENALTY
        if request.stop is not None:
            params["stop"] = request.stop
        if stream:
            params["stream"] = True
        return params

    def _append_memory(
        self,
        conversation_id: Optional[str],
        raw_user_message: Optional[str],
        assistant_text: str,
    ) -> None:
        """Append user+assistant messages to memory with dedup."""
        if not (settings.MEMORY_ENABLED and conversation_id):
            return
        new_messages: List[Dict[str, str]] = []
        if raw_user_message:
            if not self._should_skip_duplicate(conversation_id, "user", raw_user_message):
                new_messages.append({"role": "user", "content": raw_user_message})
        if assistant_text:
            if not self._should_skip_duplicate(conversation_id, "assistant", assistant_text):
                new_messages.append({"role": "assistant", "content": assistant_text})
        if new_messages:
            self._memory_store.append_messages(conversation_id, new_messages)

    @staticmethod
    def _extract_last_user_raw(messages: Iterable[Dict[str, str]]) -> Optional[str]:
        """Get last raw user message (before control cleaning)."""
        for msg in reversed(list(messages)):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"].strip()
        return None

    @staticmethod
    def _simple_chat_response(text: str, model: str) -> Dict[str, Any]:
        """Return a minimal non-streaming response payload."""
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    def _stream_simple_text(self, text: str, model: str) -> Iterable[str]:
        """Yield a simple SSE stream for short responses."""
        now = int(time.time())
        chunk_id = f"chatcmpl-{now}"
        first_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }
        yield self._format_sse(first_chunk)

        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield self._format_sse(final_chunk)
        yield "data: [DONE]\n\n"

    def _get_conversation_id(
        self, payload: ChatCompletionRequest, http_request: Request
    ) -> Optional[str]:
        """Extract conversation id from body, headers, or metadata."""
        if payload.conversation_id:
            return str(payload.conversation_id)

        header_value = http_request.headers.get(settings.MEMORY_HEADER)
        if header_value:
            return str(header_value)

        extra = payload.model_extra or {}
        metadata = extra.get("metadata") if isinstance(extra, dict) else None
        keys = ("conversation_id", "chat_id", "thread_id", "session_id")
        if isinstance(metadata, dict):
            for key in keys:
                value = metadata.get(key)
                if value:
                    return str(value)

        if isinstance(extra, dict):
            for key in keys:
                value = extra.get(key)
                if value:
                    return str(value)
            chat = extra.get("chat")
            if isinstance(chat, dict):
                for key in ("id", "chat_id"):
                    value = chat.get(key)
                    if value:
                        return str(value)

        if payload.user:
            return f"user:{payload.user}"

        if self._logger.isEnabledFor(logging.DEBUG):
            extra_keys = list(extra.keys()) if isinstance(extra, dict) else []
            metadata_keys = list(metadata.keys()) if isinstance(metadata, dict) else []
            self._logger.debug(
                "memory: no conversation id found (header=%s, extra_keys=%s, metadata_keys=%s)",
                header_value,
                extra_keys,
                metadata_keys,
            )

        return None

    @staticmethod
    def _compact_snippet(text: str, max_chars: int) -> str:
        """Compact a text snippet for debug payloads."""
        compact = " ".join(text.split())
        if len(compact) > max_chars:
            compact = compact[: max_chars - 3].rstrip() + "..."
        return compact

    def _client_key(self, http_request: Request) -> str:
        """Build a stable client key from IP + user agent."""
        host = http_request.client.host if http_request.client else "unknown"
        user_agent = http_request.headers.get("user-agent", "").strip().lower()
        if user_agent:
            return f"{host}|{user_agent}"
        return host

    def _get_or_create_auto_conversation_id(self, http_request: Request) -> tuple[str, bool]:
        """Return a sticky auto conversation id for this client."""
        key = self._client_key(http_request)
        now = time.time()
        ttl = max(settings.MEMORY_TTL_SECONDS, 0)

        existing = self._auto_conversations.get(key)
        if existing:
            conv_id, updated_at = existing
            if ttl == 0 or (now - updated_at) <= ttl:
                self._auto_conversations[key] = (conv_id, now)
                return conv_id, False

        conv_id = f"auto:{uuid4()}"
        self._auto_conversations[key] = (conv_id, now)
        self._logger.info("memory: generated conversation id %s for client %s", conv_id, key)
        return conv_id, True

    def _remember_auto_conversation(self, http_request: Request, conversation_id: str) -> None:
        """Persist an explicit conversation id for future auto mapping."""
        key = self._client_key(http_request)
        self._auto_conversations[key] = (conversation_id, time.time())

    def _should_skip_duplicate(self, conversation_id: str, role: str, content: str) -> bool:
        """Deduplicate repeated messages within a short time window."""
        window = max(settings.MEMORY_DEDUP_WINDOW_SECONDS, 0)
        if window == 0:
            return False
        normalized = " ".join(content.split())
        now = time.time()
        key = f"{conversation_id}:{role}"
        cached = self._dedup_cache.get(key)
        if cached:
            last_role, last_content, last_ts = cached
            if last_role == role and last_content == normalized and (now - last_ts) <= window:
                return True
        self._dedup_cache[key] = (role, normalized, now)
        return False

    @staticmethod
    def _conversation_headers(
        conversation_id: Optional[str], generated: bool
    ) -> Dict[str, str]:
        """Build response headers for conversation tracking."""
        headers: Dict[str, str] = {}
        if conversation_id:
            headers["X-Conversation-Id"] = conversation_id
            if generated:
                headers["X-Conversation-Id-Generated"] = "true"
        return headers

    @staticmethod
    def _apply_conversation_headers(
        response: Response, conversation_id: Optional[str], generated: bool
    ) -> None:
        """Apply conversation headers to an HTTP response."""
        if not conversation_id:
            return
        response.headers["X-Conversation-Id"] = conversation_id
        if generated:
            response.headers["X-Conversation-Id-Generated"] = "true"

    @staticmethod
    def _count_non_system(messages: Iterable[Dict[str, str]]) -> int:
        """Count non-system messages in a list."""
        return sum(1 for msg in messages if msg.get("role") != "system")

    @staticmethod
    def _format_sse(data: Dict[str, Any]) -> str:
        """Format an SSE chunk."""
        return f"data: {json.dumps(data)}\n\n"

    def _stream_with_references(
        self,
        stream: Iterable[Any],
        refs: List[Dict[str, str]],
        fallback_model: str,
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> Iterable[str]:
        """Stream model output and append references at the end."""
        last_id: str | None = None
        last_created = int(time.time())
        last_model = fallback_model
        finish_reason: str | None = None
        assistant_parts: List[str] = []

        for chunk in stream:
            chunk_dict = chunk.model_dump()
            last_id = chunk_dict.get("id") or last_id
            last_created = chunk_dict.get("created") or last_created
            last_model = chunk_dict.get("model") or last_model

            choices = chunk_dict.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    assistant_parts.append(content)
            if finish_reason:
                continue

            yield self._format_sse(chunk_dict)

        refs_block = format_references(refs)
        if refs_block:
            refs_block = "\n\n" + refs_block
            refs_chunk = {
                "id": last_id or f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion.chunk",
                "created": last_created,
                "model": last_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": refs_block},
                        "finish_reason": None,
                    }
                ],
            }
            yield self._format_sse(refs_chunk)

        if on_complete:
            on_complete("".join(assistant_parts))

        final_chunk = {
            "id": last_id or f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion.chunk",
            "created": last_created,
            "model": last_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason or "stop",
                }
            ],
        }
        yield self._format_sse(final_chunk)
        yield "data: [DONE]\n\n"
