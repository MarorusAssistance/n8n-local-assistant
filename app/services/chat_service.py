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
from ..graphs import MasterGraphRuntime
from ..llm import chat_completion, list_models, resolve_model
from ..observability import build_graph_run_config
from ..memory import MemoryStore
from ..reasoning.pipeline import run_reasoning_pipeline
from ..rag import (
    append_references,
    build_context_block,
    build_system_prompt,
    collect_references,
    extract_last_user_message,
    normalize_messages,
    retrieve_context,
)
from ..schemas import ChatCompletionRequest, DebugWorkflowRequest
from ..token_budget import PromptBudgetResult, trim_messages_to_budget
from ..workflow.control_parser import parse_control_state
from ..workflow.n8n_client import N8NClient
from ..workflow.planner import PlanScope, plan_scope
from ..workflow.subgraph_builder import build_subgraph
from ..workflow.synthesizer import build_synthesis_messages
from ..workflow.workflow_summary import build_workflow_summary
from .chat_memory import ConversationManager
from .chat_responses import ChatResponseBuilder
from .workflow_service import WorkflowService


class ChatService:
    def __init__(self, memory_store: MemoryStore) -> None:
        """Initialize service dependencies and logger."""
        self._n8n_client = N8NClient()
        self._logger = logging.getLogger("n8n-assistant")
        self._trace_logger = logging.getLogger("n8n-assistant.trace")
        self._memory = ConversationManager(memory_store, self._logger)
        self._responses = ChatResponseBuilder()
        self._workflow = WorkflowService(self._n8n_client, self._logger)
        self._graph_runtime = MasterGraphRuntime(self._workflow)

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
        return self._memory.list_chats(limit, offset)

    def get_chat(self, conversation_id: str) -> Dict[str, Any]:
        """Return a full conversation history by id."""
        return self._memory.get_chat(conversation_id)

    def debug_workflow(self, request: DebugWorkflowRequest) -> Dict[str, Any]:
        """Return planner + subgraph + docs selection for debugging."""
        return self._workflow.debug_workflow(request)

    def create_chat_completions(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
        http_response: Optional[Response] = None,
    ) -> Any:
        """Main chat completion handler (docs-only or workflow-aware)."""
        messages = normalize_messages([msg.model_dump() for msg in request.messages])
        raw_user_message = self._extract_last_user_raw(messages)
        request_id = self._resolve_request_id(http_request)

        conversation_id: Optional[str] = None
        generated_conversation_id = False
        history: List[Dict[str, str]] = []
        if settings.MEMORY_ENABLED:
            conversation_id = self._memory.resolve_conversation_id(request, http_request)
            if conversation_id and settings.MEMORY_AUTO_CREATE_CONVERSATION_ID:
                self._memory.remember_auto_conversation(http_request, conversation_id)
            if not conversation_id and settings.MEMORY_AUTO_CREATE_CONVERSATION_ID:
                conversation_id, generated_conversation_id = self._memory.get_or_create_auto_conversation_id(
                    http_request
                )
            if conversation_id:
                history = self._memory.get_recent_messages(
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

            self._memory.append_memory(conversation_id, raw_user_message, text)
            if request.stream:
                return StreamingResponse(
                    self._responses.stream_simple_text(text, model=model),
                    headers=self._memory.conversation_headers(
                        conversation_id, generated_conversation_id
                    ),
                    media_type="text/event-stream",
                )

            response_dict = self._responses.simple_chat_response(text, model=model)
            if http_response is not None:
                self._memory.apply_conversation_headers(
                    http_response, conversation_id, generated_conversation_id
                )
            return response_dict

        self._trace_request(
            request_id=request_id,
            conversation_id=conversation_id,
            workflow_id=active_workflow_id,
            stream=request.stream,
            messages_count=len(messages),
            history_count=len(history),
            user_message=user_message,
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
                request_id=request_id,
            )

        if self._use_langgraph_workflow_runtime():
            return self._handle_workflow_aware_langgraph(
                request=request,
                user_message=user_message,
                messages_for_prompt=messages_for_prompt,
                active_workflow_id=active_workflow_id,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
                request_id=request_id,
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
            request_id=request_id,
        )

    @staticmethod
    def _use_langgraph_runtime() -> bool:
        return settings.AGENT_RUNTIME == "langgraph"

    @classmethod
    def _use_langgraph_reasoning_runtime(cls) -> bool:
        return cls._use_langgraph_runtime() and settings.LANGGRAPH_REASONING_ENABLED

    @classmethod
    def _use_langgraph_workflow_runtime(cls) -> bool:
        return cls._use_langgraph_runtime() and settings.LANGGRAPH_WORKFLOW_ENABLED

    def _handle_docs_only(
        self,
        request: ChatCompletionRequest,
        messages_for_prompt: List[Dict[str, str]],
        user_message: str,
        conversation_id: Optional[str],
        generated_conversation_id: bool,
        http_response: Optional[Response],
        raw_user_message: Optional[str],
        request_id: str,
    ) -> Any:
        """RAG-only path when no active workflow is selected."""
        self._trace_logger.debug("docs-only start: id=%s conv=%s", request_id, conversation_id)
        if self._use_langgraph_reasoning_runtime():
            return self._handle_reasoning_plan_only_langgraph(
                request=request,
                user_message=user_message,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
                request_id=request_id,
            )

        if settings.REASONING_PIPELINE_ENABLED:
            return self._handle_reasoning_plan_only(
                request=request,
                user_message=user_message,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
                request_id=request_id,
            )
        try:
            chunks = retrieve_context(user_message, request_id=request_id)
        except Exception as exc:
            self._logger.exception("rag retrieval failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    "RAG failed. Check EMBEDDING_MODEL, LM Studio embeddings support, "
                    "and database mapping in .env."
                ),
            ) from exc

        def _build_docs_prompt(current_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
            context_block = build_context_block(current_chunks)
            system_prompt = build_system_prompt(context_block)
            llm_messages = [{"role": "system", "content": system_prompt}]
            llm_messages.extend(messages_for_prompt)
            return llm_messages

        llm_messages, chunks, prompt_budget = self._apply_prompt_budget_with_docs_fallback(
            docs_chunks=chunks,
            request_id=request_id,
            mode="docs",
            message_builder=_build_docs_prompt,
        )
        context_block = build_context_block(chunks)

        model = resolve_model(request.model)
        params = self._completion_params(request, stream=request.stream)

        refs = collect_references(chunks)
        self._trace_logger.debug(
            "docs-only context: id=%s chunks=%d context_chars=%d refs=%d",
            request_id,
            len(chunks),
            len(context_block),
            len(refs),
        )
        self._trace_logger.debug(
            "docs-only prompt stats: id=%s messages=%d chars=%d est_tokens=%d",
            request_id,
            len(llm_messages),
            self._messages_char_count(llm_messages),
            prompt_budget.estimated_tokens_after,
        )
        self._trace_logger.debug(
            "llm request: id=%s model=%s temp=%s max_tokens=%s top_p=%s freq_pen=%s pres_pen=%s stream=%s",
            request_id,
            model,
            params.get("temperature"),
            params.get("max_tokens"),
            params.get("top_p"),
            params.get("frequency_penalty"),
            params.get("presence_penalty"),
            request.stream,
        )
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
                self._memory.append_memory(conversation_id, raw_user_message, assistant_text)
                self._log_response_summary(request_id, model, assistant_text, refs)

            return StreamingResponse(
                self._responses.stream_with_references(stream, refs, model, _on_stream_complete),
                headers=self._memory.conversation_headers(
                    conversation_id, generated_conversation_id
                ),
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
        self._log_response_summary(request_id, model, assistant_text_raw, refs)

        self._memory.append_memory(conversation_id, raw_user_message, assistant_text_raw)

        response_dict["choices"][0]["message"]["content"] = assistant_text
        response_dict["model"] = model
        if http_response is not None:
            self._memory.apply_conversation_headers(
                http_response, conversation_id, generated_conversation_id
            )
        return response_dict

    def _handle_reasoning_plan_only(
        self,
        request: ChatCompletionRequest,
        user_message: str,
        conversation_id: Optional[str],
        generated_conversation_id: bool,
        http_response: Optional[Response],
        raw_user_message: Optional[str],
        request_id: str,
    ) -> Any:
        self._trace_logger.debug("reasoning pipeline start: id=%s conv=%s", request_id, conversation_id)

        try:
            result = run_reasoning_pipeline(
                user_prompt=user_message,
                model=request.model,
                request_id=request_id,
                existing_workflow=None,
            )
        except Exception as exc:
            self._logger.exception("reasoning pipeline failed")
            raise HTTPException(
                status_code=502,
                detail="Reasoning pipeline failed. Check local model and retrieval configuration.",
            ) from exc

        issue_count = len(result.checker.issues)
        self._trace_logger.info(
            (
                "reasoning plan-only result: id=%s intent=%s node_cards=%d doc_chunks=%d "
                "max_tokens=%d estimated_tokens=%d second_iteration=%s issues=%d"
            ),
            request_id,
            result.router.intent,
            len(result.context_pack.nodeCards),
            len(result.context_pack.docChunks),
            result.context_pack.budget.maxContextTokens,
            result.context_pack.budget.estimatedTokens,
            result.second_iteration_used,
            issue_count,
        )

        if result.checker.ok:
            payload: Dict[str, Any] = result.plan.model_dump(exclude_none=True)
        else:
            payload = {
                "plan": result.plan.model_dump(exclude_none=True),
                "checker": result.checker.model_dump(exclude_none=True),
            }

        assistant_text_raw = json.dumps(payload, ensure_ascii=False)
        model = resolve_model(request.model)

        self._memory.append_memory(conversation_id, raw_user_message, assistant_text_raw)

        if request.stream:
            return StreamingResponse(
                self._responses.stream_simple_text(assistant_text_raw, model=model),
                headers=self._memory.conversation_headers(
                    conversation_id, generated_conversation_id
                ),
                media_type="text/event-stream",
            )

        response_dict = self._responses.simple_chat_response(assistant_text_raw, model=model)
        if http_response is not None:
            self._memory.apply_conversation_headers(
                http_response, conversation_id, generated_conversation_id
            )
        return response_dict

    def _handle_reasoning_plan_only_langgraph(
        self,
        request: ChatCompletionRequest,
        user_message: str,
        conversation_id: Optional[str],
        generated_conversation_id: bool,
        http_response: Optional[Response],
        raw_user_message: Optional[str],
        request_id: str,
    ) -> Any:
        self._trace_logger.debug("reasoning graph start: id=%s conv=%s", request_id, conversation_id)

        run_config = build_graph_run_config(
            request_id=request_id,
            mode="reasoning",
            conversation_id=conversation_id,
            workflow_id=None,
            model=request.model,
        )

        try:
            result = self._graph_runtime.run_reasoning(
                user_prompt=user_message,
                model=request.model,
                request_id=request_id,
                existing_workflow=None,
                run_config=run_config,
            )
        except Exception as exc:
            self._logger.exception("reasoning graph failed")
            self._trace_logger.warning("reasoning graph fallback to legacy pipeline: id=%s err=%s", request_id, str(exc))
            return self._handle_reasoning_plan_only(
                request=request,
                user_message=user_message,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
                request_id=request_id,
            )

        issue_count = len(result.checker.issues)
        self._trace_logger.info(
            (
                "reasoning graph result: id=%s intent=%s node_cards=%d doc_chunks=%d "
                "max_tokens=%d estimated_tokens=%d second_iteration=%s issues=%d"
            ),
            request_id,
            result.router.intent,
            len(result.context_pack.nodeCards),
            len(result.context_pack.docChunks),
            result.context_pack.budget.maxContextTokens,
            result.context_pack.budget.estimatedTokens,
            result.second_iteration_used,
            issue_count,
        )

        if result.checker.ok:
            payload: Dict[str, Any] = result.plan.model_dump(exclude_none=True)
        else:
            payload = {
                "plan": result.plan.model_dump(exclude_none=True),
                "checker": result.checker.model_dump(exclude_none=True),
            }

        assistant_text_raw = json.dumps(payload, ensure_ascii=False)
        model = resolve_model(request.model)

        self._memory.append_memory(conversation_id, raw_user_message, assistant_text_raw)

        if request.stream:
            return StreamingResponse(
                self._responses.stream_simple_text(assistant_text_raw, model=model),
                headers=self._memory.conversation_headers(
                    conversation_id, generated_conversation_id
                ),
                media_type="text/event-stream",
            )

        response_dict = self._responses.simple_chat_response(assistant_text_raw, model=model)
        if http_response is not None:
            self._memory.apply_conversation_headers(
                http_response, conversation_id, generated_conversation_id
            )
        return response_dict

    def _handle_workflow_aware_langgraph(
        self,
        request: ChatCompletionRequest,
        user_message: str,
        messages_for_prompt: List[Dict[str, str]],
        active_workflow_id: str,
        conversation_id: Optional[str],
        generated_conversation_id: bool,
        http_response: Optional[Response],
        raw_user_message: Optional[str],
        request_id: str,
    ) -> Any:
        self._trace_logger.debug(
            "workflow graph start: id=%s conv=%s wf=%s",
            request_id,
            conversation_id,
            active_workflow_id,
        )

        chat_context = self._chat_context_block(messages_for_prompt, user_message)
        run_config = build_graph_run_config(
            request_id=request_id,
            mode="workflow",
            conversation_id=conversation_id,
            workflow_id=active_workflow_id,
            model=request.model,
        )

        try:
            graph_state = self._graph_runtime.run_workflow(
                question=user_message,
                workflow_id=active_workflow_id,
                request_model=request.model,
                request_id=request_id,
                chat_context=chat_context,
                run_config=run_config,
            )
        except Exception as exc:
            self._logger.exception("workflow graph failed")
            self._trace_logger.warning("workflow graph fallback to legacy runtime: id=%s err=%s", request_id, str(exc))
            return self._handle_workflow_aware(
                request=request,
                user_message=user_message,
                messages_for_prompt=messages_for_prompt,
                active_workflow_id=active_workflow_id,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
                request_id=request_id,
            )

        if graph_state.get("fallback_to_docs_only"):
            return self._handle_docs_only(
                request=request,
                messages_for_prompt=messages_for_prompt,
                user_message=user_message,
                conversation_id=conversation_id,
                generated_conversation_id=generated_conversation_id,
                http_response=http_response,
                raw_user_message=raw_user_message,
                request_id=request_id,
            )

        clarification_text = str(graph_state.get("clarification_text") or "").strip()
        if clarification_text:
            model = resolve_model(request.model)
            self._memory.append_memory(conversation_id, raw_user_message, clarification_text)
            if request.stream:
                return StreamingResponse(
                    self._responses.stream_simple_text(clarification_text, model=model),
                    headers=self._memory.conversation_headers(
                        conversation_id, generated_conversation_id
                    ),
                    media_type="text/event-stream",
                )
            response_dict = self._responses.simple_chat_response(clarification_text, model=model)
            if http_response is not None:
                self._memory.apply_conversation_headers(
                    http_response, conversation_id, generated_conversation_id
                )
            return response_dict

        llm_messages = graph_state.get("llm_messages") or []
        docs_chunks = graph_state.get("docs_chunks") or []
        refs = graph_state.get("refs") or collect_references(docs_chunks)
        prompt_budget = graph_state.get("prompt_budget")

        if not llm_messages:
            self._trace_logger.warning(
                "workflow graph produced empty prompt, fallback to legacy: id=%s wf=%s",
                request_id,
                active_workflow_id,
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
                request_id=request_id,
            )

        model = resolve_model(request.model)
        params = self._completion_params(request, stream=request.stream)
        estimated_tokens = int(getattr(prompt_budget, "estimated_tokens_after", 0) or 0)
        self._trace_logger.debug(
            "workflow graph synthesis prompt stats: id=%s messages=%d chars=%d est_tokens=%d",
            request_id,
            len(llm_messages),
            self._messages_char_count(llm_messages),
            estimated_tokens,
        )
        self._trace_logger.debug(
            "llm request: id=%s model=%s temp=%s max_tokens=%s top_p=%s freq_pen=%s pres_pen=%s stream=%s",
            request_id,
            model,
            params.get("temperature"),
            params.get("max_tokens"),
            params.get("top_p"),
            params.get("frequency_penalty"),
            params.get("presence_penalty"),
            request.stream,
        )

        if request.stream:
            params["stream"] = True
            try:
                stream = chat_completion(llm_messages, model=model, **params)
            except Exception as exc:
                self._logger.exception("lm studio request failed")
                raise HTTPException(status_code=502, detail="LM Studio request failed") from exc

            def _on_stream_complete(assistant_text: str) -> None:
                self._memory.append_memory(conversation_id, raw_user_message, assistant_text)
                self._log_response_summary(request_id, model, assistant_text, refs)

            return StreamingResponse(
                self._responses.stream_with_references(stream, refs, model, _on_stream_complete),
                headers=self._memory.conversation_headers(
                    conversation_id, generated_conversation_id
                ),
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
        self._log_response_summary(request_id, model, assistant_text_raw, refs)

        self._memory.append_memory(conversation_id, raw_user_message, assistant_text_raw)

        response_dict["choices"][0]["message"]["content"] = assistant_text
        response_dict["model"] = model
        if http_response is not None:
            self._memory.apply_conversation_headers(
                http_response, conversation_id, generated_conversation_id
            )
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
        request_id: str,
    ) -> Any:
        """Workflow-aware path: plan scope, select subgraph, analyze nodes."""
        self._trace_logger.debug(
            "workflow start: id=%s conv=%s wf=%s",
            request_id,
            conversation_id,
            active_workflow_id,
        )
        workflow = self._workflow.fetch_workflow(active_workflow_id)
        summary = build_workflow_summary(workflow, workflow_id=active_workflow_id)
        self._trace_logger.debug(
            "workflow summary: id=%s wf=%s nodes=%d edges=%d truncated=%s",
            request_id,
            active_workflow_id,
            summary.node_count,
            len(summary.edges),
            summary.truncated,
        )

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
                request_id=request_id,
            )

        self._trace_logger.debug(
            "workflow plan: id=%s wf=%s scope=%s targets=%s confidence=%.2f truncated=%s",
            request_id,
            active_workflow_id,
            plan.scope.value,
            plan.target_node_ids[:8],
            plan.confidence,
            summary.truncated,
        )

        if plan.need_clarification:
            model = resolve_model(request.model)
            text = self._workflow.clarification_text(plan, summary)
            self._memory.append_memory(conversation_id, raw_user_message, text)
            if request.stream:
                return StreamingResponse(
                    self._responses.stream_simple_text(text, model=model),
                    headers=self._memory.conversation_headers(
                        conversation_id, generated_conversation_id
                    ),
                    media_type="text/event-stream",
                )
            response_dict = self._responses.simple_chat_response(text, model=model)
            if http_response is not None:
                self._memory.apply_conversation_headers(
                    http_response, conversation_id, generated_conversation_id
                )
            return response_dict

        subgraph = build_subgraph(summary, user_message, plan)
        self._trace_logger.debug(
            "workflow subgraph: id=%s wf=%s nodes=%d truncated=%s reason=%s",
            request_id,
            active_workflow_id,
            len(subgraph.node_ids),
            subgraph.truncated,
            subgraph.reason,
        )
        self._trace_logger.debug(
            "workflow subgraph nodes: id=%s wf=%s nodes=%s",
            request_id,
            active_workflow_id,
            subgraph.node_ids[:12],
        )
        docs_chunks = self._workflow.retrieve_docs_for_subgraph(
            user_message, summary, subgraph, request_id=request_id
        )

        findings = self._workflow.analyze_nodes(
            user_message,
            summary,
            subgraph,
            docs_chunks,
            plan,
            request.model,
            request_id=request_id,
        )
        self._trace_logger.debug(
            "workflow analysis done: id=%s wf=%s findings=%d",
            request_id,
            active_workflow_id,
            len(findings),
        )
        chat_context = self._chat_context_block(messages_for_prompt, user_message)

        def _build_workflow_prompt(current_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
            return build_synthesis_messages(
                question=user_message,
                plan=plan,
                summary=summary,
                subgraph=subgraph,
                findings=findings,
                docs_chunks=current_chunks,
                chat_context=chat_context,
            )

        llm_messages, docs_chunks, prompt_budget = self._apply_prompt_budget_with_docs_fallback(
            docs_chunks=docs_chunks,
            request_id=request_id,
            mode="workflow",
            message_builder=_build_workflow_prompt,
        )
        refs = collect_references(docs_chunks)

        model = resolve_model(request.model)
        params = self._completion_params(request, stream=request.stream)
        self._trace_logger.debug(
            "workflow synthesis prompt stats: id=%s messages=%d chars=%d est_tokens=%d",
            request_id,
            len(llm_messages),
            self._messages_char_count(llm_messages),
            prompt_budget.estimated_tokens_after,
        )
        self._trace_logger.debug(
            "llm request: id=%s model=%s temp=%s max_tokens=%s top_p=%s freq_pen=%s pres_pen=%s stream=%s",
            request_id,
            model,
            params.get("temperature"),
            params.get("max_tokens"),
            params.get("top_p"),
            params.get("frequency_penalty"),
            params.get("presence_penalty"),
            request.stream,
        )

        if request.stream:
            params["stream"] = True
            try:
                stream = chat_completion(llm_messages, model=model, **params)
            except Exception as exc:
                self._logger.exception("lm studio request failed")
                raise HTTPException(status_code=502, detail="LM Studio request failed") from exc

            def _on_stream_complete(assistant_text: str) -> None:
                self._memory.append_memory(conversation_id, raw_user_message, assistant_text)
                self._log_response_summary(request_id, model, assistant_text, refs)

            return StreamingResponse(
                self._responses.stream_with_references(stream, refs, model, _on_stream_complete),
                headers=self._memory.conversation_headers(
                    conversation_id, generated_conversation_id
                ),
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
        self._log_response_summary(request_id, model, assistant_text_raw, refs)

        self._memory.append_memory(conversation_id, raw_user_message, assistant_text_raw)

        response_dict["choices"][0]["message"]["content"] = assistant_text
        response_dict["model"] = model
        if http_response is not None:
            self._memory.apply_conversation_headers(
                http_response, conversation_id, generated_conversation_id
            )
        return response_dict

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

    @staticmethod
    def _resolve_request_id(http_request: Request) -> str:
        """Resolve or generate a request id for logging correlation."""
        for header in ("x-request-id", "x-correlation-id", "x-trace-id"):
            value = http_request.headers.get(header)
            if value:
                return value
        return uuid4().hex[:10]

    @staticmethod
    def _compact_snippet(text: str, max_chars: int = 240) -> str:
        """Compact user/assistant text for logs."""
        compact = " ".join(str(text or "").split())
        if len(compact) > max_chars:
            compact = compact[: max_chars - 3].rstrip() + "..."
        return compact

    @staticmethod
    def _messages_char_count(messages: Iterable[Dict[str, str]]) -> int:
        """Count total content chars in a prompt."""
        total = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
        return total

    def _log_response_summary(
        self,
        request_id: str,
        model: str,
        assistant_text: str,
        refs: List[Dict[str, str]],
    ) -> None:
        """Log LLM response size and snippet for traceability."""
        if not self._trace_logger.isEnabledFor(logging.INFO):
            return
        meta = f"meta: model={model} chars={len(assistant_text)} refs={len(refs)}"
        body = self._trace_text_block("respuesta", assistant_text)
        self._trace_logger.info(
            "TRACE RESPONSE id=%s\n%s\n%s",
            request_id,
            meta,
            body,
        )

    def _trace_request(
        self,
        request_id: str,
        conversation_id: Optional[str],
        workflow_id: Optional[str],
        stream: bool,
        messages_count: int,
        history_count: int,
        user_message: str,
    ) -> None:
        if not self._trace_logger.isEnabledFor(logging.INFO):
            return
        meta = (
            "meta: conv={conv} wf={wf} stream={stream} messages={messages} history={history} user_len={user_len}"
        ).format(
            conv=conversation_id or "-",
            wf=workflow_id or "-",
            stream=stream,
            messages=messages_count,
            history=history_count,
            user_len=len(user_message),
        )
        body = self._trace_text_block("usuario", user_message)
        self._trace_logger.info(
            "TRACE REQUEST id=%s\n%s\n%s",
            request_id,
            meta,
            body,
        )

    def _apply_prompt_budget(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        mode: str,
        warn_on_overflow: bool = True,
    ) -> PromptBudgetResult:
        limit = max(settings.CONVERSATION_MAX_TOKENS, 256)
        result = trim_messages_to_budget(messages, max_tokens=limit)
        if result.changed:
            self._trace_logger.info(
                "prompt budget applied: id=%s mode=%s limit=%d before=%d after=%d dropped=%d truncated=%d",
                request_id,
                mode,
                limit,
                result.estimated_tokens_before,
                result.estimated_tokens_after,
                result.dropped_messages,
                result.truncated_messages,
            )
        if warn_on_overflow and result.estimated_tokens_after > limit:
            self._trace_logger.warning(
                "prompt budget overflow: id=%s mode=%s limit=%d after=%d",
                request_id,
                mode,
                limit,
                result.estimated_tokens_after,
            )
        return result

    def _apply_prompt_budget_with_docs_fallback(
        self,
        docs_chunks: List[Dict[str, Any]],
        request_id: str,
        mode: str,
        message_builder: Callable[[List[Dict[str, Any]]], List[Dict[str, str]]],
    ) -> tuple[List[Dict[str, str]], List[Dict[str, Any]], PromptBudgetResult]:
        """Trim prompt to token budget and shrink docs chunks if still overflowing."""
        limit = max(settings.CONVERSATION_MAX_TOKENS, 256)
        docs_before = len(docs_chunks)
        current_chunks = list(docs_chunks)

        while True:
            llm_messages = message_builder(current_chunks)
            result = self._apply_prompt_budget(
                llm_messages,
                request_id=request_id,
                mode=mode,
                warn_on_overflow=False,
            )
            if result.estimated_tokens_after <= limit or not current_chunks:
                break
            current_chunks = current_chunks[:-1]

        if len(current_chunks) < docs_before:
            self._trace_logger.info(
                "docs fallback applied: id=%s mode=%s docs_before=%d docs_after=%d",
                request_id,
                mode,
                docs_before,
                len(current_chunks),
            )

        if result.estimated_tokens_after > limit:
            self._trace_logger.warning(
                "prompt budget overflow after docs fallback: id=%s mode=%s limit=%d after=%d docs=%d",
                request_id,
                mode,
                limit,
                result.estimated_tokens_after,
                len(current_chunks),
            )

        return result.messages, current_chunks, result

    @staticmethod
    def _trace_text_block(label: str, text: str) -> str:
        raw = str(text or "")
        original_len = len(raw)
        max_chars = settings.TRACE_MAX_TEXT_CHARS
        if max_chars and len(raw) > max_chars:
            raw = raw[:max_chars].rstrip() + f"... [truncado {original_len - max_chars} chars]"
        indented = "\n".join("  " + line for line in raw.splitlines()) if raw else "  (vacio)"
        return f"{label}:\n{indented}"

    @staticmethod
    def _extract_last_user_raw(messages: Iterable[Dict[str, str]]) -> Optional[str]:
        """Get last raw user message (before control cleaning)."""
        for msg in reversed(list(messages)):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"].strip()
        return None

    @staticmethod
    def _count_non_system(messages: Iterable[Dict[str, str]]) -> int:
        """Count non-system messages in a list."""
        return sum(1 for msg in messages if msg.get("role") != "system")
