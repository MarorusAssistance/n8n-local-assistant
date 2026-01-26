from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Iterable, Optional
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
from ..schemas import ChatCompletionRequest


class ChatService:
    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store
        self._logger = logging.getLogger("n8n-assistant")

    def health(self) -> Dict[str, Any]:
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

    def create_chat_completions(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
        http_response: Optional[Response] = None,
    ) -> Any:
        messages = normalize_messages([msg.model_dump() for msg in request.messages])
        user_message = extract_last_user_message(messages)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        self._logger.info(
            "chat_completion request: messages=%d user_len=%d",
            len(messages),
            len(user_message),
        )

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

        conversation_id = None
        generated_conversation_id = False
        history: list[Dict[str, str]] = []
        if settings.MEMORY_ENABLED:
            conversation_id = self._get_conversation_id(request, http_request)
            if not conversation_id and settings.MEMORY_AUTO_CREATE_CONVERSATION_ID:
                conversation_id = f"auto:{uuid4()}"
                generated_conversation_id = True
                self._logger.info("memory: generated conversation id %s", conversation_id)
            if conversation_id and self._count_non_system(messages) <= 1:
                history = self._memory_store.get_recent_messages(
                    conversation_id, settings.MEMORY_MAX_MESSAGES
                )

        context_block = build_context_block(chunks)
        system_prompt = build_system_prompt(context_block)
        llm_messages = [{"role": "system", "content": system_prompt}]
        if history:
            llm_messages.extend(history)
        llm_messages.extend(messages)

        model = resolve_model(request.model)

        params: Dict[str, Any] = {}
        params["temperature"] = request.temperature if request.temperature is not None else 0.2
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            params["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            params["stop"] = request.stop

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
                if not (settings.MEMORY_ENABLED and conversation_id):
                    return
                new_messages: list[Dict[str, str]] = []
                if user_message:
                    new_messages.append({"role": "user", "content": user_message})
                if assistant_text:
                    new_messages.append({"role": "assistant", "content": assistant_text})
                if new_messages:
                    self._memory_store.append_messages(conversation_id, new_messages)

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

        if settings.MEMORY_ENABLED and conversation_id:
            new_messages: list[Dict[str, str]] = []
            if user_message:
                new_messages.append({"role": "user", "content": user_message})
            if assistant_text_raw:
                new_messages.append({"role": "assistant", "content": assistant_text_raw})
            if new_messages:
                self._memory_store.append_messages(conversation_id, new_messages)

        response_dict["choices"][0]["message"]["content"] = assistant_text
        response_dict["model"] = model
        if http_response is not None:
            self._apply_conversation_headers(
                http_response, conversation_id, generated_conversation_id
            )
        return response_dict

    def _get_conversation_id(
        self, payload: ChatCompletionRequest, http_request: Request
    ) -> Optional[str]:
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
    def _conversation_headers(
        conversation_id: Optional[str], generated: bool
    ) -> Dict[str, str]:
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
        if not conversation_id:
            return
        response.headers["X-Conversation-Id"] = conversation_id
        if generated:
            response.headers["X-Conversation-Id-Generated"] = "true"

    @staticmethod
    def _count_non_system(messages: Iterable[Dict[str, str]]) -> int:
        return sum(1 for msg in messages if msg.get("role") != "system")

    @staticmethod
    def _format_sse(data: Dict[str, Any]) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def _stream_with_references(
        self,
        stream: Iterable[Any],
        refs: list[dict[str, str]],
        fallback_model: str,
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> Iterable[str]:
        last_id: str | None = None
        last_created = int(time.time())
        last_model = fallback_model
        finish_reason: str | None = None
        assistant_parts: list[str] = []

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
