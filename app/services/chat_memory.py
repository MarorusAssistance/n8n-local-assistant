from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import HTTPException, Request, Response

from ..config import settings
from ..memory import MemoryStore
from ..schemas import ChatCompletionRequest


class ConversationManager:
    """Encapsulate memory, conversation id, and header handling."""

    _auto_conversations: Dict[str, tuple[str, float]] = {}
    _dedup_cache: Dict[str, tuple[str, str, float]] = {}

    def __init__(self, memory_store: MemoryStore, logger: logging.Logger) -> None:
        self._memory_store = memory_store
        self._logger = logger

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

    def get_recent_messages(self, conversation_id: str, limit: int) -> List[Dict[str, str]]:
        """Return recent messages for a conversation."""
        return self._memory_store.get_recent_messages(conversation_id, limit)

    def resolve_conversation_id(
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

    def get_or_create_auto_conversation_id(self, http_request: Request) -> tuple[str, bool]:
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

    def remember_auto_conversation(self, http_request: Request, conversation_id: str) -> None:
        """Persist an explicit conversation id for future auto mapping."""
        key = self._client_key(http_request)
        self._auto_conversations[key] = (conversation_id, time.time())

    def append_memory(
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
    def conversation_headers(conversation_id: Optional[str], generated: bool) -> Dict[str, str]:
        """Build response headers for conversation tracking."""
        headers: Dict[str, str] = {}
        if conversation_id:
            headers["X-Conversation-Id"] = conversation_id
            if generated:
                headers["X-Conversation-Id-Generated"] = "true"
        return headers

    @staticmethod
    def apply_conversation_headers(
        response: Response, conversation_id: Optional[str], generated: bool
    ) -> None:
        """Apply conversation headers to an HTTP response."""
        if not conversation_id:
            return
        response.headers["X-Conversation-Id"] = conversation_id
        if generated:
            response.headers["X-Conversation-Id-Generated"] = "true"

    def _client_key(self, http_request: Request) -> str:
        """Build a stable client key from IP + user agent."""
        host = http_request.client.host if http_request.client else "unknown"
        user_agent = http_request.headers.get("user-agent", "").strip().lower()
        if user_agent:
            return f"{host}|{user_agent}"
        return host

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
