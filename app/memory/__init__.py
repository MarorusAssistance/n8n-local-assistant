from __future__ import annotations

import logging
from typing import List, Optional

from .base import ConversationDetail, ConversationInfo, MemoryStore
from .in_memory import InMemoryStore
from .postgres import PostgresStore
from ..config import settings


logger = logging.getLogger("n8n-assistant")


class ResilientStore(MemoryStore):
    def __init__(self, primary: MemoryStore, fallback: MemoryStore) -> None:
        self._primary = primary
        self._fallback = fallback

    def get_recent_messages(self, conversation_id: str, limit: int) -> List[dict[str, str]]:
        try:
            return self._primary.get_recent_messages(conversation_id, limit)
        except Exception:
            logger.exception("memory primary read failed, falling back")
            return self._fallback.get_recent_messages(conversation_id, limit)

    def append_messages(self, conversation_id: str, messages: List[dict[str, str]]) -> None:
        try:
            self._primary.append_messages(conversation_id, messages)
            return
        except Exception:
            logger.exception("memory primary write failed, falling back")
            self._fallback.append_messages(conversation_id, messages)

    def list_conversations(self, limit: int, offset: int) -> List[ConversationInfo]:
        try:
            return self._primary.list_conversations(limit, offset)
        except Exception:
            logger.exception("memory primary list failed, falling back")
            return self._fallback.list_conversations(limit, offset)

    def get_conversation(self, conversation_id: str) -> Optional[ConversationDetail]:
        try:
            return self._primary.get_conversation(conversation_id)
        except Exception:
            logger.exception("memory primary get failed, falling back")
            return self._fallback.get_conversation(conversation_id)


def get_memory_store() -> MemoryStore:
    backend = settings.MEMORY_BACKEND.lower()
    if backend == "postgres":
        primary = PostgresStore(
            settings.DATABASE_URL,
            settings.MEMORY_CONVERSATIONS_TABLE,
            settings.MEMORY_MESSAGES_TABLE,
        )
        fallback = InMemoryStore(settings.MEMORY_TTL_SECONDS)
        return ResilientStore(primary, fallback)
    return InMemoryStore(settings.MEMORY_TTL_SECONDS)


__all__ = [
    "ConversationDetail",
    "ConversationInfo",
    "MemoryStore",
    "get_memory_store",
]
