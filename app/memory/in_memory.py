from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .base import ConversationDetail, ConversationInfo, MemoryStore
from .utils import derive_title, filter_messages


@dataclass
class ConversationState:
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    title: Optional[str] = None


class InMemoryStore(MemoryStore):
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = max(ttl_seconds, 0)
        self._store: Dict[str, ConversationState] = {}

    def _is_expired(self, state: ConversationState) -> bool:
        if self._ttl_seconds <= 0:
            return False
        age = datetime.now(timezone.utc) - state.updated_at
        return age.total_seconds() > self._ttl_seconds

    def _purge_if_expired(self, conversation_id: str) -> bool:
        state = self._store.get(conversation_id)
        if not state:
            return True
        if self._is_expired(state):
            self._store.pop(conversation_id, None)
            return True
        return False

    def get_recent_messages(self, conversation_id: str, limit: int) -> List[Dict[str, str]]:
        if self._purge_if_expired(conversation_id):
            return []
        state = self._store[conversation_id]
        if limit <= 0:
            return list(state.messages)
        return state.messages[-limit:]

    def append_messages(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        filtered = filter_messages(messages)
        if not filtered:
            return

        state = self._store.get(conversation_id)
        if not state:
            state = ConversationState()
            state.title = derive_title(filtered)
            self._store[conversation_id] = state
        elif not state.title:
            state.title = derive_title(filtered)

        state.messages.extend(filtered)
        state.updated_at = datetime.now(timezone.utc)

    def list_conversations(self, limit: int, offset: int) -> List[ConversationInfo]:
        expired_keys = [
            key for key, state in self._store.items() if self._is_expired(state)
        ]
        for key in expired_keys:
            self._store.pop(key, None)

        items = [
            ConversationInfo(
                conversation_id=key,
                title=state.title,
                created_at=state.created_at,
                updated_at=state.updated_at,
                message_count=len(state.messages),
            )
            for key, state in self._store.items()
        ]
        items.sort(key=lambda item: item.updated_at, reverse=True)
        if offset < 0:
            offset = 0
        if limit <= 0:
            return items[offset:]
        return items[offset : offset + limit]

    def get_conversation(self, conversation_id: str) -> Optional[ConversationDetail]:
        if self._purge_if_expired(conversation_id):
            return None
        state = self._store[conversation_id]
        return ConversationDetail(
            conversation_id=conversation_id,
            title=state.title,
            created_at=state.created_at,
            updated_at=state.updated_at,
            message_count=len(state.messages),
            messages=list(state.messages),
        )
