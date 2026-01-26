from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Protocol


@dataclass
class ConversationInfo:
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int


@dataclass
class ConversationDetail(ConversationInfo):
    messages: List[Dict[str, str]]


class MemoryStore(Protocol):
    def get_recent_messages(self, conversation_id: str, limit: int) -> List[Dict[str, str]]:
        ...

    def append_messages(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        ...

    def list_conversations(self, limit: int, offset: int) -> List[ConversationInfo]:
        ...

    def get_conversation(self, conversation_id: str) -> Optional[ConversationDetail]:
        ...
