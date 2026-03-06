from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ...schemas import ChatCompletionRequest


class ChatRuntimeMode(str, Enum):
    docs = "docs"
    reasoning = "reasoning"
    workflow = "workflow"


@dataclass(frozen=True)
class ChatModePolicyResult:
    mode: ChatRuntimeMode
    reason: str


@dataclass(frozen=True)
class ChatUseCaseInput:
    request: ChatCompletionRequest
    request_id: str
    user_message: Optional[str]
    active_workflow_id: Optional[str]
    messages_for_prompt: List[Dict[str, str]]
    raw_user_message: Optional[str]
    conversation_id: Optional[str]
    generated_conversation_id: bool
    history_count: int
    messages_count: int


@dataclass
class ChatUseCaseResult:
    mode: ChatRuntimeMode
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
