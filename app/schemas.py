from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[Any] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    user: Optional[str] = None
    conversation_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")
