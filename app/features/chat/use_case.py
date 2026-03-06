from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .contracts import (
    ChatRuntimeMode,
    ChatUseCaseInput,
    ChatUseCaseResult,
)
from .policy import ChatModePolicy


@dataclass(frozen=True)
class ChatUseCaseHandlers:
    on_missing_user_message: Callable[[ChatUseCaseInput], Any]
    on_docs_mode: Callable[[ChatUseCaseInput], Any]
    on_reasoning_mode: Callable[[ChatUseCaseInput], Any]
    on_workflow_mode: Callable[[ChatUseCaseInput], Any]


class HandleChatUseCase:
    """Application use case for chat mode dispatch."""

    def __init__(self, policy: ChatModePolicy) -> None:
        self._policy = policy

    def execute(self, chat_input: ChatUseCaseInput, handlers: ChatUseCaseHandlers) -> ChatUseCaseResult:
        if not chat_input.user_message:
            payload = handlers.on_missing_user_message(chat_input)
            return ChatUseCaseResult(mode=ChatRuntimeMode.docs, payload=payload)

        policy_result = self._policy.decide(chat_input)
        if policy_result.mode == ChatRuntimeMode.workflow:
            payload = handlers.on_workflow_mode(chat_input)
        elif policy_result.mode == ChatRuntimeMode.reasoning:
            payload = handlers.on_reasoning_mode(chat_input)
        else:
            payload = handlers.on_docs_mode(chat_input)

        return ChatUseCaseResult(
            mode=policy_result.mode,
            payload=payload,
            metadata={"policy_reason": policy_result.reason},
        )
