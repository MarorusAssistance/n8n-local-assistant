from __future__ import annotations

from app.features.chat import (
    ChatModePolicy,
    ChatRuntimeMode,
    ChatUseCaseInput,
    HandleChatUseCase,
)
from app.features.chat.use_case import ChatUseCaseHandlers
from app.schemas import ChatCompletionRequest


def _input(user_message: str | None, active_workflow_id: str | None) -> ChatUseCaseInput:
    return ChatUseCaseInput(
        request=ChatCompletionRequest(messages=[{"role": "user", "content": "hola"}]),  # type: ignore[arg-type]
        request_id="req-1",
        user_message=user_message,
        active_workflow_id=active_workflow_id,
        messages_for_prompt=[{"role": "user", "content": user_message or ""}],
        raw_user_message=user_message,
        conversation_id="conv-1",
        generated_conversation_id=False,
        history_count=0,
        messages_count=1,
    )


def test_use_case_routes_to_missing_user_handler() -> None:
    use_case = HandleChatUseCase(ChatModePolicy())
    calls: list[str] = []

    handlers = ChatUseCaseHandlers(
        on_missing_user_message=lambda _: calls.append("missing") or "missing",
        on_docs_mode=lambda _: calls.append("docs") or "docs",
        on_reasoning_mode=lambda _: calls.append("reasoning") or "reasoning",
        on_workflow_mode=lambda _: calls.append("workflow") or "workflow",
    )
    result = use_case.execute(_input(None, None), handlers)
    assert result.mode == ChatRuntimeMode.docs
    assert result.payload == "missing"
    assert calls == ["missing"]


def test_use_case_routes_to_workflow_when_active_workflow() -> None:
    use_case = HandleChatUseCase(ChatModePolicy())
    calls: list[str] = []

    handlers = ChatUseCaseHandlers(
        on_missing_user_message=lambda _: calls.append("missing") or "missing",
        on_docs_mode=lambda _: calls.append("docs") or "docs",
        on_reasoning_mode=lambda _: calls.append("reasoning") or "reasoning",
        on_workflow_mode=lambda _: calls.append("workflow") or "workflow",
    )
    result = use_case.execute(_input("analiza flujo", "wf-123"), handlers)
    assert result.mode == ChatRuntimeMode.workflow
    assert result.payload == "workflow"
    assert calls == ["workflow"]
