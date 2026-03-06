from __future__ import annotations

from app.config import settings
from app.features.chat import ChatModePolicy, ChatRuntimeMode, ChatUseCaseInput
from app.schemas import ChatCompletionRequest


def _chat_input(*, user_message: str | None, active_workflow_id: str | None) -> ChatUseCaseInput:
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


def test_chat_mode_policy_selects_workflow_when_active_workflow_exists(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "legacy", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", False, raising=False)
    policy = ChatModePolicy()
    result = policy.decide(_chat_input(user_message="revisa workflow", active_workflow_id="wf-1"))
    assert result.mode == ChatRuntimeMode.workflow


def test_chat_mode_policy_selects_reasoning_when_flag_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "legacy", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", False, raising=False)
    policy = ChatModePolicy()
    result = policy.decide(_chat_input(user_message="crea workflow", active_workflow_id=None))
    assert result.mode == ChatRuntimeMode.reasoning


def test_chat_mode_policy_selects_docs_by_default(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REASONING_PIPELINE_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "AGENT_RUNTIME", "legacy", raising=False)
    monkeypatch.setattr(settings, "LANGGRAPH_REASONING_ENABLED", False, raising=False)
    policy = ChatModePolicy()
    result = policy.decide(_chat_input(user_message="como configurar webhook", active_workflow_id=None))
    assert result.mode == ChatRuntimeMode.docs
