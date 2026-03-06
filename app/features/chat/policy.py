from __future__ import annotations

from ...config import settings
from .contracts import ChatModePolicyResult, ChatRuntimeMode, ChatUseCaseInput


class ChatModePolicy:
    """Single place to decide runtime mode for chat requests."""

    def decide(self, chat_input: ChatUseCaseInput) -> ChatModePolicyResult:
        if chat_input.active_workflow_id:
            return ChatModePolicyResult(
                mode=ChatRuntimeMode.workflow,
                reason="active_workflow_id_present",
            )
        if settings.AGENT_RUNTIME == "langgraph" and settings.LANGGRAPH_REASONING_ENABLED:
            return ChatModePolicyResult(
                mode=ChatRuntimeMode.reasoning,
                reason="langgraph_reasoning_runtime_enabled",
            )
        if settings.REASONING_PIPELINE_ENABLED:
            return ChatModePolicyResult(
                mode=ChatRuntimeMode.reasoning,
                reason="reasoning_pipeline_enabled",
            )
        return ChatModePolicyResult(
            mode=ChatRuntimeMode.docs,
            reason="default_docs_mode",
        )
