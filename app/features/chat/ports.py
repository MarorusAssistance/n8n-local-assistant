from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union

from ..reasoning.multi_agent_contracts import MultiAgentGraphResult
from ...reasoning.types import ReasoningPipelineResult


class LLMPort(Protocol):
    def resolve_model(self, requested: Optional[str]) -> str:
        ...

    def chat_completion(self, messages: List[Dict[str, str]], model: str, **kwargs: Any) -> Any:
        ...


class RetrievalPort(Protocol):
    def retrieve(self, question: str, *, request_id: Optional[str]) -> List[Dict[str, Any]]:
        ...


class ReasoningPort(Protocol):
    def run(
        self,
        *,
        user_prompt: str,
        model: Optional[str],
        request_id: Optional[str],
        existing_workflow: Any,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> Union[ReasoningPipelineResult, MultiAgentGraphResult]:
        ...


class WorkflowPort(Protocol):
    def run(
        self,
        *,
        question: str,
        workflow_id: str,
        request_model: Optional[str],
        request_id: Optional[str],
        chat_context: str,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...


class MemoryPort(Protocol):
    def append_memory(
        self,
        conversation_id: Optional[str],
        raw_user_message: Optional[str],
        assistant_text: str,
    ) -> None:
        ...
