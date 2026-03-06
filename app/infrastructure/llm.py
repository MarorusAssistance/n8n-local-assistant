from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..llm import chat_completion, create_embedding, resolve_model


def resolve_runtime_model(requested: Optional[str]) -> str:
    return resolve_model(requested)


def complete_chat(messages: List[Dict[str, str]], *, model: str, **kwargs: Any) -> Any:
    return chat_completion(messages, model=model, **kwargs)


def create_text_embedding(text: str) -> List[float]:
    return create_embedding(text)
