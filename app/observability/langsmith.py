from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..config import settings


def configure_langsmith_environment() -> None:
    """Set LangSmith environment variables from settings when not explicitly provided."""
    os.environ.setdefault("LANGSMITH_TRACING", "true" if settings.LANGSMITH_TRACING else "false")
    if settings.LANGSMITH_PROJECT:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)


def build_langsmith_metadata(
    request_id: str,
    mode: str,
    conversation_id: Optional[str],
    workflow_id: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    """Build normalized metadata for traces and graph runs."""
    return {
        "request_id": request_id,
        "mode": mode,
        "conversation_id": conversation_id or "-",
        "workflow_id": workflow_id or "-",
        "model": model or "-",
    }


def build_graph_run_config(
    *,
    request_id: str,
    mode: str,
    conversation_id: Optional[str],
    workflow_id: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    """Create a LangGraph invoke config enriched with metadata and recursion limits."""
    thread_id = conversation_id or f"req:{request_id}"
    metadata = build_langsmith_metadata(
        request_id=request_id,
        mode=mode,
        conversation_id=conversation_id,
        workflow_id=workflow_id,
        model=model,
    )
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": metadata,
        "tags": ["n8n-assistant", mode, settings.AGENT_RUNTIME],
        "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT,
    }
