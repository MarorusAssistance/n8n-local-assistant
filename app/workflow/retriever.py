from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..config import settings
from ..rag import retrieve_context


def retrieve_docs(
    question: str,
    node_types: Optional[Iterable[str]] = None,
    node_names: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve docs for a question; node hints are reserved for future boosting."""
    # RAG is intentionally kept simple for now: no hints and no reranking.
    # We keep the signature so we can add boosting/reranking later.
    _ = node_types, node_names
    return retrieve_context(question, top_k=settings.WORKFLOW_DOCS_TOP_K)
