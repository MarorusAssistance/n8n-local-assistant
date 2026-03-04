from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..rag import retrieve_context

try:  # Optional until LangChain deps are installed.
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except Exception:  # pragma: no cover - optional dependency fallback
    Document = None  # type: ignore[assignment]

    class BaseRetriever:  # type: ignore[override]
        pass


class RetrieveContextRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapper over the existing hybrid retrieval pipeline."""

    top_k: Optional[int] = None
    request_id: Optional[str] = None

    def _get_relevant_documents(self, query: str, *args: Any, **kwargs: Any) -> List[Any]:
        chunks = retrieve_context(query, top_k=self.top_k, request_id=self.request_id)
        docs: List[Any] = []
        for chunk in chunks:
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            metadata: Dict[str, Any] = {
                "doc_id": chunk.get("doc_id"),
                "url": chunk.get("url"),
                "title": chunk.get("title"),
                "section": chunk.get("section"),
                "source": chunk.get("source"),
                "context_kind": chunk.get("context_kind"),
                "linked_def_type": chunk.get("linked_def_type"),
            }
            if Document is None:
                docs.append({"page_content": text, "metadata": metadata})
            else:
                docs.append(Document(page_content=text, metadata=metadata))
        return docs
