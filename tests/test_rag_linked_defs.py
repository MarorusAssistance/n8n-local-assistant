from __future__ import annotations

from typing import Any, Dict, List

from app.config import settings
from app.db import METADATA_KEY, ROW_ID_KEY
from app.rag import _extract_doc_page_keys, collect_references, retrieve_context


def test_extract_doc_page_keys_prefers_page_id_over_url() -> None:
    chunks = [
        {
            "url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/",
            "metadata": {
                "page_id": "Docs:Webhook",
                "source_url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/",
            },
        },
        {
            "url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.if/",
            "metadata": {
                "source_url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.if/",
            },
        },
    ]

    keys = _extract_doc_page_keys(chunks, max_docs=6)
    assert keys == ["docs:webhook", "docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.if"]


def test_retrieve_context_appends_linked_definition_chunks(monkeypatch) -> None:
    captured_page_keys: List[str] = []

    monkeypatch.setattr(settings, "ENABLE_HYBRID", False, raising=False)
    monkeypatch.setattr(settings, "ENABLE_RERANK", False, raising=False)
    monkeypatch.setattr(settings, "TOP_K", 2, raising=False)
    monkeypatch.setattr(settings, "LINKED_DEFS_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "LINKED_DEFS_TOP_DOCS", 2, raising=False)

    def fake_embed(_: str) -> List[float]:
        return [0.1, 0.2]

    def fake_query_similar(
        embedding: List[float],
        top_k: int | None = None,
        request_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        _ = embedding, top_k, request_id
        return [
            {
                ROW_ID_KEY: "(0,1)",
                settings.TEXT_COLUMN: "doc 1",
                settings.URL_COLUMN: "https://docs.n8n.io/page-1/",
                settings.TITLE_COLUMN: None,
                settings.SECTION_COLUMN: None,
                METADATA_KEY: {"page_id": "Docs:PageOne"},
            },
            {
                ROW_ID_KEY: "(0,2)",
                settings.TEXT_COLUMN: "doc 2",
                settings.URL_COLUMN: "https://docs.n8n.io/page-2/",
                settings.TITLE_COLUMN: None,
                settings.SECTION_COLUMN: None,
                METADATA_KEY: {"page_id": "Docs:PageTwo"},
            },
        ]

    def fake_linked_chunks(
        page_keys: List[str],
        *,
        request_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        _ = request_id
        captured_page_keys.extend(page_keys)
        return [
            {
                "doc_id": "linked:node:n8n-nodes-base.webhook",
                "text": "Node definition text",
                "url": "",
                "title": "Node: Webhook",
                "section": "linked_by=url_exact (1.00)",
                "context_kind": "linked_def",
            }
        ]

    monkeypatch.setattr("app.rag.create_embedding", fake_embed)
    monkeypatch.setattr("app.rag.query_similar", fake_query_similar)
    monkeypatch.setattr("app.rag.query_related_definition_chunks", fake_linked_chunks)

    chunks = retrieve_context("question", top_k=2, request_id="req-1")
    assert len(chunks) == 3
    assert chunks[-1]["context_kind"] == "linked_def"
    assert captured_page_keys == ["docs:pageone", "docs:pagetwo"]


def test_collect_references_skips_linked_definition_chunks(monkeypatch) -> None:
    monkeypatch.setattr(settings, "APP_SHOW_REFERENCES", True, raising=False)

    chunks = [
        {"text": "Main doc text", "url": "https://docs.n8n.io/a", "title": "A", "section": ""},
        {
            "text": "Definition text",
            "url": "https://docs.n8n.io/defs",
            "title": "Node def",
            "section": "",
            "context_kind": "linked_def",
        },
    ]
    refs = collect_references(chunks, max_refs=8, max_chars=120)
    assert len(refs) == 1
    assert refs[0]["url"] == "https://docs.n8n.io/a"
