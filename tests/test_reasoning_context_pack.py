from __future__ import annotations

from app.config import settings
from app.reasoning.context_pack import build_context_pack
from app.reasoning.types import RouterConstraints, RouterOutput


def _router_output() -> RouterOutput:
    return RouterOutput(
        intent="create",
        goal="demo",
        constraints=RouterConstraints(),
        missing_info=[],
        complexity_score=1,
    )


def test_context_pack_respects_limits_and_token_budget(monkeypatch) -> None:
    monkeypatch.setattr(settings, "MAX_NODE_CARDS", 2, raising=False)
    monkeypatch.setattr(settings, "MAX_DOC_CHUNKS", 2, raising=False)
    monkeypatch.setattr(settings, "MAX_CONTEXT_TOKENS", 220, raising=False)

    def fake_retrieve_context(*args, **kwargs):
        _ = args, kwargs
        return [
            {
                "doc_id": "node-1",
                "text": (
                    "Kind: NODE_OVERVIEW\nNode Type: n8n-nodes-base.webhook\n"
                    "Display Name: Webhook\nRequired Credentials: -\n"
                ),
                "title": "Webhook",
                "metadata": {
                    "kind": "NODE_OVERVIEW",
                    "nodeType": "n8n-nodes-base.webhook",
                    "displayName": "Webhook",
                    "credentialTypes_required": [],
                },
            },
            {
                "doc_id": "node-2",
                "text": (
                    "Kind: NODE_OVERVIEW\nNode Type: n8n-nodes-base.googleSheets\n"
                    "Display Name: Google Sheets\nRequired Credentials: googleSheetsOAuth2Api\n"
                ),
                "title": "Google Sheets",
                "metadata": {
                    "kind": "NODE_OVERVIEW",
                    "nodeType": "n8n-nodes-base.googleSheets",
                    "displayName": "Google Sheets",
                    "credentialTypes_required": ["googleSheetsOAuth2Api"],
                },
            },
            {
                "doc_id": "doc-1",
                "text": "A" * 2000,
                "title": "Docs 1",
                "metadata": {"source": "n8n-docs"},
            },
            {
                "doc_id": "doc-2",
                "text": "B" * 1600,
                "title": "Docs 2",
                "metadata": {"source": "n8n-docs"},
            },
        ]

    monkeypatch.setattr("app.reasoning.context_pack.retrieve_context", fake_retrieve_context)

    pack = build_context_pack(
        router_output=_router_output(),
        user_prompt="Crea workflow webhook -> sheets",
        request_id="req-context-pack",
    )

    assert len(pack.nodeCards) <= settings.MAX_NODE_CARDS
    assert len(pack.docChunks) <= settings.MAX_DOC_CHUNKS
    assert pack.budget.estimatedTokens <= settings.MAX_CONTEXT_TOKENS
