from __future__ import annotations

from app.config import settings
from app.db import build_fts_queries


def test_build_fts_queries_extracts_keywords_and_removes_noise(monkeypatch) -> None:
    monkeypatch.setattr(settings, "FTS_KEYWORDS_MAX_TERMS", 8, raising=False)
    monkeypatch.setattr(settings, "FTS_KEYWORDS_MIN_TERM_LEN", 3, raising=False)
    monkeypatch.setattr(settings, "FTS_QUERY_MAX_CHARS", 220, raising=False)

    text = (
        "Quiero implementar en n8n un workflow con Webhook y Airtable para no perder leads "
        "y enviar notificacion interna"
    )
    payload = build_fts_queries(text)

    assert payload["strict_query"]
    assert len(payload["terms"]) <= 8
    assert "quiero" not in payload["terms"]
    assert "n8n" in payload["terms"]
    assert "webhook" in payload["terms"]
    assert payload["relaxed_query"]


def test_build_fts_queries_deduplicates_and_limits_output(monkeypatch) -> None:
    monkeypatch.setattr(settings, "FTS_KEYWORDS_MAX_TERMS", 3, raising=False)
    monkeypatch.setattr(settings, "FTS_KEYWORDS_MIN_TERM_LEN", 3, raising=False)
    monkeypatch.setattr(settings, "FTS_QUERY_MAX_CHARS", 30, raising=False)

    payload = build_fts_queries("webhook webhook formulario formulario crm")

    assert payload["terms"] == ["webhook", "formulario", "crm"]
    assert payload["strict_query"] == "webhook formulario crm"
    assert payload["relaxed_query"] == "webhook | formulario | crm"
    assert len(payload["strict_query"]) <= 30


def test_build_fts_queries_single_term_disables_relaxed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "FTS_KEYWORDS_MAX_TERMS", 5, raising=False)
    monkeypatch.setattr(settings, "FTS_KEYWORDS_MIN_TERM_LEN", 3, raising=False)
    monkeypatch.setattr(settings, "FTS_QUERY_MAX_CHARS", 100, raising=False)

    payload = build_fts_queries("webhook")

    assert payload["strict_query"] == "webhook"
    assert payload["relaxed_query"] == ""
