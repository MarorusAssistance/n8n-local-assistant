from __future__ import annotations

from app import reranker as reranker_mod


def test_reranker_skips_when_model_not_cached(monkeypatch) -> None:
    instance = reranker_mod.Reranker()

    monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANK", True)
    monkeypatch.setattr(reranker_mod.settings, "RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setattr(reranker_mod.settings, "RERANK_FALLBACK_MODEL", "BAAI/bge-reranker-v2.5-gemma2-lightweight")
    monkeypatch.setattr(reranker_mod, "_reranker_model_available_locally", lambda model_name: False)

    def _unexpected_load(*args, **kwargs):
        raise AssertionError("reranker should not try to download or load an uncached model")

    monkeypatch.setattr(instance, "_load_model", _unexpected_load)

    assert instance.load() is False
