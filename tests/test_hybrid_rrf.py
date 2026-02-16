from __future__ import annotations

from app.hybrid import rrf_fuse


def test_rrf_fuse_deduplicates_and_ranks_with_weights() -> None:
    vector_list = [
        {"doc_id": "A", "url": "/a", "text": "A"},
        {"doc_id": "B", "url": "/b", "text": "B"},
    ]
    fts_list = [
        {"doc_id": "B", "url": "/b", "text": "B from fts"},
        {"doc_id": "C", "url": "/c", "text": "C"},
    ]

    merged = rrf_fuse(
        vector_list,
        fts_list,
        rrf_k=60,
        vector_weight=1.0,
        fts_weight=0.6,
    )

    assert [item["doc_id"] for item in merged] == ["B", "A", "C"]
    assert merged[0]["retrieval_sources"] == ["fts", "vector"]
    assert merged[0]["vector_rank"] == 2
    assert merged[0]["fts_rank"] == 1


def test_rrf_fuse_applies_top_m_after_merge() -> None:
    vector_list = [
        {"doc_id": "v1", "url": "/v1", "text": "V1"},
        {"doc_id": "shared", "url": "/shared", "text": "Shared"},
        {"doc_id": "v3", "url": "/v3", "text": "V3"},
    ]
    fts_list = [
        {"doc_id": "shared", "url": "/shared", "title": "Shared title"},
        {"doc_id": "f2", "url": "/f2", "text": "F2"},
    ]

    merged = rrf_fuse(
        vector_list,
        fts_list,
        rrf_k=60,
        vector_weight=1.0,
        fts_weight=0.8,
        top_m=2,
    )

    assert len(merged) == 2
    assert merged[0]["doc_id"] == "shared"
    assert merged[0]["title"] == "Shared title"
