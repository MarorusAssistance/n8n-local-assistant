from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import hashlib


def rrf_fuse(
    vector_list: Iterable[Dict[str, Any]],
    fts_list: Iterable[Dict[str, Any]],
    *,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    fts_weight: float = 0.6,
    top_m: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fuse ranked lists with Reciprocal Rank Fusion (RRF)."""
    fused: Dict[str, Dict[str, Any]] = {}

    _accumulate_rrf(
        fused,
        vector_list,
        source="vector",
        weight=float(vector_weight),
        rrf_k=max(1, int(rrf_k)),
    )
    _accumulate_rrf(
        fused,
        fts_list,
        source="fts",
        weight=float(fts_weight),
        rrf_k=max(1, int(rrf_k)),
    )

    merged: List[Dict[str, Any]] = []
    for candidate in fused.values():
        sources = candidate.pop("_sources", set())
        candidate["retrieval_sources"] = sorted(str(item) for item in sources)
        merged.append(candidate)

    merged.sort(
        key=lambda item: (
            -float(item.get("rrf_score") or 0.0),
            _best_rank(item),
            str(item.get("doc_id") or ""),
        )
    )

    if top_m is not None and top_m > 0:
        return merged[:top_m]
    return merged


def _accumulate_rrf(
    fused: Dict[str, Dict[str, Any]],
    ranked_list: Iterable[Dict[str, Any]],
    *,
    source: str,
    weight: float,
    rrf_k: int,
) -> None:
    if weight <= 0:
        return

    for rank, raw_candidate in enumerate(ranked_list, start=1):
        candidate = dict(raw_candidate)
        doc_id = _resolve_doc_id(candidate)
        score = weight / float(rrf_k + rank)

        if doc_id not in fused:
            candidate["doc_id"] = doc_id
            candidate["rrf_score"] = 0.0
            candidate["vector_rank"] = None
            candidate["fts_rank"] = None
            candidate["_sources"] = set()
            fused[doc_id] = candidate

        merged = fused[doc_id]
        merged["rrf_score"] = float(merged.get("rrf_score") or 0.0) + score
        merged["_sources"].add(source)

        rank_key = "vector_rank" if source == "vector" else "fts_rank"
        if merged.get(rank_key) is None:
            merged[rank_key] = rank

        _fill_empty_fields(merged, candidate)


def _resolve_doc_id(candidate: Dict[str, Any]) -> str:
    if candidate.get("doc_id"):
        return str(candidate["doc_id"])

    raw = "|".join(
        [
            str(candidate.get("url") or ""),
            str(candidate.get("title") or ""),
            str(candidate.get("section") or ""),
            str(candidate.get("text") or ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"fp:{digest}"


def _fill_empty_fields(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in ("text", "url", "title", "section"):
        if not target.get(key) and source.get(key):
            target[key] = source[key]



def _best_rank(candidate: Dict[str, Any]) -> int:
    ranks = [candidate.get("vector_rank"), candidate.get("fts_rank")]
    valid = [int(rank) for rank in ranks if isinstance(rank, int)]
    if not valid:
        return 10**9
    return min(valid)
