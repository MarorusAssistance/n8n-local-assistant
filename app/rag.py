from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import logging
import re
import time

from .config import settings
from .db import (
    FTS_SCORE_KEY,
    METADATA_KEY,
    ROW_ID_KEY,
    query_fts,
    query_related_definition_chunks,
    query_similar,
)
from .doc_links import derive_doc_page_key
from .hybrid import rrf_fuse
from .llm import create_embedding
from .reranker import reranker


SYSTEM_PROMPT = (
    "Eres un asistente experto en n8n. Responde con pasos accionables en n8n, "
    "menciona nombres de nodos y campos, sugiere checks y posibles causas. "
    "No inventes valores, endpoints, nombres de campos o configuraciones. "
    "Si falta informacion o no estas seguro, dilo claramente y pide el dato minimo necesario. "
    "No asumas configuraciones ocultas ni llenes huecos con datos ficticios. "
    "No incluyas referencias; el backend las agregara al final."
)

FORMAT_PROMPT = (
    "Formato de salida:\n"
    "Diagnostico:\n"
    "Pasos:\n"
    "Alternativas:\n"
)

trace_logger = logging.getLogger("n8n-assistant.trace")


def _trace_truncate(text: str, max_chars: int) -> str:
    raw = str(text or "")
    if max_chars and len(raw) > max_chars:
        return raw[:max_chars].rstrip() + f"... [truncado {len(raw) - max_chars} chars]"
    return raw


def _trace_chunks(
    chunks: List[Dict[str, Any]],
    include_rerank_scores: bool = False,
    include_rrf_scores: bool = False,
) -> str:
    if not chunks:
        return "  (sin resultados)"
    max_chunks = settings.TRACE_MAX_CHUNKS or len(chunks)
    lines: List[str] = []
    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        title = _compact_header_value(chunk.get("title"), max_chars=120)
        section = _compact_header_value(chunk.get("section"), max_chars=80)
        url = _compact_header_value(chunk.get("url"), max_chars=140)
        snippet = _compact_snippet(
            chunk.get("text") or "", max_chars=settings.TRACE_MAX_CHUNK_CHARS
        )
        header_parts = []
        if include_rerank_scores and chunk.get("rerank_score") is not None:
            header_parts.append(f"rerank={float(chunk['rerank_score']):.4f}")
        if include_rrf_scores and chunk.get("rrf_score") is not None:
            header_parts.append(f"rrf={float(chunk['rrf_score']):.4f}")
        if chunk.get("doc_id"):
            header_parts.append(f"id={chunk['doc_id']}")
        if title:
            header_parts.append(f"title={title}")
        if section:
            header_parts.append(f"section={section}")
        if url:
            header_parts.append(f"url={url}")
        header = " | ".join(header_parts) if header_parts else "chunk"
        lines.append(f"[{idx}] {header}")
        lines.append(f"  {snippet}")
    if len(chunks) > max_chunks:
        lines.append(f"... {len(chunks) - max_chunks} mas")
    return "\n".join(lines)


def _trace_retrieval_stage(stage: str, chunks: List[Dict[str, Any]], max_items: int = 10) -> str:
    if not chunks:
        return f"{stage}: (sin resultados)"

    lines = [f"{stage}: count={len(chunks)}"]
    for idx, chunk in enumerate(chunks[:max_items], start=1):
        ref = _candidate_reference(chunk)
        parts = [f"rank={idx}", f"id={chunk.get('doc_id') or '-'}", f"ref={ref}"]
        if chunk.get("vector_rank") is not None:
            parts.append(f"vec_rank={chunk['vector_rank']}")
        if chunk.get("fts_rank") is not None:
            parts.append(f"fts_rank={chunk['fts_rank']}")
        if chunk.get("rrf_score") is not None:
            parts.append(f"rrf={float(chunk['rrf_score']):.4f}")
        if chunk.get("retrieval_sources"):
            parts.append(f"sources={','.join(chunk['retrieval_sources'])}")
        if chunk.get("fts_score") is not None:
            parts.append(f"fts_score={float(chunk['fts_score']):.4f}")
        lines.append("  - " + " | ".join(parts))
    return "\n".join(lines)


def _candidate_reference(chunk: Dict[str, Any]) -> str:
    for key in ("url", "title", "section"):
        value = (chunk.get(key) or "").strip() if isinstance(chunk.get(key), str) else chunk.get(key)
        if value:
            return str(value)
    return str(chunk.get("doc_id") or "-")


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        content = _normalize_content(msg.get("content"))
        if role and content is not None:
            normalized.append({"role": role, "content": content})
    return normalized


def extract_last_user_message(messages: List[Dict[str, str]]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return None


def retrieve_context(
    question: str,
    top_k: Optional[int] = None,
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    final_top_k = _resolve_top_k(top_k)
    rerank_pool_size = _resolve_pool_size(final_top_k)
    vector_limit = _resolve_vector_limit(rerank_pool_size)
    fts_limit = _resolve_fts_limit(rerank_pool_size)
    rrf_top_m = _resolve_rrf_top_m(rerank_pool_size)

    if trace_logger.isEnabledFor(logging.DEBUG):
        trace_logger.debug(
            "rag query text: id=%s text=%s",
            request_id or "-",
            _compact_snippet(question, max_chars=220),
        )

    embed_start = time.perf_counter()
    embedding = create_embedding(question)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    vector_start = time.perf_counter()
    vector_rows = query_similar(embedding, top_k=vector_limit, request_id=request_id)
    vector_ms = (time.perf_counter() - vector_start) * 1000
    vector_list = [_row_to_candidate(row) for row in vector_rows]

    fts_list: List[Dict[str, Any]] = []
    hybrid_warning = ""
    retrieval_pool = list(vector_list)

    if settings.ENABLE_HYBRID:
        try:
            fts_start = time.perf_counter()
            fts_rows = query_fts(question, top_k=fts_limit, request_id=request_id)
            fts_ms = (time.perf_counter() - fts_start) * 1000
            fts_list = [_row_to_candidate(row) for row in fts_rows]
            if trace_logger.isEnabledFor(logging.DEBUG):
                trace_logger.debug(
                    "hybrid fts done: id=%s rows=%d fts_ms=%.1f",
                    request_id or "-",
                    len(fts_list),
                    fts_ms,
                )
        except Exception as exc:
            if settings.HYBRID_STRICT_MODE:
                raise
            hybrid_warning = str(exc)
            trace_logger.warning(
                "hybrid fts failed, fallback vector-only: id=%s error=%s",
                request_id or "-",
                exc,
            )

        retrieval_pool = rrf_fuse(
            vector_list,
            fts_list,
            rrf_k=settings.RRF_K,
            vector_weight=settings.RRF_VECTOR_WEIGHT,
            fts_weight=settings.RRF_FTS_WEIGHT,
            top_m=rrf_top_m,
        )
        if not retrieval_pool:
            retrieval_pool = list(vector_list)[:rrf_top_m]

    if settings.RETRIEVAL_DEBUG and trace_logger.isEnabledFor(logging.INFO):
        trace_logger.info(
            "TRACE RETRIEVAL id=%s\n%s",
            request_id or "-",
            _trace_retrieval_stage("vector_list", vector_list),
        )
        if settings.ENABLE_HYBRID:
            trace_logger.info(
                "TRACE RETRIEVAL id=%s\n%s",
                request_id or "-",
                _trace_retrieval_stage("fts_list", fts_list),
            )
            trace_logger.info(
                "TRACE RETRIEVAL id=%s\n%s",
                request_id or "-",
                _trace_retrieval_stage("merged_pool", retrieval_pool),
            )

    pre_rerank = list(retrieval_pool)
    if settings.ENABLE_RERANK:
        results = reranker.rerank(
            question, retrieval_pool, top_k=final_top_k, request_id=request_id
        )
    else:
        results = retrieval_pool[:final_top_k]

    docs_results = list(results)
    linked_defs: List[Dict[str, Any]] = []
    page_keys = _extract_doc_page_keys(
        docs_results,
        max_docs=settings.LINKED_DEFS_TOP_DOCS,
    )
    if settings.LINKED_DEFS_ENABLED and page_keys:
        try:
            linked_defs = query_related_definition_chunks(page_keys, request_id=request_id)
        except Exception as exc:
            trace_logger.warning(
                "linked defs retrieval failed: id=%s error=%s",
                request_id or "-",
                exc,
            )
            linked_defs = []
    if linked_defs:
        results = _merge_unique_chunks(docs_results, linked_defs)

    linked_defs_count = max(0, len(results) - len(docs_results))

    if trace_logger.isEnabledFor(logging.INFO):
        meta = (
            "meta: q_len={q_len} top_k={top_k} pool={pool} rows={rows} embed_dim={embed_dim} "
            "embed_ms={embed_ms:.1f} vector_ms={vector_ms:.1f} model={model} "
            "rerank={rerank} rerank_model={rerank_model} "
            "hybrid={hybrid} n_vec={n_vec} n_fts={n_fts} rrf_top_m={rrf_top_m} "
            "rows_vec={rows_vec} rows_fts={rows_fts} rows_docs={rows_docs} rows_linked_defs={rows_linked_defs}"
        ).format(
            q_len=len(question),
            top_k=final_top_k,
            pool=len(pre_rerank),
            rows=len(results),
            embed_dim=len(embedding),
            embed_ms=embed_ms,
            vector_ms=vector_ms,
            model=settings.EMBEDDING_MODEL,
            rerank=settings.ENABLE_RERANK,
            rerank_model=settings.RERANK_MODEL if settings.ENABLE_RERANK else "-",
            hybrid=settings.ENABLE_HYBRID,
            n_vec=vector_limit,
            n_fts=fts_limit if settings.ENABLE_HYBRID else 0,
            rrf_top_m=rrf_top_m if settings.ENABLE_HYBRID else 0,
            rows_vec=len(vector_list),
            rows_fts=len(fts_list),
            rows_docs=len(docs_results),
            rows_linked_defs=linked_defs_count,
        )
        query_text = _trace_truncate(question, settings.TRACE_MAX_TEXT_CHARS)
        query_block = "\n  ".join(query_text.splitlines()) if query_text else "(vacio)"
        if settings.ENABLE_RERANK:
            pool_block = _trace_chunks(
                pre_rerank,
                include_rrf_scores=settings.ENABLE_HYBRID,
            )
            final_block = _trace_chunks(
                results,
                include_rerank_scores=True,
                include_rrf_scores=settings.ENABLE_HYBRID,
            )
            trace_logger.info(
                "TRACE RAG id=%s\n%s\nquery_text:\n  %s\nchunks_pool:\n%s\nchunks_final:\n%s",
                request_id or "-",
                meta,
                query_block,
                pool_block,
                final_block,
            )
        else:
            chunks_block = _trace_chunks(
                results,
                include_rrf_scores=settings.ENABLE_HYBRID,
            )
            trace_logger.info(
                "TRACE RAG id=%s\n%s\nquery_text:\n  %s\nchunks:\n%s",
                request_id or "-",
                meta,
                query_block,
                chunks_block,
            )

    if hybrid_warning and settings.RETRIEVAL_DEBUG and trace_logger.isEnabledFor(logging.INFO):
        trace_logger.info(
            "TRACE RETRIEVAL id=%s\nhybrid_warning: %s",
            request_id or "-",
            hybrid_warning,
        )

    return results


def _row_to_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    candidate: Dict[str, Any] = {
        "doc_id": str(row.get(ROW_ID_KEY) or "").strip(),
        "text": row.get(settings.TEXT_COLUMN),
        "url": row.get(settings.URL_COLUMN),
        "title": row.get(settings.TITLE_COLUMN) if settings.TITLE_COLUMN else None,
        "section": row.get(settings.SECTION_COLUMN) if settings.SECTION_COLUMN else None,
    }
    metadata = row.get(METADATA_KEY)
    if isinstance(metadata, dict):
        candidate["metadata"] = metadata
    if row.get(FTS_SCORE_KEY) is not None:
        candidate["fts_score"] = float(row[FTS_SCORE_KEY])
    if not candidate["doc_id"]:
        candidate["doc_id"] = _candidate_reference(candidate)
    return candidate


def _resolve_top_k(top_k: Optional[int]) -> int:
    resolved = top_k or settings.TOP_K
    if resolved <= 0:
        resolved = settings.TOP_K
    if settings.ENABLE_RERANK and settings.RERANK_TOP_K and settings.RERANK_TOP_K > 0:
        return settings.RERANK_TOP_K
    return resolved


def _resolve_pool_size(final_top_k: int) -> int:
    if not settings.ENABLE_RERANK:
        return final_top_k
    pool_size = settings.RERANK_POOL_SIZE or final_top_k
    if pool_size <= 0:
        pool_size = final_top_k
    return max(pool_size, final_top_k)


def _resolve_vector_limit(pool_size: int) -> int:
    if not settings.ENABLE_HYBRID:
        return pool_size
    n_vec = settings.N_VEC or pool_size
    if n_vec <= 0:
        n_vec = pool_size
    return max(n_vec, pool_size)


def _resolve_fts_limit(pool_size: int) -> int:
    if not settings.ENABLE_HYBRID:
        return 0
    n_fts = settings.N_FTS or pool_size
    if n_fts <= 0:
        n_fts = pool_size
    return max(n_fts, pool_size)


def _resolve_rrf_top_m(pool_size: int) -> int:
    if not settings.ENABLE_HYBRID:
        return pool_size
    top_m = settings.RRF_TOP_M or pool_size
    if top_m <= 0:
        top_m = pool_size
    return max(top_m, pool_size)


def _extract_doc_page_keys(chunks: List[Dict[str, Any]], max_docs: int) -> List[str]:
    if max_docs <= 0:
        return []
    page_keys: List[str] = []
    seen = set()
    for chunk in chunks[:max_docs]:
        metadata = chunk.get("metadata")
        metadata_map = metadata if isinstance(metadata, dict) else {}
        row_url = str(chunk.get("url") or "")
        page_key, _, _ = derive_doc_page_key(metadata_map, row_url=row_url)
        if not page_key or page_key in seen:
            continue
        seen.add(page_key)
        page_keys.append(page_key)
    return page_keys


def _chunk_identity(chunk: Dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(chunk.get("doc_id") or "").strip(),
        str(chunk.get("url") or "").strip(),
        str(chunk.get("title") or "").strip(),
        str(chunk.get("section") or "").strip(),
    )


def _merge_unique_chunks(
    docs_chunks: List[Dict[str, Any]],
    linked_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = list(docs_chunks)
    seen = {_chunk_identity(chunk) for chunk in merged}
    for chunk in linked_chunks:
        identity = _chunk_identity(chunk)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(chunk)
    return merged


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    max_chars = settings.MAX_CONTEXT_CHARS
    used = 0
    lines: List[str] = ["Contexto de documentacion y definiciones relacionadas (citado):"]

    for index, chunk in enumerate(chunks, start=1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        header_parts: List[str] = []
        if chunk.get("title"):
            header_parts.append(
                f"title: {_compact_header_value(str(chunk['title']), max_chars=120)}"
            )
        if chunk.get("section"):
            header_parts.append(
                f"section: {_compact_header_value(str(chunk['section']), max_chars=80)}"
            )
        if chunk.get("url"):
            header_parts.append(
                f"url: {_compact_header_value(str(chunk['url']), max_chars=120)}"
            )

        header = " | ".join(header_parts) if header_parts else "chunk"
        entry = f"[{index}] {header}\n{text}\n"

        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(entry) > remaining:
            entry = entry[:remaining]
        lines.append(entry)
        used += len(entry)

        if used >= max_chars:
            break

    return "\n".join(lines)


def build_system_prompt(context_block: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{FORMAT_PROMPT}\n{context_block}"


_MD_HEADER_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_MD_QUOTE_RE = re.compile(r"(?m)^\s*>\s?")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _sanitize_markdown(text: str) -> str:
    # Prevent markdown headers/bullets from rendering as huge titles in clients.
    cleaned = _MD_HEADER_RE.sub("", text)
    cleaned = _MD_BULLET_RE.sub("", cleaned)
    cleaned = _MD_QUOTE_RE.sub("", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    return cleaned


def _compact_snippet(text: str, max_chars: int) -> str:
    snippet = _sanitize_markdown(text)
    snippet = " ".join(snippet.split())
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def _compact_header_value(value: Optional[str], max_chars: int = 120) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact


def _references_enabled() -> bool:
    return settings.references_enabled()


def collect_references(
    chunks: List[Dict[str, Any]],
    max_refs: int = 8,
    max_chars: int = 280,
) -> List[Dict[str, str]]:
    if not _references_enabled():
        return []
    refs: List[Dict[str, str]] = []
    seen = set()
    for chunk in chunks:
        if chunk.get("context_kind") == "linked_def":
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        snippet = _compact_snippet(text, max_chars)
        url = chunk.get("url") or ""
        title = chunk.get("title") or ""
        section = chunk.get("section") or ""
        key = (url, snippet)
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            {
                "snippet": snippet,
                "url": url,
                "title": _compact_header_value(title, max_chars=120),
                "section": _compact_header_value(section, max_chars=80),
            }
        )
        if len(refs) >= max_refs:
            break
    return refs


def format_references(refs: List[Dict[str, str]]) -> str:
    if not _references_enabled():
        return ""
    if not refs:
        return ""
    lines: List[str] = ["Referencias:"]
    for ref in refs:
        header_parts: List[str] = []
        if ref.get("title"):
            header_parts.append(
                f"title: {_compact_header_value(ref.get('title'), max_chars=120)}"
            )
        if ref.get("section"):
            header_parts.append(
                f"section: {_compact_header_value(ref.get('section'), max_chars=80)}"
            )
        if ref.get("url"):
            header_parts.append(
                f"url: {_compact_header_value(ref.get('url'), max_chars=120)}"
            )
        header = " | ".join(header_parts)
        if header:
            lines.append(f"- {header}")
            lines.append(f"  {ref['snippet']}")
        else:
            lines.append(f"- {ref['snippet']}")
    return "\n".join(lines)


def append_references(text: str, refs: List[Dict[str, str]]) -> str:
    if not _references_enabled():
        return text
    refs_block = format_references(refs)
    if not refs_block:
        return text
    return text.rstrip() + "\n\n" + refs_block
