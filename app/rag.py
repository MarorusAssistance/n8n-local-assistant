from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import re
import time

from .config import settings
from .db import query_similar
from .llm import create_embedding


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


def _trace_chunks(chunks: List[Dict[str, Any]]) -> str:
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
    effective_top_k = top_k or settings.TOP_K
    if effective_top_k <= 0:
        effective_top_k = settings.TOP_K

    if trace_logger.isEnabledFor(logging.DEBUG):
        trace_logger.debug(
            "rag query text: id=%s text=%s",
            request_id or "-",
            _compact_snippet(question, max_chars=220),
        )

    embed_start = time.perf_counter()
    embedding = create_embedding(question)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    query_start = time.perf_counter()
    rows = query_similar(embedding, top_k=top_k, request_id=request_id)
    query_ms = (time.perf_counter() - query_start) * 1000

    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                ##ToDo: Review columns names
            
                "text": row.get(settings.TEXT_COLUMN),
                "url": row.get(settings.URL_COLUMN),
                "title": row.get(settings.TITLE_COLUMN) if settings.TITLE_COLUMN else None,
                "section": row.get(settings.SECTION_COLUMN) if settings.SECTION_COLUMN else None,
            }
        )

    if trace_logger.isEnabledFor(logging.INFO):
        meta = (
            "meta: q_len={q_len} top_k={top_k} rows={rows} embed_dim={embed_dim} "
            "embed_ms={embed_ms:.1f} query_ms={query_ms:.1f} model={model}"
        ).format(
            q_len=len(question),
            top_k=effective_top_k,
            rows=len(results),
            embed_dim=len(embedding),
            embed_ms=embed_ms,
            query_ms=query_ms,
            model=settings.EMBEDDING_MODEL,
        )
        query_text = _trace_truncate(question, settings.TRACE_MAX_TEXT_CHARS)
        chunks_block = _trace_chunks(results)
        trace_logger.info(
            "TRACE RAG id=%s\n%s\nquery_text:\n  %s\nchunks:\n%s",
            request_id or "-",
            meta,
            "\n  ".join(query_text.splitlines()) if query_text else "(vacio)",
            chunks_block,
        )
    return results


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    max_chars = settings.MAX_CONTEXT_CHARS
    used = 0
    lines: List[str] = ["Contexto de documentacion (citado):"]

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
