from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import settings
from .db import query_similar
from .llm import create_embedding


SYSTEM_PROMPT = (
    "Eres un asistente experto en n8n. Responde con pasos accionables en n8n, "
    "menciona nombres de nodos y campos, sugiere checks y posibles causas. "
    "Si falta informacion, dilo y propone hipotesis. No inventes. "
    "No incluyas referencias; el backend las agregara al final."
)

FORMAT_PROMPT = (
    "Formato de salida:\n"
    "Diagnostico:\n"
    "Pasos:\n"
    "Alternativas:\n"
)


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


def retrieve_context(question: str) -> List[Dict[str, Any]]:
    embedding = create_embedding(question)
    rows = query_similar(embedding)

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
            header_parts.append(f"title: {chunk['title']}")
        if chunk.get("section"):
            header_parts.append(f"section: {chunk['section']}")
        if chunk.get("url"):
            header_parts.append(f"url: {chunk['url']}")

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


def _compact_snippet(text: str, max_chars: int) -> str:
    snippet = " ".join(text.split())
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def collect_references(
    chunks: List[Dict[str, Any]],
    max_refs: int = 8,
    max_chars: int = 280,
) -> List[Dict[str, str]]:
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
                "title": title,
                "section": section,
            }
        )
        if len(refs) >= max_refs:
            break
    return refs


def format_references(refs: List[Dict[str, str]]) -> str:
    if not refs:
        return ""
    lines: List[str] = ["Referencias:"]
    for ref in refs:
        header_parts: List[str] = []
        if ref.get("title"):
            header_parts.append(f"title: {ref['title']}")
        if ref.get("section"):
            header_parts.append(f"section: {ref['section']}")
        if ref.get("url"):
            header_parts.append(f"url: {ref['url']}")
        header = " | ".join(header_parts)
        if header:
            lines.append(f"- {header}")
            lines.append(f"  {ref['snippet']}")
        else:
            lines.append(f"- {ref['snippet']}")
    return "\n".join(lines)


def append_references(text: str, refs: List[Dict[str, str]]) -> str:
    refs_block = format_references(refs)
    if not refs_block:
        return text
    return text.rstrip() + "\n\n" + refs_block
