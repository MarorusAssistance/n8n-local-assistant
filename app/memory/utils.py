from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ..config import settings


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _is_task_prefix(text: str, prefixes: List[str]) -> bool:
    lowered = text.lower()
    for prefix in prefixes:
        if not prefix:
            continue
        if lowered.startswith(prefix.lower()):
            return True
    return False


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _looks_like_json(text: str) -> bool:
    stripped = _strip_code_fences(text)
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_like_task_response(text: str, json_keys: List[str]) -> bool:
    if not _looks_like_json(text):
        return False
    lowered = _strip_code_fences(text).lower()
    for key in json_keys:
        if not key:
            continue
        token = f"\"{key.lower()}\""
        if token in lowered:
            return True
    return False


def filter_messages(messages: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    skip_prefixes = _split_csv(settings.MEMORY_SKIP_PREFIXES)
    assistant_json_keys = _split_csv(settings.MEMORY_SKIP_ASSISTANT_JSON_KEYS)

    skip_next_assistant = False
    for msg in list(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        trimmed = content.strip()
        if not trimmed:
            continue

        if _is_task_prefix(trimmed, skip_prefixes):
            if role == "user":
                skip_next_assistant = True
            continue

        if role == "assistant" and skip_next_assistant:
            if _looks_like_task_response(trimmed, assistant_json_keys):
                skip_next_assistant = False
                continue
            skip_next_assistant = False

        filtered.append({"role": role, "content": trimmed})
    return filtered


def compact_text(text: str, max_chars: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) > max_chars:
        compacted = compacted[: max_chars - 3].rstrip() + "..."
    return compacted


def derive_title(messages: Iterable[Dict[str, str]], max_chars: int = 80) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            content = msg["content"].strip()
            if content:
                return compact_text(content, max_chars)
    return None
