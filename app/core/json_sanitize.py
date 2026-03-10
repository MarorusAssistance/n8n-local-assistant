from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

try:  # Optional import at runtime boundaries.
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional fallback
    BaseModel = None  # type: ignore[assignment]


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def sanitize_text(value: Any, *, max_chars: int = 6000) -> str:
    text = html.unescape(str(value or ""))
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lowered = text.lower()
    if ("node_type" in lowered or "workflow_context" in lowered or '"confidence"' in lowered) and (
        "http://" in lowered or "https://" in lowered
    ):
        match = _URL_RE.search(text)
        if match:
            text = match.group(0)
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text.strip()


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return sanitize_text(value)
    if BaseModel is not None and isinstance(value, BaseModel):
        return sanitize_for_json(value.model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_for_json(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [sanitize_for_json(item) for item in value]
    return sanitize_text(value)


def safe_json_dumps(value: Any) -> tuple[str, Any]:
    sanitized = sanitize_for_json(value)
    try:
        raw = json.dumps(sanitized, ensure_ascii=False)
        json.loads(raw)
        return raw, sanitized
    except Exception:
        fallback = {
            "entry_intent": "unknown",
            "target_stage": None,
            "confidence": 0.0,
            "routing_signals": ["json_sanitization_fallback"],
            "current_stage": None,
            "missing_user_inputs": [
                "The response payload became invalid and was replaced by a safe fallback."
            ],
            "status": "unknown_terminal",
        }
        raw = json.dumps(fallback, ensure_ascii=False)
        return raw, fallback
