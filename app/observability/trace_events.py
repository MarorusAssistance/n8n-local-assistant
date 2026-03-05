from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional


_DEFAULT_TEXT_LIMIT = 24000


def _trim_text(value: Any, max_chars: int = _DEFAULT_TEXT_LIMIT) -> str:
    text = str(value or "")
    if max_chars > 0 and len(text) > max_chars:
        omitted = len(text) - max_chars
        return text[:max_chars].rstrip() + f"... [trimmed {omitted} chars]"
    return text


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    return str(value)


def emit_trace_event(
    trace_logger: logging.Logger,
    *,
    event: str,
    request_id: Optional[str],
    stage: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    if not trace_logger.isEnabledFor(level):
        return

    body: dict[str, Any] = {
        "event": str(event),
        "request_id": str(request_id or "-"),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    if stage:
        body["stage"] = str(stage)
    if payload:
        for key, value in payload.items():
            body[str(key)] = _safe_json(value)

    trace_logger.log(level, "TRACE EVENT %s", json.dumps(body, ensure_ascii=False, sort_keys=True))


def emit_llm_prompt_event(
    trace_logger: logging.Logger,
    *,
    request_id: Optional[str],
    stage: str,
    model: Optional[str],
    messages: Iterable[Mapping[str, Any]],
    estimated_tokens: Optional[int] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> None:
    event_messages = []
    total_chars = 0
    for msg in messages:
        role = str(msg.get("role") or "unknown")
        content = _trim_text(msg.get("content") or "")
        total_chars += len(content)
        event_messages.append(
            {
                "role": role,
                "chars": len(content),
                "content": content,
            }
        )

    payload: dict[str, Any] = {
        "model": str(model or ""),
        "message_count": len(event_messages),
        "total_chars": total_chars,
        "messages": event_messages,
    }
    if estimated_tokens is not None:
        payload["estimated_tokens"] = int(estimated_tokens)
    if params:
        payload["params"] = {str(k): _safe_json(v) for k, v in params.items()}

    emit_trace_event(
        trace_logger,
        event="llm_prompt",
        request_id=request_id,
        stage=stage,
        payload=payload,
    )


def emit_llm_output_event(
    trace_logger: logging.Logger,
    *,
    request_id: Optional[str],
    stage: str,
    model: Optional[str],
    latency_ms: Optional[float],
    content: Optional[str],
    usage: Optional[Mapping[str, Any]],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "model": str(model or ""),
        "content_chars": len(str(content or "")),
        "content_preview": _trim_text(content or "", max_chars=4000),
    }
    if latency_ms is not None:
        payload["latency_ms"] = round(float(latency_ms), 3)
    if usage:
        payload["usage"] = {str(k): _safe_json(v) for k, v in usage.items()}
    if extra:
        for key, value in extra.items():
            payload[str(key)] = _safe_json(value)

    emit_trace_event(
        trace_logger,
        event="llm_output",
        request_id=request_id,
        stage=stage,
        payload=payload,
    )
