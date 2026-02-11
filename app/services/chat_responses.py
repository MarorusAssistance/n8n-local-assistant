from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..rag import format_references


class ChatResponseBuilder:
    """Builds streaming and non-streaming chat responses."""

    @staticmethod
    def simple_chat_response(text: str, model: str) -> Dict[str, Any]:
        """Return a minimal non-streaming response payload."""
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    @staticmethod
    def stream_simple_text(text: str, model: str) -> Iterable[str]:
        """Yield a simple SSE stream for short responses."""
        now = int(time.time())
        chunk_id = f"chatcmpl-{now}"
        first_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }
        yield ChatResponseBuilder._format_sse(first_chunk)

        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield ChatResponseBuilder._format_sse(final_chunk)
        yield "data: [DONE]\n\n"

    @staticmethod
    def stream_with_references(
        stream: Iterable[Any],
        refs: List[Dict[str, str]],
        fallback_model: str,
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> Iterable[str]:
        """Stream model output and append references at the end."""
        last_id: str | None = None
        last_created = int(time.time())
        last_model = fallback_model
        finish_reason: str | None = None
        assistant_parts: List[str] = []

        for chunk in stream:
            chunk_dict = chunk.model_dump()
            last_id = chunk_dict.get("id") or last_id
            last_created = chunk_dict.get("created") or last_created
            last_model = chunk_dict.get("model") or last_model

            choices = chunk_dict.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    assistant_parts.append(content)
            if finish_reason:
                continue

            yield ChatResponseBuilder._format_sse(chunk_dict)

        refs_block = format_references(refs)
        if refs_block:
            refs_block = "\n\n" + refs_block
            refs_chunk = {
                "id": last_id or f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion.chunk",
                "created": last_created,
                "model": last_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": refs_block},
                        "finish_reason": None,
                    }
                ],
            }
            yield ChatResponseBuilder._format_sse(refs_chunk)

        if on_complete:
            on_complete("".join(assistant_parts))

        final_chunk = {
            "id": last_id or f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion.chunk",
            "created": last_created,
            "model": last_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason or "stop",
                }
            ],
        }
        yield ChatResponseBuilder._format_sse(final_chunk)
        yield "data: [DONE]\n\n"

    @staticmethod
    def _format_sse(data: Dict[str, Any]) -> str:
        """Format an SSE chunk."""
        return f"data: {json.dumps(data)}\n\n"
