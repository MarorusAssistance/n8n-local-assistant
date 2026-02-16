from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


# Conservative estimator: ~3 UTF-8 bytes per token.
_BYTES_PER_TOKEN = 3
# ChatML-style per-message overhead.
_TOKENS_PER_MESSAGE_OVERHEAD = 4
# Priming overhead for assistant reply.
_TOKENS_REPLY_PRIMING = 3


@dataclass
class PromptBudgetResult:
    messages: List[Dict[str, str]]
    estimated_tokens_before: int
    estimated_tokens_after: int
    dropped_messages: int
    truncated_messages: int

    @property
    def changed(self) -> bool:
        return self.dropped_messages > 0 or self.truncated_messages > 0


def estimate_text_tokens(text: str) -> int:
    value = str(text or "")
    if not value:
        return 0
    byte_len = len(value.encode("utf-8"))
    return max(1, (byte_len + (_BYTES_PER_TOKEN - 1)) // _BYTES_PER_TOKEN)


def estimate_messages_tokens(messages: Iterable[Dict[str, str]]) -> int:
    total = _TOKENS_REPLY_PRIMING
    for msg in messages:
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "")
        total += _TOKENS_PER_MESSAGE_OVERHEAD
        total += estimate_text_tokens(role)
        total += estimate_text_tokens(content)
    return total


def trim_messages_to_budget(
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> PromptBudgetResult:
    normalized: List[Dict[str, str]] = [
        {
            "role": str(msg.get("role") or ""),
            "content": str(msg.get("content") or ""),
        }
        for msg in messages
    ]

    before = estimate_messages_tokens(normalized)
    if max_tokens <= 0 or before <= max_tokens:
        return PromptBudgetResult(
            messages=normalized,
            estimated_tokens_before=before,
            estimated_tokens_after=before,
            dropped_messages=0,
            truncated_messages=0,
        )

    token_by_index = [estimate_messages_tokens([msg]) - _TOKENS_REPLY_PRIMING for msg in normalized]
    system_indices = [idx for idx, msg in enumerate(normalized) if msg["role"] == "system"]
    last_user_idx = next(
        (idx for idx in range(len(normalized) - 1, -1, -1) if normalized[idx]["role"] == "user"),
        None,
    )

    required = set(system_indices)
    if last_user_idx is not None:
        required.add(last_user_idx)

    used = sum(token_by_index[idx] for idx in required) + _TOKENS_REPLY_PRIMING
    kept_indices = set(required)

    for idx in range(len(normalized) - 1, -1, -1):
        if idx in kept_indices:
            continue
        msg_tokens = token_by_index[idx]
        if used + msg_tokens <= max_tokens:
            kept_indices.add(idx)
            used += msg_tokens

    trimmed = [normalized[idx].copy() for idx in sorted(kept_indices)]
    dropped = len(normalized) - len(trimmed)
    after = estimate_messages_tokens(trimmed)
    truncated = 0

    if after > max_tokens and trimmed:
        after, truncated = _truncate_to_fit(trimmed, max_tokens)

    return PromptBudgetResult(
        messages=trimmed,
        estimated_tokens_before=before,
        estimated_tokens_after=after,
        dropped_messages=dropped,
        truncated_messages=truncated,
    )


def _truncate_to_fit(messages: List[Dict[str, str]], max_tokens: int) -> tuple[int, int]:
    truncated = 0
    total = estimate_messages_tokens(messages)
    if total <= max_tokens:
        return total, truncated

    for idx, msg in enumerate(messages):
        if total <= max_tokens:
            break

        content = msg.get("content") or ""
        content_tokens = estimate_text_tokens(content)
        if content_tokens <= 1:
            continue

        overflow = total - max_tokens
        target_tokens = max(1, content_tokens - overflow - 4)
        truncated_content = _truncate_text_to_tokens(content, target_tokens)
        if truncated_content == content:
            continue

        messages[idx]["content"] = truncated_content
        truncated += 1
        total = estimate_messages_tokens(messages)

    return total, truncated


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    raw = str(text or "")
    if max_tokens <= 0:
        return ""

    max_bytes = max_tokens * _BYTES_PER_TOKEN
    encoded = raw.encode("utf-8")
    if len(encoded) <= max_bytes:
        return raw

    # Keep valid UTF-8 while trimming to approximate token budget.
    compact = encoded[:max(0, max_bytes - 3)].decode("utf-8", errors="ignore").rstrip()
    if not compact:
        return ""
    if compact == raw:
        return compact
    return compact + "..."
