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

    system_indices = [idx for idx, msg in enumerate(normalized) if msg["role"] == "system"]
    last_user_idx = next(
        (idx for idx in range(len(normalized) - 1, -1, -1) if normalized[idx]["role"] == "user"),
        None,
    )

    required = set(system_indices)
    if last_user_idx is not None:
        required.add(last_user_idx)

    kept_indices = list(range(len(normalized)))

    # Strict FIFO trimming: drop oldest non-required conversation message first.
    while True:
        trimmed = [normalized[idx].copy() for idx in kept_indices]
        after = estimate_messages_tokens(trimmed)
        if after <= max_tokens:
            break

        drop_pos = next((pos for pos, idx in enumerate(kept_indices) if idx not in required), None)
        if drop_pos is None:
            break

        kept_indices.pop(drop_pos)

    trimmed = [normalized[idx].copy() for idx in kept_indices]
    dropped = len(normalized) - len(trimmed)
    after = estimate_messages_tokens(trimmed)

    return PromptBudgetResult(
        messages=trimmed,
        estimated_tokens_before=before,
        estimated_tokens_after=after,
        dropped_messages=dropped,
        truncated_messages=0,
    )
