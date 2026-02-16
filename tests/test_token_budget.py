from __future__ import annotations

from app.token_budget import estimate_messages_tokens, trim_messages_to_budget


def test_trim_messages_to_budget_keeps_recent_history() -> None:
    messages = [
        {"role": "system", "content": "Instrucciones base"},
        {"role": "user", "content": "u1 " * 180},
        {"role": "assistant", "content": "a1 " * 180},
        {"role": "user", "content": "u2 " * 180},
        {"role": "assistant", "content": "a2 " * 180},
        {"role": "user", "content": "u3 pregunta final"},
    ]

    result = trim_messages_to_budget(messages, max_tokens=700)

    assert result.changed
    assert result.dropped_messages > 0
    assert result.messages[0]["role"] == "system"
    assert result.messages[-1]["role"] == "user"
    assert result.messages[-1]["content"] == "u3 pregunta final"
    assert all(msg["content"] != "u1 " * 180 for msg in result.messages)
    assert result.estimated_tokens_after <= 700


def test_trim_messages_to_budget_truncates_when_single_message_is_too_large() -> None:
    messages = [{"role": "user", "content": "x" * 9000}]

    result = trim_messages_to_budget(messages, max_tokens=220)

    assert result.truncated_messages == 1
    assert result.dropped_messages == 0
    assert result.messages[0]["role"] == "user"
    assert len(result.messages[0]["content"]) < 9000
    assert estimate_messages_tokens(result.messages) <= 220
