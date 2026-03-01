from __future__ import annotations

from app.token_budget import estimate_messages_tokens, trim_messages_to_budget


def test_trim_messages_to_budget_drops_oldest_message_first() -> None:
    messages = [
        {"role": "system", "content": "Instrucciones base"},
        {"role": "user", "content": "u1 " * 180},
        {"role": "assistant", "content": "a1 " * 180},
        {"role": "user", "content": "u2 " * 180},
        {"role": "assistant", "content": "a2 " * 180},
        {"role": "user", "content": "u3 pregunta final"},
    ]

    total = estimate_messages_tokens(messages)
    first_message_cost = estimate_messages_tokens([messages[1]]) - 3
    limit = total - first_message_cost

    result = trim_messages_to_budget(messages, max_tokens=limit)

    assert result.changed
    assert result.dropped_messages == 1
    assert result.truncated_messages == 0
    assert result.messages[0] == messages[0]
    assert result.messages[1:] == messages[2:]
    assert result.estimated_tokens_after <= limit


def test_trim_messages_to_budget_drops_until_fit_preserving_system_and_last_user() -> None:
    messages = [
        {"role": "system", "content": "Instrucciones base"},
        {"role": "user", "content": "u1 " * 180},
        {"role": "assistant", "content": "a1 " * 180},
        {"role": "user", "content": "u2 " * 180},
        {"role": "assistant", "content": "a2 " * 180},
        {"role": "user", "content": "u3 pregunta final"},
    ]

    limit = estimate_messages_tokens([messages[0], messages[-1]])
    result = trim_messages_to_budget(messages, max_tokens=limit)

    assert result.changed
    assert result.truncated_messages == 0
    assert result.messages == [messages[0], messages[-1]]
    assert result.estimated_tokens_after <= limit


def test_trim_messages_to_budget_does_not_truncate_protected_messages() -> None:
    messages = [
        {"role": "system", "content": "s" * 9000},
        {"role": "user", "content": "x" * 9000},
    ]

    result = trim_messages_to_budget(messages, max_tokens=220)

    assert result.truncated_messages == 0
    assert result.dropped_messages == 0
    assert result.messages == messages
    assert estimate_messages_tokens(result.messages) > 220
