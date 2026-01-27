from __future__ import annotations

from app.workflow.control_parser import parse_control_state


def test_last_wf_command_wins_and_is_removed_from_prompt() -> None:
    messages = [
        {"role": "user", "content": "/wf wf-1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "revisa el nodo http"},
        {"role": "user", "content": "/wf wf-2"},
    ]

    state = parse_control_state(messages)

    assert state.active_workflow_id == "wf-2"
    assert all("/wf" not in msg["content"] for msg in state.cleaned_messages)


def test_embedded_command_preserves_rest_of_message() -> None:
    messages = [
        {
            "role": "user",
            "content": "por favor /wf my-workflow revisa el merge",
        }
    ]

    state = parse_control_state(messages)

    assert state.active_workflow_id == "my-workflow"
    assert state.cleaned_messages == [
        {"role": "user", "content": "por favor revisa el merge"}
    ]
