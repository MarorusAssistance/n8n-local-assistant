from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# /wf <workflow_id> where id can be uuid-like or slug-like.
WF_COMMAND_RE = re.compile(r"(^|\s)/wf\s+([A-Za-z0-9_-]+)", re.IGNORECASE)


def _find_workflow_ids(text: str) -> List[str]:
    return [match[1] for match in WF_COMMAND_RE.findall(text)]


def _strip_commands(text: str) -> Tuple[str, List[str]]:
    ids = _find_workflow_ids(text)
    if not ids:
        return text, []

    stripped = WF_COMMAND_RE.sub(" ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped, ids


@dataclass
class ControlState:
    active_workflow_id: Optional[str]
    cleaned_messages: List[Dict[str, str]]
    control_messages_count: int

    @property
    def has_active_workflow(self) -> bool:
        return bool(self.active_workflow_id)


def parse_control_state(messages: Iterable[Dict[str, str]]) -> ControlState:
    active_workflow_id: Optional[str] = None
    cleaned: List[Dict[str, str]] = []
    control_messages_count = 0

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant", "system"):
            continue
        if not isinstance(content, str):
            continue

        if role == "user":
            stripped, ids = _strip_commands(content)
            if ids:
                control_messages_count += 1
                active_workflow_id = ids[-1]
            if stripped:
                cleaned.append({"role": role, "content": stripped})
            continue

        cleaned.append({"role": role, "content": content})

    return ControlState(
        active_workflow_id=active_workflow_id,
        cleaned_messages=cleaned,
        control_messages_count=control_messages_count,
    )


def strip_control_commands(messages: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove /wf commands from user messages without altering non-control content."""
    return parse_control_state(messages).cleaned_messages
