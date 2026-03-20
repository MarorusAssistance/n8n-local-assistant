from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.workflow_trial_harness import (  # noqa: E402
    DEFAULT_EMAIL_URGENCY_REQUEST,
    run_email_urgency_trial,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an automated multi-turn trial against the local chat service for the email urgency workflow."
    )
    parser.add_argument(
        "--model",
        default="local-model",
        help="Model id to send to /v1/chat/completions.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Maximum number of turns to attempt before stopping.",
    )
    parser.add_argument(
        "--request",
        default=DEFAULT_EMAIL_URGENCY_REQUEST,
        help="Initial user request to send.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path to write the full harness result as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_email_urgency_trial(
        model=args.model,
        max_turns=args.max_turns,
        initial_request=args.request,
    )

    print(f"Conversation: {result.conversation_id}")
    print(f"Turns used: {result.turns_used}")
    if result.workflow_url:
        print(f"Workflow URL: {result.workflow_url}")
    if result.workflow_id:
        print(f"Workflow ID: {result.workflow_id}")
    print("")
    print("Transcript")
    print("----------")
    for turn in result.transcript:
        print(f"[{turn.turn_index}] user: {turn.user_text}")
        print(f"[{turn.turn_index}] assistant: {turn.assistant_text}")
        if turn.current_stage or turn.last_block_cause:
            print(
                f"[{turn.turn_index}] meta: stage={turn.current_stage or '-'} "
                f"block={turn.last_block_cause or '-'} slots={','.join(turn.pending_slot_keys) or '-'}"
            )
        print("")

    print("Validation")
    print("----------")
    print(f"Success: {result.success}")
    if result.envelope:
        current_stage = str(result.envelope.get("current_stage") or "").strip() or "-"
        entry_intent = str(result.envelope.get("entry_intent") or "").strip() or "-"
        last_block_cause = str(result.envelope.get("last_block_cause") or "").strip() or "-"
        print(
            "Envelope summary: "
            f"entry_intent={entry_intent} current_stage={current_stage} last_block_cause={last_block_cause}"
        )
    if result.structural_issues:
        print("Structural issues:")
        for issue in result.structural_issues:
            print(f"- {issue}")
    else:
        print("Structural issues: none")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote result JSON to {output_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
