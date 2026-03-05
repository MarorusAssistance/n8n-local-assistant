from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List
from uuid import uuid4

from .adapter import PipelineAdapter
from .checks import load_catalog, run_checks
from .io import append_jsonl, ensure_dir, load_cases, load_experiments, relpath_str, write_json, write_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark experiment in an isolated process.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--experiments", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-cases", type=int, default=3)
    parser.add_argument("--quick-repetitions", type=int, default=1)
    parser.add_argument("--http-base-url", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    cases = load_cases(Path(args.cases))
    experiments = load_experiments(Path(args.experiments))
    target = next((exp for exp in experiments if exp.name == args.experiment_name), None)
    if target is None:
        raise ValueError(f"experiment not found: {args.experiment_name}")

    if args.quick:
        cases = cases[: max(1, int(args.quick_cases))]
        warmup_count = 0
        repetitions = max(1, int(args.quick_repetitions))
    else:
        warmup_count = target.warmup
        repetitions = target.repetitions

    catalog = load_catalog(Path("nodes.json"), Path("credentials.json"))
    adapter = PipelineAdapter(
        adapter=target.adapter,
        http_base_url=args.http_base_url,
        model=args.model,
    )
    runs_path = run_dir / "runs.jsonl"

    try:
        for case in cases:
            _run_case(
                run_dir=run_dir,
                runs_path=runs_path,
                adapter=adapter,
                catalog=catalog,
                experiment_name=target.name,
                case=case,
                warmup_count=warmup_count,
                repetitions=repetitions,
                experiment_json_only=target.json_only,
            )
    finally:
        adapter.close()
    return 0


def _run_case(
    *,
    run_dir: Path,
    runs_path: Path,
    adapter: PipelineAdapter,
    catalog,
    experiment_name: str,
    case,
    warmup_count: int,
    repetitions: int,
    experiment_json_only: bool,
) -> None:
    items: List[Dict[str, int | bool]] = []
    for rep in range(1, warmup_count + 1):
        items.append({"rep": rep, "is_warmup": True})
    for rep in range(1, repetitions + 1):
        items.append({"rep": rep, "is_warmup": False})

    for item in items:
        rep = int(item["rep"])
        is_warmup = bool(item["is_warmup"])
        rep_label = f"warmup-{rep}" if is_warmup else str(rep)

        started_at = datetime.now(timezone.utc)
        prompt = adapter.build_user_message(
            user_message=case.user_message,
            json_only=bool(case.json_only or experiment_json_only),
        )
        conv_id = f"bench-{experiment_name}-{case.id}-{rep_label}-{uuid4().hex[:8]}"
        trace_request_id = (
            f"bench-{experiment_name}-{case.id}-{rep_label}-{uuid4().hex[:6]}"
            .replace(" ", "-")
            .replace("/", "-")
        )
        t0 = perf_counter()
        adapter_response = adapter.invoke(
            user_message=prompt,
            conversation_id=conv_id,
            request_id=trace_request_id,
        )
        latency_ms = (perf_counter() - t0) * 1000.0
        ended_at = datetime.now(timezone.utc)

        response_path = run_dir / "responses" / experiment_name / case.id / f"{rep_label}.txt"
        write_text(response_path, adapter_response.assistant_text)

        check_result = run_checks(
            adapter_response.assistant_text,
            case=case,
            catalog=catalog,
        )

        workflow_path = None
        workflow_payload = check_result.get("workflow")
        if isinstance(workflow_payload, dict):
            workflow_path = run_dir / "workflows" / experiment_name / case.id / f"{rep_label}.json"
            write_json(workflow_path, workflow_payload)

        checks_path = run_dir / "checks" / experiment_name / case.id / f"{rep_label}.json"
        write_json(
            checks_path,
            {
                "experiment": experiment_name,
                "case_id": case.id,
                "rep": rep,
                "is_warmup": is_warmup,
                **check_result,
            },
        )

        record = {
            "experiment": experiment_name,
            "case_id": case.id,
            "rep": rep,
            "is_warmup": is_warmup,
            "conversation_id": conv_id,
            "trace_request_id": trace_request_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "latency_ms": round(latency_ms, 3),
            "success": adapter_response.success,
            "error": adapter_response.error,
            "http_status": adapter_response.http_status,
            "parse_ok": bool(check_result.get("parse_ok", False)),
            "compliance_score": float(check_result.get("compliance_score", 0.0)),
            "breakdown": check_result.get("breakdown", {}),
            "artifacts": {
                "response": relpath_str(response_path, run_dir),
                "checks": relpath_str(checks_path, run_dir),
                "workflow": relpath_str(workflow_path, run_dir) if workflow_path else None,
                "diff": None,
                "logs_html": None,
                "trace_events": None,
                "trace_slice": None,
                "prompts": None,
            },
        }
        append_jsonl(runs_path, record)


def main() -> int:
    args = _parse_args()
    ensure_dir(Path(args.run_dir))
    try:
        return run_worker(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
