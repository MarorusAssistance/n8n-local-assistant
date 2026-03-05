from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from bench.report import generate_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_report_correlates_runs_with_trace_request_id() -> None:
    run_dir = Path.cwd() / f"bench_log_corr_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "experiment": "baseline",
            "case_id": "case_a",
            "rep": 1,
            "is_warmup": False,
            "conversation_id": "conv-baseline-case_a-1",
            "trace_request_id": "req-baseline-1",
            "started_at": "2026-03-05T10:00:00+00:00",
            "ended_at": "2026-03-05T10:00:10+00:00",
            "latency_ms": 100.0,
            "success": True,
            "error": None,
            "http_status": 200,
            "parse_ok": True,
            "compliance_score": 90.0,
            "breakdown": {},
            "artifacts": {
                "response": "responses/baseline/case_a/1.txt",
                "checks": "checks/baseline/case_a/1.json",
                "workflow": None,
                "diff": None,
            },
        }
    ]
    _write_jsonl(run_dir / "runs.jsonl", rows)

    run_meta = {
        "experiments": ["baseline"],
        "experiment_configs": {"baseline": {"adapter": "direct"}},
        "trace_capture": {
            "raw_files": {"baseline": "logs/raw/baseline.log"},
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    trace_text = """2026-03-05 10:00:00,000 INFO n8n-assistant.trace: TRACE REQUEST id=req-baseline-1
meta: conv=conv-baseline-case_a-1 wf=- stream=False messages=1 history=0 user_len=4
usuario:
  hola
2026-03-05 10:00:00,050 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_prompt","request_id":"req-baseline-1","stage":"docs_only.synthesis","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hola"}],"total_chars":7}
2026-03-05 10:00:00,120 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_output","request_id":"req-baseline-1","stage":"docs_only.synthesis","latency_ms":12.5}
"""
    raw_trace = run_dir / "logs" / "raw" / "baseline.log"
    raw_trace.parent.mkdir(parents=True, exist_ok=True)
    raw_trace.write_text(trace_text, encoding="utf-8")

    try:
        generate_report(run_dir)
        updated_rows = [
            json.loads(line)
            for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        artifacts = updated_rows[0]["artifacts"]
        assert artifacts["logs_html"]
        assert artifacts["trace_events"]
        assert artifacts["trace_slice"]
        assert artifacts["prompts"]
    finally:
        for path in sorted(run_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
