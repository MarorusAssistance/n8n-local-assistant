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


def test_generate_logs_report_and_prompt_diff() -> None:
    run_dir = Path.cwd() / f"bench_log_report_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "experiment": "baseline",
            "case_id": "case_a",
            "rep": 1,
            "is_warmup": False,
            "conversation_id": "conv-b",
            "trace_request_id": "req-b",
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
        },
        {
            "experiment": "exp2",
            "case_id": "case_a",
            "rep": 1,
            "is_warmup": False,
            "conversation_id": "conv-e",
            "trace_request_id": "req-e",
            "started_at": "2026-03-05T10:01:00+00:00",
            "ended_at": "2026-03-05T10:01:10+00:00",
            "latency_ms": 120.0,
            "success": True,
            "error": None,
            "http_status": 200,
            "parse_ok": True,
            "compliance_score": 80.0,
            "breakdown": {},
            "artifacts": {
                "response": "responses/exp2/case_a/1.txt",
                "checks": "checks/exp2/case_a/1.json",
                "workflow": None,
                "diff": None,
            },
        },
    ]
    _write_jsonl(run_dir / "runs.jsonl", rows)
    run_meta = {
        "experiments": ["baseline", "exp2"],
        "experiment_configs": {
            "baseline": {"adapter": "direct"},
            "exp2": {"adapter": "direct"},
        },
        "trace_capture": {
            "raw_files": {
                "baseline": "logs/raw/baseline.log",
                "exp2": "logs/raw/exp2.log",
            },
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    baseline_trace = """2026-03-05 10:00:00,000 INFO n8n-assistant.trace: TRACE REQUEST id=req-b
meta: conv=conv-b wf=- stream=False messages=1 history=0 user_len=4
usuario:
  hola
2026-03-05 10:00:00,050 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_prompt","request_id":"req-b","stage":"docs_only.synthesis","messages":[{"role":"system","content":"sys baseline"},{"role":"user","content":"hola"}],"total_chars":16}
2026-03-05 10:00:00,120 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_output","request_id":"req-b","stage":"docs_only.synthesis","latency_ms":11.1}
"""
    exp_trace = """2026-03-05 10:01:00,000 INFO n8n-assistant.trace: TRACE REQUEST id=req-e
meta: conv=conv-e wf=- stream=False messages=1 history=0 user_len=4
usuario:
  hola
2026-03-05 10:01:00,050 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_prompt","request_id":"req-e","stage":"docs_only.synthesis","messages":[{"role":"system","content":"sys exp2"},{"role":"user","content":"hola"}],"total_chars":12}
2026-03-05 10:01:00,120 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_output","request_id":"req-e","stage":"docs_only.synthesis","latency_ms":13.3}
"""
    raw_dir = run_dir / "logs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "baseline.log").write_text(baseline_trace, encoding="utf-8")
    (raw_dir / "exp2.log").write_text(exp_trace, encoding="utf-8")

    try:
        outputs = generate_report(run_dir)
        assert Path(outputs["summary_csv"]).exists()
        assert Path(outputs["stage_metrics_csv"]).exists()
        assert Path(outputs["report_html"]).exists()
        assert Path(outputs["logs_report_html"]).exists()

        prompt_diff = run_dir / "logs" / "diffs" / "case_a" / "baseline__vs__exp2__rep1_prompts.html"
        assert prompt_diff.exists()
        report_html = (run_dir / "report.html").read_text(encoding="utf-8")
        assert "diff prompts" in report_html
        run_logs_html = run_dir / "logs" / "html" / "baseline" / "case_a" / "1.html"
        assert run_logs_html.exists()
        run_logs = run_logs_html.read_text(encoding="utf-8")
        assert "Prompts Por Fase" in run_logs
        assert "Retrieval (Solo contenido + score)" in run_logs
    finally:
        for path in sorted(run_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
