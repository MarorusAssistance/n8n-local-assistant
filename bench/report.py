from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from difflib import HtmlDiff
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .io import ensure_dir, percentile, read_jsonl, write_json, write_text
from .trace import (
    compute_log_metrics,
    entries_for_request,
    extract_prompt_rows,
    extract_retrieval_views,
    parse_trace_file,
    redact_text_light,
    sort_entries,
)


@dataclass(frozen=True)
class SummaryRow:
    experiment: str
    runs_count: int
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    compliance_p50: float
    compliance_p95: float


@dataclass(frozen=True)
class LogSummaryRow:
    experiment: str
    runs_with_logs: int
    llm_calls_p50: float
    prompt_chars_p50: float
    retrieval_chunks_p50: float
    linked_node_chunks_p50: float
    linked_credential_chunks_p50: float
    budget_trim_rate: float
    docs_fallback_rate: float
    reasoning_second_iteration_rate: float


def generate_report(run_dir: Path) -> Dict[str, str]:
    run_dir = run_dir.resolve()
    runs = read_jsonl(run_dir / "runs.jsonl")
    meta = _load_run_meta(run_dir)
    experiment_order = meta.get("experiments") or _discover_experiment_order(runs)

    summary_rows = _build_summary_rows(runs, experiment_order)
    summary_csv_path = run_dir / "summary.csv"
    _write_summary_csv(summary_csv_path, summary_rows)

    workflow_diff_index = _generate_workflow_diffs(run_dir, runs, experiment_order)
    log_outputs = _generate_log_artifacts(run_dir, runs, experiment_order, meta)
    prompt_diff_index = _generate_prompt_diffs(
        run_dir=run_dir,
        runs=runs,
        experiment_order=experiment_order,
        per_run_log_artifacts=log_outputs["per_run_artifacts"],
    )

    merged_runs = _attach_log_artifacts(
        runs=runs,
        per_run_log_artifacts=log_outputs["per_run_artifacts"],
        prompt_diff_index=prompt_diff_index,
    )
    _write_runs_jsonl(run_dir / "runs.jsonl", merged_runs)

    report_html_path = run_dir / "report.html"
    report_html = _render_report_html(
        runs=merged_runs,
        summary_rows=summary_rows,
        experiment_order=experiment_order,
        workflow_diff_index=workflow_diff_index,
        prompt_diff_index=prompt_diff_index,
    )
    write_text(report_html_path, report_html)

    logs_report_path = run_dir / "logs" / "report.html"
    logs_report_html = _render_logs_report_html(
        runs=merged_runs,
        experiment_order=experiment_order,
        per_run_log_metrics=log_outputs["per_run_metrics"],
        prompt_diff_index=prompt_diff_index,
    )
    write_text(logs_report_path, logs_report_html)

    return {
        "summary_csv": str(summary_csv_path),
        "report_html": str(report_html_path),
        "logs_report_html": str(logs_report_path),
    }


def _load_run_meta(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _discover_experiment_order(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = set()
    order: List[str] = []
    for row in runs:
        name = str(row.get("experiment") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        order.append(name)
    return order


def _build_summary_rows(
    runs: Sequence[Mapping[str, Any]],
    experiment_order: Sequence[str],
) -> List[SummaryRow]:
    valid_runs = [row for row in runs if not bool(row.get("is_warmup", False))]
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in valid_runs:
        grouped[str(row.get("experiment") or "")].append(row)

    rows: List[SummaryRow] = []
    ordered_names = list(experiment_order) or sorted(grouped.keys())
    for experiment in ordered_names:
        exp_runs = grouped.get(experiment, [])
        if not exp_runs:
            rows.append(
                SummaryRow(
                    experiment=experiment,
                    runs_count=0,
                    error_rate=0.0,
                    latency_p50_ms=0.0,
                    latency_p95_ms=0.0,
                    compliance_p50=0.0,
                    compliance_p95=0.0,
                )
            )
            continue

        error_count = sum(1 for row in exp_runs if not bool(row.get("success", False)))
        latencies = [float(row.get("latency_ms") or 0.0) for row in exp_runs]
        compliances = [float(row.get("compliance_score") or 0.0) for row in exp_runs]
        rows.append(
            SummaryRow(
                experiment=experiment,
                runs_count=len(exp_runs),
                error_rate=round(error_count / max(1, len(exp_runs)), 6),
                latency_p50_ms=round(percentile(latencies, 0.5), 3),
                latency_p95_ms=round(percentile(latencies, 0.95), 3),
                compliance_p50=round(percentile(compliances, 0.5), 3),
                compliance_p95=round(percentile(compliances, 0.95), 3),
            )
        )
    return rows


def _write_summary_csv(path: Path, rows: Sequence[SummaryRow]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "runs_count",
                "error_rate",
                "latency_p50_ms",
                "latency_p95_ms",
                "compliance_p50",
                "compliance_p95",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment": row.experiment,
                    "runs_count": row.runs_count,
                    "error_rate": row.error_rate,
                    "latency_p50_ms": row.latency_p50_ms,
                    "latency_p95_ms": row.latency_p95_ms,
                    "compliance_p50": row.compliance_p50,
                    "compliance_p95": row.compliance_p95,
                }
            )


def _generate_workflow_diffs(
    run_dir: Path,
    runs: Sequence[Mapping[str, Any]],
    experiment_order: Sequence[str],
) -> Dict[Tuple[str, str], str]:
    if len(experiment_order) < 2:
        return {}

    baseline = experiment_order[0]
    rep1_runs: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in runs:
        if bool(row.get("is_warmup", False)):
            continue
        if int(row.get("rep") or 0) != 1:
            continue
        key = (str(row.get("case_id") or ""), str(row.get("experiment") or ""))
        if key[0] and key[1]:
            rep1_runs[key] = row

    diff_index: Dict[Tuple[str, str], str] = {}
    cases = sorted({case_id for case_id, _ in rep1_runs.keys()})
    for case_id in cases:
        baseline_run = rep1_runs.get((case_id, baseline))
        if not baseline_run:
            continue
        baseline_workflow = _load_workflow_text(run_dir, baseline_run)
        if baseline_workflow is None:
            continue

        for experiment in experiment_order[1:]:
            current_run = rep1_runs.get((case_id, experiment))
            if not current_run:
                continue
            current_workflow = _load_workflow_text(run_dir, current_run)
            if current_workflow is None:
                continue

            diff_html = HtmlDiff(tabsize=2, wrapcolumn=120).make_file(
                baseline_workflow.splitlines(),
                current_workflow.splitlines(),
                fromdesc=f"{baseline} / {case_id} / rep1",
                todesc=f"{experiment} / {case_id} / rep1",
                context=True,
                numlines=2,
            )
            diff_path = (
                run_dir
                / "diffs"
                / case_id
                / f"{baseline}__vs__{experiment}__rep1.html"
            )
            write_text(diff_path, diff_html)
            diff_index[(case_id, experiment)] = diff_path.relative_to(run_dir).as_posix()
    return diff_index


def _load_workflow_text(run_dir: Path, row: Mapping[str, Any]) -> str | None:
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    workflow_rel = artifacts.get("workflow")
    if not workflow_rel:
        return None
    workflow_path = run_dir / str(workflow_rel)
    if not workflow_path.exists():
        return None
    return workflow_path.read_text(encoding="utf-8")


def _generate_log_artifacts(
    run_dir: Path,
    runs: Sequence[Mapping[str, Any]],
    experiment_order: Sequence[str],
    meta: Mapping[str, Any],
) -> Dict[str, Any]:
    experiment_configs = meta.get("experiment_configs", {})
    trace_capture = meta.get("trace_capture", {})
    raw_files: Dict[str, str] = {}
    if isinstance(trace_capture, Mapping):
        maybe_raw = trace_capture.get("raw_files")
        if isinstance(maybe_raw, Mapping):
            raw_files = {str(k): str(v) for k, v in maybe_raw.items()}

    parsed_by_experiment: Dict[str, List[Any]] = {}
    for experiment in experiment_order:
        rel = raw_files.get(experiment, f"logs/raw/{experiment}.log")
        path = run_dir / rel
        parsed_by_experiment[experiment] = parse_trace_file(path)

    per_run_artifacts: Dict[Tuple[str, str, int, bool], Dict[str, Optional[str]]] = {}
    per_run_metrics: Dict[Tuple[str, str, int, bool], Dict[str, Any]] = {}

    for row in runs:
        experiment = str(row.get("experiment") or "")
        case_id = str(row.get("case_id") or "")
        rep = int(row.get("rep") or 0)
        is_warmup = bool(row.get("is_warmup", False))
        if not experiment or not case_id or rep <= 0:
            continue

        key = (experiment, case_id, rep, is_warmup)
        rep_label = f"warmup-{rep}" if is_warmup else str(rep)
        request_id = str(row.get("trace_request_id") or "") or None
        conversation_id = str(row.get("conversation_id") or "") or None
        adapter = "direct"
        if isinstance(experiment_configs, Mapping):
            cfg = experiment_configs.get(experiment, {})
            if isinstance(cfg, Mapping):
                adapter = str(cfg.get("adapter") or "direct")

        entries = entries_for_request(
            parsed_by_experiment.get(experiment, []),
            request_id=request_id,
            conversation_id=conversation_id,
        )
        metrics = compute_log_metrics(entries)
        per_run_metrics[key] = metrics

        if not entries:
            per_run_artifacts[key] = {
                "logs_html": None,
                "trace_events": None,
                "trace_slice": None,
                "prompts": None,
                "logs_note": (
                    "logs_unavailable_http_adapter" if adapter == "http" else "no_trace_entries"
                ),
            }
            continue

        slice_path = run_dir / "logs" / "slices" / experiment / case_id / f"{rep_label}.log"
        event_path = run_dir / "logs" / "events" / experiment / case_id / f"{rep_label}.json"
        prompts_path = run_dir / "logs" / "prompts" / experiment / case_id / f"{rep_label}.json"
        html_path = run_dir / "logs" / "html" / experiment / case_id / f"{rep_label}.html"

        raw_slice = "\n".join(entry.raw for entry in sort_entries(entries)) + "\n"
        write_text(slice_path, redact_text_light(raw_slice))
        prompt_rows = extract_prompt_rows(entries)
        serialized_events = [_trace_entry_to_json(entry) for entry in sort_entries(entries)]
        write_json(event_path, serialized_events)
        write_json(prompts_path, prompt_rows)
        write_text(
            html_path,
            _render_run_logs_html(
                run_row=row,
                entries=sort_entries(entries),
                prompt_rows=prompt_rows,
                metrics=metrics,
            ),
        )

        per_run_artifacts[key] = {
            "logs_html": html_path.relative_to(run_dir).as_posix(),
            "trace_events": event_path.relative_to(run_dir).as_posix(),
            "trace_slice": slice_path.relative_to(run_dir).as_posix(),
            "prompts": prompts_path.relative_to(run_dir).as_posix(),
            "logs_note": None,
        }

    return {
        "per_run_artifacts": per_run_artifacts,
        "per_run_metrics": per_run_metrics,
    }


def _trace_entry_to_json(entry: Any) -> Dict[str, Any]:
    return {
        "timestamp": entry.timestamp,
        "level": entry.level,
        "logger": entry.logger,
        "request_id": entry.request_id,
        "event_type": entry.event_type,
        "stage": entry.stage,
        "payload": entry.payload,
        "message": redact_text_light(entry.message),
    }


def _generate_prompt_diffs(
    *,
    run_dir: Path,
    runs: Sequence[Mapping[str, Any]],
    experiment_order: Sequence[str],
    per_run_log_artifacts: Mapping[Tuple[str, str, int, bool], Mapping[str, Optional[str]]],
) -> Dict[Tuple[str, str], str]:
    if len(experiment_order) < 2:
        return {}

    baseline = experiment_order[0]
    rep1_runs: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for row in runs:
        if bool(row.get("is_warmup", False)):
            continue
        if int(row.get("rep") or 0) != 1:
            continue
        case_id = str(row.get("case_id") or "")
        experiment = str(row.get("experiment") or "")
        if case_id and experiment:
            rep1_runs[(case_id, experiment)] = row

    diff_index: Dict[Tuple[str, str], str] = {}
    case_ids = sorted({case_id for case_id, _ in rep1_runs.keys()})
    for case_id in case_ids:
        base_prompts_rel = per_run_log_artifacts.get((baseline, case_id, 1, False), {}).get("prompts")
        if not base_prompts_rel:
            continue
        base_lines = _prompt_lines(run_dir / base_prompts_rel)
        if not base_lines:
            continue

        for experiment in experiment_order[1:]:
            prompts_rel = per_run_log_artifacts.get((experiment, case_id, 1, False), {}).get("prompts")
            if not prompts_rel:
                continue
            current_lines = _prompt_lines(run_dir / prompts_rel)
            if not current_lines:
                continue

            diff_html = HtmlDiff(tabsize=2, wrapcolumn=140).make_file(
                base_lines,
                current_lines,
                fromdesc=f"{baseline} / {case_id} / rep1 prompts",
                todesc=f"{experiment} / {case_id} / rep1 prompts",
                context=True,
                numlines=2,
            )
            diff_path = (
                run_dir
                / "logs"
                / "diffs"
                / case_id
                / f"{baseline}__vs__{experiment}__rep1_prompts.html"
            )
            write_text(diff_path, diff_html)
            diff_index[(case_id, experiment)] = diff_path.relative_to(run_dir).as_posix()
    return diff_index


def _prompt_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    lines: List[str] = []
    if not isinstance(payload, list):
        return lines
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        stage = str(row.get("stage") or "unknown")
        role = str(row.get("role") or "unknown")
        content = str(row.get("content") or "")
        lines.append(f"[{stage}] {role}: {content}")
    return lines


def _attach_log_artifacts(
    *,
    runs: Sequence[Mapping[str, Any]],
    per_run_log_artifacts: Mapping[Tuple[str, str, int, bool], Mapping[str, Optional[str]]],
    prompt_diff_index: Mapping[Tuple[str, str], str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in runs:
        exp = str(row.get("experiment") or "")
        case_id = str(row.get("case_id") or "")
        rep = int(row.get("rep") or 0)
        is_warmup = bool(row.get("is_warmup", False))
        key = (exp, case_id, rep, is_warmup)
        log_artifacts = per_run_log_artifacts.get(key, {})

        cloned = dict(row)
        artifacts = dict(cloned.get("artifacts") or {})
        artifacts["logs_html"] = log_artifacts.get("logs_html")
        artifacts["trace_events"] = log_artifacts.get("trace_events")
        artifacts["trace_slice"] = log_artifacts.get("trace_slice")
        artifacts["prompts"] = log_artifacts.get("prompts")
        artifacts["logs_note"] = log_artifacts.get("logs_note")
        if rep == 1 and not is_warmup:
            prompt_diff = prompt_diff_index.get((case_id, exp))
            if prompt_diff:
                artifacts["prompt_diff"] = prompt_diff
        cloned["artifacts"] = artifacts
        output.append(cloned)
    return output


def _write_runs_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_report_html(
    *,
    runs: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[SummaryRow],
    experiment_order: Sequence[str],
    workflow_diff_index: Mapping[Tuple[str, str], str],
    prompt_diff_index: Mapping[Tuple[str, str], str],
) -> str:
    non_warmup = [row for row in runs if not bool(row.get("is_warmup", False))]
    cases = sorted({str(row.get("case_id") or "") for row in non_warmup if row.get("case_id")})
    by_case_exp: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in non_warmup:
        case_id = str(row.get("case_id") or "")
        experiment = str(row.get("experiment") or "")
        if not case_id or not experiment:
            continue
        by_case_exp[(case_id, experiment)].append(row)

    head = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bench Report</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #111; }
    h1, h2 { margin: 0 0 12px 0; }
    h2 { margin-top: 28px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 18px; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; }
    code { background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }
    .muted { color: #555; font-size: 12px; }
    .ok { color: #0a7c2f; font-weight: 600; }
    .bad { color: #b42318; font-weight: 600; }
  </style>
</head>
<body>
"""
    parts: List[str] = [head, "<h1>Phase 1 Quality Harness Report</h1>"]
    parts.append(f"<p class='muted'>Total runs (no warmup): {len(non_warmup)}</p>")
    parts.append("<p class='muted'><a href='logs/report.html'>Open logs comparison report</a></p>")

    parts.append("<h2>Summary by Experiment</h2>")
    parts.append("<table>")
    parts.append(
        "<tr>"
        "<th>experiment</th><th>runs_count</th><th>error_rate</th>"
        "<th>latency_p50_ms</th><th>latency_p95_ms</th>"
        "<th>compliance_p50</th><th>compliance_p95</th>"
        "</tr>"
    )
    for row in summary_rows:
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(row.experiment)}</code></td>"
            f"<td>{row.runs_count}</td>"
            f"<td>{row.error_rate:.3f}</td>"
            f"<td>{row.latency_p50_ms:.3f}</td>"
            f"<td>{row.latency_p95_ms:.3f}</td>"
            f"<td>{row.compliance_p50:.3f}</td>"
            f"<td>{row.compliance_p95:.3f}</td>"
            "</tr>"
        )
    parts.append("</table>")

    for case_id in cases:
        parts.append(f"<h2>Case: <code>{html.escape(case_id)}</code></h2>")
        parts.append("<table>")
        parts.append(
            "<tr>"
            "<th>experiment</th><th>runs</th><th>errors</th><th>latency_p50_ms</th>"
            "<th>compliance_p50</th><th>rep1 success</th><th>artifacts</th>"
            "<th>workflow diff</th><th>prompt diff</th>"
            "</tr>"
        )
        for experiment in experiment_order:
            rows = sorted(
                by_case_exp.get((case_id, experiment), []),
                key=lambda row: int(row.get("rep") or 0),
            )
            if not rows:
                parts.append(
                    "<tr>"
                    f"<td><code>{html.escape(experiment)}</code></td>"
                    "<td colspan='8' class='muted'>no runs</td>"
                    "</tr>"
                )
                continue
            latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
            compliances = [float(row.get("compliance_score") or 0.0) for row in rows]
            errors = sum(1 for row in rows if not bool(row.get("success", False)))
            rep1 = rows[0]
            artifacts_html = _artifacts_html(rep1)
            rep1_success = bool(rep1.get("success", False))
            rep1_css = "ok" if rep1_success else "bad"
            rep1_label = "ok" if rep1_success else "error"

            workflow_diff_rel = workflow_diff_index.get((case_id, experiment))
            workflow_diff_html = "-"
            if workflow_diff_rel:
                workflow_diff_html = f"<a href='{html.escape(workflow_diff_rel)}'>diff rep1</a>"

            prompt_diff_rel = prompt_diff_index.get((case_id, experiment))
            prompt_diff_html = "-"
            if prompt_diff_rel:
                prompt_diff_html = f"<a href='{html.escape(prompt_diff_rel)}'>diff prompts</a>"

            parts.append(
                "<tr>"
                f"<td><code>{html.escape(experiment)}</code></td>"
                f"<td>{len(rows)}</td>"
                f"<td>{errors}</td>"
                f"<td>{percentile(latencies, 0.5):.3f}</td>"
                f"<td>{percentile(compliances, 0.5):.3f}</td>"
                f"<td class='{rep1_css}'>{rep1_label}</td>"
                f"<td>{artifacts_html}</td>"
                f"<td>{workflow_diff_html}</td>"
                f"<td>{prompt_diff_html}</td>"
                "</tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _artifacts_html(run_row: Mapping[str, Any]) -> str:
    artifacts = run_row.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return "-"
    links: List[str] = []
    for key in ("response", "workflow", "checks", "logs_html", "trace_events", "prompts"):
        rel = artifacts.get(key)
        if not rel:
            continue
        rel_text = html.escape(str(rel))
        links.append(f"<a href='{rel_text}'>{key}</a>")
    note = artifacts.get("logs_note")
    if note:
        links.append(html.escape(str(note)))
    return " | ".join(links) if links else "-"


def _build_log_summary_rows(
    runs: Sequence[Mapping[str, Any]],
    experiment_order: Sequence[str],
    per_run_log_metrics: Mapping[Tuple[str, str, int, bool], Mapping[str, Any]],
) -> List[LogSummaryRow]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[str(row.get("experiment") or "")].append(row)

    rows: List[LogSummaryRow] = []
    for experiment in experiment_order:
        exp_runs = grouped.get(experiment, [])
        metrics = [
            per_run_log_metrics.get((experiment, str(row.get("case_id") or ""), int(row.get("rep") or 0), False), {})
            for row in exp_runs
        ]
        metrics = [item for item in metrics if item]
        if not metrics:
            rows.append(
                LogSummaryRow(
                    experiment=experiment,
                    runs_with_logs=0,
                    llm_calls_p50=0.0,
                    prompt_chars_p50=0.0,
                    retrieval_chunks_p50=0.0,
                    linked_node_chunks_p50=0.0,
                    linked_credential_chunks_p50=0.0,
                    budget_trim_rate=0.0,
                    docs_fallback_rate=0.0,
                    reasoning_second_iteration_rate=0.0,
                )
            )
            continue

        llm_calls = [float(item.get("llm_calls") or 0.0) for item in metrics]
        prompt_chars = [float(item.get("prompt_chars") or 0.0) for item in metrics]
        retrieval_chunks = [float(item.get("retrieval_chunks") or 0.0) for item in metrics]
        linked_node_chunks = [float(item.get("linked_node_chunks") or 0.0) for item in metrics]
        linked_credential_chunks = [float(item.get("linked_credential_chunks") or 0.0) for item in metrics]
        budget_trim_rate = (
            sum(1 for item in metrics if float(item.get("budget_trim_count") or 0.0) > 0)
            / max(1, len(metrics))
        )
        docs_fallback_rate = (
            sum(1 for item in metrics if float(item.get("docs_fallback_count") or 0.0) > 0)
            / max(1, len(metrics))
        )
        reasoning_second_iteration_rate = (
            sum(1 for item in metrics if bool(item.get("reasoning_second_iteration")))
            / max(1, len(metrics))
        )
        rows.append(
            LogSummaryRow(
                experiment=experiment,
                runs_with_logs=len(metrics),
                llm_calls_p50=round(percentile(llm_calls, 0.5), 3),
                prompt_chars_p50=round(percentile(prompt_chars, 0.5), 3),
                retrieval_chunks_p50=round(percentile(retrieval_chunks, 0.5), 3),
                linked_node_chunks_p50=round(percentile(linked_node_chunks, 0.5), 3),
                linked_credential_chunks_p50=round(percentile(linked_credential_chunks, 0.5), 3),
                budget_trim_rate=round(budget_trim_rate, 6),
                docs_fallback_rate=round(docs_fallback_rate, 6),
                reasoning_second_iteration_rate=round(reasoning_second_iteration_rate, 6),
            )
        )
    return rows


def _render_logs_report_html(
    *,
    runs: Sequence[Mapping[str, Any]],
    experiment_order: Sequence[str],
    per_run_log_metrics: Mapping[Tuple[str, str, int, bool], Mapping[str, Any]],
    prompt_diff_index: Mapping[Tuple[str, str], str],
) -> str:
    non_warmup_runs = [row for row in runs if not bool(row.get("is_warmup", False))]
    summary_rows = _build_log_summary_rows(non_warmup_runs, experiment_order, per_run_log_metrics)

    by_case_exp: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in non_warmup_runs:
        case_id = str(row.get("case_id") or "")
        experiment = str(row.get("experiment") or "")
        if case_id and experiment:
            by_case_exp[(case_id, experiment)].append(row)
    case_ids = sorted({str(row.get("case_id") or "") for row in non_warmup_runs if row.get("case_id")})

    head = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bench Logs Report</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #111; }
    h1, h2 { margin: 0 0 12px 0; }
    h2 { margin-top: 28px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 18px; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; }
    code { background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }
    .muted { color: #555; font-size: 12px; }
  </style>
</head>
<body>
"""
    parts: List[str] = [head, "<h1>Bench Logs Comparison</h1>"]
    parts.append("<p class='muted'>Log-derived metrics over non-warmup runs.</p>")
    parts.append("<h2>Summary by Experiment</h2>")
    parts.append("<table>")
    parts.append(
        "<tr>"
        "<th>experiment</th><th>runs_with_logs</th><th>llm_calls_p50</th>"
        "<th>prompt_chars_p50</th><th>retrieval_chunks_p50</th><th>linked_node_chunks_p50</th>"
        "<th>linked_credential_chunks_p50</th><th>budget_trim_rate</th><th>docs_fallback_rate</th>"
        "<th>reasoning_second_iteration_rate</th>"
        "</tr>"
    )
    for row in summary_rows:
        parts.append(
            "<tr>"
            f"<td><code>{html.escape(row.experiment)}</code></td>"
            f"<td>{row.runs_with_logs}</td>"
            f"<td>{row.llm_calls_p50:.3f}</td>"
            f"<td>{row.prompt_chars_p50:.3f}</td>"
            f"<td>{row.retrieval_chunks_p50:.3f}</td>"
            f"<td>{row.linked_node_chunks_p50:.3f}</td>"
            f"<td>{row.linked_credential_chunks_p50:.3f}</td>"
            f"<td>{row.budget_trim_rate:.3f}</td>"
            f"<td>{row.docs_fallback_rate:.3f}</td>"
            f"<td>{row.reasoning_second_iteration_rate:.3f}</td>"
            "</tr>"
        )
    parts.append("</table>")

    for case_id in case_ids:
        parts.append(f"<h2>Case: <code>{html.escape(case_id)}</code></h2>")
        parts.append("<table>")
        parts.append(
            "<tr>"
            "<th>experiment</th><th>llm_calls_p50</th><th>prompt_chars_p50</th>"
            "<th>retrieval_chunks_p50</th><th>budget_trim_rate</th><th>docs_fallback_rate</th>"
            "<th>rep1 logs</th><th>prompt diff vs baseline</th>"
            "</tr>"
        )
        for experiment in experiment_order:
            rows = sorted(
                by_case_exp.get((case_id, experiment), []),
                key=lambda item: int(item.get("rep") or 0),
            )
            if not rows:
                parts.append(
                    "<tr>"
                    f"<td><code>{html.escape(experiment)}</code></td>"
                    "<td colspan='7' class='muted'>no runs</td>"
                    "</tr>"
                )
                continue

            metrics = [
                per_run_log_metrics.get((experiment, case_id, int(row.get("rep") or 0), False), {})
                for row in rows
            ]
            llm_calls = [float(item.get("llm_calls") or 0.0) for item in metrics]
            prompt_chars = [float(item.get("prompt_chars") or 0.0) for item in metrics]
            retrieval_chunks = [float(item.get("retrieval_chunks") or 0.0) for item in metrics]
            budget_trim_rate = (
                sum(1 for item in metrics if float(item.get("budget_trim_count") or 0.0) > 0)
                / max(1, len(metrics))
            )
            docs_fallback_rate = (
                sum(1 for item in metrics if float(item.get("docs_fallback_count") or 0.0) > 0)
                / max(1, len(metrics))
            )

            rep1 = rows[0]
            artifacts = rep1.get("artifacts") if isinstance(rep1.get("artifacts"), Mapping) else {}
            logs_html_rel = artifacts.get("logs_html") if isinstance(artifacts, Mapping) else None
            logs_html = "-"
            if logs_html_rel:
                logs_html = f"<a href='../{html.escape(str(logs_html_rel))}'>rep1 logs</a>"

            prompt_diff_rel = prompt_diff_index.get((case_id, experiment))
            prompt_diff_html = "-"
            if prompt_diff_rel:
                prompt_diff_html = f"<a href='../{html.escape(prompt_diff_rel)}'>prompt diff</a>"

            parts.append(
                "<tr>"
                f"<td><code>{html.escape(experiment)}</code></td>"
                f"<td>{percentile(llm_calls, 0.5):.3f}</td>"
                f"<td>{percentile(prompt_chars, 0.5):.3f}</td>"
                f"<td>{percentile(retrieval_chunks, 0.5):.3f}</td>"
                f"<td>{budget_trim_rate:.3f}</td>"
                f"<td>{docs_fallback_rate:.3f}</td>"
                f"<td>{logs_html}</td>"
                f"<td>{prompt_diff_html}</td>"
                "</tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _render_run_logs_html(
    *,
    run_row: Mapping[str, Any],
    entries: Sequence[Any],
    prompt_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> str:
    experiment = html.escape(str(run_row.get("experiment") or ""))
    case_id = html.escape(str(run_row.get("case_id") or ""))
    rep = html.escape(str(run_row.get("rep") or ""))
    request_id = html.escape(str(run_row.get("trace_request_id") or ""))
    conversation_id = html.escape(str(run_row.get("conversation_id") or ""))
    latency = html.escape(str(run_row.get("latency_ms") or ""))
    success = bool(run_row.get("success", False))
    retrieval_views = extract_retrieval_views(entries)

    filtered_prompt_rows = [
        row
        for row in prompt_rows
        if str(row.get("stage") or "") not in ("request.input", "response.output")
    ]
    if not filtered_prompt_rows:
        filtered_prompt_rows = list(prompt_rows)

    prompts_by_stage: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in filtered_prompt_rows:
        stage = str(row.get("stage") or "unknown")
        role = str(row.get("role") or "unknown")
        prompts_by_stage[stage][role].append(row)

    prompt_sections: List[str] = []
    role_order = ["system", "user", "assistant"]
    for stage in sorted(prompts_by_stage.keys()):
        role_map = prompts_by_stage[stage]
        section_parts = [
            "<section class='card'>",
            f"<h3><code>{html.escape(stage)}</code></h3>",
        ]
        known_roles = [r for r in role_order if r in role_map]
        other_roles = sorted(role for role in role_map.keys() if role not in role_order)
        for role in known_roles + other_roles:
            rows = role_map[role]
            section_parts.append(
                f"<h4>{html.escape(role)} <span class='muted'>({len(rows)})</span></h4>"
            )
            for item in rows:
                content = str(item.get("content") or "")
                section_parts.append("<article class='message'>")
                section_parts.append(
                    f"<div class='meta'>{html.escape(str(item.get('timestamp') or ''))}</div>"
                )
                section_parts.append(_render_prompt_content(content))
                section_parts.append("</article>")
        section_parts.append("</section>")
        prompt_sections.append("\n".join(section_parts))
    if not prompt_sections:
        prompt_sections.append("<p class='muted'>No prompt events found.</p>")

    retrieval_sections: List[str] = []
    retrieval_sections.append(
        _render_retrieval_card(
            "Pre-rerank: API docs",
            retrieval_views.get("pre_docs", []),
        )
    )
    retrieval_sections.append(
        _render_retrieval_card(
            "Pre-rerank: linked node defs (nodes.json)",
            retrieval_views.get("pre_linked_node", []),
        )
    )
    retrieval_sections.append(
        _render_retrieval_card(
            "Pre-rerank: linked credential defs (credentials.json)",
            retrieval_views.get("pre_linked_credential", []),
        )
    )
    retrieval_sections.append(
        _render_retrieval_card(
            "Post-rerank: API docs",
            retrieval_views.get("post_docs", []),
        )
    )
    retrieval_sections.append(
        _render_retrieval_card(
            "Post-rerank: linked node defs (nodes.json)",
            retrieval_views.get("post_linked_node", []),
        )
    )
    retrieval_sections.append(
        _render_retrieval_card(
            "Post-rerank: linked credential defs (credentials.json)",
            retrieval_views.get("post_linked_credential", []),
        )
    )

    success_label = "ok" if success else "error"
    success_css = "ok" if success else "bad"

    return (
        "<!doctype html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='utf-8' />\n"
        "  <title>Bench Trace View</title>\n"
        "  <style>\n"
        "    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; color: #111; background: #f4f6f8; }\n"
        "    .wrap { max-width: 1320px; margin: 0 auto; padding: 24px; }\n"
        "    h1, h2, h3, h4 { margin: 0; }\n"
        "    h1 { font-size: 24px; margin-bottom: 8px; }\n"
        "    h2 { margin: 28px 0 12px 0; font-size: 18px; }\n"
        "    h3 { margin-bottom: 8px; font-size: 15px; }\n"
        "    h4 { margin: 10px 0 6px 0; font-size: 13px; }\n"
        "    code { background: #eef2f6; padding: 2px 6px; border-radius: 6px; }\n"
        "    .meta-grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 10px; margin: 10px 0 18px 0; }\n"
        "    .meta-item { background: #fff; border: 1px solid #d8dde3; border-radius: 10px; padding: 10px; font-size: 12px; }\n"
        "    .muted { color: #5f6b76; font-size: 12px; }\n"
        "    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }\n"
        "    .card { background: #fff; border: 1px solid #d8dde3; border-radius: 12px; padding: 12px; }\n"
        "    .message { border: 1px solid #e7ebef; border-radius: 8px; padding: 8px; margin-bottom: 8px; background: #fbfcfd; }\n"
        "    .message .meta { font-size: 11px; color: #64707c; margin-bottom: 6px; }\n"
        "    .preview { border-left: 3px solid #dce4eb; padding-left: 8px; }\n"
        "    details { margin-top: 8px; }\n"
        "    summary { cursor: pointer; color: #0f5b99; font-size: 12px; }\n"
        "    pre { white-space: pre-wrap; margin: 0; font-family: Consolas, monospace; font-size: 12px; line-height: 1.45; }\n"
        "    table { border-collapse: collapse; width: 100%; margin-top: 8px; }\n"
        "    th, td { border: 1px solid #e1e6eb; padding: 8px; font-size: 12px; vertical-align: top; }\n"
        "    th { background: #f7f9fb; text-align: left; }\n"
        "    .score { width: 140px; white-space: nowrap; }\n"
        "    .ok { color: #0a7c2f; font-weight: 600; }\n"
        "    .bad { color: #b42318; font-weight: 600; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "<div class='wrap'>\n"
        "<h1>Bench Trace View</h1>\n"
        "<p class='muted'>Prompts organizados por fase de reasoning y retrieval pre/post-rerank.</p>\n"
        "<div class='meta-grid'>\n"
        f"<div class='meta-item'>experiment<br><code>{experiment}</code></div>"
        f"<div class='meta-item'>case_id<br><code>{case_id}</code></div>"
        f"<div class='meta-item'>rep<br><code>{rep}</code></div>"
        f"<div class='meta-item'>request_id<br><code>{request_id}</code></div>"
        f"<div class='meta-item'>conversation_id<br><code>{conversation_id}</code></div>"
        f"<div class='meta-item'>latency/status<br><code>{latency}</code> ms | <span class='{success_css}'>{success_label}</span></div>"
        "</div>\n"
        "<h2>Prompts Por Fase</h2>\n"
        + "\n".join(prompt_sections)
        + "\n<h2>Retrieval (Solo contenido + score)</h2>\n"
        "<div class='grid'>\n"
        + "\n".join(retrieval_sections)
        + "\n</div>\n"
        "<p class='muted'>metrics: "
        f"llm_calls={metrics.get('llm_calls', 0)}, "
        f"retrieval_chunks={metrics.get('retrieval_chunks', 0)}, "
        f"budget_trim_count={metrics.get('budget_trim_count', 0)}, "
        f"docs_fallback_count={metrics.get('docs_fallback_count', 0)}"
        "</p>\n"
        "</div>\n"
        "</body></html>\n"
    )


def _render_retrieval_card(title: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return (
            "<section class='card'>"
            f"<h3>{html.escape(title)}</h3>"
            "<p class='muted'>No data.</p>"
            "</section>"
        )

    table_rows: List[str] = []
    for row in rows:
        score = row.get("score")
        score_label = str(row.get("score_label") or "-")
        if isinstance(score, (int, float)):
            score_text = f"{score_label}: {float(score):.4f}"
        else:
            score_text = "-"
        table_rows.append(
            "<tr>"
            f"<td class='score'>{html.escape(score_text)}</td>"
            f"<td><pre>{html.escape(str(row.get('content') or ''))}</pre></td>"
            "</tr>"
        )

    return (
        "<section class='card'>"
        f"<h3>{html.escape(title)}</h3>"
        "<table>"
        "<tr><th class='score'>score</th><th>content</th></tr>"
        + "".join(table_rows)
        + "</table>"
        "</section>"
    )


def _render_prompt_content(content: str) -> str:
    value = content or ""
    if len(value) <= 650:
        return f"<pre>{html.escape(value)}</pre>"
    preview = value[:420].rstrip() + " ... [recortado para vista]"
    return (
        f"<pre class='preview'>{html.escape(preview)}</pre>"
        "<details><summary>Ver prompt completo</summary>"
        f"<pre>{html.escape(value)}</pre>"
        "</details>"
    )
