from __future__ import annotations

import argparse
from pathlib import Path

from .runner import regenerate_report, run_benchmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="Phase 1 Quality Harness for n8n workflow JSON generation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute benchmark experiments.")
    run_parser.add_argument("--cases", default="bench/cases.yaml")
    run_parser.add_argument("--experiments", default="bench/experiments.yaml")
    run_parser.add_argument("--out", default="bench/results")
    run_parser.add_argument("--quick", action="store_true")
    run_parser.add_argument("--quick-cases", type=int, default=3)
    run_parser.add_argument("--quick-repetitions", type=int, default=1)
    run_parser.add_argument("--http-base-url", default=None)
    run_parser.add_argument("--model", default=None)

    report_parser = subparsers.add_parser("report", help="Regenerate summary/report from a run dir.")
    report_parser.add_argument("--in", dest="run_dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_dir = run_benchmarks(
            cases_path=Path(args.cases).resolve(),
            experiments_path=Path(args.experiments).resolve(),
            out_root=Path(args.out).resolve(),
            quick=bool(args.quick),
            quick_cases=args.quick_cases,
            quick_repetitions=args.quick_repetitions,
            http_base_url=args.http_base_url,
            model=args.model,
        )
        print(f"Run completed: {run_dir}")
        print(f"Summary: {run_dir / 'summary.csv'}")
        print(f"Report: {run_dir / 'report.html'}")
        print(f"Logs report: {run_dir / 'logs' / 'report.html'}")
        return 0

    if args.command == "report":
        outputs = regenerate_report(Path(args.run_dir))
        print(f"Summary: {outputs['summary_csv']}")
        print(f"Report: {outputs['report_html']}")
        logs_path = outputs.get("logs_report_html")
        if logs_path:
            print(f"Logs report: {logs_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
