from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from .io import ensure_dir, get_git_sha, load_experiments, make_run_id, write_json
from .report import generate_report


TRACE_PRIVACY_MODE = "light"


def run_benchmarks(
    *,
    cases_path: Path,
    experiments_path: Path,
    out_root: Path,
    quick: bool,
    quick_cases: int = 3,
    quick_repetitions: int = 1,
    http_base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Path:
    repo_root = Path.cwd().resolve()
    experiments = load_experiments(experiments_path)

    git_sha = get_git_sha(repo_root)
    run_id = make_run_id(git_sha)
    run_dir = ensure_dir(out_root / run_id)
    trace_raw_files = {
        exp.name: f"logs/raw/{exp.name}.log"
        for exp in experiments
    }

    write_json(
        run_dir / "run_meta.json",
        {
            "run_id": run_id,
            "git_sha": git_sha,
            "cases_path": str(cases_path),
            "experiments_path": str(experiments_path),
            "quick": quick,
            "experiments": [exp.name for exp in experiments],
            "experiment_configs": {
                exp.name: {
                    "adapter": exp.adapter,
                    "json_only": exp.json_only,
                    "warmup": exp.warmup,
                    "repetitions": exp.repetitions,
                }
                for exp in experiments
            },
            "trace_capture": {
                "enabled": True,
                "privacy_mode": TRACE_PRIVACY_MODE,
                "raw_files": trace_raw_files,
                "defaults_applied": {
                    "TRACE_LOG_ENABLED": "true",
                    "TRACE_LOG_LEVEL": "INFO",
                    "TRACE_LOG_FILE": "results/<run_id>/logs/raw/<experiment>.log",
                    "TRACE_LOG_MAX_BYTES": "200000000",
                    "TRACE_LOG_BACKUPS": "1",
                    "RETRIEVAL_DEBUG": "true",
                },
            },
        },
    )

    for experiment in experiments:
        _run_experiment_worker(
            experiment_name=experiment.name,
            env_overrides=experiment.env_overrides,
            cases_path=cases_path,
            experiments_path=experiments_path,
            run_dir=run_dir,
            quick=quick,
            quick_cases=quick_cases,
            quick_repetitions=quick_repetitions,
            http_base_url=http_base_url,
            model=model,
        )

    generate_report(run_dir)
    return run_dir


def regenerate_report(run_dir: Path) -> Dict[str, str]:
    return generate_report(run_dir.resolve())


def _run_experiment_worker(
    *,
    experiment_name: str,
    env_overrides: Dict[str, str],
    cases_path: Path,
    experiments_path: Path,
    run_dir: Path,
    quick: bool,
    quick_cases: int,
    quick_repetitions: int,
    http_base_url: Optional[str],
    model: Optional[str],
) -> None:
    trace_raw_file = (run_dir / "logs" / "raw" / f"{experiment_name}.log").resolve()
    ensure_dir(trace_raw_file.parent)
    trace_defaults = {
        "TRACE_LOG_ENABLED": "true",
        "TRACE_LOG_LEVEL": "INFO",
        "TRACE_LOG_FILE": str(trace_raw_file),
        "TRACE_LOG_MAX_BYTES": "200000000",
        "TRACE_LOG_BACKUPS": "1",
        "RETRIEVAL_DEBUG": "true",
    }

    command = [
        sys.executable,
        "-m",
        "bench.worker",
        "--cases",
        str(cases_path),
        "--experiments",
        str(experiments_path),
        "--experiment-name",
        experiment_name,
        "--run-dir",
        str(run_dir),
    ]
    if quick:
        command.extend(
            [
                "--quick",
                "--quick-cases",
                str(max(1, quick_cases)),
                "--quick-repetitions",
                str(max(1, quick_repetitions)),
            ]
        )
    if http_base_url:
        command.extend(["--http-base-url", http_base_url])
    if model:
        command.extend(["--model", model])

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in env_overrides.items()})
    for key, value in trace_defaults.items():
        env.setdefault(key, value)
    result = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"worker failed for experiment '{experiment_name}' "
            f"(exit={result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
