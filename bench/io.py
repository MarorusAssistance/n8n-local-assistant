from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .models import CaseSpec, CasesFile, ExperimentSpec, ExperimentsFile


def load_structured_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)

    # Default: try YAML first, then JSON.
    try:
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def load_cases(path: Path) -> List[CaseSpec]:
    raw = load_structured_file(path)
    file_model = CasesFile.from_raw(raw)
    cases = file_model.cases
    if not cases:
        raise ValueError("cases file is empty")

    seen = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicated case id: {case.id}")
        seen.add(case.id)
    return cases


def load_experiments(path: Path) -> List[ExperimentSpec]:
    raw = load_structured_file(path)
    file_model = ExperimentsFile.from_raw(raw)
    experiments = file_model.experiments
    if not experiments:
        raise ValueError("experiments file is empty")

    seen = set()
    for exp in experiments:
        if exp.name in seen:
            raise ValueError(f"duplicated experiment name: {exp.name}")
        seen.add(exp.name)
    return experiments


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(git_sha: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{git_sha}"


def get_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        value = out.strip()
        return value or "nogit"
    except Exception:
        return "nogit"


def relpath_str(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(1.0, q))
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    weight_hi = idx - lo
    return vals[lo] * (1.0 - weight_hi) + vals[hi] * weight_hi
