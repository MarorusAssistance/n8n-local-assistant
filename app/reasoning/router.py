from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from ..config import settings
from ..llm import chat_completion, resolve_model
from .types import RouterConstraints, RouterOutput


logger = logging.getLogger("n8n-assistant")

_TOKEN_RE = re.compile(r"[a-z0-9_\-]+", re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

_INTENT_FIX_HINTS = (
    "fix",
    "arregla",
    "arreglar",
    "falla",
    "error",
    "bug",
    "debug",
    "repair",
    "corregir",
    "broken",
    "issue",
)
_INTENT_EXTEND_HINTS = (
    "extiende",
    "extend",
    "agrega",
    "anade",
    "añade",
    "add",
    "suma",
    "append",
    "modifica",
    "modificar",
    "enhance",
)
_INTENT_CREATE_HINTS = (
    "crea",
    "crear",
    "create",
    "new workflow",
    "nuevo workflow",
    "construye",
    "build",
)

_TRIGGER_HINTS = (
    "trigger",
    "webhook",
    "cron",
    "schedule",
    "scheduler",
    "manual trigger",
    "interval",
)

_SERVICE_PATTERNS: Dict[str, Sequence[str]] = {
    "Google Sheets": (r"\bgoogle\s+sheets?\b",),
    "Slack": (r"\bslack\b",),
    "Gmail": (r"\bgmail\b", r"\bgoogle\s+mail\b"),
    "Notion": (r"\bnotion\b",),
    "Airtable": (r"\bairtable\b",),
    "Postgres": (r"\bpostgres(?:ql)?\b",),
    "MySQL": (r"\bmysql\b",),
    "HTTP API": (r"\bhttp\b", r"\bapi\b", r"\bendpoint\b"),
    "Webhook": (r"\bwebhook\b",),
    "S3": (r"\bs3\b", r"\bamazon\s+s3\b"),
}

_NON_FUNCTIONAL_HINTS: Dict[str, Sequence[str]] = {
    "retry": ("retry", "reintento", "retries"),
    "idempotency": ("idempotent", "idempotencia", "idempotency"),
    "performance": ("performance", "latency", "throughput", "rendimiento"),
    "security": ("secure", "seguridad", "encryption", "encrypt", "auth hardening"),
    "observability": ("logging", "tracing", "monitoring", "observability"),
}

_BRANCHING_HINTS = ("if", "condicion", "condición", "switch", "branch", "route", "rama")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _contains_any(text: str, hints: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(hint in lowered for hint in hints)


def _detect_intent(text: str, existing_workflow: Any) -> str:
    lowered = (text or "").lower()
    if _contains_any(lowered, _INTENT_FIX_HINTS):
        return "fix"
    if _contains_any(lowered, _INTENT_EXTEND_HINTS):
        return "extend"
    if _contains_any(lowered, _INTENT_CREATE_HINTS):
        return "create"
    if existing_workflow is not None:
        return "extend"
    return "create"


def _extract_services(text: str) -> List[str]:
    lowered = (text or "").lower()
    services: List[str] = []
    for service, patterns in _SERVICE_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            services.append(service)
    return services


def _extract_inputs(text: str) -> List[str]:
    lowered = (text or "").lower()
    inputs: List[str] = []
    if "webhook" in lowered:
        inputs.append("webhook payload")
    if any(word in lowered for word in ("archivo", "file", "csv", "excel")):
        inputs.append("file input")
    if any(word in lowered for word in ("email", "correo")):
        inputs.append("email message")
    if any(word in lowered for word in ("form", "formulario")):
        inputs.append("form submission")
    return inputs


def _extract_outputs(text: str) -> List[str]:
    lowered = (text or "").lower()
    outputs: List[str] = []
    if "google sheets" in lowered:
        outputs.append("google sheets row(s)")
    if "slack" in lowered:
        outputs.append("slack message")
    if any(word in lowered for word in ("email", "correo")):
        outputs.append("email notification")
    if any(word in lowered for word in ("database", "db", "postgres", "mysql")):
        outputs.append("database write")
    return outputs


def _extract_non_functional(text: str) -> List[str]:
    lowered = (text or "").lower()
    tags: List[str] = []
    for name, hints in _NON_FUNCTIONAL_HINTS.items():
        if any(hint in lowered for hint in hints):
            tags.append(name)
    return tags


def _missing_info(intent: str, text: str, constraints: RouterConstraints) -> List[str]:
    missing: List[str] = []
    lowered = (text or "").lower()
    if intent == "create" and not _contains_any(lowered, _TRIGGER_HINTS):
        missing.append("No trigger specified (webhook/schedule/manual trigger).")
    if intent in {"create", "extend"} and not constraints.outputs:
        missing.append("Output destination is not explicit.")
    if intent in {"fix", "extend"} and "workflow" not in lowered and "node" not in lowered:
        missing.append("Target workflow or node is not explicit.")
    return missing


def _complexity_score(text: str, constraints: RouterConstraints) -> int:
    lowered = (text or "").lower()
    score = 1
    service_count = len(constraints.services)
    has_branching = any(token in lowered for token in _BRANCHING_HINTS)
    has_extra_logic = any(
        token in lowered
        for token in ("approval", "loop", "iterate", "batch", "parallel", "fallback")
    )

    if service_count >= 2 or has_branching:
        score = 2
    if (service_count >= 3 and has_branching) or has_extra_logic:
        score = 3
    return score


def _heuristic_router_output(user_prompt: str, existing_workflow: Any) -> RouterOutput:
    text = (user_prompt or "").strip()
    intent = _detect_intent(text, existing_workflow=existing_workflow)
    constraints = RouterConstraints(
        services=_extract_services(text),
        inputs=_extract_inputs(text),
        outputs=_extract_outputs(text),
        nonFunctional=_extract_non_functional(text),
    )
    return RouterOutput(
        intent=intent,  # type: ignore[arg-type]
        goal=text or "User needs a workflow plan.",
        constraints=constraints,
        missing_info=_missing_info(intent, text, constraints),
        complexity_score=_complexity_score(text, constraints),  # type: ignore[arg-type]
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _route_with_llm(
    user_prompt: str,
    model: Optional[str],
) -> RouterOutput:
    system_prompt = (
        "You are a strict router for n8n assistant requests. "
        "Classify the intent and extract a compact JSON object. "
        "Return ONLY valid JSON, no prose."
    )
    user_payload = (
        "Output schema:\n"
        "{\n"
        '  "intent": "create" | "fix" | "extend",\n'
        '  "goal": "string",\n'
        '  "constraints": {\n'
        '    "services": ["string"],\n'
        '    "inputs": ["string"],\n'
        '    "outputs": ["string"],\n'
        '    "nonFunctional": ["string"]\n'
        "  },\n"
        '  "missing_info": ["string"],\n'
        '  "complexity_score": 1 | 2 | 3\n'
        "}\n\n"
        "Rules:\n"
        "- Keep arrays empty if unknown.\n"
        "- Do not include extra keys.\n"
        "- Keep goal concise.\n\n"
        f"User request:\n{(user_prompt or '').strip()}"
    )
    resolved_model = resolve_model(model)
    response = chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        model=resolved_model,
        temperature=0.0,
    )
    content = (
        response.model_dump().get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    payload = _extract_json(str(content))
    if payload is None:
        raise ValueError("router llm returned invalid json")
    return RouterOutput.model_validate(payload)


def route_prompt(
    user_prompt: str,
    existing_workflow: Any = None,
    model: Optional[str] = None,
) -> RouterOutput:
    if not settings.ROUTER_USE_LLM:
        return _heuristic_router_output(user_prompt, existing_workflow)
    fallback = _heuristic_router_output(user_prompt, existing_workflow)
    try:
        return _route_with_llm(user_prompt, model=model)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("router llm fallback to heuristics: %s", str(exc))
        return fallback
