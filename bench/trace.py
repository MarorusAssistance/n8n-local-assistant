from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_HEADER_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
    r"(?P<level>[A-Z]+) "
    r"(?P<logger>[^:]+): "
    r"(?P<message>.*)$"
)
_REQUEST_ID_RE = re.compile(r"\bid=([A-Za-z0-9._:-]+)")
_TRACE_EVENT_PREFIX = "TRACE EVENT "
_PROMPT_BUDGET_RE = re.compile(
    r"dropped=(?P<dropped>\d+)\s+truncated=(?P<truncated>\d+)"
)

_REDACT_PATTERNS = [
    (
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9\-._~+/]+=*"),
        r"\1***REDACTED***",
    ),
    (
        re.compile(
            r"(?i)\b(api[_\-\s]?key|token|secret|password)\b(\s*[:=]\s*)([^\s\"']+|\"[^\"]+\"|'[^']+')"
        ),
        r"\1\2***REDACTED***",
    ),
]


@dataclass(frozen=True)
class TraceEntry:
    timestamp: str
    level: str
    logger: str
    message: str
    raw: str
    request_id: Optional[str]
    event_type: str
    stage: Optional[str]
    payload: Dict[str, Any]


def parse_trace_file(path: Path) -> List[TraceEntry]:
    if not path.exists():
        return []
    return parse_trace_text(path.read_text(encoding="utf-8"))


def parse_trace_text(text: str) -> List[TraceEntry]:
    lines = text.splitlines()
    entries: List[TraceEntry] = []

    current_header: Optional[Dict[str, str]] = None
    current_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current_header, current_lines
        if current_header is None:
            return
        message = "\n".join(current_lines)
        request_id = _extract_request_id(message)
        event_type, stage, payload = _classify_event(message)
        if not request_id:
            payload_request_id = payload.get("request_id") if isinstance(payload, Mapping) else None
            if payload_request_id:
                request_id = str(payload_request_id).strip() or None
        raw = (
            f"{current_header['ts']} {current_header['level']} "
            f"{current_header['logger']}: {message}"
        )
        entries.append(
            TraceEntry(
                timestamp=current_header["ts"],
                level=current_header["level"],
                logger=current_header["logger"],
                message=message,
                raw=raw,
                request_id=request_id,
                event_type=event_type,
                stage=stage,
                payload=payload,
            )
        )
        current_header = None
        current_lines = []

    for line in lines:
        header_match = _HEADER_RE.match(line)
        if header_match:
            flush_current()
            current_header = {
                "ts": header_match.group("ts"),
                "level": header_match.group("level"),
                "logger": header_match.group("logger"),
            }
            current_lines = [header_match.group("message")]
            continue
        if current_header is not None:
            current_lines.append(line)

    flush_current()
    return entries


def sort_entries(entries: Iterable[TraceEntry]) -> List[TraceEntry]:
    def _key(item: TraceEntry) -> tuple[datetime, int]:
        try:
            dt = datetime.strptime(item.timestamp, "%Y-%m-%d %H:%M:%S,%f")
        except Exception:
            dt = datetime.min
        return dt, 0

    return sorted(entries, key=_key)


def redact_text_light(text: str) -> str:
    value = str(text or "")
    for pattern, replacement in _REDACT_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def redact_mapping_light(payload: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in payload.items():
        output[str(key)] = _redact_any(value)
    return output


def entries_for_request(
    entries: Iterable[TraceEntry],
    *,
    request_id: Optional[str],
    conversation_id: Optional[str] = None,
) -> List[TraceEntry]:
    matched: List[TraceEntry] = []
    for entry in entries:
        if request_id and entry.request_id == request_id:
            matched.append(entry)
            continue
        if conversation_id and f"conv={conversation_id}" in entry.message:
            matched.append(entry)
    return sort_entries(matched)


def extract_prompt_rows(entries: Iterable[TraceEntry]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in sort_entries(entries):
        if entry.event_type == "llm_prompt":
            stage = str(entry.stage or entry.payload.get("stage") or "unknown")
            for msg in entry.payload.get("messages", []):
                if not isinstance(msg, Mapping):
                    continue
                content = redact_text_light(str(msg.get("content") or ""))
                rows.append(
                    {
                        "timestamp": entry.timestamp,
                        "stage": stage,
                        "role": str(msg.get("role") or "unknown"),
                        "content": content,
                    }
                )
            continue

        if entry.event_type == "trace_request":
            content = _extract_label_block(entry.message, "usuario")
            if content:
                rows.append(
                    {
                        "timestamp": entry.timestamp,
                        "stage": "request.input",
                        "role": "user",
                        "content": redact_text_light(content),
                    }
                )
            continue

        if entry.event_type == "trace_response":
            content = _extract_label_block(entry.message, "respuesta")
            if content:
                rows.append(
                    {
                        "timestamp": entry.timestamp,
                        "stage": "response.output",
                        "role": "assistant",
                        "content": redact_text_light(content),
                    }
                )
    return rows


def compute_log_metrics(entries: Iterable[TraceEntry]) -> Dict[str, Any]:
    llm_calls = 0
    prompt_chars = 0
    retrieval_chunks = 0
    linked_node_chunks = 0
    linked_credential_chunks = 0
    budget_trim_count = 0
    docs_fallback_count = 0
    reasoning_second_iteration = False

    for entry in entries:
        if entry.event_type == "llm_output":
            llm_calls += 1

        if entry.event_type == "llm_prompt":
            total_chars = entry.payload.get("total_chars")
            if isinstance(total_chars, (int, float)):
                prompt_chars += int(total_chars)
            else:
                for msg in entry.payload.get("messages", []):
                    if isinstance(msg, Mapping):
                        prompt_chars += len(str(msg.get("content") or ""))

        if entry.event_type == "retrieval_final":
            chunks = entry.payload.get("chunks")
            if isinstance(chunks, list):
                retrieval_chunks += len(chunks)
                for chunk in chunks:
                    if not isinstance(chunk, Mapping):
                        continue
                    linked_type = str(chunk.get("linked_def_type") or "").lower()
                    if linked_type == "node":
                        linked_node_chunks += 1
                    if linked_type == "credential":
                        linked_credential_chunks += 1

        if entry.event_type == "prompt_budget_applied":
            match = _PROMPT_BUDGET_RE.search(entry.message)
            if match:
                dropped = int(match.group("dropped"))
                truncated = int(match.group("truncated"))
                if dropped > 0 or truncated > 0:
                    budget_trim_count += 1

        if entry.event_type == "docs_fallback_applied":
            docs_fallback_count += 1

        if entry.event_type == "reasoning_result":
            if "second_iteration=True" in entry.message or "second_iteration=true" in entry.message.lower():
                reasoning_second_iteration = True
            if "second_iteration=True" not in entry.message and "second_iteration=False" not in entry.message:
                if "second_iteration=True" in json.dumps(entry.payload):
                    reasoning_second_iteration = True

    return {
        "llm_calls": llm_calls,
        "prompt_chars": prompt_chars,
        "retrieval_chunks": retrieval_chunks,
        "linked_node_chunks": linked_node_chunks,
        "linked_credential_chunks": linked_credential_chunks,
        "budget_trim_count": budget_trim_count,
        "docs_fallback_count": docs_fallback_count,
        "reasoning_second_iteration": reasoning_second_iteration,
    }


def extract_retrieval_views(entries: Iterable[TraceEntry]) -> Dict[str, List[Dict[str, Any]]]:
    latest_payload: Optional[Mapping[str, Any]] = None
    latest_trace_rag_message: Optional[str] = None
    for entry in sort_entries(entries):
        if entry.event_type == "retrieval_final":
            latest_payload = entry.payload
        elif entry.event_type == "trace_rag":
            latest_trace_rag_message = entry.message

    if latest_payload:
        pre = _normalize_chunk_records(latest_payload.get("pre_rerank_chunks"))
        post = _normalize_chunk_records(latest_payload.get("post_rerank_chunks"))
        final = _normalize_chunk_records(
            latest_payload.get("final_chunks") or latest_payload.get("chunks")
        )

        if not post and final:
            post = [
                item
                for item in final
                if item.get("linked_def_type") not in ("node", "credential")
            ]

        pre_linked_node = [
            item for item in pre if item.get("linked_def_type") == "node"
        ]
        pre_linked_credential = [
            item for item in pre if item.get("linked_def_type") == "credential"
        ]
        post_linked_node = [
            item for item in final if item.get("linked_def_type") == "node"
        ]
        post_linked_credential = [
            item for item in final if item.get("linked_def_type") == "credential"
        ]

        return {
            "pre_docs": [
                item
                for item in pre
                if item.get("linked_def_type") not in ("node", "credential")
            ],
            "pre_linked_node": pre_linked_node,
            "pre_linked_credential": pre_linked_credential,
            "post_docs": [
                item
                for item in post
                if item.get("linked_def_type") not in ("node", "credential")
            ],
            "post_linked_node": post_linked_node,
            "post_linked_credential": post_linked_credential,
        }

    if latest_trace_rag_message:
        pre_pool = _parse_trace_rag_chunks(latest_trace_rag_message, section="chunks_pool")
        post_final = _parse_trace_rag_chunks(latest_trace_rag_message, section="chunks_final")
        if not post_final:
            post_final = _parse_trace_rag_chunks(latest_trace_rag_message, section="chunks")
        return {
            "pre_docs": pre_pool,
            "pre_linked_node": [],
            "pre_linked_credential": [],
            "post_docs": post_final,
            "post_linked_node": [],
            "post_linked_credential": [],
        }

    return {
        "pre_docs": [],
        "pre_linked_node": [],
        "pre_linked_credential": [],
        "post_docs": [],
        "post_linked_node": [],
        "post_linked_credential": [],
    }


def _classify_event(message: str) -> tuple[str, Optional[str], Dict[str, Any]]:
    stage: Optional[str] = None
    payload: Dict[str, Any] = {}

    if message.startswith(_TRACE_EVENT_PREFIX):
        raw_json = message[len(_TRACE_EVENT_PREFIX):].strip()
        try:
            parsed = json.loads(raw_json)
        except Exception:
            parsed = {"raw": raw_json}
        if isinstance(parsed, Mapping):
            payload = {str(k): _redact_any(v) for k, v in parsed.items()}
            stage = str(parsed.get("stage") or "") or None
            event_name = str(parsed.get("event") or "trace_event")
            return event_name, stage, payload
        return "trace_event", None, {"raw": redact_text_light(raw_json)}

    if message.startswith("TRACE REQUEST id="):
        return "trace_request", None, payload
    if message.startswith("TRACE RESPONSE id="):
        return "trace_response", None, payload
    if message.startswith("TRACE RAG id="):
        return "trace_rag", None, payload
    if message.startswith("TRACE RETRIEVAL id="):
        return "trace_retrieval", None, payload
    if message.startswith("TRACE DB QUERY id="):
        return "trace_db_query", None, payload
    if message.startswith("TRACE DB FTS PREP id="):
        return "trace_db_fts_prep", None, payload
    if message.startswith("TRACE DB FTS id="):
        return "trace_db_fts", None, payload
    if message.startswith("prompt budget applied:"):
        return "prompt_budget_applied", None, payload
    if message.startswith("docs fallback applied:"):
        return "docs_fallback_applied", None, payload
    if message.startswith("reasoning graph result:") or message.startswith("reasoning pipeline:"):
        return "reasoning_result", None, payload
    return "other", None, payload


def _normalize_chunk_records(raw_chunks: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_chunks, list):
        return []
    output: List[Dict[str, Any]] = []
    for item in raw_chunks:
        if not isinstance(item, Mapping):
            continue
        score_value, score_label = _pick_score(item)
        content = str(
            item.get("content")
            or item.get("snippet")
            or item.get("text")
            or ""
        ).strip()
        output.append(
            {
                "score": score_value,
                "score_label": score_label,
                "content": redact_text_light(content),
                "linked_def_type": str(item.get("linked_def_type") or "").strip().lower(),
            }
        )
    return output


def _pick_score(item: Mapping[str, Any]) -> tuple[Optional[float], str]:
    for key, label in (
        ("rerank_score", "rerank"),
        ("rrf_score", "rrf"),
        ("fts_score", "fts"),
    ):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value), label
        if isinstance(value, str):
            try:
                return float(value), label
            except ValueError:
                pass
    return None, "-"


def _parse_trace_rag_chunks(message: str, *, section: str) -> List[Dict[str, Any]]:
    section_text = _extract_section_text(message, section)
    if not section_text:
        return []
    lines = section_text.splitlines()
    rows: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("["):
            i += 1
            continue
        header = line
        content = ""
        if i + 1 < len(lines):
            maybe_content = lines[i + 1].strip()
            if maybe_content and not maybe_content.startswith("["):
                content = maybe_content
                i += 1
        score: Optional[float] = None
        score_label = "-"
        rerank_match = re.search(r"\brerank=([-+]?[0-9]*\.?[0-9]+)", header)
        rrf_match = re.search(r"\brrf=([-+]?[0-9]*\.?[0-9]+)", header)
        if rerank_match:
            score = float(rerank_match.group(1))
            score_label = "rerank"
        elif rrf_match:
            score = float(rrf_match.group(1))
            score_label = "rrf"
        rows.append(
            {
                "score": score,
                "score_label": score_label,
                "content": redact_text_light(content),
                "linked_def_type": "",
            }
        )
        i += 1
    return rows


def _extract_section_text(message: str, section: str) -> str:
    token = f"{section}:\n"
    start = message.find(token)
    if start < 0:
        return ""
    start += len(token)
    remainder = message[start:]
    next_idx = len(remainder)
    for marker in ("\nchunks_pool:\n", "\nchunks_final:\n", "\nchunks:\n"):
        idx = remainder.find(marker)
        if idx >= 0:
            next_idx = min(next_idx, idx)
    return remainder[:next_idx].strip()


def _extract_request_id(message: str) -> Optional[str]:
    match = _REQUEST_ID_RE.search(message)
    if not match:
        return None
    value = str(match.group(1) or "").strip()
    return value or None


def _extract_label_block(message: str, label: str) -> str:
    token = f"{label}:\n"
    idx = message.find(token)
    if idx < 0:
        return ""
    chunk = message[idx + len(token):]
    lines = [line[2:] if line.startswith("  ") else line for line in chunk.splitlines()]
    return "\n".join(lines).strip()


def _redact_any(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return redact_text_light(value)
    if isinstance(value, Mapping):
        return {str(k): _redact_any(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_any(item) for item in value]
    return value
