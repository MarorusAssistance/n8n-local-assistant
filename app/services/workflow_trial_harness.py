from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from fastapi.testclient import TestClient

import app.api.routes as routes

from ..main import app
from ..memory.in_memory import InMemoryStore
from .chat_service import ChatService

DEFAULT_EMAIL_URGENCY_REQUEST = (
    "crea un workflow para coger los emails que vaya recibiendo y que los clasifique "
    "por nivel de urgencia con IA"
)

_MASTER_CLARIFICATION = (
    "Los correos llegan desde mi Gmail y el workflow debe ejecutarse siempre que entre un email nuevo. "
    "Usa una IA o LLM para leer asunto y cuerpo del correo y clasificar semanticamente la urgencia en "
    "bajo, medio, alto o critico. Despues añade al propio correo una etiqueta visible en Gmail con ese "
    "nivel de urgencia para que yo pueda filtrarlos desde la UI de Gmail. Si no se puede decidir el nivel, "
    "usa por defecto la etiqueta Review. Prefiero un LLM local si hay un nodo claro; si no, usa OpenAI. "
    "Si aun falta algun detalle no critico, inventa valores razonables y continua."
)
_ESCALATED_CLARIFICATION = (
    "No necesito mas aclaraciones funcionales. Cierra el workflow con esta estructura minima obligatoria: "
    "1) un trigger de Gmail o email entrante que se ejecute con cada correo recibido, 2) un nodo LLM que "
    "clasifique semanticamente asunto y cuerpo en bajo, medio, alto o critico, y 3) un nodo que aplique "
    "al mismo correo una etiqueta visible en Gmail con el nivel devuelto. Si no puedes resolver un detalle "
    "tecnico no critico, usa defaults razonables. Si falta una credencial, deja ese dato pendiente pero "
    "continua con el resto del workflow."
)
_TECHNICAL_FALLBACK = (
    "No tengo mas datos tecnicos. Si te falta alguna credencial o parametro no critico, usa defaults "
    "razonables, deja la credencial sin asignar si hace falta y continua con el resto del workflow."
)

_TRIGGER_HINTS = (
    "gmailtrigger",
    "emailreadimap",
    "outlooktrigger",
    "sendinbluetrigger",
    "emeliatrigger",
    "email trigger",
    "gmail trigger",
    "incoming email",
)
_TRIGGER_CONTEXT_HINTS = ("gmail", "email", "correo", "imap", "outlook", "inbox")
_LLM_HINTS = (
    "openai",
    "ollama",
    "mistral",
    "anthropic",
    "gemini",
    "llm",
    "aitransform",
    "basicllmchain",
    "ai chain",
)
_CLASSIFY_HINTS = ("classif", "urgenc", "semantic", "semant", "priority")
_APPLY_HINTS = ("label", "tag", "etiquet", "review", "addlabel", "add label")
_SOURCE_APPLY_HINTS = ("gmail", "email", "correo", "message")


@dataclass
class WorkflowTrialTurn:
    turn_index: int
    user_text: str
    assistant_text: str
    current_stage: Optional[str] = None
    last_block_cause: Optional[str] = None
    pending_slot_keys: List[str] = field(default_factory=list)
    missing_user_inputs: List[str] = field(default_factory=list)


@dataclass
class WorkflowTrialResult:
    success: bool
    conversation_id: str
    turns_used: int
    workflow_id: Optional[str]
    workflow_url: Optional[str]
    public_text: str
    structural_success: bool = False
    persistence_success: bool = False
    structural_issues: List[str] = field(default_factory=list)
    transcript: List[WorkflowTrialTurn] = field(default_factory=list)
    envelope: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "conversation_id": self.conversation_id,
            "turns_used": self.turns_used,
            "workflow_id": self.workflow_id,
            "workflow_url": self.workflow_url,
            "public_text": self.public_text,
            "structural_success": self.structural_success,
            "persistence_success": self.persistence_success,
            "structural_issues": list(self.structural_issues),
            "transcript": [asdict(turn) for turn in self.transcript],
            "envelope": self.envelope,
        }


def create_trial_client() -> Tuple[TestClient, ChatService]:
    store = InMemoryStore(ttl_seconds=0)
    service = ChatService(store)
    routes.chat_service = service
    return TestClient(app), service


def read_temporal_result(service: ChatService) -> Dict[str, Any]:
    return json.loads(service._temporal_result_path.read_text(encoding="utf-8"))  # noqa: SLF001


def run_email_urgency_trial(
    *,
    client: Optional[TestClient] = None,
    service: Optional[ChatService] = None,
    initial_request: str = DEFAULT_EMAIL_URGENCY_REQUEST,
    model: str = "local-model",
    max_turns: int = 8,
    conversation_id: Optional[str] = None,
) -> WorkflowTrialResult:
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")
    if (client is None) != (service is None):
        raise ValueError("client and service must be provided together")

    if client is None or service is None:
        client, service = create_trial_client()

    try:
        service._temporal_result_path.unlink()  # noqa: SLF001
    except FileNotFoundError:
        pass

    conv_id = conversation_id or f"workflow-trial-{uuid4().hex[:10]}"
    next_user_text = initial_request
    turns: List[WorkflowTrialTurn] = []
    block_counts: Dict[str, int] = {}
    envelope: Dict[str, Any] = {}
    public_text = ""

    for turn_index in range(1, max_turns + 1):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "conversation_id": conv_id,
                "messages": [{"role": "user", "content": next_user_text}],
            },
        )
        if response.status_code != 200:
            return WorkflowTrialResult(
                success=False,
                conversation_id=conv_id,
                turns_used=turn_index,
                workflow_id=None,
                workflow_url=None,
                public_text=f"HTTP {response.status_code}",
                structural_success=False,
                persistence_success=False,
                structural_issues=[f"service returned HTTP {response.status_code}"],
                transcript=turns,
                envelope=envelope,
            )

        response_payload = response.json()
        public_text = str(response_payload["choices"][0]["message"]["content"] or "").strip()
        envelope = read_temporal_result(service)
        turns.append(
            WorkflowTrialTurn(
                turn_index=turn_index,
                user_text=next_user_text,
                assistant_text=public_text,
                current_stage=_as_text(envelope.get("current_stage")) or None,
                last_block_cause=_as_text(envelope.get("last_block_cause")) or None,
                pending_slot_keys=_pending_slot_keys(envelope),
                missing_user_inputs=_string_list(envelope.get("missing_user_inputs")),
            )
        )

        if _workflow_ready(envelope):
            break

        block_signature = _block_signature(envelope, public_text)
        block_counts[block_signature] = block_counts.get(block_signature, 0) + 1
        follow_up = build_trial_followup(
            envelope=envelope,
            assistant_text=public_text,
            repeat_count=block_counts[block_signature],
        )
        if not follow_up:
            break
        next_user_text = follow_up

    workflow_id, workflow_url = _workflow_reference(envelope)
    structural_issues = validate_email_urgency_workflow(envelope)
    persistence_success = bool(envelope.get("workflow_persisted"))
    non_persistence_issues = [issue for issue in structural_issues if issue != "workflow_not_persisted"]
    structural_success = not non_persistence_issues
    success = structural_success and persistence_success
    return WorkflowTrialResult(
        success=success,
        conversation_id=conv_id,
        turns_used=len(turns),
        workflow_id=workflow_id,
        workflow_url=workflow_url,
        public_text=public_text,
        structural_success=structural_success,
        persistence_success=persistence_success,
        structural_issues=structural_issues,
        transcript=turns,
        envelope=envelope,
    )


def build_trial_followup(
    *,
    envelope: Dict[str, Any],
    assistant_text: str,
    repeat_count: int = 1,
) -> Optional[str]:
    if not _requires_user_followup(envelope):
        return None

    targeted = _targeted_followup_from_questions(envelope)
    current_stage = _as_text(envelope.get("current_stage"))
    if repeat_count >= 2:
        return "\n".join(part for part in (_ESCALATED_CLARIFICATION, targeted, _TECHNICAL_FALLBACK) if part)

    if current_stage == "architect_agent" and _has_semantic_blockers(envelope):
        return "\n".join(part for part in (_ESCALATED_CLARIFICATION, targeted) if part)

    if _has_technical_blockers(envelope):
        return "\n\n".join(part for part in (targeted or _MASTER_CLARIFICATION, _TECHNICAL_FALLBACK) if part)

    if _has_semantic_blockers(envelope) or assistant_text:
        return targeted or _MASTER_CLARIFICATION

    return None


def validate_email_urgency_workflow(payload: Dict[str, Any]) -> List[str]:
    workflow_draft = payload.get("workflow_draft")
    if not isinstance(workflow_draft, dict):
        return ["workflow_draft_missing"]

    nodes = [
        item for item in workflow_draft.get("nodes") or [] if isinstance(item, dict)
    ]
    if not nodes:
        return ["workflow_nodes_missing"]

    connections = [
        item for item in workflow_draft.get("connections") or [] if isinstance(item, dict)
    ]
    stage_kind_by_id = _stage_kind_by_id(payload)

    trigger_ids = {
        node_id
        for node in nodes
        if (node_id := _as_text(node.get("node_id"))) and _is_email_trigger_node(node, stage_kind_by_id)
    }
    llm_ids = {
        node_id
        for node in nodes
        if (node_id := _as_text(node.get("node_id"))) and _is_llm_classifier_node(node, stage_kind_by_id)
    }
    apply_ids = {
        node_id
        for node in nodes
        if (node_id := _as_text(node.get("node_id"))) and _is_label_apply_node(node, stage_kind_by_id)
    }

    issues: List[str] = []
    if not trigger_ids:
        issues.append("missing_email_trigger_node")
    if not llm_ids:
        issues.append("missing_llm_classification_node")
    if not apply_ids:
        issues.append("missing_label_apply_node")
    if _as_text(payload.get("implementation_status")) == "blocked_waiting_user":
        issues.append("workflow_blocked_waiting_user")
    if _string_list(payload.get("missing_user_inputs")):
        issues.append("workflow_requires_user_followup")
    if not bool(payload.get("workflow_persisted")) and isinstance(payload.get("workflow_api_sync_result"), dict):
        sync_error = _as_text(payload["workflow_api_sync_result"].get("error"))
        if sync_error:
            issues.append("workflow_not_persisted")

    adjacency = _adjacency_map(connections)
    if trigger_ids and llm_ids and not _has_any_path(trigger_ids, llm_ids, adjacency):
        issues.append("missing_path_from_trigger_to_llm")
    if llm_ids and apply_ids and not _has_any_path(llm_ids, apply_ids, adjacency):
        issues.append("missing_path_from_llm_to_label_apply")
    return issues


def _requires_user_followup(envelope: Dict[str, Any]) -> bool:
    if _string_list(envelope.get("missing_user_inputs")):
        return True
    if _pending_slot_keys(envelope):
        return True
    if _missing_input_details(envelope):
        return True
    return _as_text(envelope.get("implementation_status")) == "blocked_waiting_user"


def _workflow_ready(envelope: Dict[str, Any]) -> bool:
    if not bool(envelope.get("workflow_persisted")):
        return False
    if _requires_user_followup(envelope):
        return False
    return bool(_workflow_reference(envelope)[0] or _workflow_reference(envelope)[1])


def _workflow_reference(envelope: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    reference = envelope.get("workflow_reference")
    workflow_id = None
    workflow_url = None
    if isinstance(reference, dict):
        workflow_id = _as_text(reference.get("id")) or None
        workflow_url = _as_text(reference.get("url")) or None
    workflow_id = workflow_id or (_as_text(envelope.get("active_workflow_id")) or None)
    workflow_url = workflow_url or (_as_text(envelope.get("active_workflow_url")) or None)
    return workflow_id, workflow_url


def _block_signature(envelope: Dict[str, Any], assistant_text: str) -> str:
    signature = {
        "stage": _as_text(envelope.get("current_stage")),
        "cause": _as_text(envelope.get("last_block_cause")),
        "slots": _pending_slot_keys(envelope),
        "missing_user_inputs": _string_list(envelope.get("missing_user_inputs")),
        "assistant_text": _as_text(assistant_text),
    }
    return json.dumps(signature, sort_keys=True, ensure_ascii=True)


def _stage_kind_by_id(payload: Dict[str, Any]) -> Dict[str, str]:
    plan = payload.get("architecture_plan")
    if not isinstance(plan, dict):
        return {}
    result: Dict[str, str] = {}
    for stage in plan.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        stage_id = _as_text(stage.get("id"))
        stage_kind = _as_text(stage.get("stage_kind"))
        if stage_id and stage_kind:
            result[stage_id] = stage_kind
    return result


def _is_email_trigger_node(node: Dict[str, Any], stage_kind_by_id: Dict[str, str]) -> bool:
    stage_kind = stage_kind_by_id.get(_as_text(node.get("stage_id")))
    text = _node_identity_text(node)
    if stage_kind == "trigger_intake" and any(hint in text for hint in _TRIGGER_CONTEXT_HINTS):
        return True
    return any(hint in text for hint in _TRIGGER_HINTS) and any(
        ctx in text for ctx in _TRIGGER_CONTEXT_HINTS
    )


def _is_llm_classifier_node(node: Dict[str, Any], stage_kind_by_id: Dict[str, str]) -> bool:
    stage_kind = stage_kind_by_id.get(_as_text(node.get("stage_id")))
    identity = _node_identity_text(node)
    if _looks_like_trigger_identity(identity):
        return False
    has_ai_identity = any(hint in identity for hint in _LLM_HINTS)
    has_classification_identity = any(hint in identity for hint in _CLASSIFY_HINTS)
    params = _merged_node_parameters(node)
    has_model_signal = bool(_as_text(params.get("model")) or _as_text(params.get("promptType")))
    if stage_kind == "classify_decision":
        return has_ai_identity or (has_model_signal and has_classification_identity)
    return (has_ai_identity or has_model_signal) and has_classification_identity


def _is_label_apply_node(node: Dict[str, Any], stage_kind_by_id: Dict[str, str]) -> bool:
    stage_kind = stage_kind_by_id.get(_as_text(node.get("stage_id")))
    identity = _node_identity_text(node)
    if _looks_like_trigger_identity(identity):
        return False
    params = _merged_node_parameters(node)
    resource = _as_text(params.get("resource")).lower()
    operation = _as_text(params.get("operation")).lower()
    has_explicit_apply = resource == "message" and operation in {"addlabel", "add_label", "label"}
    has_apply_identity = any(hint in identity for hint in ("addlabel", "add label", "label", "tag", "etiquet"))
    has_source = any(hint in identity for hint in _SOURCE_APPLY_HINTS)
    if stage_kind == "apply_update_source":
        return has_explicit_apply or (has_apply_identity and has_source)
    return has_explicit_apply or (has_apply_identity and has_source)


def _adjacency_map(connections: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    adjacency: Dict[str, List[str]] = {}
    for connection in connections:
        source = _as_text(connection.get("source_node_id"))
        target = _as_text(connection.get("target_node_id"))
        if not source or not target:
            continue
        adjacency.setdefault(source, []).append(target)
    return adjacency


def _has_any_path(
    source_ids: Iterable[str],
    target_ids: Iterable[str],
    adjacency: Dict[str, List[str]],
) -> bool:
    targets = set(target_ids)
    for source in source_ids:
        if _has_path(source, targets, adjacency):
            return True
    return False


def _has_path(start: str, targets: set[str], adjacency: Dict[str, List[str]]) -> bool:
    visited: set[str] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in targets:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency.get(current, []))
    return False


def _node_text(node: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "node_id",
        "name",
        "node_type",
        "purpose",
    ):
        value = node.get(key)
        if value:
            parts.append(str(value))
    for list_key in ("notes", "expected_inputs", "expected_outputs", "parameters_unresolved"):
        value = node.get(list_key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
    for map_key in ("parameters_known", "parameters_inferred", "credential_refs", "implementation_hints"):
        value = node.get(map_key)
        if isinstance(value, dict):
            parts.append(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return " ".join(parts).lower()


def _node_identity_text(node: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("node_id", "name", "node_type", "purpose"):
        value = node.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _looks_like_trigger_identity(text: str) -> bool:
    return any(hint in text for hint in _TRIGGER_HINTS)


def _merged_node_parameters(node: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key in ("parameters_inferred", "parameters_known"):
        value = node.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _node_tokens(node: Dict[str, Any]) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", _node_text(node)) if token]


def _pending_slot_keys(envelope: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    for slot in _all_pending_slots(envelope):
        if not isinstance(slot, dict):
            continue
        slot_key = _as_text(slot.get("slot_key"))
        if slot_key:
            result.append(slot_key)
    return result


def _all_pending_slots(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    for item in envelope.get("pending_decision_slots") or []:
        if isinstance(item, dict):
            slots.append(item)
    workflow_context = envelope.get("workflow_context")
    if isinstance(workflow_context, dict):
        for item in workflow_context.get("pending_decision_slots") or []:
            if isinstance(item, dict):
                slots.append(item)
    pm_state = envelope.get("pm_clarification_state")
    if isinstance(pm_state, dict):
        for item in pm_state.get("pending_slots") or []:
            if isinstance(item, dict):
                slots.append(item)
    architect_state = envelope.get("architect_clarification_state")
    if isinstance(architect_state, dict):
        for item in architect_state.get("pending_slots") or []:
            if isinstance(item, dict):
                slots.append(item)
    return slots


def _missing_input_details(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item
        for item in envelope.get("missing_user_input_details") or []
        if isinstance(item, dict)
    ]


def _has_technical_blockers(envelope: Dict[str, Any]) -> bool:
    for item in _missing_input_details(envelope):
        if _as_text(item.get("category")) in {"credential", "parameter"}:
            return True
    return False


def _has_semantic_blockers(envelope: Dict[str, Any]) -> bool:
    if _pending_slot_keys(envelope):
        return True
    for item in _missing_input_details(envelope):
        if _as_text(item.get("category")) in {"decision", "business_rule", "handoff"}:
            return True
    return False


def _targeted_followup_from_questions(envelope: Dict[str, Any]) -> str:
    questions = _question_texts(envelope)
    lowered_questions = [question.lower() for question in questions]
    lines: List[str] = []

    def add_line(text: str) -> None:
        if text not in lines:
            lines.append(text)

    if any(_question_mentions(question, "email system", "gmail", "outlook", "imap", "incoming emails") for question in lowered_questions):
        add_line("Usa Gmail como origen y procesa cada email nuevo en cuanto llegue.")
    if any(_question_mentions(question, "how many urgency levels", "urgency levels", "critical", "high", "medium", "low") for question in lowered_questions):
        add_line("Los niveles de urgencia son bajo, medio, alto y critico.")
    if any(_question_mentions(question, "moved to a new folder", "remain in the inbox", "folder", "inbox with labels") for question in lowered_questions):
        add_line("Los correos deben quedarse en inbox y llevar una etiqueta visible en Gmail con su nivel de urgencia.")
    if any(
        _question_mentions(
            question,
            "what should happen",
            "result next",
            "final business outcome",
            "outcome handling",
            "apply it to the source item",
        )
        for question in lowered_questions
    ):
        add_line(
            "Aplica el resultado al elemento origen: al mismo correo de Gmail. No hace falta guardarlo aparte ni "
            "notificar; solo etiquetarlo en Gmail para poder filtrarlo despues."
        )
    if any(_question_mentions(question, "cannot confidently classify", "confidence threshold", "manual review", "review") for question in lowered_questions):
        add_line("Si la IA no puede clasificar con seguridad, usa por defecto la etiqueta Review.")
    if any(_question_mentions(question, "local llm", "openai", "preferred", "provider", "both should be supported") for question in lowered_questions):
        add_line("Prefiero un LLM local si existe un nodo claro; si no, usa OpenAI.")
    if any(_question_mentions(question, "exact name or format of the label", "canonical name", "canonical label", "visual appearance", "color scheme", "label names") for question in lowered_questions):
        add_line(
            "No hace falta color ni iconos especiales. Usa etiquetas visibles con estos nombres exactos: "
            "Urgency/Low, Urgency/Medium, Urgency/High, Urgency/Critical y Review."
        )
    if any(_question_mentions(question, "already labeled", "overwrite existing", "skip labeling", "existing urgency-related labels") for question in lowered_questions):
        add_line("Si el correo ya tiene una etiqueta de urgencia anterior, sobreescribela por la nueva.")
    if any(_question_mentions(question, "notify", "silent operation", "notification", "alert") for question in lowered_questions):
        add_line("La operacion debe ser silenciosa; no quiero notificaciones adicionales.")
    if any(_question_mentions(question, "workflow goal", "full request", "abstract workflow plan", "coherent abstract workflow plan", "current issue:") for question in lowered_questions):
        add_line(
            "El objetivo completo es este: 1) trigger de Gmail para cada email entrante, 2) un nodo LLM que "
            "clasifique semantica y contextualmente asunto y cuerpo en bajo, medio, alto o critico, y 3) un "
            "nodo de Gmail que aplique al mismo correo una etiqueta visible con ese nivel para poder filtrarlo "
            "despues desde la UI. Si no se puede decidir el nivel, usa Review."
        )
    if not lines:
        add_line(_MASTER_CLARIFICATION)
    return "\n".join(lines)


def _question_texts(envelope: Dict[str, Any]) -> List[str]:
    questions = _string_list(envelope.get("missing_user_inputs"))
    for slot in _all_pending_slots(envelope):
        question_text = _as_text(slot.get("question_text"))
        if question_text and question_text not in questions:
            questions.append(question_text)
    return questions


def _question_mentions(question: str, *fragments: str) -> bool:
    return any(fragment in question for fragment in fragments)


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _as_text(value: Any) -> str:
    return str(value or "").strip()
