from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    DecisionSlot,
    DecisionSlotAnswerStatus,
)

_SPANISH_HINTS = (
    " que ",
    " como ",
    " cuando ",
    " donde ",
    " por que ",
    " porque ",
    " para ",
    " quiero ",
    " necesito ",
    " workflow",
    " correo",
    " correos",
    " nodo",
    " etapa",
    " accion",
    " activar",
    " clasificar",
    " guardar",
    " notificar",
    " en espanol",
    " en español",
    " traduc",
    " no entiendo",
)
_QUESTION_REPHRASE_HINTS = (
    "no entiendo",
    "no se entiende",
    "en espanol",
    "en español",
    "traduc",
    "traduce",
    "traducelo",
    "tradúcelo",
    "explica",
    "aclara",
    "reformula",
    "rephrase",
    "translate",
    "what do you mean",
    "que quieres decir",
    "qué quieres decir",
)
_SPANISH_REPHRASE_PREFIXES = (
    "te lo reformulo en espanol:",
    "te lo reformulo en español:",
    "te la reformulo en espanol:",
    "te la reformulo en español:",
    "pregunta original:",
)
_TRANSLATION_RULES: List[tuple[re.Pattern[str], Any]] = [
    (
        re.compile(r"^Which concrete system or trigger should start the stage '([^']+)'\?$", re.IGNORECASE),
        lambda match: f"Que sistema o trigger concreto debe iniciar la etapa '{match.group(1)}'?",
    ),
    (
        re.compile(r"^Which concrete app or action should stage '([^']+)' use\?$", re.IGNORECASE),
        lambda match: f"Que app, nodo o accion concreta debe usar la etapa '{match.group(1)}'?",
    ),
    (
        re.compile(
            r"^I could not find a clear trigger-capable standard n8n node for stage '([^']+)'\. Which system or event should start this workflow\?$",
            re.IGNORECASE,
        ),
        lambda match: (
            f"No he encontrado un nodo trigger estandar de n8n suficientemente claro para la etapa '{match.group(1)}'. "
            "Que sistema o evento debe iniciar este workflow?"
        ),
    ),
    (
        re.compile(
            r"^I could not find a clear standard n8n node for stage '([^']+)'\. Which concrete app or action should this stage use\?$",
            re.IGNORECASE,
        ),
        lambda match: (
            f"No he encontrado un nodo estandar de n8n suficientemente claro para la etapa '{match.group(1)}'. "
            "Que app, nodo o accion concreta debe usar esta etapa?"
        ),
    ),
    (
        re.compile(
            r"^The workflow currently ends at stage '([^']+)' with a classification result\. What should happen with that classification next: apply it to the source item, save it somewhere, notify someone, or route it to a downstream action\?$",
            re.IGNORECASE,
        ),
        lambda match: (
            f"El workflow ahora mismo termina en la etapa '{match.group(1)}' con una clasificacion. "
            "Que quieres hacer despues con ese resultado: aplicarlo al elemento de origen, guardarlo, notificar a alguien o enrutarlo a una accion concreta?"
        ),
    ),
    (
        re.compile(
            r"^The workflow currently ends at stage '([^']+)' without a clear final business outcome\. What should happen with that result next: apply it to the source item, save it somewhere, notify someone, or route it to a concrete downstream action\?$",
            re.IGNORECASE,
        ),
        lambda match: (
            f"El workflow ahora mismo termina en la etapa '{match.group(1)}' sin una accion final clara de negocio. "
            "Que quieres hacer despues con ese resultado: aplicarlo al elemento de origen, guardarlo, notificar a alguien o enrutarlo a una accion concreta?"
        ),
    ),
    (
        re.compile(
            r"^What specific actions should be taken with classified emails\? For example: Should they be stored in different folders, routed to other systems, or notified to the user\?$",
            re.IGNORECASE,
        ),
        lambda _match: (
            "Que accion concreta quieres hacer con los emails ya clasificados? "
            "Por ejemplo: guardarlos en carpetas distintas, enviarlos a otros sistemas o notificar al usuario."
        ),
    ),
    (
        re.compile(
            r"^Are there any special rules for determining urgency levels \(e\.g\., keywords, sender priority, time-sensitive indicators\)\?$",
            re.IGNORECASE,
        ),
        lambda _match: (
            "Hay reglas especiales para determinar el nivel de urgencia? "
            "Por ejemplo: palabras clave, prioridad del remitente o indicadores de tiempo."
        ),
    ),
    (
        re.compile(
            r"^What specific rules or criteria should the classification model use to determine urgency levels\? For example, keywords, sentiment analysis, or predefined patterns\?$",
            re.IGNORECASE,
        ),
        lambda _match: (
            "Que reglas o criterios concretos debe usar el modelo para determinar el nivel de urgencia? "
            "Por ejemplo: palabras clave, analisis de sentimiento o patrones predefinidos."
        ),
    ),
    (
        re.compile(r"^Which business rule defines urgency\?$", re.IGNORECASE),
        lambda _match: "Que regla de negocio define la urgencia?",
    ),
    (
        re.compile(r"^What source should provide the incoming emails\?$", re.IGNORECASE),
        lambda _match: "Que origen debe proporcionar los emails entrantes?",
    ),
    (
        re.compile(r"^Provide value for parameter '([^']+)' required by node '([^']+)'\.$", re.IGNORECASE),
        lambda match: f"Indica el valor del parametro '{match.group(1)}' que necesita el nodo '{match.group(2)}'.",
    ),
    (
        re.compile(r"^Confirm the value for '([^']+)' required by node '([^']+)'\.$", re.IGNORECASE),
        lambda match: f"Confirma el valor de '{match.group(1)}' que necesita el nodo '{match.group(2)}'.",
    ),
    (
        re.compile(
            r"^Provide the credential reference to use for '([^']+)' in node '([^']+)'\.$",
            re.IGNORECASE,
        ),
        lambda match: (
            f"Indica la referencia de credencial que debe usar el nodo '{match.group(2)}' para '{match.group(1)}'."
        ),
    ),
]
_SEMANTIC_SLOT_KEYWORDS = {
    "source_system": (
        "gmail",
        "outlook",
        "imap",
        "email",
        "correo",
        "mailbox",
        "slack",
        "notion",
        "drive",
        "airtable",
    ),
    "classification_method": (
        "ia",
        "ai",
        "llm",
        "openai",
        "ollama",
        "local",
        "modelo",
        "model",
        "heuristic",
        "heuristics",
        "rule-based",
        "rules",
        "semantica",
        "semantics",
    ),
    "result_application_mode": (
        "apply",
        "aplicar",
        "aplica",
        "tag",
        "tags",
        "label",
        "labels",
        "etiqueta",
        "etiquetas",
        "etiquetar",
        "folder",
        "carpeta",
        "move",
        "mover",
        "mueve",
        "guardar",
        "guarda",
        "save",
        "filter",
        "filtrar",
        "visible",
        "ver",
        "archivar",
        "archiva",
        "archive",
    ),
    "storage_destination": (
        "database",
        "db",
        "sheet",
        "sheets",
        "table",
        "tabla",
        "airtable",
        "notion",
        "guardar",
        "save",
        "persist",
        "almacenar",
    ),
    "notification_policy": (
        "notify",
        "notification",
        "alert",
        "sms",
        "email",
        "mail",
        "push",
        "notificar",
        "aviso",
        "alerta",
    ),
    "model_preference": (
        "openai",
        "ollama",
        "mistral",
        "claude",
        "gpt",
        "local",
        "hosteado",
        "hosted",
        "modelo",
        "model",
    ),
}
_SLOT_QUESTION_TEMPLATES = {
    "workflow_goal": "Reformula el objetivo completo del workflow incluyendo trigger, procesamiento y resultado final.",
    "source_system": "Que sistema u origen debe proporcionar los datos de esta etapa?",
    "classification_method": "Que metodo semantico debe usar esta etapa: IA, reglas deterministas o algun criterio concreto?",
    "result_application_mode": "Que quieres hacer con el resultado de esta etapa: aplicarlo al elemento origen, etiquetarlo, moverlo, guardarlo o notificarlo?",
    "storage_destination": "Donde debe guardarse o persistirse el resultado de esta etapa?",
    "notification_policy": "A quien o como debe notificarse el resultado de esta etapa?",
    "model_preference": "Tienes preferencia de modelo o proveedor para esta etapa, por ejemplo OpenAI o un LLM local?",
}


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _strip_rephrase_prefix(text: str) -> str:
    normalized = _clean_text(text)
    lowered = normalized.lower()
    for prefix in _SPANISH_REPHRASE_PREFIXES:
        if lowered.startswith(prefix):
            return _clean_text(normalized[len(prefix):])
    return normalized


def user_prefers_spanish(*texts: Any) -> bool:
    combined = f" {' '.join(_clean_text(item).lower() for item in texts if _clean_text(item))} "
    if not combined.strip():
        return False
    if any(char in combined for char in ("ñ", "á", "é", "í", "ó", "ú", "¿")):
        return True
    return any(hint in combined for hint in _SPANISH_HINTS)


def is_question_rephrase_request(text: Any) -> bool:
    normalized = _clean_text(text).lower()
    if not normalized:
        return False
    for hint in _QUESTION_REPHRASE_HINTS:
        if " " in hint:
            if hint in normalized:
                return True
            continue
        if re.search(rf"\b{re.escape(hint)}\b", normalized):
            return True
    return False


def localize_question_text(
    question: Any,
    *,
    user_query: Any = "",
    context_texts: Sequence[Any] | None = None,
) -> str:
    normalized = _strip_rephrase_prefix(_clean_text(question))
    if not normalized:
        return ""
    context_values = list(context_texts or [])
    wants_spanish = user_prefers_spanish(user_query, normalized, *context_values) or is_question_rephrase_request(
        user_query
    )
    if not wants_spanish:
        return normalized
    if user_prefers_spanish(normalized):
        return normalized
    for pattern, builder in _TRANSLATION_RULES:
        match = pattern.match(normalized)
        if match:
            return _clean_text(builder(match))
    return normalized


def localize_question_list(
    questions: Iterable[Any],
    *,
    user_query: Any = "",
    context_texts: Sequence[Any] | None = None,
) -> List[str]:
    output: List[str] = []
    seen = set()
    for question in questions:
        localized = localize_question_text(
            question,
            user_query=user_query,
            context_texts=context_texts,
        )
        if not localized or localized in seen:
            continue
        seen.add(localized)
        output.append(localized)
    return output


def infer_decision_slot_key(
    question: Any,
    *,
    stage_name: Any = "",
    question_intent: Any = "",
) -> str:
    combined = _clean_text(question).lower()
    if not combined:
        combined = _clean_text(stage_name).lower()
    combined = f"{combined} {_clean_text(question_intent).lower()}".strip()
    if any(
        token in combined
        for token in (
            "workflow goal",
            "full workflow goal",
            "full request",
            "restate the workflow goal",
            "restate the full workflow",
            "coherent abstract workflow plan",
            "abstract workflow plan",
            "unknown target stage",
            "unknown source stage",
            "depends on unknown stage",
            "data flow references",
            "current issue:",
        )
    ):
        return "workflow_goal"
    if any(
        token in combined
        for token in (
            "visual appearance",
            "color scheme",
            "label names",
            "canonical label",
            "canonical name",
            "exact name or format of the label",
            "exact canonical name",
            "already labeled",
            "overwrite existing",
            "skip labeling",
        )
    ):
        return "result_application_mode"
    if any(
        token in combined
        for token in (
            "downstream action",
            "downstream actions",
            "after classification",
            "after triage",
            "after the classification",
            "after the result",
            "additional downstream actions",
            "routing to a different mailbox",
            "forwarding",
            "different mailbox",
            "what happens after",
            "what should happen after",
            "que debe pasar despues",
            "que quieres hacer despues",
            "accion posterior",
            "acciones posteriores",
        )
    ):
        return "result_application_mode"
    if any(
        token in combined
        for token in ("what should happen", "que quieres hacer", "result next", "apply", "tag", "label", "save", "guardar", "notify", "route")
    ):
        return "result_application_mode"
    if any(token in combined for token in ("openai", "ollama", "provider", "proveedor", "local llm", "local model")):
        return "model_preference"
    if any(token in combined for token in ("classification", "clasific", "urgency", "modelo", "model", "ia", "llm")):
        return "classification_method"
    if any(token in combined for token in ("source", "origen", "trigger", "incoming emails", "gmail", "imap")):
        return "source_system"
    if any(token in combined for token in ("notify", "notification", "alert", "notific", "silent operation")):
        return "notification_policy"
    if any(token in combined for token in ("store", "persist", "save it somewhere", "where should", "guardar", "persistir")):
        return "storage_destination"
    return "result_application_mode"


def decision_slot_question(
    slot_key: str,
    *,
    stage_name: Any = "",
    default_question: Any = "",
    user_query: Any = "",
    context_texts: Sequence[Any] | None = None,
) -> str:
    question = _clean_text(default_question) or _SLOT_QUESTION_TEMPLATES.get(slot_key, "")
    if stage_name and question and "{stage_name}" in question:
        question = question.format(stage_name=_clean_text(stage_name))
    return localize_question_text(
        question,
        user_query=user_query,
        context_texts=context_texts,
    )


def build_decision_slot(
    *,
    slot_key: str,
    owner_agent: AgentStage,
    question_text: Any,
    stage_id: Any = None,
    question_intent: Any = None,
    user_query: Any = "",
    context_texts: Sequence[Any] | None = None,
) -> DecisionSlot:
    return DecisionSlot(
        slot_key=_clean_text(slot_key) or infer_decision_slot_key(question_text, stage_name=stage_id, question_intent=question_intent),
        owner_agent=owner_agent,
        stage_id=_clean_text(stage_id) or None,
        question_text=decision_slot_question(
            _clean_text(slot_key),
            stage_name=stage_id,
            default_question=question_text,
            user_query=user_query,
            context_texts=context_texts,
        ),
        question_intent=_clean_text(question_intent) or None,
        answer_status=DecisionSlotAnswerStatus.pending,
        answer=None,
    )


def merge_decision_slots(
    existing: Iterable[DecisionSlot],
    new_slots: Iterable[DecisionSlot],
) -> List[DecisionSlot]:
    merged: List[DecisionSlot] = []
    seen = set()
    for slot in list(existing) + list(new_slots):
        key = (slot.owner_agent.value, slot.stage_id or "", slot.slot_key)
        if key in seen:
            continue
        seen.add(key)
        merged.append(slot)
    return merged


def pending_slot_questions(slots: Iterable[DecisionSlot]) -> List[str]:
    output: List[str] = []
    seen = set()
    for slot in slots:
        question = _clean_text(slot.question_text)
        if not question or question in seen:
            continue
        seen.add(question)
        output.append(question)
    return output


def infer_resolved_slot_answers(answer_text: Any) -> dict[str, str]:
    normalized = f" {_clean_text(answer_text).lower()} "
    resolved: dict[str, str] = {}
    if not normalized.strip():
        return resolved
    for slot_key, keywords in _SEMANTIC_SLOT_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            resolved[slot_key] = _clean_text(answer_text)
    return resolved


def resolve_pending_slots_from_answer(
    pending_slots: Sequence[DecisionSlot],
    *,
    answer_text: Any,
) -> tuple[List[DecisionSlot], List[DecisionSlot]]:
    if not pending_slots:
        return [], []
    answer = _clean_text(answer_text)
    if not answer or is_question_rephrase_request(answer):
        return list(pending_slots), []

    inferred_answers = infer_resolved_slot_answers(answer)
    remaining: List[DecisionSlot] = []
    resolved: List[DecisionSlot] = []
    one_pending = len(pending_slots) == 1

    for slot in pending_slots:
        matches = slot.slot_key in inferred_answers
        if not matches and slot.question_intent:
            matches = slot.question_intent in inferred_answers
        if not matches and one_pending:
            matches = True
        if matches:
            resolved.append(
                slot.model_copy(
                    update={
                        "answer_status": DecisionSlotAnswerStatus.resolved,
                        "answer": answer,
                    }
                )
            )
        else:
            remaining.append(slot)
    return remaining, resolved
