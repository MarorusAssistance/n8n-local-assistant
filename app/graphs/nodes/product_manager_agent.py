from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ...config import settings
from ...features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    EntryIntent,
    PMClarificationState,
    PMClarificationTurn,
    PMProgressState,
    PMStagePlan,
    PMStatus,
    UseCase,
    WorkflowContext,
)
from ...llm import get_langchain_chat_model
from ...observability import emit_llm_output_event, emit_llm_prompt_event, emit_trace_event
from ...token_budget import estimate_messages_tokens
from ..multi_agent_state import MultiAgentGraphState

logger = logging.getLogger("n8n-assistant")
trace_logger = logging.getLogger("n8n-assistant.trace")


class _AbstractPlanningOutput(BaseModel):
    workflow_summary: str = ""
    stages: List[ArchitectureStage] = Field(default_factory=list)
    data_flow: List[ArchitectureDataFlowItem] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    handoff_notes: List[str] = Field(default_factory=list)
    planning_ready: bool = True


def _pm_int(name: str, default: int) -> int:
    raw = getattr(settings, name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _sanitize_text(text: Any) -> str:
    raw = html.unescape(str(text or ""))
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def _compact(text: Any, max_chars: int = 260) -> str:
    value = " ".join(_sanitize_text(text).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _safe_list(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = _sanitize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _stage_trace_summary(stages: Sequence[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for stage in stages:
        purpose = getattr(stage, "purpose", None) or getattr(stage, "objective", "")
        output.append(
            {
                "id": getattr(stage, "id", None),
                "name": getattr(stage, "name", None),
                "purpose": _compact(purpose, max_chars=180),
                "required_capabilities": _safe_list(getattr(stage, "required_capabilities", []) or [])[:5],
                "expected_inputs": _safe_list(getattr(stage, "expected_inputs", []) or [])[:5],
                "expected_outputs": _safe_list(getattr(stage, "expected_outputs", []) or [])[:5],
                "dependencies": _safe_list(getattr(stage, "dependencies", []) or []),
                "success_criteria": _safe_list(getattr(stage, "success_criteria", []) or [])[:4],
            }
        )
    return output


def _runtime_context(state: MultiAgentGraphState) -> Tuple[Optional[str], Optional[str]]:
    runtime_context = state.get("runtime_context")
    if not isinstance(runtime_context, dict):
        runtime_context = state.get("workflow_context")
    if not isinstance(runtime_context, dict):
        return None, None
    model = runtime_context.get("model")
    request_id = runtime_context.get("request_id")
    return (
        model if isinstance(model, str) or model is None else None,
        request_id if isinstance(request_id, str) or request_id is None else None,
    )


def _normalize_use_case(value: Any) -> Optional[UseCase]:
    if isinstance(value, UseCase):
        return value
    if isinstance(value, dict):
        try:
            return UseCase.model_validate(value)
        except Exception:
            return None
    return None


def _derive_use_case_from_direct_build_request(user_query: str) -> Optional[UseCase]:
    query = _compact(user_query, max_chars=140)
    if not query:
        return None
    return UseCase(
        id="direct_build_request",
        title=_compact(query, max_chars=80),
        business_problem=f"User requested a new workflow for: {query}",
        desired_outcome=f"Deliver an abstract workflow plan that satisfies: {query}",
        expected_value="Provide a high-quality workflow plan ready for architectural grounding.",
        feasibility="unknown",
        priority_score=70.0,
        why_selected="Direct workflow build request routed to product manager.",
    )


def _normalize_pm_status(value: Any) -> Optional[PMStatus]:
    if isinstance(value, PMStatus):
        return value
    if isinstance(value, str):
        try:
            return PMStatus(value)
        except ValueError:
            return None
    return None


def _normalize_model(value: Any, model_cls: Any) -> Optional[Any]:
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        try:
            return model_cls.model_validate(value)
        except Exception:
            return None
    return None


def _record_clarification_answer(
    clarification_state: PMClarificationState,
    user_query: str,
) -> PMClarificationState:
    pending = list(clarification_state.pending_questions)
    turns = list(clarification_state.turns)
    if not pending:
        return clarification_state
    question = pending.pop(0)
    answer = _compact(user_query, max_chars=320)
    for idx in range(len(turns) - 1, -1, -1):
        if turns[idx].question == question and not turns[idx].answer:
            turns[idx] = turns[idx].model_copy(update={"answer": answer})
            break
    else:
        turns.append(PMClarificationTurn(question=question, answer=answer))
    return clarification_state.model_copy(update={"pending_questions": pending, "turns": turns})


def _problem_statement(
    *,
    use_case: UseCase,
    user_query: str,
    clarification_state: PMClarificationState,
) -> str:
    clarification_lines = [
        f"Question: {turn.question}\nAnswer: {turn.answer}"
        for turn in clarification_state.turns
        if turn.answer
    ]
    clarification_text = "\n\n".join(clarification_lines[:4])
    return (
        f"User request: {_compact(user_query, max_chars=500)}\n"
        f"Use case title: {use_case.title}\n"
        f"Business problem: {use_case.business_problem}\n"
        f"Desired outcome: {use_case.desired_outcome}\n"
        f"Expected business value: {use_case.expected_value}\n"
        f"Clarifications already provided:\n{clarification_text or '-'}"
    )


def _invoke_structured_output(
    *,
    system_prompt: str,
    user_prompt: str,
    output_model: Any,
    model: Optional[str],
    request_id: Optional[str],
    temperature: float,
    stage: str,
) -> Any:
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError("No model configured for PM structured output")

    llm = get_langchain_chat_model(model=model, temperature=temperature)
    if llm is None:
        raise RuntimeError("LangChain chat model is unavailable")

    try:
        structured = llm.with_structured_output(output_model)
    except Exception as exc:
        raise RuntimeError(f"Structured output unavailable: {exc}") from exc

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_tokens = estimate_messages_tokens(messages)
    emit_llm_prompt_event(
        trace_logger,
        request_id=request_id,
        stage=stage,
        model=model,
        messages=messages,
        estimated_tokens=prompt_tokens,
        params={"temperature": temperature, "structured": True},
    )

    response = structured.invoke(messages)
    emit_llm_output_event(
        trace_logger,
        request_id=request_id,
        stage=stage,
        model=model,
        latency_ms=None,
        content=response.model_dump_json(exclude_none=True) if hasattr(response, "model_dump_json") else str(response),
        usage=None,
        extra={"structured": True},
    )
    return response


def _heuristic_abstract_plan(use_case: UseCase) -> _AbstractPlanningOutput:
    combined_context = " ".join(
        [
            _sanitize_text(use_case.title),
            _sanitize_text(use_case.business_problem),
            _sanitize_text(use_case.desired_outcome),
        ]
    ).lower()
    processing_label = "Decisioning"
    processing_purpose = "Apply the core business logic to classify, decide, or transform the payload."
    if any(token in combined_context for token in ("heuristic", "rule-based", "rule based", "deterministic")):
        processing_label = "Rule-Based Decisioning"
        processing_purpose = "Apply the explicit heuristic or rule-based business logic requested by the user."
    stages = [
        ArchitectureStage(
            id="stage_intake",
            name="Intake",
            purpose="Capture the incoming event or source item and normalize the payload.",
            required_capabilities=["Receive the input", "Normalize the working payload"],
            expected_inputs=["Incoming trigger or source item"],
            expected_outputs=["Normalized workflow payload"],
            dependencies=[],
            success_criteria=["The workflow receives the relevant item reliably."],
        ),
        ArchitectureStage(
            id="stage_processing",
            name=processing_label,
            purpose=processing_purpose,
            required_capabilities=["Interpret the payload", "Apply workflow business rules"],
            expected_inputs=["Normalized workflow payload"],
            expected_outputs=["Decision result or enriched payload"],
            dependencies=["stage_intake"],
            success_criteria=["The workflow produces a clear business decision or enrichment."],
        ),
        ArchitectureStage(
            id="stage_delivery",
            name="Outcome Handling",
            purpose="Persist, notify, or otherwise record the final outcome of the workflow.",
            required_capabilities=["Execute the final outcome step", "Record the result"],
            expected_inputs=["Decision result or enriched payload"],
            expected_outputs=["Stored outcome or outbound action result"],
            dependencies=["stage_processing"],
            success_criteria=["The final outcome is executed and traceable."],
        ),
    ]
    return _AbstractPlanningOutput(
        workflow_summary=_compact(
            f"Abstract workflow plan for '{use_case.title}' with intake, decisioning, and outcome handling stages.",
            max_chars=320,
        ),
        stages=stages,
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_processing",
                data_items=["normalized workflow payload"],
                notes="Normalized payload feeds the business logic stage.",
            ),
            ArchitectureDataFlowItem(
                source_stage_id="stage_processing",
                target_stage_id="stage_delivery",
                data_items=["decision result", "enriched payload"],
                notes="Decision result is used to drive the final workflow outcome.",
            ),
        ],
        assumptions=[
            "The trigger/source for the workflow can be identified during later architecture work.",
            "The final persistence or notification target can be grounded in the architect stage.",
        ],
        missing_information=[],
        handoff_notes=[
            "Architect agent should ground each stage into node candidates without changing the abstract stage intent."
        ],
        planning_ready=True,
    )


def _plan_abstract_workflow_with_structured_output(
    *,
    use_case: UseCase,
    user_query: str,
    clarification_state: PMClarificationState,
    model: Optional[str],
    request_id: Optional[str],
) -> _AbstractPlanningOutput:
    system_prompt = (
        "You are product_manager_agent for an n8n workflow assistant. "
        "Your role is abstract workflow planning only. "
        "Design how the workflow should behave stage by stage, but do not select node types, "
        "do not mention n8n nodes, do not mention credentials, do not mention parameter names, "
        "do not mention API endpoints, and do not generate workflow JSON. "
        "You are preparing a clean handoff for a future architect_agent that will later ground the plan into nodes. "
        "Preserve the user's exact semantics instead of broadening the request."
    )
    user_prompt = (
        "Create an abstract workflow plan from this request context.\n\n"
        f"{_problem_statement(use_case=use_case, user_query=user_query, clarification_state=clarification_state)}\n\n"
        "Rules:\n"
        "- Output planning-level stages only.\n"
        "- Use 2 to 6 stages when possible.\n"
        "- Each stage must define purpose, required_capabilities, expected_inputs, expected_outputs, dependencies, and success_criteria.\n"
        "- Stages must describe behavior, not implementation details.\n"
        "- Preserve the exact source system, trigger style, processing mode, and final outcome named by the user.\n"
        "- If the user said heuristic, rule-based, deterministic, manual, inbound, outgoing, receive, or send, keep that distinction explicit in the plan.\n"
        "- Distinguish receiving email from sending email; they are not interchangeable.\n"
        "- Do not invent validation, feedback, manual review, refinement, or approval stages unless the user explicitly asked for them.\n"
        "- Never output concrete node types or workflow JSON.\n"
        "- If the user explicitly named systems like Gmail, Slack, or Google Sheets, treat them as business/system context, not node choices.\n"
        "- Set planning_ready=true only if the abstract workflow plan is coherent enough to hand off to architect_agent.\n"
        "- If essential planning information is missing, set planning_ready=false and list the missing_information as concise user-facing clarification questions.\n"
        "- handoff_notes must explain what architect_agent should preserve when grounding this plan into nodes.\n"
    )
    output = _invoke_structured_output(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=_AbstractPlanningOutput,
        model=model,
        request_id=request_id,
        temperature=0.2,
        stage="multi_agent.product_manager.abstract_plan",
    )
    if not output.stages:
        raise RuntimeError("PM abstract planning returned no stages")
    return output


def _build_architecture_plan(
    *,
    use_case: UseCase,
    plan_output: _AbstractPlanningOutput,
) -> ArchitecturePlan:
    return ArchitecturePlan(
        use_case_id=use_case.id,
        title=use_case.title,
        business_objective=_compact(use_case.business_problem, max_chars=260),
        desired_outcome=_compact(use_case.desired_outcome, max_chars=260),
        workflow_summary=_compact(plan_output.workflow_summary, max_chars=320),
        stages=list(plan_output.stages),
        data_flow=list(plan_output.data_flow),
        assumptions=_safe_list(plan_output.assumptions),
        missing_information=_safe_list(plan_output.missing_information),
        implementation_notes_for_engineer=_safe_list(plan_output.handoff_notes),
        required_nodes=[],
    )


def _derive_pm_stage_plan(plan: ArchitecturePlan) -> List[PMStagePlan]:
    return [
        PMStagePlan(
            id=stage.id,
            name=stage.name,
            objective=stage.purpose,
            expected_inputs=list(stage.expected_inputs),
            expected_outputs=list(stage.expected_outputs),
            success_criteria=list(stage.success_criteria),
            dependencies=list(stage.dependencies),
        )
        for stage in plan.stages
    ]


def _build_stage_progress(stage_plan: List[PMStagePlan], *, completed: bool) -> PMProgressState:
    stage_ids = [stage.id for stage in stage_plan]
    return PMProgressState(
        total_stages=len(stage_plan),
        current_stage_id=None if completed else (stage_ids[0] if stage_ids else None),
        completed_stage_ids=stage_ids if completed else [],
        blocked_stage_ids=[] if completed else stage_ids[:1],
        passes_by_stage={},
    )


def _planning_summary(
    *,
    use_case: UseCase,
    stage_plan: List[PMStagePlan],
    pm_status: PMStatus,
) -> str:
    return _compact(
        (
            f"PM status={pm_status.value}; use_case={use_case.id}; "
            f"abstract_stages={len(stage_plan)}; handoff_target=architect_agent."
        ),
        max_chars=260,
    )


def product_manager_agent_node(state: MultiAgentGraphState) -> Dict[str, Any]:
    model, request_id = _runtime_context(state)
    routing_signals = list(state.get("routing_signals") or [])
    if "entered_product_manager_agent" not in routing_signals:
        routing_signals.append("entered_product_manager_agent")

    missing_user_inputs = list(state.get("missing_user_inputs") or [])
    entry_intent = state.get("entry_intent")
    user_query = str(state.get("user_query") or "")

    selected_use_case = _normalize_use_case(state.get("selected_use_case"))
    if selected_use_case is None and entry_intent == EntryIntent.workflow_build_request:
        selected_use_case = _derive_use_case_from_direct_build_request(user_query)
        if selected_use_case is not None:
            routing_signals.append("pm_use_case_derived_from_direct_build_request")

    if selected_use_case is None:
        message = "A selected use case or direct workflow build request is required before PM planning can continue."
        missing_user_inputs = _safe_list(missing_user_inputs + [message])
        routing_signals.append("pm_missing_selected_use_case")
        workflow_context = WorkflowContext(
            use_case_id="unknown",
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            notes=["product_manager_requires_selected_use_case", "abstract_plan_only"],
        )
        return {
            "current_stage": "product_manager_agent",
            "pm_status": PMStatus.pm_failed_no_solution,
            "pm_stage_plan": [],
            "pm_stage_selections": [],
            "pm_stage_progress": PMProgressState(total_stages=0),
            "pm_clarification_state": PMClarificationState(
                attempts_used=0,
                max_attempts=_pm_int("PM_MAX_USER_CLARIFICATIONS", 2),
                pending_questions=[],
                turns=[],
            ),
            "pm_stage_search_history": [],
            "pm_reasoning_trace_full": [],
            "architecture_plan": None,
            "workflow_context": workflow_context,
            "planning_summary": None,
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    pm_status = _normalize_pm_status(state.get("pm_status"))
    clarification_state = _normalize_model(state.get("pm_clarification_state"), PMClarificationState)
    if clarification_state is None:
        clarification_state = PMClarificationState(
            attempts_used=0,
            max_attempts=_pm_int("PM_MAX_USER_CLARIFICATIONS", 2),
            pending_questions=[],
            turns=[],
        )

    if pm_status == PMStatus.pm_blocked_waiting_user and clarification_state.pending_questions and user_query:
        clarification_state = _record_clarification_answer(clarification_state, user_query)
        if "pm_clarification_answer_received" not in routing_signals:
            routing_signals.append("pm_clarification_answer_received")

    try:
        plan_output = _plan_abstract_workflow_with_structured_output(
            use_case=selected_use_case,
            user_query=user_query,
            clarification_state=clarification_state,
            model=model,
            request_id=request_id,
        )
    except Exception as exc:
        logger.warning("pm abstract planning fallback triggered: %s", str(exc))
        plan_output = _heuristic_abstract_plan(selected_use_case)

    architecture_plan = _build_architecture_plan(use_case=selected_use_case, plan_output=plan_output)
    stage_plan = _derive_pm_stage_plan(architecture_plan)
    clarification_questions = _safe_list(plan_output.missing_information)

    reasoning_trace = [
        {
            "planning_ready": bool(plan_output.planning_ready),
            "stage_count": len(stage_plan),
            "missing_information_count": len(clarification_questions),
        }
    ]

    if not stage_plan:
        pm_status = PMStatus.pm_failed_no_solution
        missing_user_inputs = _safe_list(
            missing_user_inputs + ["PM could not derive any abstract workflow stages from the request."]
        )
        routing_signals.append("pm_failed_no_solution")
        workflow_context = WorkflowContext(
            use_case_id=selected_use_case.id,
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            notes=["pm_failed_no_solution", "abstract_plan_only"],
        )
        return {
            "current_stage": "product_manager_agent",
            "pm_status": pm_status,
            "pm_stage_plan": [],
            "pm_stage_selections": [],
            "pm_stage_progress": PMProgressState(total_stages=0),
            "pm_clarification_state": clarification_state,
            "pm_stage_search_history": [],
            "pm_reasoning_trace_full": reasoning_trace,
            "architecture_plan": None,
            "workflow_context": workflow_context,
            "planning_summary": None,
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    if clarification_questions or not plan_output.planning_ready:
        if clarification_state.attempts_used < clarification_state.max_attempts:
            clarification_state = clarification_state.model_copy(
                update={
                    "attempts_used": clarification_state.attempts_used + 1,
                    "pending_questions": clarification_questions,
                    "turns": list(clarification_state.turns)
                    + [PMClarificationTurn(question=question, answer=None) for question in clarification_questions],
                }
            )
            pm_status = PMStatus.pm_blocked_waiting_user
            missing_user_inputs = _safe_list(missing_user_inputs + clarification_questions)
            if "pm_blocked_waiting_user" not in routing_signals:
                routing_signals.append("pm_blocked_waiting_user")
        else:
            pm_status = PMStatus.pm_failed_no_solution
            missing_user_inputs = _safe_list(missing_user_inputs + clarification_questions)
            if "pm_failed_no_solution" not in routing_signals:
                routing_signals.append("pm_failed_no_solution")

        workflow_context = WorkflowContext(
            use_case_id=selected_use_case.id,
            planning_ready=False,
            handoff_target=None,
            required_node_types=[],
            unresolved_inputs=missing_user_inputs,
            notes=[
                f"pm_status={pm_status.value}",
                "abstract_plan_only",
                "waiting_for_clarification",
            ],
        )
        stage_progress = _build_stage_progress(stage_plan, completed=False)
        emit_trace_event(
            trace_logger,
            event="pm_abstract_plan",
            request_id=request_id,
            stage="multi_agent.product_manager",
            payload={
                "pm_status": pm_status.value,
                "use_case_id": selected_use_case.id,
                "title": architecture_plan.title,
                "workflow_summary": architecture_plan.workflow_summary,
                "stage_count": len(stage_plan),
                "stages": _stage_trace_summary(stage_plan),
                "missing_information": _safe_list(architecture_plan.missing_information),
                "assumptions": _safe_list(architecture_plan.assumptions)[:6],
                "clarification_attempts": clarification_state.attempts_used,
                "planning_ready": False,
                "handoff_target": None,
            },
        )
        return {
            "current_stage": "product_manager_agent",
            "pm_status": pm_status,
            "pm_stage_plan": stage_plan,
            "pm_stage_selections": [],
            "pm_stage_progress": stage_progress,
            "pm_clarification_state": clarification_state,
            "pm_stage_search_history": [],
            "pm_reasoning_trace_full": reasoning_trace,
            "architecture_plan": architecture_plan,
            "workflow_context": workflow_context,
            "planning_summary": _planning_summary(
                use_case=selected_use_case,
                stage_plan=stage_plan,
                pm_status=pm_status,
            ),
            "missing_user_inputs": missing_user_inputs,
            "routing_signals": routing_signals,
            "target_stage": None,
            "proposed_nodes": [],
            "required_credentials": [],
        }

    pm_status = PMStatus.pm_completed
    if "handoff_ready_architect" not in routing_signals:
        routing_signals.append("handoff_ready_architect")

    workflow_context = WorkflowContext(
        use_case_id=selected_use_case.id,
        planning_ready=True,
        handoff_target=AgentStage.architect_agent,
        required_node_types=[],
        unresolved_inputs=[],
        notes=[
            f"pm_status={pm_status.value}",
            "abstract_plan_only",
            "handoff_target=architect_agent",
        ],
    )
    stage_progress = _build_stage_progress(stage_plan, completed=True)

    emit_trace_event(
        trace_logger,
        event="pm_abstract_plan",
        request_id=request_id,
        stage="multi_agent.product_manager",
        payload={
            "pm_status": pm_status.value,
            "use_case_id": selected_use_case.id,
            "title": architecture_plan.title,
            "workflow_summary": architecture_plan.workflow_summary,
            "stage_count": len(stage_plan),
            "stages": _stage_trace_summary(stage_plan),
            "missing_information": _safe_list(architecture_plan.missing_information),
            "assumptions": _safe_list(architecture_plan.assumptions)[:6],
            "clarification_attempts": clarification_state.attempts_used,
            "planning_ready": True,
            "handoff_target": AgentStage.architect_agent.value,
        },
    )

    return {
        "current_stage": "product_manager_agent",
        "pm_status": pm_status,
        "pm_stage_plan": stage_plan,
        "pm_stage_selections": [],
        "pm_stage_progress": stage_progress,
        "pm_clarification_state": clarification_state,
        "pm_stage_search_history": [],
        "pm_reasoning_trace_full": reasoning_trace,
        "architecture_plan": architecture_plan,
        "workflow_context": workflow_context,
        "planning_summary": _planning_summary(
            use_case=selected_use_case,
            stage_plan=stage_plan,
            pm_status=pm_status,
        ),
        "missing_user_inputs": [],
        "routing_signals": routing_signals,
        "target_stage": None,
        "proposed_nodes": [],
        "required_credentials": [],
    }
