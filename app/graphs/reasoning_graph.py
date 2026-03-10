from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from ..core.json_sanitize import sanitize_for_json
from ..features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitecturePlan,
    BlockedNode,
    BusinessContextSummary,
    EntryIntent,
    ImplementationQueueItem,
    ImplementationStatus,
    ImplementedNode,
    PMClarificationState,
    PMProgressState,
    PMStagePlan,
    PMStageSearchState,
    PMStageSelection,
    PMStatus,
    MissingUserInput,
    MultiAgentGraphResult,
    ProposedNode,
    RequiredCredential,
    UseCase,
    VariableDefinition,
    WorkflowDraft,
    WorkflowVersion,
    WorkflowContext,
)
from .multi_agent_state import MultiAgentGraphState
from .nodes.commercial_agent import commercial_agent_node
from .nodes.engineer_agent import engineer_agent_node
from .nodes.multi_agent_router import route_entry_intent
from .nodes.product_manager_agent import product_manager_agent_node
from .nodes.multi_agent_stubs import consultant_agent_node, qa_agent_node

try:  # Optional until langgraph dependency is installed.
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency fallback
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]

try:  # Optional checkpointer support.
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover - optional dependency fallback
    MemorySaver = None  # type: ignore[assignment]


logger = logging.getLogger("n8n-assistant")

T = TypeVar("T")


def _sanitize_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = sanitize_for_json(updates)
    if isinstance(sanitized, dict):
        return sanitized
    return {}


def _stage_value(stage: Any) -> Optional[str]:
    if isinstance(stage, AgentStage):
        return stage.value
    if isinstance(stage, str):
        return stage
    return None


def _thread_id_from_run_config(run_config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(run_config, dict):
        return None
    configurable = run_config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id.strip()
    return None


def _resolve_existing_workflow_ref(
    existing_workflow: Any,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if isinstance(existing_workflow, str) and existing_workflow.strip():
        return existing_workflow.strip(), None, None
    if isinstance(existing_workflow, dict):
        workflow_id = existing_workflow.get("id") or existing_workflow.get("workflow_id")
        workflow_name = existing_workflow.get("name")
        workflow_url = existing_workflow.get("url")
        resolved_id = str(workflow_id).strip() if isinstance(workflow_id, (str, int)) else None
        resolved_name = str(workflow_name).strip() if isinstance(workflow_name, str) else None
        resolved_url = str(workflow_url).strip() if isinstance(workflow_url, str) else None
        return resolved_id, resolved_name, resolved_url
    return None, None, None


def _runtime_context(state: MultiAgentGraphState) -> Dict[str, Any]:
    context = state.get("runtime_context")
    if isinstance(context, dict):
        return context
    workflow_context = state.get("workflow_context")
    if isinstance(workflow_context, dict):
        return workflow_context
    return {}


def _normalize_entry_intent(value: Any) -> EntryIntent:
    if isinstance(value, EntryIntent):
        return value
    if isinstance(value, str):
        try:
            return EntryIntent(value)
        except ValueError:
            return EntryIntent.unknown
    return EntryIntent.unknown


def _normalize_impl_status(value: Any) -> Optional[ImplementationStatus]:
    if isinstance(value, ImplementationStatus):
        return value
    if isinstance(value, str):
        try:
            return ImplementationStatus(value)
        except ValueError:
            return None
    return None


def _normalize_pm_status(value: Any) -> Optional[PMStatus]:
    if isinstance(value, PMStatus):
        return value
    if isinstance(value, str):
        try:
            return PMStatus(value)
        except ValueError:
            return None
    return None


def _model_or_none(value: Any, model_cls: Type[T]) -> Optional[T]:
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        try:
            return model_cls.model_validate(value)  # type: ignore[attr-defined]
        except Exception:
            return None
    return None


def _model_list(values: Any, model_cls: Type[T]) -> List[T]:
    if not isinstance(values, list):
        return []
    output: List[T] = []
    for value in values:
        model = _model_or_none(value, model_cls)
        if model is not None:
            output.append(model)
    return output


def _is_engineer_blocked_state(state: Optional[MultiAgentGraphState]) -> bool:
    if not isinstance(state, dict):
        return False
    if state.get("current_stage") != "engineer_agent":
        return False
    status = _normalize_impl_status(state.get("implementation_status"))
    return status == ImplementationStatus.blocked_waiting_user


def _is_pm_blocked_state(state: Optional[MultiAgentGraphState]) -> bool:
    if not isinstance(state, dict):
        return False
    if state.get("current_stage") != "product_manager_agent":
        return False
    status = _normalize_pm_status(state.get("pm_status"))
    return status == PMStatus.pm_blocked_waiting_user


def _route_after_entry(state: MultiAgentGraphState) -> str:
    target = _stage_value(state.get("target_stage"))
    if target == "commercial_agent":
        return "commercial_agent"
    if target == "consultant_agent":
        return "consultant_agent"
    if target == "product_manager_agent":
        return "product_manager_agent"
    if target == "engineer_agent":
        return "engineer_agent"
    if target == "qa_agent":
        return "qa_agent"
    return "unknown"


def _route_after_commercial(state: MultiAgentGraphState) -> str:
    target = _stage_value(state.get("target_stage"))
    selected_use_case = state.get("selected_use_case")
    if target == "product_manager_agent" and selected_use_case is not None:
        return "product_manager_agent"
    return "end"


def _route_after_product_manager(state: MultiAgentGraphState) -> str:
    target = _stage_value(state.get("target_stage"))
    if target == "engineer_agent":
        return "engineer_agent"
    return "end"


class ReasoningGraphRuntime:
    """Entry graph runtime for multi-agent routing and stage handoff."""

    def __init__(self) -> None:
        self._state_cache_by_thread: Dict[str, MultiAgentGraphState] = {}
        self._checkpointer = MemorySaver() if MemorySaver is not None else None
        self._graph = self._compile_graph()

    def _compile_graph(self) -> Any:
        if StateGraph is None:
            return None

        graph = StateGraph(MultiAgentGraphState)
        graph.add_node("entry_router", self._entry_router_node)
        graph.add_node("commercial_agent", self._commercial_node)
        graph.add_node("consultant_agent", self._consultant_node)
        graph.add_node("product_manager_agent", self._product_manager_node)
        graph.add_node("engineer_agent", self._engineer_node)
        graph.add_node("qa_agent", self._qa_node)

        graph.add_edge(START, "entry_router")
        graph.add_conditional_edges(
            "entry_router",
            _route_after_entry,
            {
                "commercial_agent": "commercial_agent",
                "consultant_agent": "consultant_agent",
                "product_manager_agent": "product_manager_agent",
                "engineer_agent": "engineer_agent",
                "qa_agent": "qa_agent",
                "unknown": END,
            },
        )
        graph.add_conditional_edges(
            "commercial_agent",
            _route_after_commercial,
            {
                "product_manager_agent": "product_manager_agent",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "product_manager_agent",
            _route_after_product_manager,
            {
                "engineer_agent": "engineer_agent",
                "end": END,
            },
        )
        graph.add_edge("consultant_agent", END)
        graph.add_edge("engineer_agent", END)
        graph.add_edge("qa_agent", END)

        compile_kwargs: Dict[str, Any] = {}
        if self._checkpointer is not None:
            compile_kwargs["checkpointer"] = self._checkpointer
        return graph.compile(**compile_kwargs)

    def _entry_router_node(self, state: MultiAgentGraphState) -> Dict[str, Any]:
        if bool(state.get("resume_requested")) and _is_engineer_blocked_state(state):
            signals = list(state.get("routing_signals") or [])
            if "resume_engineer_from_checkpoint" not in signals:
                signals.append("resume_engineer_from_checkpoint")
            return _sanitize_updates(
                {
                    "entry_intent": state.get("entry_intent") or EntryIntent.workflow_edit_request,
                    "target_stage": AgentStage.engineer_agent,
                    "confidence": float(state.get("confidence", 0.0) or 0.0),
                    "routing_signals": signals,
                    "current_stage": None,
                }
            )
        if bool(state.get("resume_requested")) and _is_pm_blocked_state(state):
            signals = list(state.get("routing_signals") or [])
            if "resume_pm_from_checkpoint" not in signals:
                signals.append("resume_pm_from_checkpoint")
            return _sanitize_updates(
                {
                    "entry_intent": state.get("entry_intent") or EntryIntent.workflow_build_request,
                    "target_stage": AgentStage.product_manager_agent,
                    "confidence": float(state.get("confidence", 0.0) or 0.0),
                    "routing_signals": signals,
                    "current_stage": None,
                }
            )

        runtime_context = _runtime_context(state)
        model = runtime_context.get("model")
        request_id = runtime_context.get("request_id")
        decision = route_entry_intent(
            user_query=state.get("user_query") or "",
            model=model if isinstance(model, str) or model is None else None,
            request_id=request_id if isinstance(request_id, str) or request_id is None else None,
        )
        return _sanitize_updates(
            {
                "entry_intent": decision.entry_intent,
                "target_stage": decision.target_stage,
                "confidence": decision.confidence,
                "routing_signals": list(decision.routing_signals),
                "missing_user_inputs": list(decision.missing_user_inputs),
                "current_stage": None,
            }
        )

    def _commercial_node(self, state: MultiAgentGraphState) -> Dict[str, Any]:
        return _sanitize_updates(commercial_agent_node(state))

    def _consultant_node(self, state: MultiAgentGraphState) -> Dict[str, Any]:
        return _sanitize_updates(consultant_agent_node(state))

    def _product_manager_node(self, state: MultiAgentGraphState) -> Dict[str, Any]:
        return _sanitize_updates(product_manager_agent_node(state))

    def _engineer_node(self, state: MultiAgentGraphState) -> Dict[str, Any]:
        return _sanitize_updates(engineer_agent_node(state))

    def _qa_node(self, state: MultiAgentGraphState) -> Dict[str, Any]:
        return _sanitize_updates(qa_agent_node(state))

    def _run_sequential(self, state: MultiAgentGraphState) -> MultiAgentGraphState:
        current = dict(state)
        current.update(self._entry_router_node(current))
        route = _route_after_entry(current)
        if route == "commercial_agent":
            current.update(self._commercial_node(current))
            if _route_after_commercial(current) == "product_manager_agent":
                current.update(self._product_manager_node(current))
                if _route_after_product_manager(current) == "engineer_agent":
                    current.update(self._engineer_node(current))
        elif route == "consultant_agent":
            current.update(self._consultant_node(current))
        elif route == "product_manager_agent":
            current.update(self._product_manager_node(current))
            if _route_after_product_manager(current) == "engineer_agent":
                current.update(self._engineer_node(current))
        elif route == "engineer_agent":
            current.update(self._engineer_node(current))
        elif route == "qa_agent":
            current.update(self._qa_node(current))
        return current

    @staticmethod
    def _build_result(state: MultiAgentGraphState) -> MultiAgentGraphResult:
        target_stage = _stage_value(state.get("target_stage"))
        target_stage_enum: Optional[AgentStage] = None
        if isinstance(target_stage, str):
            try:
                target_stage_enum = AgentStage(target_stage)
            except ValueError:
                target_stage_enum = None

        entry_intent = _normalize_entry_intent(state.get("entry_intent"))
        current_stage = state.get("current_stage")
        selected_use_case = _model_or_none(state.get("selected_use_case"), UseCase)
        architecture_plan = _model_or_none(state.get("architecture_plan"), ArchitecturePlan)
        workflow_context = _model_or_none(state.get("workflow_context"), WorkflowContext)
        implementation_status = _normalize_impl_status(state.get("implementation_status"))
        pm_status = _normalize_pm_status(state.get("pm_status"))
        final_workflow_json = (
            dict(state.get("final_workflow_json"))
            if isinstance(state.get("final_workflow_json"), dict)
            else {}
        )

        if current_stage == "commercial_agent" and selected_use_case is None:
            status = "unknown_terminal"
        elif current_stage == "product_manager_agent":
            planning_ready = bool(workflow_context and workflow_context.planning_ready)
            if not workflow_context:
                planning_ready = bool(
                    architecture_plan is not None and target_stage_enum == AgentStage.engineer_agent
                )
            if pm_status in (PMStatus.pm_blocked_waiting_user, PMStatus.pm_failed_no_solution):
                status = "unknown_terminal"
            else:
                status = "stub_routed" if planning_ready else "unknown_terminal"
        elif current_stage == "engineer_agent":
            if implementation_status == ImplementationStatus.completed and final_workflow_json:
                status = "stub_routed"
            elif implementation_status in (
                ImplementationStatus.blocked_waiting_user,
                ImplementationStatus.failed,
            ):
                status = "unknown_terminal"
            else:
                status = "stub_routed" if target_stage_enum is not None else "unknown_terminal"
        else:
            status = "stub_routed" if target_stage_enum is not None and current_stage else "unknown_terminal"

        return MultiAgentGraphResult(
            user_query=state.get("user_query") or "",
            entry_intent=entry_intent,
            target_stage=target_stage_enum,
            confidence=float(state.get("confidence", 0.0)),
            routing_signals=list(state.get("routing_signals") or []),
            current_stage=current_stage,
            missing_user_inputs=list(state.get("missing_user_inputs") or []),
            business_context_summary=_model_or_none(
                state.get("business_context_summary"), BusinessContextSummary
            ),
            discovered_use_cases=_model_list(state.get("discovered_use_cases"), UseCase),
            selected_use_case=selected_use_case,
            alternative_use_cases=_model_list(state.get("alternative_use_cases"), UseCase),
            selection_reason=state.get("selection_reason"),
            architecture_plan=architecture_plan,
            workflow_context=workflow_context,
            planning_summary=state.get("planning_summary"),
            pm_status=pm_status,
            pm_stage_plan=_model_list(state.get("pm_stage_plan"), PMStagePlan),
            pm_stage_selections=_model_list(state.get("pm_stage_selections"), PMStageSelection),
            pm_stage_progress=_model_or_none(state.get("pm_stage_progress"), PMProgressState),
            pm_clarification_state=_model_or_none(
                state.get("pm_clarification_state"), PMClarificationState
            ),
            proposed_nodes=_model_list(state.get("proposed_nodes"), ProposedNode),
            required_credentials=_model_list(state.get("required_credentials"), RequiredCredential),
            workflow_draft=_model_or_none(state.get("workflow_draft"), WorkflowDraft),
            workflow_versions=_model_list(state.get("workflow_versions"), WorkflowVersion),
            node_implementation_queue=_model_list(
                state.get("node_implementation_queue"), ImplementationQueueItem
            ),
            implemented_nodes=_model_list(state.get("implemented_nodes"), ImplementedNode),
            blocked_nodes=_model_list(state.get("blocked_nodes"), BlockedNode),
            variable_registry=_model_list(state.get("variable_registry"), VariableDefinition),
            missing_user_input_details=_model_list(
                state.get("missing_user_input_details"), MissingUserInput
            ),
            implementation_status=implementation_status,
            engineer_notes=list(state.get("engineer_notes") or []),
            final_workflow_json=final_workflow_json,
            active_workflow_id=(
                str(state.get("active_workflow_id")).strip()
                if isinstance(state.get("active_workflow_id"), str) and str(state.get("active_workflow_id")).strip()
                else None
            ),
            active_workflow_name=(
                str(state.get("active_workflow_name")).strip()
                if isinstance(state.get("active_workflow_name"), str) and str(state.get("active_workflow_name")).strip()
                else None
            ),
            active_workflow_url=(
                str(state.get("active_workflow_url")).strip()
                if isinstance(state.get("active_workflow_url"), str) and str(state.get("active_workflow_url")).strip()
                else None
            ),
            workflow_persisted=bool(state.get("workflow_persisted", False)),
            workflow_persist_action=(
                str(state.get("workflow_persist_action")).strip()
                if isinstance(state.get("workflow_persist_action"), str) and str(state.get("workflow_persist_action")).strip()
                else None
            ),
            workflow_api_sync_result=(
                dict(state.get("workflow_api_sync_result"))
                if isinstance(state.get("workflow_api_sync_result"), dict)
                else {}
            ),
            qa_enabled=bool(state.get("qa_enabled", False)),
            needs_replan=bool(state.get("needs_replan", False)),
            status=status,
        )

    def _load_previous_state(
        self,
        thread_id: Optional[str],
    ) -> Optional[MultiAgentGraphState]:
        if not thread_id:
            return None
        cached = self._state_cache_by_thread.get(thread_id)
        if isinstance(cached, dict):
            return dict(cached)

        if self._graph is None or not hasattr(self._graph, "get_state"):
            return None
        try:
            snapshot = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        except Exception:  # pragma: no cover - optional runtime path
            return None
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            return dict(values)
        return None

    def run(
        self,
        *,
        user_prompt: str,
        model: Optional[str],
        request_id: Optional[str],
        existing_workflow: Any,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> MultiAgentGraphResult:
        existing_workflow_id, existing_workflow_name, existing_workflow_url = _resolve_existing_workflow_ref(
            existing_workflow
        )
        thread_id = _thread_id_from_run_config(run_config)
        previous_state = self._load_previous_state(thread_id)
        resume_from_blocked = _is_engineer_blocked_state(previous_state) or _is_pm_blocked_state(previous_state)

        if resume_from_blocked and isinstance(previous_state, dict):
            state: MultiAgentGraphState = dict(previous_state)
            state["user_query"] = user_prompt or ""
            state["runtime_context"] = {
                "model": model,
                "request_id": request_id,
                "persist_to_n8n": True,
            }
            state["resume_requested"] = True
        else:
            state = {
                "user_query": user_prompt or "",
                "entry_intent": EntryIntent.unknown,
                "target_stage": None,
                "confidence": 0.0,
                "routing_signals": [],
                "current_stage": None,
                "business_context_summary": None,
                "discovered_use_cases": [],
                "selected_use_case": None,
                "alternative_use_cases": [],
                "selection_reason": None,
                "workflow_context": None,
                "architecture_plan": None,
                "planning_summary": None,
                "pm_status": None,
                "pm_stage_plan": [],
                "pm_stage_selections": [],
                "pm_stage_progress": None,
                "pm_clarification_state": None,
                "pm_stage_search_history": [],
                "pm_reasoning_trace_full": [],
                "proposed_nodes": [],
                "required_credentials": [],
                "workflow_draft": None,
                "workflow_versions": [],
                "node_implementation_queue": [],
                "implemented_nodes": [],
                "blocked_nodes": [],
                "variable_registry": [],
                "missing_user_inputs": [],
                "missing_user_input_details": [],
                "implementation_status": None,
                "engineer_notes": [],
                "qa_enabled": True,
                "qa_result": {},
                "needs_replan": False,
                "final_workflow_json": {},
                "active_workflow_id": existing_workflow_id,
                "active_workflow_name": existing_workflow_name,
                "active_workflow_url": existing_workflow_url,
                "workflow_persisted": False,
                "workflow_persist_action": None,
                "workflow_api_sync_result": {},
                "runtime_context": {
                    "model": model,
                    "request_id": request_id,
                    "persist_to_n8n": True,
                },
                "resume_requested": False,
            }

        if self._graph is None:
            result_state = self._run_sequential(state)
        else:
            try:
                result_state = self._graph.invoke(state, config=run_config)
            except Exception as exc:  # pragma: no cover - runtime fallback
                logger.warning("reasoning graph fallback to sequential runtime: %s", str(exc))
                result_state = self._run_sequential(state)

        if thread_id:
            self._state_cache_by_thread[thread_id] = dict(result_state)

        return self._build_result(result_state)
