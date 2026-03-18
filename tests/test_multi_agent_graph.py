from __future__ import annotations

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectStatus,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    BusinessContextSummary,
    EntryIntent,
    EntryRouterDecision,
    ImplementationStatus,
    NodeRequirement,
    UseCase,
    WorkflowContext,
)
from app.graphs.reasoning_graph import ReasoningGraphRuntime


def _decision(intent: EntryIntent, stage: AgentStage | None) -> EntryRouterDecision:
    return EntryRouterDecision(
        entry_intent=intent,
        target_stage=stage,
        confidence=0.8 if stage is not None else 0.3,
        routing_signals=["test_route"],
        missing_user_inputs=[],
    )


@pytest.mark.parametrize(
    ("intent", "stage"),
    [
        (EntryIntent.workflow_fix_request, AgentStage.qa_agent),
        (EntryIntent.information_request, AgentStage.consultant_agent),
    ],
)
def test_graph_routes_start_to_expected_stub(
    monkeypatch: pytest.MonkeyPatch,
    intent: EntryIntent,
    stage: AgentStage,
) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr("app.graphs.reasoning_graph.route_entry_intent", lambda **kwargs: _decision(intent, stage))
    if stage == AgentStage.consultant_agent:
        monkeypatch.setattr(
            "app.graphs.reasoning_graph.consultant_agent_node",
            lambda state: {
                "current_stage": "consultant_agent",
                "routing_signals": list(state.get("routing_signals") or []) + ["entered_consultant_agent"],
                "consultant_response": {"text": "Informational answer."},
                "consultant_selected_sources": ["conversation_history"],
                "consultant_tools_used": [],
                "consultant_retrieval_results": [],
                "consultant_used_retrieval": False,
                "consultant_notes": ["test_stub"],
            },
        )

    result = runtime.run(
        user_prompt="route me",
        model=None,
        request_id="req-1",
        existing_workflow=None,
    )
    assert result.entry_intent == intent
    assert result.target_stage == stage
    assert result.current_stage == stage.value
    assert result.status == "stub_routed"


def test_graph_routes_direct_build_request_to_architect_via_product_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.product_manager_agent_node",
        lambda state: {
            "current_stage": "product_manager_agent",
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_architect"],
            "pm_status": "pm_completed",
            "workflow_context": WorkflowContext(
                use_case_id="uc_build",
                planning_ready=True,
                handoff_target=AgentStage.architect_agent,
                required_node_types=[],
                unresolved_inputs=[],
                notes=["test"],
            ),
            "architecture_plan": ArchitecturePlan(
                use_case_id="uc_build",
                title="Build flow",
                business_objective="Deliver build request",
                desired_outcome="Produce workflow candidate",
                workflow_summary="PM handoff summary",
                stages=[
                    ArchitectureStage(
                        id="stage_1",
                        name="Stage 1",
                        purpose="Capture input",
                        required_capabilities=["Capture event"],
                        expected_inputs=["user request"],
                        expected_outputs=["normalized payload"],
                        dependencies=[],
                        success_criteria=["Input is captured and normalized."],
                    )
                ],
                data_flow=[],
                assumptions=[],
                missing_information=[],
                implementation_notes_for_engineer=[],
                required_nodes=[],
            ),
        },
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.architect_agent_node",
        lambda state: {
            "current_stage": "architect_agent",
            "target_stage": AgentStage.engineer_agent,
            "architect_status": ArchitectStatus.architect_completed,
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_engineer"],
            "workflow_context": WorkflowContext(
                use_case_id="uc_build",
                planning_ready=True,
                handoff_target=AgentStage.engineer_agent,
                required_node_types=["n8n-nodes-base.manualTrigger"],
                unresolved_inputs=[],
                notes=["architect_complete"],
            ),
            "workflow_draft": {
                "name": "Build flow",
                "nodes": [{"node_id": "an_1", "name": "manual_trigger_1", "node_type": "n8n-nodes-base.manualTrigger"}],
                "connections": [],
            },
            "final_workflow_json": {"name": "Build flow", "nodes": [{"id": "an_1"}], "connections": {}},
        },
    )

    result = runtime.run(
        user_prompt="Build a new workflow from scratch.",
        model=None,
        request_id="req-build-1",
        existing_workflow=None,
    )
    assert result.current_stage == "architect_agent"
    assert result.target_stage == AgentStage.engineer_agent
    assert result.workflow_context is not None
    assert result.workflow_context.handoff_target == AgentStage.engineer_agent
    assert result.status == "stub_routed"


def test_graph_unknown_route_ends_without_forcing_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.unknown, None),
    )

    result = runtime.run(
        user_prompt="ambiguous",
        model=None,
        request_id="req-2",
        existing_workflow=None,
    )
    assert result.entry_intent == EntryIntent.unknown
    assert result.target_stage is None
    assert result.current_stage is None
    assert result.status == "unknown_terminal"


def test_graph_compiles_or_uses_sequential_fallback() -> None:
    runtime = ReasoningGraphRuntime()
    assert runtime._graph is None or hasattr(runtime._graph, "invoke")


def test_graph_business_discovery_sets_handoff_ready_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.commercial_agent_node",
        lambda state: {
            "current_stage": "commercial_agent",
            "business_context_summary": BusinessContextSummary(
                process_scope="Finance operations",
                pain_points=["manual approvals"],
                desired_outcomes=["faster approvals"],
                constraints=[],
            ),
            "discovered_use_cases": [
                UseCase(
                    id="uc_1",
                    title="Invoice approval reminders",
                    business_problem="Invoice approvals are delayed due to manual follow-up.",
                    desired_outcome="Automate reminders and escalation for pending approvals.",
                    expected_value="Reduce payment delays and improve throughput.",
                    feasibility="medium",
                    priority_score=86.0,
                    why_selected="Selected for high business value.",
                )
            ],
            "selected_use_case": UseCase(
                id="uc_1",
                title="Invoice approval reminders",
                business_problem="Invoice approvals are delayed due to manual follow-up.",
                desired_outcome="Automate reminders and escalation for pending approvals.",
                expected_value="Reduce payment delays and improve throughput.",
                feasibility="medium",
                priority_score=86.0,
                why_selected="Selected for high business value.",
            ),
            "alternative_use_cases": [],
            "selection_reason": "Highest value and actionable business outcome.",
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_product_manager"],
            "target_stage": AgentStage.product_manager_agent,
            "missing_user_inputs": [],
        },
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.product_manager_agent_node",
        lambda state: {
            "current_stage": "product_manager_agent",
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_architect"],
            "pm_status": "pm_completed",
            "architecture_plan": ArchitecturePlan(
                use_case_id="uc_1",
                title="Invoice approval reminders",
                business_objective="Reduce approval delays",
                desired_outcome="Automate reminders and escalation",
                workflow_summary="Three-stage architecture from intake to delivery.",
                stages=[
                    ArchitectureStage(
                        id="stage_intake",
                        name="Intake",
                        purpose="Capture approval event",
                        required_capabilities=["Capture trigger event"],
                        expected_inputs=["Approval event"],
                        expected_outputs=["Normalized payload"],
                        dependencies=[],
                        success_criteria=["The approval event is normalized for downstream stages."],
                    )
                ],
                data_flow=[
                    ArchitectureDataFlowItem(
                        source_stage_id="stage_intake",
                        target_stage_id="stage_intake",
                        data_items=["payload"],
                    )
                ],
                assumptions=["Approval source is stable"],
                missing_information=[],
                implementation_notes_for_engineer=["Architect agent should preserve this stage intent."],
                required_nodes=[],
            ),
            "workflow_context": WorkflowContext(
                use_case_id="uc_1",
                planning_ready=True,
                handoff_target=AgentStage.architect_agent,
                required_node_types=[],
                unresolved_inputs=[],
                notes=["abstract_plan_only"],
            ),
            "planning_summary": "Handoff ready for architect with abstract stages.",
        },
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.architect_agent_node",
        lambda state: {
            "current_stage": "architect_agent",
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_engineer"],
            "target_stage": AgentStage.engineer_agent,
            "architect_status": ArchitectStatus.architect_completed,
            "workflow_context": WorkflowContext(
                use_case_id="uc_1",
                planning_ready=True,
                handoff_target=AgentStage.engineer_agent,
                required_node_types=["n8n-nodes-base.webhook"],
                unresolved_inputs=[],
                notes=["architect_complete"],
            ),
            "workflow_draft": {
                "name": "Invoice approval reminders",
                "nodes": [{"node_id": "an_1", "node_type": "n8n-nodes-base.webhook"}],
                "connections": [],
            },
            "final_workflow_json": {"nodes": [{"id": "an_1"}], "connections": {}},
        },
    )
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
    )

    result = runtime.run(
        user_prompt="We have manual approval bottlenecks in finance.",
        model=None,
        request_id="req-commercial-1",
        existing_workflow=None,
    )
    assert result.current_stage == "architect_agent"
    assert result.selected_use_case is not None
    assert result.target_stage == AgentStage.engineer_agent
    assert "handoff_ready_product_manager" in result.routing_signals
    assert "handoff_ready_architect" in result.routing_signals
    assert "handoff_ready_engineer" in result.routing_signals
    assert result.architecture_plan is not None
    assert result.workflow_context is not None
    assert result.workflow_context.handoff_target == AgentStage.engineer_agent
    assert result.status == "stub_routed"


def test_graph_business_discovery_no_actionable_case_ends_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.commercial_agent_node",
        lambda state: {
            "current_stage": "commercial_agent",
            "business_context_summary": BusinessContextSummary(
                process_scope="General operations",
                pain_points=[],
                desired_outcomes=[],
                constraints=[],
            ),
            "discovered_use_cases": [],
            "selected_use_case": None,
            "alternative_use_cases": [],
            "selection_reason": "No actionable use case was extracted from the current input.",
            "routing_signals": list(state.get("routing_signals") or []) + ["no_actionable_use_case"],
            "target_stage": None,
            "missing_user_inputs": ["Please provide one concrete process and desired outcome."],
        },
    )
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.business_discovery_conversation, AgentStage.commercial_agent),
    )

    result = runtime.run(
        user_prompt="We should automate something eventually.",
        model=None,
        request_id="req-commercial-2",
        existing_workflow=None,
    )
    assert result.current_stage == "commercial_agent"
    assert result.selected_use_case is None
    assert result.alternative_use_cases == []
    assert "no actionable use case" in (result.selection_reason or "").lower()
    assert result.status == "unknown_terminal"


def test_router_fix_path_does_not_call_commercial_or_pm_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.workflow_fix_request, AgentStage.qa_agent),
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.commercial_agent_node",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("commercial agent should not run on router fix path")
        ),
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.product_manager_agent_node",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("product manager should not run on router fix path")
        ),
    )

    result = runtime.run(
        user_prompt="Fix this workflow, timeout error on HTTP Request.",
        model=None,
        request_id="req-router-no-ext",
        existing_workflow=None,
    )
    assert result.entry_intent == EntryIntent.workflow_fix_request
    assert result.current_stage == "qa_agent"
    assert result.status == "stub_routed"


def test_direct_engineer_route_blocks_when_handoff_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.workflow_edit_request, AgentStage.engineer_agent),
    )

    result = runtime.run(
        user_prompt="Update this workflow and fix mappings.",
        model=None,
        request_id="req-engineer-block",
        existing_workflow=None,
    )
    assert result.current_stage == "engineer_agent"
    assert result.implementation_status == ImplementationStatus.blocked_waiting_user
    assert result.target_stage is None
    assert result.status == "unknown_terminal"


def test_runtime_resumes_blocked_engineer_state_on_same_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.workflow_edit_request, AgentStage.engineer_agent),
    )

    call_state = {"count": 0}

    def _engineer_node(state):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return {
                "current_stage": "engineer_agent",
                "target_stage": None,
                "implementation_status": ImplementationStatus.blocked_waiting_user,
                "routing_signals": list(state.get("routing_signals") or []),
                "final_workflow_json": {},
            }
        return {
            "current_stage": "engineer_agent",
            "target_stage": AgentStage.qa_agent,
            "implementation_status": ImplementationStatus.completed,
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_qa"],
            "final_workflow_json": {"nodes": [{"id": "n1"}], "connections": {}},
        }

    monkeypatch.setattr("app.graphs.reasoning_graph.engineer_agent_node", _engineer_node)

    run_config = {"configurable": {"thread_id": "conv-resume-1"}}
    first = runtime.run(
        user_prompt="Update this workflow to call an API.",
        model=None,
        request_id="req-resume-1",
        existing_workflow=None,
        run_config=run_config,
    )
    assert first.current_stage == "engineer_agent"
    assert first.implementation_status == ImplementationStatus.blocked_waiting_user
    assert first.final_workflow_json == {}

    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.unknown, None),
    )
    second = runtime.run(
        user_prompt="parameter:pn_1:url=https://api.example.com/events",
        model=None,
        request_id="req-resume-2",
        existing_workflow=None,
        run_config=run_config,
    )
    assert second.current_stage == "engineer_agent"
    assert second.implementation_status == ImplementationStatus.completed
    assert second.final_workflow_json.get("nodes")
    assert second.target_stage == AgentStage.qa_agent


def test_runtime_resumes_blocked_pm_state_on_same_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ReasoningGraphRuntime()
    route_calls = {"count": 0}

    def _route(**kwargs):
        route_calls["count"] += 1
        return _decision(EntryIntent.workflow_build_request, AgentStage.product_manager_agent)

    monkeypatch.setattr("app.graphs.reasoning_graph.route_entry_intent", _route)

    def _pm_node(state):
        if state.get("pm_status") == "pm_blocked_waiting_user":
            return {
                "current_stage": "product_manager_agent",
                "pm_status": "pm_completed",
                "pm_stage_plan": [
                    {
                        "id": "stage_1",
                        "name": "Intake",
                        "objective": "Capture input",
                        "expected_inputs": ["event"],
                        "expected_outputs": ["payload"],
                        "success_criteria": ["captured"],
                        "dependencies": [],
                    }
                ],
                "pm_stage_selections": [],
                "pm_stage_progress": {
                    "total_stages": 1,
                    "current_stage_id": None,
                    "completed_stage_ids": ["stage_1"],
                    "blocked_stage_ids": [],
                    "passes_by_stage": {},
                },
                "pm_clarification_state": {
                    "attempts_used": 1,
                    "max_attempts": 2,
                    "pending_questions": [],
                    "turns": [
                        {
                            "stage_id": "stage_1",
                            "question": "Need details",
                            "answer": "Use webhook + Google Sheets",
                        }
                    ],
                },
                "target_stage": None,
                "workflow_context": WorkflowContext(
                    use_case_id="uc_pm_resume",
                    planning_ready=True,
                    handoff_target=AgentStage.architect_agent,
                    required_node_types=[],
                    unresolved_inputs=[],
                    notes=["pm-resumed"],
                ),
                "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_architect"],
            }
        return {
            "current_stage": "product_manager_agent",
            "pm_status": "pm_blocked_waiting_user",
            "pm_stage_plan": [],
            "pm_stage_selections": [],
            "pm_stage_progress": {
                "total_stages": 1,
                "current_stage_id": "stage_1",
                "completed_stage_ids": [],
                "blocked_stage_ids": ["stage_1"],
                "passes_by_stage": {"stage_1": 3},
            },
            "pm_clarification_state": {
                "attempts_used": 1,
                "max_attempts": 2,
                "pending_questions": ["Need details"],
                "turns": [{"stage_id": "stage_1", "question": "Need details", "answer": None}],
            },
            "target_stage": None,
            "routing_signals": list(state.get("routing_signals") or []) + ["pm_blocked_waiting_user"],
        }

    monkeypatch.setattr("app.graphs.reasoning_graph.product_manager_agent_node", _pm_node)
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.architect_agent_node",
        lambda state: {
            "current_stage": "architect_agent",
            "architect_status": ArchitectStatus.architect_completed,
            "target_stage": AgentStage.engineer_agent,
            "workflow_context": WorkflowContext(
                use_case_id="uc_pm_resume",
                planning_ready=True,
                handoff_target=AgentStage.engineer_agent,
                required_node_types=["n8n-nodes-base.webhook"],
                unresolved_inputs=[],
                notes=["architect_complete"],
            ),
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_engineer"],
            "workflow_draft": {
                "name": "PM Resume",
                "nodes": [{"node_id": "an_1", "node_type": "n8n-nodes-base.webhook"}],
                "connections": [],
            },
            "final_workflow_json": {"name": "PM Resume", "nodes": [{"id": "an_1"}], "connections": {}},
        },
    )

    run_config = {"configurable": {"thread_id": "thread-pm-resume-1"}}
    first = runtime.run(
        user_prompt="Build a workflow",
        model=None,
        request_id="req-pm-1",
        existing_workflow=None,
        run_config=run_config,
    )
    assert first.pm_status == "pm_blocked_waiting_user"
    assert first.current_stage == "product_manager_agent"

    second = runtime.run(
        user_prompt="Use webhook and persist into sheets",
        model=None,
        request_id="req-pm-2",
        existing_workflow=None,
        run_config=run_config,
    )
    assert route_calls["count"] == 1
    assert second.current_stage == "architect_agent"
    assert second.pm_status == "pm_completed"
    assert second.target_stage == AgentStage.engineer_agent
    assert second.workflow_context is not None
    assert second.workflow_context.handoff_target == AgentStage.engineer_agent
    assert "resume_pm_from_checkpoint" in second.routing_signals
    assert "handoff_ready_engineer" in second.routing_signals


def test_runtime_resumes_blocked_architect_state_on_same_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
    )

    monkeypatch.setattr(
        "app.graphs.reasoning_graph.product_manager_agent_node",
        lambda state: {
            "current_stage": "product_manager_agent",
            "pm_status": "pm_completed",
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_architect"],
            "workflow_context": WorkflowContext(
                use_case_id="uc_arch",
                planning_ready=True,
                handoff_target=AgentStage.architect_agent,
                required_node_types=[],
                unresolved_inputs=[],
                notes=["pm_complete"],
            ),
            "architecture_plan": ArchitecturePlan(
                use_case_id="uc_arch",
                title="Build flow",
                business_objective="Build a flow",
                desired_outcome="Produce workflow draft",
                workflow_summary="summary",
                stages=[
                    ArchitectureStage(
                        id="stage_1",
                        name="Trigger",
                        purpose="Receive input",
                        required_capabilities=["Receive input"],
                        expected_inputs=["event"],
                        expected_outputs=["payload"],
                        dependencies=[],
                        success_criteria=["captured"],
                    )
                ],
                data_flow=[],
                assumptions=[],
                missing_information=[],
                implementation_notes_for_engineer=[],
                required_nodes=[],
            ),
        },
    )

    calls = {"count": 0}

    def _architect_node(state):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "current_stage": "architect_agent",
                "architect_status": ArchitectStatus.architect_blocked_waiting_user,
                "target_stage": None,
                "routing_signals": list(state.get("routing_signals") or []),
                "missing_user_inputs": ["Which trigger should start the workflow?"],
                "architect_clarification_state": {
                    "attempts_used": 1,
                    "max_attempts": 3,
                    "pending_questions": ["Which trigger should start the workflow?"],
                    "turns": [{"stage_id": "stage_1", "question": "Which trigger should start the workflow?", "answer": None}],
                },
                "workflow_context": WorkflowContext(
                    use_case_id="uc_arch",
                    planning_ready=False,
                    handoff_target=None,
                    required_node_types=[],
                    unresolved_inputs=["Which trigger should start the workflow?"],
                    notes=["architect_blocked"],
                ),
            }
        return {
            "current_stage": "architect_agent",
            "architect_status": ArchitectStatus.architect_completed,
            "target_stage": AgentStage.engineer_agent,
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_engineer"],
            "workflow_context": WorkflowContext(
                use_case_id="uc_arch",
                planning_ready=True,
                handoff_target=AgentStage.engineer_agent,
                required_node_types=["n8n-nodes-base.webhook"],
                unresolved_inputs=[],
                notes=["architect_complete"],
            ),
            "final_workflow_json": {"nodes": [{"id": "an_1"}], "connections": {}},
        }

    monkeypatch.setattr("app.graphs.reasoning_graph.architect_agent_node", _architect_node)

    run_config = {"configurable": {"thread_id": "thread-architect-resume-1"}}
    first = runtime.run(
        user_prompt="Build a workflow to receive leads.",
        model=None,
        request_id="req-architect-1",
        existing_workflow=None,
        run_config=run_config,
    )
    assert first.current_stage == "architect_agent"
    assert first.architect_status == ArchitectStatus.architect_blocked_waiting_user

    second = runtime.run(
        user_prompt="Use a webhook trigger.",
        model=None,
        request_id="req-architect-2",
        existing_workflow=None,
        run_config=run_config,
    )
    assert second.current_stage == "architect_agent"
    assert second.architect_status == ArchitectStatus.architect_completed
    assert second.target_stage == AgentStage.engineer_agent
    assert "resume_architect_from_checkpoint" in second.routing_signals
