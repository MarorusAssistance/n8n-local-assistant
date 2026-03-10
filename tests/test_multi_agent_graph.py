from __future__ import annotations

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
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


def test_graph_routes_direct_build_request_to_product_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = ReasoningGraphRuntime()
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.product_manager_agent_node",
        lambda state: {
            "current_stage": "product_manager_agent",
            "target_stage": AgentStage.engineer_agent,
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_engineer"],
            "workflow_context": WorkflowContext(
                use_case_id="uc_build",
                planning_ready=True,
                handoff_target=AgentStage.engineer_agent,
                required_node_types=["n8n-nodes-base.webhook"],
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
                    )
                ],
                data_flow=[],
                assumptions=[],
                missing_information=[],
                implementation_notes_for_engineer=[],
                required_nodes=[
                    NodeRequirement(
                        node_type="n8n-nodes-base.webhook",
                        why_required="Required trigger",
                        evidence_chunk_ids=["chunk-1"],
                        evidence_refs=["ref-1"],
                        evidence_confidence=0.8,
                    )
                ],
            ),
        },
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.engineer_agent_node",
        lambda state: {
            "current_stage": "engineer_agent",
            "target_stage": AgentStage.qa_agent,
            "implementation_status": ImplementationStatus.completed,
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_qa"],
            "final_workflow_json": {"nodes": [{"id": "n1"}], "connections": {}},
        },
    )

    result = runtime.run(
        user_prompt="Build a new workflow from scratch.",
        model=None,
        request_id="req-build-1",
        existing_workflow=None,
    )
    assert result.current_stage == "engineer_agent"
    assert result.target_stage == AgentStage.qa_agent
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
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_engineer"],
            "target_stage": AgentStage.engineer_agent,
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
                implementation_notes_for_engineer=["Configure node parameters in engineering phase."],
                required_nodes=[
                    NodeRequirement(
                        node_type="n8n-nodes-base.webhook",
                        why_required="Evidence-backed node for intake",
                        evidence_chunk_ids=["linked:node:n8n-nodes-base.webhook"],
                        evidence_refs=["Node: Webhook"],
                        evidence_confidence=0.86,
                        rerank_confidence=0.71,
                        blended_confidence=0.82,
                    )
                ],
            ),
            "workflow_context": WorkflowContext(
                use_case_id="uc_1",
                planning_ready=True,
                handoff_target=AgentStage.engineer_agent,
                required_node_types=["n8n-nodes-base.webhook"],
                unresolved_inputs=[],
                notes=["required_nodes=1"],
            ),
            "planning_summary": "Handoff ready for engineer with evidence-backed nodes.",
        },
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.engineer_agent_node",
        lambda state: {
            "current_stage": "engineer_agent",
            "target_stage": AgentStage.qa_agent,
            "implementation_status": ImplementationStatus.completed,
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_qa"],
            "final_workflow_json": {"nodes": [{"id": "n1"}], "connections": {}},
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
    assert result.current_stage == "engineer_agent"
    assert result.selected_use_case is not None
    assert result.target_stage == AgentStage.qa_agent
    assert "handoff_ready_product_manager" in result.routing_signals
    assert "handoff_ready_engineer" in result.routing_signals
    assert "handoff_ready_qa" in result.routing_signals
    assert result.architecture_plan is not None
    assert result.workflow_context is not None
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
        "app.graphs.nodes.engineer_agent.N8NClient.create_workflow",
        lambda self, payload: {"id": "wf_resume_1", "name": payload.get("name"), "url": "http://localhost:5678/workflow/wf_resume_1"},
    )

    monkeypatch.setattr(
        "app.graphs.reasoning_graph.route_entry_intent",
        lambda **kwargs: _decision(EntryIntent.workflow_build_request, AgentStage.product_manager_agent),
    )
    monkeypatch.setattr(
        "app.graphs.reasoning_graph.product_manager_agent_node",
        lambda state: {
            "current_stage": "product_manager_agent",
            "target_stage": AgentStage.engineer_agent,
            "routing_signals": list(state.get("routing_signals") or []) + ["handoff_ready_engineer"],
            "architecture_plan": ArchitecturePlan(
                use_case_id="uc_resume",
                title="HTTP integration flow",
                business_objective="Push events to external API",
                desired_outcome="Create API call per incoming event",
                workflow_summary="Two stages",
                stages=[
                    ArchitectureStage(
                        id="s1",
                        name="Trigger",
                        purpose="Receive event",
                        required_capabilities=["Receive"],
                        expected_inputs=["event"],
                        expected_outputs=["payload"],
                        dependencies=[],
                    ),
                    ArchitectureStage(
                        id="s2",
                        name="Call API",
                        purpose="Send HTTP request",
                        required_capabilities=["HTTP call"],
                        expected_inputs=["payload"],
                        expected_outputs=["api_response"],
                        dependencies=["s1"],
                    ),
                ],
                data_flow=[
                    ArchitectureDataFlowItem(
                        source_stage_id="s1",
                        target_stage_id="s2",
                        data_items=["payload"],
                    )
                ],
                assumptions=[],
                missing_information=[],
                implementation_notes_for_engineer=[],
                required_nodes=[
                    NodeRequirement(
                        node_type="n8n-nodes-base.httpRequest",
                        why_required="Required outbound API call",
                        evidence_chunk_ids=["chunk-http"],
                        evidence_refs=["ref-http"],
                        evidence_confidence=0.8,
                    )
                ],
            ),
            "workflow_context": WorkflowContext(
                use_case_id="uc_resume",
                planning_ready=True,
                handoff_target=AgentStage.engineer_agent,
                required_node_types=["n8n-nodes-base.httpRequest"],
                unresolved_inputs=[],
                notes=[],
            ),
            "proposed_nodes": [],
            "required_credentials": [],
        },
    )

    run_config = {"configurable": {"thread_id": "conv-resume-1"}}
    first = runtime.run(
        user_prompt="Build an API sync workflow",
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
