from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    EntryIntent,
    ImplementationStatus,
    NodeRequirement,
    WorkflowContext,
)
from app.graphs.nodes.engineer_agent import engineer_agent_node


def _requirement(node_type: str, idx: int) -> NodeRequirement:
    return NodeRequirement(
        node_type=node_type,
        why_required=f"Required node {node_type}",
        evidence_chunk_ids=[f"chunk-{idx}"],
        evidence_refs=[f"ref-{idx}"],
        evidence_confidence=0.8,
    )


def _plan(node_types: List[str], *, title: str = "Automation Flow") -> ArchitecturePlan:
    stages: List[ArchitectureStage] = []
    data_flow: List[ArchitectureDataFlowItem] = []
    requirements: List[NodeRequirement] = []
    for idx, node_type in enumerate(node_types, start=1):
        stage_id = f"stage_{idx}"
        prev_stage = f"stage_{idx - 1}" if idx > 1 else None
        stages.append(
            ArchitectureStage(
                id=stage_id,
                name=f"Stage {idx}",
                purpose=f"Implement {node_type}",
                required_capabilities=[f"Capability for {node_type}"],
                expected_inputs=["input_payload"] if idx == 1 else [f"stage_{idx - 1}_output"],
                expected_outputs=[f"stage_{idx}_output"],
                dependencies=[prev_stage] if prev_stage else [],
            )
        )
        if prev_stage:
            data_flow.append(
                ArchitectureDataFlowItem(
                    source_stage_id=prev_stage,
                    target_stage_id=stage_id,
                    data_items=[f"stage_{idx - 1}_output"],
                )
            )
        requirements.append(_requirement(node_type, idx))
    return ArchitecturePlan(
        use_case_id="uc_engineer",
        title=title,
        business_objective="Automate process",
        desired_outcome="Deliver workflow result",
        workflow_summary="Engineer implementation plan",
        stages=stages,
        data_flow=data_flow,
        assumptions=[],
        missing_information=[],
        implementation_notes_for_engineer=[],
        required_nodes=requirements,
    )


def _state(
    plan: ArchitecturePlan | None,
    *,
    entry_intent: EntryIntent = EntryIntent.workflow_build_request,
    user_query: str = "Implement workflow",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    required_node_types = [item.node_type for item in plan.required_nodes] if plan else []
    state: Dict[str, Any] = {
        "user_query": user_query,
        "entry_intent": entry_intent,
        "current_stage": "product_manager_agent",
        "target_stage": AgentStage.engineer_agent if plan else None,
        "routing_signals": ["handoff_ready_engineer"],
        "architecture_plan": plan,
        "workflow_context": WorkflowContext(
            use_case_id=(plan.use_case_id if plan else "unknown"),
            planning_ready=bool(plan),
            handoff_target=AgentStage.engineer_agent if plan else None,
            required_node_types=required_node_types,
            unresolved_inputs=[],
            notes=[],
        ),
        "missing_user_inputs": [],
        "missing_user_input_details": [],
        "proposed_nodes": [],
        "required_credentials": [],
        "workflow_draft": None,
        "workflow_versions": [],
        "node_implementation_queue": [],
        "implemented_nodes": [],
        "blocked_nodes": [],
        "variable_registry": [],
        "implementation_status": None,
        "engineer_notes": [],
        "final_workflow_json": {},
        "runtime_context": {"model": None, "request_id": "req-engineer-test"},
        "resume_requested": False,
    }
    if extra:
        state.update(extra)
    return state


@pytest.mark.parametrize(
    "node_types",
    [
        ["n8n-nodes-base.webhook", "n8n-nodes-base.set"],
        ["n8n-nodes-base.webhook", "n8n-nodes-base.if", "n8n-nodes-base.set"],
        ["n8n-nodes-base.set", "n8n-nodes-base.switch"],
    ],
)
def test_engineer_successful_iterative_construction(node_types: List[str]) -> None:
    updates = engineer_agent_node(_state(_plan(node_types)))

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["final_workflow_json"]["nodes"]
    assert len(updates["final_workflow_json"]["nodes"]) == len(node_types)
    assert updates["target_stage"] == AgentStage.qa_agent
    assert "handoff_ready_qa" in updates["routing_signals"]
    assert all(item.status == "implemented" for item in updates["node_implementation_queue"])


@pytest.mark.parametrize(
    "plan_obj",
    [
        _plan(["n8n-nodes-base.httpRequest"], title="HTTP sync flow"),
        _plan(["n8n-nodes-base.slack"], title="Slack alert flow"),
        _plan(["n8n-nodes-base.googleSheets"], title="Row logging flow"),
    ],
)
def test_engineer_blocks_when_critical_inputs_missing(plan_obj: ArchitecturePlan) -> None:
    updates = engineer_agent_node(_state(plan_obj))

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert updates["blocked_nodes"]
    assert updates["missing_user_input_details"]
    assert updates["final_workflow_json"] == {}
    assert updates["target_stage"] is None


@pytest.mark.parametrize(
    "plan_obj",
    [
        _plan(["n8n-nodes-base.googleSheets"], title="Google Sheet sync"),
        _plan(["n8n-nodes-base.postgres"], title="Database write flow"),
    ],
)
def test_engineer_does_not_infer_missing_credentials_from_node_type_heuristics(
    plan_obj: ArchitecturePlan,
) -> None:
    updates = engineer_agent_node(_state(plan_obj))

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["final_workflow_json"]["nodes"]
    categories = {item.category for item in updates["missing_user_input_details"]}
    assert "credential" not in categories


def test_variable_registry_tracks_origins_and_consumers() -> None:
    updates = engineer_agent_node(
        _state(_plan(["n8n-nodes-base.webhook", "n8n-nodes-base.set"]))
    )

    registry = updates["variable_registry"]
    assert registry
    assert any(item.destination_node_ids for item in registry)
    assert all(item.origin_node_id for item in registry)
    assert all(item.semantic_meaning for item in registry)


def test_variable_registry_across_three_nodes() -> None:
    updates = engineer_agent_node(
        _state(_plan(["n8n-nodes-base.webhook", "n8n-nodes-base.if", "n8n-nodes-base.set"]))
    )
    queue_ids = {item.queue_id for item in updates["node_implementation_queue"]}
    downstream_ids = {dest for item in updates["variable_registry"] for dest in item.destination_node_ids}
    assert "pn_2" in queue_ids and "pn_3" in queue_ids
    assert "pn_2" in downstream_ids or "pn_3" in downstream_ids


def test_workflow_versions_are_incremental_and_reasoned() -> None:
    updates = engineer_agent_node(
        _state(_plan(["n8n-nodes-base.webhook", "n8n-nodes-base.set", "n8n-nodes-base.if"]))
    )
    versions = updates["workflow_versions"]

    assert len(versions) >= 5
    version_ids = [item.version for item in versions]
    assert version_ids == sorted(version_ids)
    assert any("implemented node" in item.reason for item in versions)
    assert versions[0].reason.startswith("initialized engineer workflow draft")


def test_workflow_versions_grow_after_resume_completion() -> None:
    initial_state = _state(_plan(["n8n-nodes-base.httpRequest"], title="HTTP flow"))
    first = engineer_agent_node(initial_state)
    assert first["implementation_status"] == ImplementationStatus.blocked_waiting_user
    first_versions = len(first["workflow_versions"])

    resumed_state = dict(initial_state)
    resumed_state.update(first)
    resumed_state["user_query"] = "parameter:pn_1:url=https://api.example.com/resource"
    resumed_state["resume_requested"] = True
    resumed_state["runtime_context"] = {"model": None, "request_id": "req-engineer-resume"}
    second = engineer_agent_node(resumed_state)

    assert second["implementation_status"] == ImplementationStatus.completed
    assert len(second["workflow_versions"]) > first_versions
    assert second["final_workflow_json"]["nodes"]


@pytest.mark.parametrize(
    "entry_intent,expected_status",
    [
        (EntryIntent.workflow_build_request, ImplementationStatus.failed),
        (EntryIntent.workflow_edit_request, ImplementationStatus.blocked_waiting_user),
    ],
)
def test_invalid_or_missing_pm_handoff_fails_cleanly(
    entry_intent: EntryIntent,
    expected_status: ImplementationStatus,
) -> None:
    updates = engineer_agent_node(_state(None, entry_intent=entry_intent))

    assert updates["implementation_status"] == expected_status
    assert updates["final_workflow_json"] == {}
    if expected_status == ImplementationStatus.blocked_waiting_user:
        assert updates["missing_user_input_details"]


def test_engineer_does_not_execute_qa_logic_on_completion() -> None:
    updates = engineer_agent_node(_state(_plan(["n8n-nodes-base.webhook", "n8n-nodes-base.set"])))
    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["current_stage"] == "engineer_agent"
    assert "entered_qa_agent" not in updates["routing_signals"]
    assert updates["target_stage"] == AgentStage.qa_agent


def test_engineer_does_not_execute_qa_logic_when_blocked() -> None:
    updates = engineer_agent_node(_state(_plan(["n8n-nodes-base.httpRequest"])))
    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert updates["target_stage"] is None
    assert "entered_qa_agent" not in updates["routing_signals"]


def test_engineer_does_not_skip_iterative_steps_to_final_json() -> None:
    updates = engineer_agent_node(
        _state(_plan(["n8n-nodes-base.webhook", "n8n-nodes-base.if", "n8n-nodes-base.set"]))
    )

    assert len(updates["node_implementation_queue"]) == 3
    assert len(updates["implemented_nodes"]) == 3
    assert len(updates["workflow_versions"]) >= 5
    assert updates["final_workflow_json"]["connections"] is not None


def test_engineer_iterative_history_contains_per_node_updates() -> None:
    updates = engineer_agent_node(
        _state(_plan(["n8n-nodes-base.webhook", "n8n-nodes-base.set"]))
    )
    reasons = [item.reason for item in updates["workflow_versions"]]
    assert any("implemented node pn_1" in reason for reason in reasons)
    assert any("implemented node pn_2" in reason for reason in reasons)


def test_engineer_persists_new_workflow_in_n8n_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(["n8n-nodes-base.webhook", "n8n-nodes-base.set"], title="Persist Create")
    state = _state(
        plan,
        extra={"runtime_context": {"model": None, "request_id": "req-create", "persist_to_n8n": True}},
    )

    captured: Dict[str, Any] = {}

    def _create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        captured["payload"] = payload
        assert "active" not in payload
        assert "settings" in payload
        return {
            "id": "wf_100",
            "name": payload.get("name"),
            "url": "http://localhost:5678/workflow/wf_100",
        }

    monkeypatch.setattr("app.graphs.nodes.engineer_agent.N8NClient.create_workflow", _create)

    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["workflow_persisted"] is True
    assert updates["workflow_persist_action"] == "created"
    assert updates["active_workflow_id"] == "wf_100"
    assert updates["active_workflow_url"] == "http://localhost:5678/workflow/wf_100"
    assert captured["payload"]["name"] == "Persist Create"


def test_engineer_updates_existing_workflow_in_n8n_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(["n8n-nodes-base.webhook", "n8n-nodes-base.set"], title="Persist Update")
    state = _state(
        plan,
        extra={
            "active_workflow_id": "wf_existing",
            "runtime_context": {"model": None, "request_id": "req-update", "persist_to_n8n": True},
        },
    )
    seen: Dict[str, Any] = {}

    def _update(self, workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        seen["workflow_id"] = workflow_id
        seen["name"] = payload.get("name")
        assert "active" not in payload
        assert "settings" in payload
        return {
            "id": workflow_id,
            "name": payload.get("name"),
            "url": f"http://localhost:5678/workflow/{workflow_id}",
        }

    monkeypatch.setattr("app.graphs.nodes.engineer_agent.N8NClient.update_workflow", _update)

    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["workflow_persisted"] is True
    assert updates["workflow_persist_action"] == "updated"
    assert seen["workflow_id"] == "wf_existing"
    assert updates["active_workflow_id"] == "wf_existing"
