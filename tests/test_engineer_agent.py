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
    ProposedNode,
    WorkflowContext,
    WorkflowDraft,
    WorkflowDraftConnection,
    WorkflowDraftNode,
)
from app.graphs.nodes import engineer_agent as engineer_mod
from app.graphs.nodes.engineer_agent import (
    DeveloperCredentialDefinition,
    DeveloperNodeDefinition,
    DeveloperParameterDefinition,
    engineer_agent_node,
)


def _requirement(node_type: str, idx: int) -> NodeRequirement:
    return NodeRequirement(
        node_type=node_type,
        why_required=f"Required node {node_type}",
        evidence_chunk_ids=[],
        evidence_refs=[],
        evidence_confidence=0.0,
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
                success_criteria=[f"Stage {idx} is configured correctly."],
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


def _draft(node_types: List[str], *, title: str = "Automation Flow") -> WorkflowDraft:
    nodes: List[WorkflowDraftNode] = []
    connections: List[WorkflowDraftConnection] = []
    for idx, node_type in enumerate(node_types, start=1):
        node_id = f"an_{idx}"
        nodes.append(
            WorkflowDraftNode(
                node_id=node_id,
                name=f"{node_type.split('.')[-1]}_{idx}",
                node_type=node_type,
                type_version=2 if node_type.endswith(".code") else 1,
                purpose=f"Purpose for {node_type}",
                stage_id=f"stage_{idx}",
                parameters_known={},
                parameters_inferred={},
                parameters_unresolved=[],
                credential_refs={},
                expected_inputs=["input_payload"] if idx == 1 else [f"stage_{idx - 1}_output"],
                expected_outputs=[f"stage_{idx}_output"],
                dependencies=[f"an_{idx - 1}"] if idx > 1 else [],
                position=[260 * idx, 300],
                notes=["architect_stage"],
            )
        )
        if idx > 1:
            connections.append(
                WorkflowDraftConnection(
                    source_node_id=f"an_{idx - 1}",
                    target_node_id=node_id,
                )
            )
    return WorkflowDraft(
        name=title,
        use_case_id="uc_engineer",
        summary="Architect draft",
        nodes=nodes,
        connections=connections,
        metadata={},
    )


def _proposed_nodes(node_types: List[str]) -> List[ProposedNode]:
    return [
        ProposedNode(
            node_id=f"an_{idx}",
            node_type=node_type,
            stage_id=f"stage_{idx}",
            purpose=f"Purpose for {node_type}",
            depends_on=[f"an_{idx - 1}"] if idx > 1 else [],
            expected_inputs=["input_payload"] if idx == 1 else [f"stage_{idx - 1}_output"],
            expected_outputs=[f"stage_{idx}_output"],
        )
        for idx, node_type in enumerate(node_types, start=1)
    ]


def _state(
    node_types: List[str] | None,
    *,
    entry_intent: EntryIntent = EntryIntent.workflow_build_request,
    user_query: str = "Implement workflow",
    runtime_context: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    plan = _plan(node_types or []) if node_types else None
    draft = _draft(node_types or [], title=plan.title if plan else "Workflow Draft") if node_types else None
    proposed = _proposed_nodes(node_types or []) if node_types else []
    state: Dict[str, Any] = {
        "user_query": user_query,
        "entry_intent": entry_intent,
        "current_stage": "architect_agent" if plan else None,
        "target_stage": AgentStage.engineer_agent if plan else None,
        "routing_signals": ["handoff_ready_engineer"] if plan else [],
        "architecture_plan": plan,
        "workflow_context": WorkflowContext(
            use_case_id=(plan.use_case_id if plan else "unknown"),
            planning_ready=bool(plan),
            handoff_target=AgentStage.engineer_agent if plan else None,
            required_node_types=[item.node_type for item in plan.required_nodes] if plan else [],
            unresolved_inputs=[],
            notes=["architect_complete"] if plan else [],
        ) if plan else None,
        "missing_user_inputs": [],
        "missing_user_input_details": [],
        "proposed_nodes": proposed,
        "required_credentials": [],
        "workflow_draft": draft,
        "workflow_versions": [],
        "node_implementation_queue": [],
        "implemented_nodes": [],
        "blocked_nodes": [],
        "variable_registry": [],
        "implementation_status": None,
        "engineer_notes": [],
        "final_workflow_json": {},
        "runtime_context": runtime_context or {"model": None, "request_id": "req-engineer-test", "persist_to_n8n": False},
        "resume_requested": False,
        "active_workflow_id": "wf_existing" if plan else None,
        "active_workflow_name": draft.name if draft else None,
        "active_workflow_url": "http://localhost:5678/workflow/wf_existing" if plan else None,
        "workflow_persisted": False,
        "workflow_persist_action": None,
        "workflow_api_sync_result": {},
    }
    if extra:
        state.update(extra)
    return state


def _node_def(
    node_type: str,
    *,
    required_params: List[str] | None = None,
    credential_types: List[str] | None = None,
    type_version: int = 1,
) -> DeveloperNodeDefinition:
    params = [
        DeveloperParameterDefinition(name=name, required=True, description=f"Required param {name}")
        for name in (required_params or [])
    ]
    return DeveloperNodeDefinition(
        node_type=node_type,
        display_name=node_type.split(".")[-1],
        type_version=type_version,
        summary=f"Definition for {node_type}",
        parameter_schema=params,
        credential_types_required=list(credential_types or []),
        source_refs=["n8n://definition"],
        raw_chunks=[],
    )


def _stub_definition_lookups(
    monkeypatch: pytest.MonkeyPatch,
    specs: Dict[str, Dict[str, Any]],
) -> None:
    defs: Dict[str, DeveloperNodeDefinition] = {}
    creds: Dict[str, List[DeveloperCredentialDefinition]] = {}
    for node_type, spec in specs.items():
        defs[node_type] = _node_def(
            node_type,
            required_params=spec.get("required_params", []),
            credential_types=spec.get("credential_types", []),
            type_version=spec.get("type_version", 1),
        )
        creds[node_type] = [
            DeveloperCredentialDefinition(
                credential_type=item,
                display_name=spec.get("credential_display_names", {}).get(item, item),
                field_names=["apiKey"],
                summary=f"Credential for {item}",
            )
            for item in spec.get("credential_types", [])
        ]

    monkeypatch.setattr(engineer_mod, "get_node_definition", lambda node_type: defs.get(node_type))
    monkeypatch.setattr(
        engineer_mod,
        "get_node_parameter_schema",
        lambda node_type: list((defs.get(node_type) or _node_def(node_type)).parameter_schema),
    )
    monkeypatch.setattr(
        engineer_mod,
        "get_node_credential_requirements",
        lambda node_type: list(creds.get(node_type, [])),
    )


def test_engineer_success_from_architect_handoff_simple(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.set": {},
        },
    )

    updates = engineer_agent_node(
        _state(["n8n-nodes-base.webhook", "n8n-nodes-base.set"])
    )

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["target_stage"] == AgentStage.qa_agent
    assert len(updates["implemented_nodes"]) == 2
    assert updates["workflow_draft"].nodes[0].node_id == "an_1"
    assert updates["workflow_draft"].connections


def test_engineer_success_preserves_architect_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.code": {"type_version": 2},
            "n8n-nodes-base.googleSheets": {},
        },
    )

    state = _state(
        ["n8n-nodes-base.webhook", "n8n-nodes-base.code", "n8n-nodes-base.googleSheets"]
    )
    updates = engineer_agent_node(state)

    draft = updates["workflow_draft"]
    assert updates["implementation_status"] == ImplementationStatus.completed
    assert [node.node_id for node in draft.nodes] == ["an_1", "an_2", "an_3"]
    assert [node.position for node in draft.nodes] == [[260, 300], [520, 300], [780, 300]]
    assert draft.nodes[1].type_version == 2
    assert len(draft.connections) == 2


def test_engineer_success_with_existing_known_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.httpRequest": {"required_params": ["url"]},
        },
    )

    state = _state(["n8n-nodes-base.httpRequest"])
    state["workflow_draft"].nodes[0].parameters_known = {"url": "https://api.example.com"}
    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["workflow_draft"].nodes[0].parameters_known["url"] == "https://api.example.com"


@pytest.mark.parametrize(
    "node_type,required_param",
    [
        ("n8n-nodes-base.httpRequest", "url"),
        ("n8n-nodes-base.slack", "text"),
        ("n8n-nodes-base.googleSheets", "sheetName"),
    ],
)
def test_engineer_blocks_when_required_parameter_missing(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    required_param: str,
) -> None:
    _stub_definition_lookups(monkeypatch, {node_type: {"required_params": [required_param]}})

    updates = engineer_agent_node(_state([node_type]))

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert updates["blocked_nodes"]
    assert updates["missing_user_input_details"]
    assert updates["missing_user_input_details"][0].category == "parameter"
    assert required_param in updates["missing_user_input_details"][0].question


@pytest.mark.parametrize(
    "node_type,credential_type",
    [
        ("n8n-nodes-base.googleSheets", "googleSheetsOAuth2Api"),
        ("n8n-nodes-base.postgres", "postgres"),
    ],
)
def test_engineer_blocks_when_credential_reference_missing(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    credential_type: str,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {node_type: {"credential_types": [credential_type]}},
    )

    updates = engineer_agent_node(_state([node_type]))

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert updates["blocked_nodes"]
    assert any(item.category == "credential" for item in updates["missing_user_input_details"])
    assert updates["required_credentials"][0].credential_key == credential_type


def test_variable_registry_tracks_downstream_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.set": {},
            "n8n-nodes-base.if": {},
        },
    )

    updates = engineer_agent_node(
        _state(["n8n-nodes-base.webhook", "n8n-nodes-base.set", "n8n-nodes-base.if"])
    )

    assert updates["implementation_status"] == ImplementationStatus.completed
    registry = updates["variable_registry"]
    assert registry
    assert any(item.destination_node_ids for item in registry)
    assert all(item.origin_node_id for item in registry)


def test_workflow_versions_capture_incremental_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.set": {},
        },
    )

    updates = engineer_agent_node(_state(["n8n-nodes-base.webhook", "n8n-nodes-base.set"]))
    reasons = [item.reason for item in updates["workflow_versions"]]

    assert len(updates["workflow_versions"]) >= 4
    assert any("implemented node an_1" in reason for reason in reasons)
    assert any("implemented node an_2" in reason for reason in reasons)


def test_engineer_resume_after_parameter_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {"n8n-nodes-base.httpRequest": {"required_params": ["url"]}},
    )

    initial = _state(["n8n-nodes-base.httpRequest"])
    first = engineer_agent_node(initial)
    assert first["implementation_status"] == ImplementationStatus.blocked_waiting_user

    resumed = dict(initial)
    resumed.update(first)
    resumed["user_query"] = "parameter:an_1:url=https://api.example.com/resource"
    resumed["resume_requested"] = True
    second = engineer_agent_node(resumed)

    assert second["implementation_status"] == ImplementationStatus.completed
    assert second["workflow_draft"].nodes[0].parameters_known["url"] == "https://api.example.com/resource"


def test_engineer_resume_after_credential_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {"n8n-nodes-base.googleSheets": {"credential_types": ["googleSheetsOAuth2Api"]}},
    )

    initial = _state(["n8n-nodes-base.googleSheets"])
    first = engineer_agent_node(initial)
    assert first["implementation_status"] == ImplementationStatus.blocked_waiting_user

    resumed = dict(initial)
    resumed.update(first)
    resumed["user_query"] = "credential:an_1:googleSheetsOAuth2Api=cred_google"
    resumed["resume_requested"] = True
    second = engineer_agent_node(resumed)

    assert second["implementation_status"] == ImplementationStatus.completed
    assert second["workflow_draft"].nodes[0].credential_refs["googleSheetsOAuth2Api"] == "cred_google"


@pytest.mark.parametrize(
    "entry_intent,active_workflow_id,expected_status",
    [
        (EntryIntent.workflow_build_request, None, ImplementationStatus.failed),
        (EntryIntent.workflow_edit_request, None, ImplementationStatus.blocked_waiting_user),
    ],
)
def test_engineer_handles_missing_handoff_cleanly(
    entry_intent: EntryIntent,
    active_workflow_id: str | None,
    expected_status: ImplementationStatus,
) -> None:
    updates = engineer_agent_node(
        _state(
            None,
            entry_intent=entry_intent,
            extra={"active_workflow_id": active_workflow_id},
        )
    )

    assert updates["implementation_status"] == expected_status
    assert updates["final_workflow_json"] == {}


def test_engineer_bootstraps_from_active_workflow_on_edit_request(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow_payload = {
        "id": "wf_edit_1",
        "name": "Existing Workflow",
        "nodes": [
            {
                "id": "node_1",
                "name": "Webhook",
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 1,
                "position": [260, 300],
                "parameters": {},
            },
            {
                "id": "node_2",
                "name": "Set",
                "type": "n8n-nodes-base.set",
                "typeVersion": 1,
                "position": [520, 300],
                "parameters": {},
            },
        ],
        "connections": {
            "Webhook": {
                "main": [[{"node": "Set", "type": "main", "index": 0}]]
            }
        },
    }
    monkeypatch.setattr(engineer_mod, "get_active_workflow", lambda workflow_id: workflow_payload)
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.set": {},
        },
    )

    updates = engineer_agent_node(
        _state(
            None,
            entry_intent=EntryIntent.workflow_edit_request,
            extra={
                "active_workflow_id": "wf_edit_1",
                "active_workflow_name": None,
                "active_workflow_url": None,
            },
        )
    )

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert "engineer_bootstrapped_from_active_workflow" in updates["routing_signals"]
    assert updates["workflow_draft"].nodes[0].node_id == "node_1"
    assert updates["active_workflow_id"] == "wf_edit_1"


def test_engineer_persists_incremental_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.set": {},
        },
    )

    calls: List[tuple[str, str]] = []

    def _update(self, workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((workflow_id, payload["name"]))
        return {
            "id": workflow_id,
            "name": payload["name"],
            "url": f"http://localhost:5678/workflow/{workflow_id}",
        }

    monkeypatch.setattr("app.graphs.nodes.engineer_agent.N8NClient.update_workflow", _update)

    updates = engineer_agent_node(
        _state(
            ["n8n-nodes-base.webhook", "n8n-nodes-base.set"],
            runtime_context={"model": None, "request_id": "req-persist", "persist_to_n8n": True},
        )
    )

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["workflow_persisted"] is True
    assert updates["workflow_persist_action"] == "updated"
    assert len(calls) >= 3


def test_engineer_does_not_execute_qa_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.set": {},
        },
    )

    updates = engineer_agent_node(_state(["n8n-nodes-base.webhook", "n8n-nodes-base.set"]))

    assert updates["implementation_status"] == ImplementationStatus.completed
    assert updates["current_stage"] == "engineer_agent"
    assert "entered_qa_agent" not in updates["routing_signals"]
    assert updates["target_stage"] == AgentStage.qa_agent


def test_engineer_relies_on_definition_lookup_not_hardcoded_heuristics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.httpRequest": {"required_params": ["customUrl"]},
        },
    )

    updates = engineer_agent_node(_state(["n8n-nodes-base.httpRequest"]))

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert updates["missing_user_input_details"][0].missing_item == "customUrl"

