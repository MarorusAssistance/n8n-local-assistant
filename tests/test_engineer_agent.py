from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitecturePlan,
    ArchitectureStage,
    BlockedNode,
    EntryIntent,
    ImplementationQueueItem,
    ImplementationStatus,
    MissingUserInput,
    NodeRequirement,
    ProposedNode,
    StageKind,
    WorkflowContext,
    WorkflowDraft,
    WorkflowDraftConnection,
    WorkflowDraftNode,
)
from app.graphs.nodes import engineer_agent as engineer_mod
from app.graphs.nodes.engineer_agent import (
    DeveloperCredentialDefinition,
    DeveloperMissingInputDecision,
    DeveloperNodeDefinition,
    DeveloperParameterDefinition,
    NodeImplementationDecision,
    engineer_agent_node,
    get_node_definition,
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
    param_defaults: Dict[str, Any] | None = None,
    credential_types: List[str] | None = None,
    type_version: int = 1,
) -> DeveloperNodeDefinition:
    defaults = param_defaults or {}
    params = [
        DeveloperParameterDefinition(
            name=name,
            required=True,
            description=f"Required param {name}",
            default_value=defaults.get(name),
        )
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
            param_defaults=spec.get("param_defaults", {}),
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


def test_get_node_definition_accepts_list_version_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "title": "AI Transform",
            "url": "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.aitransform/",
            "text": "Overview for AI Transform",
            "metadata": {
                "kind": "NODE_OVERVIEW",
                "displayName": "AI Transform",
                "version": [1],
                "credentialTypes_required": [],
            },
        }
    ]
    monkeypatch.setattr(engineer_mod, "query_definition_chunks_by_entity", lambda **kwargs: rows)

    definition = get_node_definition("n8n-nodes-base.aiTransform")

    assert definition is not None
    assert definition.display_name == "AI Transform"
    assert definition.type_version == 1


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


def test_engineer_does_not_store_freeform_text_as_credential_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.googleSheets": {
                "credential_types": ["googleSheetsOAuth2Api"],
                "credential_display_names": {"googleSheetsOAuth2Api": "Google Sheets OAuth2 API"},
            },
        },
    )

    state = _state(
        ["n8n-nodes-base.googleSheets"],
        user_query=(
            "No tengo mas datos tecnicos. Si te falta alguna credencial o parametro no critico, "
            "usa defaults razonables y continua."
        ),
        extra={
            "resume_requested": True,
            "missing_user_inputs": [
                "Indica la referencia de credencial que debe usar el nodo 'an_1' para 'Google Sheets OAuth2 API'."
            ],
            "missing_user_input_details": [
                MissingUserInput(
                    input_id="credential:an_1:googleSheetsOAuth2Api",
                    input_key="credential:an_1:googleSheetsOAuth2Api",
                    missing_item="googleSheetsOAuth2Api",
                    reason="Credential is required.",
                    blocking_node_id="an_1",
                    category="credential",
                    question="Indica la referencia de credencial.",
                )
            ],
        },
    )

    updates = engineer_agent_node(state)
    node = updates["workflow_draft"].nodes[0]

    assert node.credential_refs == {}
    assert not any("Applied freeform user response" in note for note in updates["engineer_notes"])


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


def test_engineer_rephrases_pending_question_in_spanish_without_consuming_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {"n8n-nodes-base.googleSheets": {"credential_types": ["googleSheetsOAuth2Api"]}},
    )

    initial = _state(["n8n-nodes-base.googleSheets"])
    first = engineer_agent_node(initial)

    assert first["implementation_status"] == ImplementationStatus.blocked_waiting_user

    resumed = dict(initial)
    resumed.update(first)
    resumed["user_query"] = "No entiendo la pregunta, me la puedes hacer en espanol?"
    resumed["resume_requested"] = True
    second = engineer_agent_node(resumed)

    assert second["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert second["missing_user_input_details"][0].question.startswith("Indica la referencia de credencial")
    assert "engineer_question_rephrased" in second["routing_signals"]


def test_engineer_default_continue_uses_schema_default_for_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.httpRequest": {
                "required_params": ["method", "responseFormat"],
                "param_defaults": {"method": "GET", "responseFormat": "json"},
            }
        },
    )

    initial = _state(["n8n-nodes-base.httpRequest"])
    first = engineer_agent_node(initial)

    assert first["implementation_status"] == ImplementationStatus.blocked_waiting_user

    resumed = dict(initial)
    resumed.update(first)
    resumed["user_query"] = "usa default"
    resumed["resume_requested"] = True
    second = engineer_agent_node(resumed)

    assert second["implementation_status"] == ImplementationStatus.completed
    assert second["workflow_draft"].nodes[0].parameters_inferred["method"] == "GET"
    assert second["workflow_draft"].nodes[0].parameters_inferred["responseFormat"] == "json"
    assert second["missing_user_input_details"] == []
    assert "parameter:an_1:method" in second["workflow_draft"].metadata["defaulted_input_keys"]


def test_engineer_default_continue_leaves_missing_credential_empty_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.googleSheets": {
                "required_params": ["sheetName"],
                "param_defaults": {"sheetName": "Sheet1"},
                "credential_types": ["googleSheetsOAuth2Api", "googleDriveOAuth2Api"],
            }
        },
    )

    initial = _state(["n8n-nodes-base.googleSheets"])
    first = engineer_agent_node(initial)

    assert first["implementation_status"] == ImplementationStatus.blocked_waiting_user

    resumed = dict(initial)
    resumed.update(first)
    resumed["user_query"] = "lo que veas"
    resumed["resume_requested"] = True
    second = engineer_agent_node(resumed)

    node = second["workflow_draft"].nodes[0]
    warnings = second["workflow_draft"].metadata["developer_warnings"]

    assert second["implementation_status"] == ImplementationStatus.completed
    assert node.credential_refs == {}
    assert "credential_unresolved_after_user_declined:googleSheetsOAuth2Api" in node.notes
    assert any("googleSheetsOAuth2Api" in item for item in warnings)
    assert second["missing_user_input_details"] == []


def test_engineer_strips_invented_credential_refs_from_structured_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.mistralAi": {
                "credential_types": ["mistralCloudApi"],
            }
        },
    )
    monkeypatch.setattr(
        engineer_mod,
        "_decide_node_implementation_with_structured_output",
        lambda **_kwargs: NodeImplementationDecision(
            parameters_known={},
            parameters_inferred={},
            parameters_unresolved=[],
            credential_refs={
                "mistralCloudApi": "MISSING",
                "can_apply": "true",
            },
            missing_inputs=[],
            variable_outputs=[],
            notes=["bogus_credential_ref"],
            can_apply=False,
        ),
    )

    updates = engineer_agent_node(_state(["n8n-nodes-base.mistralAi"]))
    node = updates["workflow_draft"].nodes[0]

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert node.credential_refs == {}
    assert all(item.category == "credential" for item in updates["missing_user_input_details"])
    assert updates["missing_user_input_details"][0].missing_item == "mistralCloudApi"


def test_engineer_default_continue_preserves_unresolved_behavior_param_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {"n8n-nodes-base.cron": {"credential_types": ["dummyCredential"]}},
    )
    monkeypatch.setattr(
        engineer_mod,
        "_decide_node_implementation_with_structured_output",
        lambda **_kwargs: NodeImplementationDecision(
            parameters_known={},
            parameters_inferred={"triggerTimes.item.mode": "everyHour"},
            parameters_unresolved=[],
            credential_refs={},
            missing_inputs=[],
            variable_outputs=[],
            notes=["test_behavior_inference"],
            can_apply=True,
        ),
    )

    initial = _state(["n8n-nodes-base.cron"])
    first = engineer_agent_node(initial)

    assert first["implementation_status"] == ImplementationStatus.blocked_waiting_user

    resumed = dict(initial)
    resumed.update(first)
    resumed["user_query"] = "hazlo tú"
    resumed["resume_requested"] = True
    second = engineer_agent_node(resumed)

    node = second["workflow_draft"].nodes[0]

    assert second["implementation_status"] == ImplementationStatus.completed
    assert "triggerTimes.item.mode" in node.parameters_unresolved
    assert "left_unresolved_after_user_declined:triggerTimes.item.mode" in node.notes
    assert second["missing_user_input_details"] == []


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


def test_engineer_blocks_when_architect_handoff_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.webhook": {},
            "n8n-nodes-base.set": {},
        },
    )

    state = _state(["n8n-nodes-base.webhook", "n8n-nodes-base.set"])
    state["proposed_nodes"] = state["proposed_nodes"][:1]
    state["workflow_draft"].nodes = state["workflow_draft"].nodes[:1]
    state["workflow_draft"].connections = []

    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert "engineer_incomplete_architect_handoff" in updates["routing_signals"]
    assert updates["missing_user_input_details"][0].category == "handoff"


def test_engineer_blocks_when_source_update_handoff_lacks_concrete_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.gmailTrigger": {},
            "n8n-nodes-base.gmail": {"required_params": ["resource", "operation"]},
        },
    )

    state = _state(["n8n-nodes-base.gmailTrigger", "n8n-nodes-base.gmail"])
    state["architecture_plan"].stages[1].stage_kind = StageKind.apply_update_source
    state["architecture_plan"].stages[1].purpose = "Apply the urgency result back onto Gmail."
    state["architecture_plan"].stages[1].target_entity = "gmail"

    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert "engineer_incomplete_architect_handoff" in updates["routing_signals"]
    assert updates["missing_user_input_details"][0].category == "handoff"


def test_engineer_blocks_multipurpose_action_nodes_without_material_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.gmailTrigger": {},
            "n8n-nodes-base.gmail": {"required_params": ["resource", "operation"]},
        },
    )

    state = _state(["n8n-nodes-base.gmailTrigger", "n8n-nodes-base.gmail"])
    stage = state["architecture_plan"].stages[1]
    stage.stage_kind = StageKind.apply_update_source
    stage.purpose = "Apply an urgency label to the Gmail message."
    stage.target_entity = "gmail"
    hints = {
        "semantic_action": "apply_label",
        "require_action_selection": True,
        "required_parameter_keys": ["resource", "operation"],
        "allow_inferred_parameter_keys": ["resource", "operation"],
        "selector_guidance": "Configure this node to apply a label to the source Gmail message.",
    }
    state["proposed_nodes"][1].implementation_hints = dict(hints)
    state["workflow_draft"].nodes[1].implementation_hints = dict(hints)

    def _decide(**kwargs):
        queue_item = kwargs["queue_item"]
        if queue_item.node_type == "n8n-nodes-base.gmailTrigger":
            return NodeImplementationDecision(
                parameters_known={},
                parameters_inferred={},
                parameters_unresolved=[],
                credential_refs={},
                missing_inputs=[],
                variable_outputs=[],
                notes=["trigger_ok"],
                can_apply=True,
            )
        return NodeImplementationDecision(
            parameters_known={},
            parameters_inferred={},
            parameters_unresolved=[],
            credential_refs={},
            missing_inputs=[],
            variable_outputs=[],
            notes=["gmail_missing_operation"],
            can_apply=True,
        )

    monkeypatch.setattr(engineer_mod, "_decide_node_implementation_with_structured_output", _decide)

    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert {item.missing_item for item in updates["missing_user_input_details"]} >= {"resource", "operation"}


def test_engineer_allows_inferred_action_selectors_when_architect_resolved_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.gmailTrigger": {},
            "n8n-nodes-base.gmail": {"required_params": ["resource", "operation"]},
        },
    )

    state = _state(["n8n-nodes-base.gmailTrigger", "n8n-nodes-base.gmail"])
    stage = state["architecture_plan"].stages[1]
    stage.stage_kind = StageKind.apply_update_source
    stage.purpose = "Apply an urgency label to the Gmail message."
    stage.target_entity = "gmail"
    hints = {
        "semantic_action": "apply_label",
        "require_action_selection": True,
        "required_parameter_keys": ["resource", "operation"],
        "allow_inferred_parameter_keys": ["resource", "operation"],
        "selector_guidance": "Configure this node to apply a label to the source Gmail message.",
    }
    state["proposed_nodes"][1].implementation_hints = dict(hints)
    state["workflow_draft"].nodes[1].implementation_hints = dict(hints)

    def _decide(**kwargs):
        queue_item = kwargs["queue_item"]
        if queue_item.node_type == "n8n-nodes-base.gmailTrigger":
            return NodeImplementationDecision(
                parameters_known={},
                parameters_inferred={},
                parameters_unresolved=[],
                credential_refs={},
                missing_inputs=[],
                variable_outputs=[],
                notes=["trigger_ok"],
                can_apply=True,
            )
        return NodeImplementationDecision(
            parameters_known={},
            parameters_inferred={"resource": "message", "operation": "addLabel"},
            parameters_unresolved=[],
            credential_refs={},
            missing_inputs=[],
            variable_outputs=[],
            notes=["gmail_apply_label"],
            can_apply=True,
        )

    monkeypatch.setattr(engineer_mod, "_decide_node_implementation_with_structured_output", _decide)

    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.completed
    gmail_node = updates["workflow_draft"].nodes[1]
    assert gmail_node.parameters_inferred["resource"] == "message"
    assert gmail_node.parameters_inferred["operation"] == "addLabel"


def test_engineer_default_continue_infers_gmail_apply_label_action_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {
            "n8n-nodes-base.gmailTrigger": {},
            "n8n-nodes-base.gmail": {"required_params": ["resource", "operation"]},
        },
    )

    state = _state(["n8n-nodes-base.gmailTrigger", "n8n-nodes-base.gmail"])
    stage = state["architecture_plan"].stages[1]
    stage.stage_kind = StageKind.apply_update_source
    stage.purpose = "Apply an urgency label to the Gmail message."
    stage.target_entity = "gmail"
    hints = {
        "semantic_action": "apply_label",
        "require_action_selection": True,
        "required_parameter_keys": ["resource", "operation"],
        "allow_inferred_parameter_keys": ["resource", "operation"],
        "preferred_resource": "message",
        "selector_guidance": "Configure this node to apply a label to the source Gmail message.",
    }
    state["proposed_nodes"][1].implementation_hints = dict(hints)
    state["workflow_draft"].nodes[1].implementation_hints = dict(hints)
    state["workflow_draft"].metadata["developer_default_continue_requested"] = True
    state["blocked_nodes"] = [
        BlockedNode(
            queue_id="stage_apply_gmail",
            node_type="n8n-nodes-base.gmail",
            reason="Need action selectors.",
            missing_input_ids=[
                "parameter:stage_apply_gmail:resource",
                "parameter:stage_apply_gmail:operation",
            ],
        )
    ]

    def _decide(**kwargs):
        queue_item = kwargs["queue_item"]
        if queue_item.node_type == "n8n-nodes-base.gmailTrigger":
            return NodeImplementationDecision(
                parameters_known={},
                parameters_inferred={},
                parameters_unresolved=[],
                credential_refs={},
                missing_inputs=[],
                variable_outputs=[],
                notes=["trigger_ok"],
                can_apply=True,
            )
        return NodeImplementationDecision(
            parameters_known={},
            parameters_inferred={},
            parameters_unresolved=[],
            credential_refs={},
            missing_inputs=[],
            variable_outputs=[],
            notes=["gmail_missing_operation"],
            can_apply=True,
        )

    monkeypatch.setattr(engineer_mod, "_decide_node_implementation_with_structured_output", _decide)

    updates = engineer_agent_node(state)

    assert updates["implementation_status"] == ImplementationStatus.completed
    gmail_node = updates["workflow_draft"].nodes[1]
    assert gmail_node.parameters_inferred["resource"] == "message"
    assert gmail_node.parameters_inferred["operation"] == "addLabel"
    assert "resource" not in gmail_node.parameters_unresolved
    assert "operation" not in gmail_node.parameters_unresolved
    assert "defaulted_after_user_declined:resource" in gmail_node.notes
    assert "defaulted_after_user_declined:operation" in gmail_node.notes


def test_engineer_blocks_behavior_defining_inferred_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_definition_lookups(
        monkeypatch,
        {"n8n-nodes-base.cron": {}},
    )
    monkeypatch.setattr(
        engineer_mod,
        "_decide_node_implementation_with_structured_output",
        lambda **_kwargs: NodeImplementationDecision(
            parameters_known={},
            parameters_inferred={"triggerTimes.item.mode": "everyHour"},
            parameters_unresolved=[],
            credential_refs={},
            missing_inputs=[],
            variable_outputs=[],
            notes=["test_behavior_inference"],
            can_apply=True,
        ),
    )

    updates = engineer_agent_node(_state(["n8n-nodes-base.cron"]))

    assert updates["implementation_status"] == ImplementationStatus.blocked_waiting_user
    assert updates["missing_user_input_details"]
    assert updates["missing_user_input_details"][0].missing_item == "triggerTimes.item.mode"


def test_decision_prompt_payload_includes_focused_parameter_schema() -> None:
    plan = ArchitecturePlan(
        use_case_id="uc_prompt",
        title="Apply Gmail label",
        business_objective="Apply urgency labels to Gmail messages.",
        desired_outcome="Each message is labeled in Gmail.",
        workflow_summary="Classify then apply Gmail label.",
        stages=[
            ArchitectureStage(
                id="stage_apply",
                name="Apply Gmail Label",
                purpose="Apply the label to the same Gmail message.",
                stage_kind=StageKind.apply_update_source,
                expected_inputs=["Urgency classification result", "Source email identifiers"],
                expected_outputs=["Updated Gmail message"],
                dependencies=[],
                success_criteria=["The Gmail label is visible."],
            )
        ],
        data_flow=[],
        assumptions=[],
        missing_information=[],
        implementation_notes_for_engineer=[],
        required_nodes=[],
    )
    queue_item = ImplementationQueueItem(
        queue_id="an_1",
        node_type="n8n-nodes-base.gmail",
        stage_id="stage_apply",
        purpose="Apply Gmail label",
        dependencies=[],
        expected_inputs=["Urgency classification result", "Source email identifiers"],
        expected_outputs=["Updated Gmail message"],
        status="pending",
        implementation_hints={
            "semantic_action": "apply_label",
            "parameter_focus": ["resource", "operation", "messageId", "labelId"],
            "fallback_label_name": "Review",
            "allowed_label_values": ["bajo", "medio", "alto", "critico"],
        },
    )
    current_node = WorkflowDraftNode(
        node_id="an_1",
        name="Gmail",
        node_type="n8n-nodes-base.gmail",
        type_version=1,
        purpose="Apply Gmail label",
        stage_id="stage_apply",
        parameters_known={},
        parameters_inferred={},
        parameters_unresolved=[],
        credential_refs={},
        expected_inputs=["Urgency classification result", "Source email identifiers"],
        expected_outputs=["Updated Gmail message"],
        dependencies=[],
        position=[260, 300],
        implementation_hints=dict(queue_item.implementation_hints),
    )
    node_definition = _node_def("n8n-nodes-base.gmail")
    parameter_schema = [
        DeveloperParameterDefinition(name="resource"),
        DeveloperParameterDefinition(name="operation"),
        DeveloperParameterDefinition(name="messageId"),
        DeveloperParameterDefinition(name="labelId"),
        DeveloperParameterDefinition(name="htmlMessage"),
    ]

    payload = engineer_mod._decision_prompt_payload(
        user_query="Aplica una etiqueta visible en Gmail.",
        architecture_plan=plan,
        queue_item=queue_item,
        current_node=current_node,
        node_definition=node_definition,
        parameter_schema=parameter_schema,
        credential_requirements=[],
        upstream_variables=[],
        resolved_inputs={},
        resolved_decision_slots=[],
        downstream_queue_ids=[],
        stage_bundle_map={"stage_apply": ["an_1"]},
    )
    parsed = json.loads(payload)

    assert parsed["implementation_hints"]["fallback_label_name"] == "Review"
    assert parsed["implementation_hints"]["allowed_label_values"] == ["bajo", "medio", "alto", "critico"]
    assert [item["name"] for item in parsed["focused_parameter_schema"]] == [
        "resource",
        "operation",
        "messageId",
        "labelId",
    ]


def test_engineer_missing_input_category_is_normalized() -> None:
    item = DeveloperMissingInputDecision(
        key_name="labelIds",
        category="required_parameter_keys",
        reason="Need label ids.",
        question="Provide label ids.",
    )

    assert item.category == "parameter"


def test_engineer_semantic_parameter_default_uses_architect_poll_hint() -> None:
    queue_item = ImplementationQueueItem(
        queue_id="gmail_trigger",
        node_type="n8n-nodes-base.gmailTrigger",
        stage_id="stage_trigger",
        purpose="Receive emails",
        dependencies=[],
        expected_inputs=[],
        expected_outputs=["email"],
        implementation_hints={
            "semantic_action": "receive_incoming_item",
            "preferred_poll_mode": "everyMinute",
        },
    )
    current_node = WorkflowDraftNode(
        node_id="gmail_trigger",
        name="Gmail Trigger",
        node_type="n8n-nodes-base.gmailTrigger",
        type_version=1,
        purpose="Receive emails",
        stage_id="stage_trigger",
        parameters_known={},
        parameters_inferred={},
        parameters_unresolved=[],
        credential_refs={},
        expected_inputs=[],
        expected_outputs=["email"],
        dependencies=[],
        position=[260, 300],
        implementation_hints=dict(queue_item.implementation_hints),
    )

    value = engineer_mod._semantic_parameter_default_value(
        queue_item=queue_item,
        current_node=current_node,
        key_name="pollTimes.item.mode",
    )

    assert value == "everyMinute"
