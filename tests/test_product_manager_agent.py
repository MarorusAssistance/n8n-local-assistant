from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.features.reasoning.multi_agent_contracts import (
    AgentStage,
    ArchitectureDataFlowItem,
    ArchitectureStage,
    EntryIntent,
    UseCase,
)
from app.graphs.nodes import product_manager_agent as pm


def _use_case(
    *,
    use_case_id: str = "uc_1",
    title: str = "Webhook to Sheets",
    business_problem: str = "Operations team manually copies incoming form submissions into spreadsheets.",
    desired_outcome: str = "Automatically capture submissions and persist them in the reporting sheet.",
    expected_value: str = "Reduce manual work and improve data consistency.",
) -> UseCase:
    return UseCase(
        id=use_case_id,
        title=title,
        business_problem=business_problem,
        desired_outcome=desired_outcome,
        expected_value=expected_value,
        feasibility="medium",
        priority_score=82.0,
        why_selected="Selected in commercial stage.",
    )


def _state(
    *,
    selected_use_case: UseCase | Dict[str, Any] | None = None,
    alternative_use_cases: List[UseCase] | None = None,
    missing_user_inputs: List[str] | None = None,
    entry_intent: EntryIntent | None = None,
    user_query: str = "Build a workflow that receives webhooks and stores data in Sheets.",
) -> Dict[str, Any]:
    return {
        "entry_intent": entry_intent,
        "user_query": user_query,
        "selected_use_case": selected_use_case,
        "alternative_use_cases": alternative_use_cases or [],
        "routing_signals": ["entered_commercial_agent", "handoff_ready_product_manager"],
        "missing_user_inputs": list(missing_user_inputs or []),
        "workflow_context": {"model": None, "request_id": "req-pm-test"},
    }


def _docs_with_explicit_node_evidence(
    node_types: List[str],
    *,
    input_types_by_node: Dict[str, List[str]] | None = None,
    usable_as_tool_by_node: Dict[str, bool] | None = None,
    page_key_by_node: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for idx, node_type in enumerate(node_types, start=1):
        input_types = (
            list((input_types_by_node or {}).get(node_type, ["main"]))
            if input_types_by_node is not None
            else ["main"]
        )
        usable_as_tool = (usable_as_tool_by_node or {}).get(node_type)
        usable_as_tool_text = "true" if usable_as_tool else "false"
        page_key = (page_key_by_node or {}).get(node_type, f"page-{idx}")
        docs.append(
            {
                "doc_id": f"linked:node:{node_type}",
                "title": f"Node: {node_type}",
                "url": f"https://docs.n8n.io/{idx}",
                "text": (
                    "Kind: NODE_OVERVIEW\n"
                    f"Node Type: {node_type}\n"
                    f"Usable As Tool: {usable_as_tool_text}\n"
                    f"Inputs: {json.dumps(input_types)}\n"
                    "Description: Sample."
                ),
                "context_kind": "linked_def",
                "linked_def_type": "node",
                "linked_entity_id": node_type,
                "link_confidence": 0.91,
                "link_doc_page_key": page_key,
                "metadata": {
                    "kind": "NODE_OVERVIEW",
                    "nodeType": node_type,
                    "displayName": node_type.split(".")[-1],
                    "inputs": input_types,
                },
            }
        )
        if usable_as_tool is not None:
            docs[-1]["metadata"]["usableAsTool"] = usable_as_tool
    return docs


def _docs_chunk_with_rerank(
    *,
    page_key: str,
    rerank_score: float,
    text: str = "General API docs evidence for this node family.",
) -> Dict[str, Any]:
    return {
        "doc_id": f"doc:{page_key}",
        "title": f"Docs {page_key}",
        "url": f"https://docs.n8n.io/{page_key}",
        "text": text,
        "rerank_score": rerank_score,
        "metadata": {
            "kind": "DOCS_PAGE",
            "page_id": page_key,
        },
    }


def _docs_without_explicit_node_evidence() -> List[Dict[str, Any]]:
    return [
        {
            "doc_id": "doc-1",
            "title": "General docs",
            "url": "https://docs.n8n.io/general",
            "text": "You can build workflows with many node types.",
            "metadata": {"kind": "DOCS_PAGE"},
        }
    ]


def _draft(
    *,
    missing_information: List[str] | None = None,
    notes: List[str] | None = None,
) -> pm.ArchitecturePlanDraft:
    return pm.ArchitecturePlanDraft(
        workflow_summary="Planning-level architecture for selected use case.",
        stages=[
            ArchitectureStage(
                id="stage_intake",
                name="Intake",
                purpose="Capture and normalize events.",
                required_capabilities=["Capture trigger", "Normalize payload"],
                expected_inputs=["Event payload"],
                expected_outputs=["Normalized event payload"],
                dependencies=[],
            ),
            ArchitectureStage(
                id="stage_delivery",
                name="Delivery",
                purpose="Deliver normalized payload to target system.",
                required_capabilities=["Write target data", "Record outcome"],
                expected_inputs=["Normalized event payload"],
                expected_outputs=["Delivery result"],
                dependencies=["stage_intake"],
            ),
        ],
        data_flow=[
            ArchitectureDataFlowItem(
                source_stage_id="stage_intake",
                target_stage_id="stage_delivery",
                data_items=["normalized payload"],
            )
        ],
        assumptions=["Event source is available."],
        missing_information=missing_information or [],
        implementation_notes_for_engineer=notes or ["Configure node parameters in engineering stage."],
        planning_summary="Planning ready for engineer handoff.",
    )


@pytest.mark.parametrize(
    "selected",
    [
        _use_case(use_case_id="uc_a", title="Lead intake automation"),
        _use_case(use_case_id="uc_b", title="Incident escalation automation"),
        _use_case(use_case_id="uc_c", title="Collections reminder automation"),
    ],
)
def test_pm_generates_well_formed_architecture_plan_for_valid_use_case(
    monkeypatch: pytest.MonkeyPatch,
    selected: UseCase,
) -> None:
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(
            ["n8n-nodes-base.webhook", "n8n-nodes-base.googleSheets"]
        ),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["current_stage"] == "product_manager_agent"
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert updates["architecture_plan"] is not None
    assert updates["architecture_plan"].use_case_id == selected.id
    assert len(updates["architecture_plan"].required_nodes) >= 1
    assert updates["workflow_context"].planning_ready is True


@pytest.mark.parametrize(
    "selected",
    [
        _use_case(
            use_case_id="uc_incomplete_1",
            business_problem="Manual work",
            desired_outcome="Automate it",
        ),
        _use_case(
            use_case_id="uc_incomplete_2",
            business_problem="Need better process",
            desired_outcome="Improve workflow",
        ),
    ],
)
def test_pm_records_missing_inputs_for_incomplete_use_case(
    monkeypatch: pytest.MonkeyPatch,
    selected: UseCase,
) -> None:
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["architecture_plan"] is not None
    assert updates["architecture_plan"].missing_information
    assert updates["missing_user_inputs"]


@pytest.mark.parametrize(
    "selected",
    [
        None,
        {"id": "bad", "title": "invalid payload"},
    ],
)
def test_pm_fails_cleanly_when_selected_use_case_is_missing_or_invalid(
    monkeypatch: pytest.MonkeyPatch,
    selected: Any,
) -> None:
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieve should not run")),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["architecture_plan"] is None
    assert updates["target_stage"] is None
    assert "pm_missing_selected_use_case" in updates["routing_signals"]


@pytest.mark.parametrize(
    "selected",
    [
        _use_case(use_case_id="uc_no_json_1"),
        _use_case(use_case_id="uc_no_json_2"),
    ],
)
def test_pm_does_not_generate_workflow_json_payload(
    monkeypatch: pytest.MonkeyPatch,
    selected: UseCase,
) -> None:
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    plan_payload = updates["architecture_plan"].model_dump()

    assert "nodes" not in plan_payload
    assert "connections" not in plan_payload
    assert "final_workflow_json" not in plan_payload


def test_missing_user_inputs_only_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_complete_inputs")
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["missing_user_inputs"] == []


def test_missing_user_inputs_populated_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(
        use_case_id="uc_missing_inputs",
        business_problem="Need automation",
        desired_outcome="Do it",
    )
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft(missing_information=[]))

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["missing_user_inputs"]


def test_pm_ignores_alternative_use_cases_and_uses_only_selected_case(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_primary")
    alternatives = [_use_case(use_case_id="uc_alt_1"), _use_case(use_case_id="uc_alt_2")]
    seen: Dict[str, Any] = {}

    def _retrieve(use_case: UseCase, request_id: str | None = None) -> List[Dict[str, Any]]:
        seen["use_case_id"] = use_case.id
        return _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"])

    monkeypatch.setattr(pm, "retrieve_pm_api_docs", _retrieve)
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(
        _state(selected_use_case=selected, alternative_use_cases=alternatives)
    )

    assert seen["use_case_id"] == "uc_primary"
    assert updates["architecture_plan"].use_case_id == "uc_primary"


def test_pm_ignores_alternative_use_cases_even_when_high_value(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_primary_2")
    high_value_alt = _use_case(
        use_case_id="uc_alt_high",
        title="High value alt",
        expected_value="Very high value but not selected.",
    )
    calls: List[str] = []

    def _retrieve(use_case: UseCase, request_id: str | None = None) -> List[Dict[str, Any]]:
        calls.append(use_case.id)
        return _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"])

    monkeypatch.setattr(pm, "retrieve_pm_api_docs", _retrieve)
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    pm.product_manager_agent_node(
        _state(selected_use_case=selected, alternative_use_cases=[high_value_alt])
    )
    assert calls == ["uc_primary_2"]


def test_pm_plan_stays_planning_level_and_strips_json_like_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_planning_level_1")
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(
        pm,
        "_plan_with_structured_output",
        lambda **_: _draft(notes=['{"nodes":[{"type":"n8n-nodes-base.webhook"}]} do not emit this JSON']),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    notes = " ".join(updates["architecture_plan"].implementation_notes_for_engineer).lower()
    assert '"nodes"' not in notes
    assert "{ " not in notes


def test_pm_plan_does_not_include_parameter_or_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_planning_level_2")
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(
        pm,
        "_plan_with_structured_output",
        lambda **_: _draft(notes=["Do not set credentials or API keys here."]),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["architecture_plan"] is not None
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert "final_workflow_json" not in updates


def test_strict_evidence_policy_blocks_inferred_nodes_when_docs_have_no_explicit_node_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(use_case_id="uc_strict_evidence_1")
    monkeypatch.setattr(pm, "retrieve_pm_api_docs", lambda *args, **kwargs: _docs_without_explicit_node_evidence())
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert updates["architecture_plan"] is None
    assert updates["target_stage"] is None
    assert "pm_no_explicit_node_evidence" in updates["routing_signals"]


def test_strict_evidence_policy_ignores_node_type_line_without_explicit_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(use_case_id="uc_strict_evidence_2")
    docs = [
        {
            "doc_id": "doc-with-node-line-only",
            "title": "How to build automations",
            "url": "https://docs.n8n.io/guides",
            "text": "Node Type: n8n-nodes-base.webhook",
            "metadata": {"kind": "DOCS_PAGE"},
        }
    ]
    monkeypatch.setattr(pm, "retrieve_pm_api_docs", lambda *args, **kwargs: docs)
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["architecture_plan"] is None
    assert updates["target_stage"] is None


def test_extract_required_nodes_uses_only_explicit_evidence_and_sorts_deterministically() -> None:
    docs = _docs_with_explicit_node_evidence(
        ["n8n-nodes-base.webhook", "n8n-nodes-base.googleSheets", "n8n-nodes-base.webhook"]
    ) + _docs_without_explicit_node_evidence()

    required_nodes = pm.extract_required_nodes_from_docs(docs)

    assert required_nodes
    assert required_nodes[0].node_type == "n8n-nodes-base.webhook"
    assert all(node.node_type.startswith("n8n-nodes-base.") for node in required_nodes)
    assert isinstance(required_nodes[0].evidence_confidence, float)
    assert required_nodes[0].rerank_confidence is None
    assert required_nodes[0].blended_confidence == required_nodes[0].evidence_confidence


def test_extract_required_nodes_classifies_node_usage_modes() -> None:
    docs = _docs_with_explicit_node_evidence(
        [
            "n8n-nodes-base.actionNode",
            "n8n-nodes-base.toolNode",
            "n8n-nodes-base.hybridNode",
        ],
        input_types_by_node={
            "n8n-nodes-base.actionNode": ["main"],
            "n8n-nodes-base.toolNode": ["ai_tool"],
            "n8n-nodes-base.hybridNode": ["main"],
        },
        usable_as_tool_by_node={
            "n8n-nodes-base.actionNode": False,
            "n8n-nodes-base.toolNode": False,
            "n8n-nodes-base.hybridNode": True,
        },
    )
    required_nodes = pm.extract_required_nodes_from_docs(docs)
    by_type = {item.node_type: item for item in required_nodes}

    assert by_type["n8n-nodes-base.actionNode"].usage_mode == "action_only"
    assert by_type["n8n-nodes-base.actionNode"].has_main_input is True
    assert by_type["n8n-nodes-base.actionNode"].usable_as_tool is False

    assert by_type["n8n-nodes-base.toolNode"].usage_mode == "tool_only"
    assert by_type["n8n-nodes-base.toolNode"].has_main_input is False
    assert "ai_tool" in by_type["n8n-nodes-base.toolNode"].input_connection_types

    assert by_type["n8n-nodes-base.hybridNode"].usage_mode == "both"
    assert by_type["n8n-nodes-base.hybridNode"].has_main_input is True
    assert by_type["n8n-nodes-base.hybridNode"].usable_as_tool is True


def test_pm_filters_tool_only_nodes_for_non_agentic_use_case(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(
        use_case_id="uc_non_agentic",
        title="Email digest automation",
        business_problem="Too much manual email triage.",
        desired_outcome="Generate digest and send by email.",
    )
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda *args, **kwargs: _docs_with_explicit_node_evidence(
            ["n8n-nodes-base.emailReadImap", "n8n-nodes-base.emailReadImapTool"],
            input_types_by_node={
                "n8n-nodes-base.emailReadImap": ["main"],
                "n8n-nodes-base.emailReadImapTool": ["ai_tool"],
            },
            usable_as_tool_by_node={
                "n8n-nodes-base.emailReadImap": False,
                "n8n-nodes-base.emailReadImapTool": False,
            },
        ),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    required_types = [node.node_type for node in updates["architecture_plan"].required_nodes]

    assert "n8n-nodes-base.emailReadImap" in required_types
    assert "n8n-nodes-base.emailReadImapTool" not in required_types
    assert any(signal.startswith("pm_filtered_tool_only_nodes:") for signal in updates["routing_signals"])


def test_pm_keeps_tool_only_nodes_for_agentic_use_case(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(
        use_case_id="uc_agentic",
        title="AI agent tool calling for inbox handling",
        business_problem="Need an AI agent that calls tools over incoming messages.",
        desired_outcome="Route tasks through AI tool-calling subnodes.",
    )
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda *args, **kwargs: _docs_with_explicit_node_evidence(
            ["n8n-nodes-base.emailReadImapTool"],
            input_types_by_node={"n8n-nodes-base.emailReadImapTool": ["ai_tool"]},
            usable_as_tool_by_node={"n8n-nodes-base.emailReadImapTool": False},
        ),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    required_types = [node.node_type for node in updates["architecture_plan"].required_nodes]

    assert "n8n-nodes-base.emailReadImapTool" in required_types
    assert not any(signal.startswith("pm_filtered_tool_only_nodes:") for signal in updates["routing_signals"])


def test_extract_required_nodes_includes_rerank_confidence_when_available() -> None:
    docs = [
        _docs_chunk_with_rerank(page_key="webhook-doc", rerank_score=0.82),
        *_docs_with_explicit_node_evidence(
            ["n8n-nodes-base.webhook"],
            page_key_by_node={"n8n-nodes-base.webhook": "webhook-doc"},
        ),
    ]

    required_nodes = pm.extract_required_nodes_from_docs(docs)

    assert len(required_nodes) == 1
    assert required_nodes[0].evidence_confidence > 0.0
    assert required_nodes[0].rerank_confidence == 0.82
    assert required_nodes[0].blended_confidence == round(
        (0.75 * required_nodes[0].evidence_confidence)
        + (0.25 * required_nodes[0].rerank_confidence),
        2,
    )


def test_extract_required_nodes_sets_null_rerank_when_only_linked_def_evidence_exists() -> None:
    docs = _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"])
    required_nodes = pm.extract_required_nodes_from_docs(docs)

    assert len(required_nodes) == 1
    assert required_nodes[0].rerank_confidence is None


def test_extract_required_nodes_does_not_use_link_confidence_as_rerank_proxy() -> None:
    docs = _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"])
    docs[0]["link_confidence"] = 0.99
    required_nodes = pm.extract_required_nodes_from_docs(docs)

    assert len(required_nodes) == 1
    assert required_nodes[0].rerank_confidence is None


def test_extract_required_nodes_generates_functional_why_required_text() -> None:
    docs = _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"])
    required_nodes = pm.extract_required_nodes_from_docs(docs)
    why_required = required_nodes[0].why_required.lower()

    assert "retrieved api docs provided" not in why_required
    assert any(token in why_required for token in ("capabilities", "use", "workflow"))


def test_extract_required_nodes_sanitizes_malformed_reference_spillover() -> None:
    docs = [
        {
            "doc_id": "linked:node:n8n-nodes-base.emailSend",
            "context_kind": "linked_def",
            "linked_def_type": "node",
            "linked_entity_id": "n8n-nodes-base.emailSend",
            "metadata": {
                "kind": "NODE_OVERVIEW",
                "nodeType": "n8n-nodes-base.emailSend",
                "displayName": "Send Email",
            },
            "url": (
                "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.sendemail/"
                '&quot;], "confidence": 0.66}, {"node_type":"n8n-nodes-base.code"}'
            ),
            "text": "Kind: NODE_OVERVIEW\nNode Type: n8n-nodes-base.emailSend",
            "link_confidence": 0.91,
        }
    ]
    required_nodes = pm.extract_required_nodes_from_docs(docs)

    assert len(required_nodes) == 1
    assert required_nodes[0].evidence_refs == [
        "https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.sendemail/"
    ]


def test_pm_retrieval_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_single_retrieval")
    calls = {"retrieve": 0}

    def _retrieve(*args, **kwargs):
        calls["retrieve"] += 1
        return _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"])

    monkeypatch.setattr(pm, "retrieve_pm_api_docs", _retrieve)
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))

    assert calls["retrieve"] == 1
    assert updates["target_stage"] == AgentStage.engineer_agent


def test_pm_uses_single_planning_call_when_prompt_fits_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(use_case_id="uc_budget_fit")
    calls = {"plan": 0, "compact": 0, "summary": 0}

    def _plan_call(**kwargs):
        calls["plan"] += 1
        return _draft()

    def _compact_call(**kwargs):
        calls["compact"] += 1
        return kwargs["node_summaries"]

    def _summary_call(**kwargs):
        calls["summary"] += 1
        return kwargs["node_summaries"]

    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda *args, **kwargs: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", _plan_call)
    monkeypatch.setattr(pm, "_compact_node_evidence_with_structured_output", _compact_call)
    monkeypatch.setattr(pm, "_summarize_node_evidence_with_structured_output", _summary_call)
    monkeypatch.setattr(pm.settings, "MAX_CONTEXT_TOKENS", 6000)

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert calls["compact"] == 0
    assert calls["plan"] == 1
    assert calls["summary"] == 0


def test_pm_uses_compaction_then_planning_when_prompt_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(use_case_id="uc_budget_overflow")
    calls = {"plan": 0, "compact": 0, "summary": 0}
    large_text = " ".join(["This node supports many operations and constraints."] * 300)
    docs = _docs_with_explicit_node_evidence(
        ["n8n-nodes-base.webhook", "n8n-nodes-base.googleSheets", "n8n-nodes-base.httpRequest"]
    )
    for item in docs:
        item["text"] = f"{item['text']}\n{large_text}"

    def _plan_call(**kwargs):
        calls["plan"] += 1
        return _draft()

    def _compact_call(**kwargs):
        calls["compact"] += 1
        summaries = dict(kwargs["node_summaries"])
        return {
            key: pm._deterministic_compact_summary(value)  # noqa: SLF001
            for key, value in summaries.items()
        }

    def _summary_call(**kwargs):
        calls["summary"] += 1
        return kwargs["node_summaries"]

    monkeypatch.setattr(pm, "retrieve_pm_api_docs", lambda *args, **kwargs: docs)
    monkeypatch.setattr(pm, "_plan_with_structured_output", _plan_call)
    monkeypatch.setattr(pm, "_compact_node_evidence_with_structured_output", _compact_call)
    monkeypatch.setattr(pm, "_summarize_node_evidence_with_structured_output", _summary_call)
    monkeypatch.setattr(pm.settings, "MAX_CONTEXT_TOKENS", 300)

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert calls["compact"] == 1
    assert calls["plan"] == 1
    assert calls["summary"] == 0


def test_pm_never_uses_third_llm_call_even_when_still_overflow_after_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(use_case_id="uc_budget_still_overflow")
    calls = {"plan": 0, "compact": 0, "summary": 0}
    huge = " ".join(["Capability and limitation details."] * 500)
    docs = _docs_with_explicit_node_evidence(
        [
            "n8n-nodes-base.webhook",
            "n8n-nodes-base.googleSheets",
            "n8n-nodes-base.httpRequest",
            "n8n-nodes-base.code",
            "n8n-nodes-base.set",
        ]
    )
    for item in docs:
        item["text"] = f"{item['text']}\n{huge}"

    def _plan_call(**kwargs):
        calls["plan"] += 1
        return _draft()

    def _compact_call(**kwargs):
        calls["compact"] += 1
        # Return unchanged summaries so deterministic trimming path runs.
        return kwargs["node_summaries"]

    def _summary_call(**kwargs):
        calls["summary"] += 1
        return kwargs["node_summaries"]

    monkeypatch.setattr(pm, "retrieve_pm_api_docs", lambda *args, **kwargs: docs)
    monkeypatch.setattr(pm, "_plan_with_structured_output", _plan_call)
    monkeypatch.setattr(pm, "_compact_node_evidence_with_structured_output", _compact_call)
    monkeypatch.setattr(pm, "_summarize_node_evidence_with_structured_output", _summary_call)
    monkeypatch.setattr(pm.settings, "MAX_CONTEXT_TOKENS", 160)

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["architecture_plan"] is not None
    assert calls["compact"] == 1
    assert calls["plan"] == 1
    assert calls["summary"] == 0


def test_pm_uses_llm_node_summary_when_model_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_llm_summary")
    calls = {"summary": 0}

    def _summary_call(**kwargs):
        calls["summary"] += 1
        summaries = dict(kwargs["node_summaries"])
        summary = summaries["n8n-nodes-base.webhook"]
        summaries["n8n-nodes-base.webhook"] = summary.model_copy(
            update={
                "purpose": "Receives incoming HTTP events as workflow trigger.",
                "capabilities": ["Receives webhook calls", "Exposes endpoint path"],
                "limitations": ["Requires public endpoint reachability"],
            }
        )
        return summaries

    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda *args, **kwargs: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(pm, "_summarize_node_evidence_with_structured_output", _summary_call)
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    state = _state(selected_use_case=selected)
    state["workflow_context"] = {"model": "mistralai/ministral-3-14b-reasoning", "request_id": "req-pm-test"}
    updates = pm.product_manager_agent_node(state)

    assert calls["summary"] == 1
    required = updates["architecture_plan"].required_nodes[0]
    assert "receives incoming http events" in required.why_required.lower()


def test_pm_allows_direct_build_request_without_selected_use_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda use_case, request_id=None: _docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]),
    )
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=None,
            entry_intent=EntryIntent.workflow_build_request,
            user_query="Create a workflow that receives a webhook and writes rows to Google Sheets.",
        )
    )

    assert updates["architecture_plan"] is not None
    assert updates["target_stage"] == AgentStage.engineer_agent
    assert "pm_use_case_derived_from_direct_build_request" in updates["routing_signals"]


def test_pm_still_requires_selected_use_case_on_commercial_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieve should not run")),
    )

    updates = pm.product_manager_agent_node(
        _state(
            selected_use_case=None,
            entry_intent=EntryIntent.business_discovery_conversation,
            user_query="We have many manual processes in operations.",
        )
    )

    assert updates["architecture_plan"] is None
    assert updates["target_stage"] is None
    assert "pm_missing_selected_use_case" in updates["routing_signals"]


def test_pm_handles_malformed_chunks_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = _use_case(use_case_id="uc_malformed_chunks")
    malformed_docs = [
        {"doc_id": "bad-1", "text": None, "metadata": "invalid"},
        {"doc_id": "bad-2", "metadata": {"kind": "NODE_OVERVIEW"}},  # missing nodeType
        {"doc_id": "bad-3", "metadata": {"nodeType": "n8n-nodes-base.webhook"}},
        "not-a-dict",
    ]
    monkeypatch.setattr(pm, "retrieve_pm_api_docs", lambda *args, **kwargs: malformed_docs)
    monkeypatch.setattr(pm, "_plan_with_structured_output", lambda **_: _draft())

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["current_stage"] == "product_manager_agent"
    assert "routing_signals" in updates
    assert isinstance(updates["routing_signals"], list)


def test_pm_structured_output_failure_uses_deterministic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = _use_case(use_case_id="uc_pm_fallback")
    monkeypatch.setattr(
        pm,
        "retrieve_pm_api_docs",
        lambda *args, **kwargs: _docs_with_explicit_node_evidence(
            ["n8n-nodes-base.webhook", "n8n-nodes-base.googleSheets"]
        ),
    )
    monkeypatch.setattr(
        pm,
        "_plan_with_structured_output",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("structured output unavailable")),
    )

    updates = pm.product_manager_agent_node(_state(selected_use_case=selected))
    assert updates["architecture_plan"] is not None
    assert updates["architecture_plan"].stages
    assert updates["workflow_context"].planning_ready is True
    assert updates["target_stage"] == AgentStage.engineer_agent


def test_pm_noisy_evidence_respects_limit_and_order_stability() -> None:
    docs = _docs_with_explicit_node_evidence(
        [
            "n8n-nodes-base.webhook",
            "n8n-nodes-base.googleSheets",
            "n8n-nodes-base.httpRequest",
            "n8n-nodes-base.slack",
            "n8n-nodes-base.postgres",
            "n8n-nodes-base.mysql",
            "n8n-nodes-base.emailSend",
            "n8n-nodes-base.if",
            "n8n-nodes-base.switch",
            "n8n-nodes-base.function",
            "n8n-nodes-base.set",
            "n8n-nodes-base.merge",
            "n8n-nodes-base.wait",
            "n8n-nodes-base.cron",
        ]
    )
    docs.extend(_docs_with_explicit_node_evidence(["n8n-nodes-base.webhook"]))
    docs.extend(_docs_without_explicit_node_evidence())

    required_nodes_a = pm.extract_required_nodes_from_docs(docs)
    required_nodes_b = pm.extract_required_nodes_from_docs(docs)

    assert len(required_nodes_a) <= pm._MAX_REQUIRED_NODES  # noqa: SLF001
    assert [node.node_type for node in required_nodes_a] == [
        node.node_type for node in required_nodes_b
    ]
    assert required_nodes_a[0].node_type == "n8n-nodes-base.webhook"
