from __future__ import annotations

from app.workflow.planner import Plan, PlanScope, plan_scope
from app.workflow.subgraph_builder import build_subgraph
from app.workflow.workflow_summary import build_workflow_summary


def _sample_workflow() -> dict:
    return {
        "id": "wf-1",
        "name": "Sample Workflow",
        "nodes": [
            {
                "id": "1",
                "name": "Webhook Trigger",
                "type": "n8n-nodes-base.webhook",
                "position": [0, 0],
                "parameters": {"path": "hook"},
            },
            {
                "id": "2",
                "name": "HTTP Request",
                "type": "n8n-nodes-base.httpRequest",
                "position": [200, 0],
                "parameters": {"url": "https://api.example.com"},
                "credentials": {"httpBasicAuth": {"id": "cred-1"}},
            },
            {
                "id": "3",
                "name": "If Router",
                "type": "n8n-nodes-base.if",
                "position": [400, 0],
                "parameters": {"conditions": {"string": [{"value1": "={{$json.ok}}"}]}},
            },
            {
                "id": "4",
                "name": "Code Transform",
                "type": "n8n-nodes-base.code",
                "position": [600, 0],
                "parameters": {"jsCode": "return items;"},
            },
            {
                "id": "5",
                "name": "Merge Results",
                "type": "n8n-nodes-base.merge",
                "position": [800, 0],
                "parameters": {"mode": "append"},
            },
            {
                "id": "6",
                "name": "Slack Notify",
                "type": "n8n-nodes-base.slack",
                "position": [1000, 0],
                "parameters": {"channel": "alerts"},
            },
        ],
        "connections": {
            "Webhook Trigger": {
                "main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]
            },
            "HTTP Request": {
                "main": [[{"node": "If Router", "type": "main", "index": 0}]]
            },
            "If Router": {
                "main": [
                    [{"node": "Code Transform", "type": "main", "index": 0}],
                    [{"node": "Slack Notify", "type": "main", "index": 0}],
                ]
            },
            "Code Transform": {
                "main": [[{"node": "Merge Results", "type": "main", "index": 0}]]
            },
            "Slack Notify": {
                "main": [[{"node": "Merge Results", "type": "main", "index": 1}]]
            },
        },
    }


def _summary():
    workflow = _sample_workflow()
    return build_workflow_summary(workflow, workflow_id=workflow["id"])


def test_plan_global_when_question_is_global() -> None:
    summary = _summary()
    plan = plan_scope("Como mejoro mi flujo en general?", summary)
    assert plan.scope == PlanScope.workflow_global


def test_plan_node_specific_detects_http_request() -> None:
    summary = _summary()
    plan = plan_scope("El HTTP Request falla con auth", summary)
    assert plan.scope in (PlanScope.node_specific, PlanScope.subgraph)
    assert "2" in plan.target_node_ids


def test_plan_subgraph_when_two_nodes_are_mentioned() -> None:
    summary = _summary()
    plan = plan_scope("Entre If Router y Slack Notify hay un problema", summary)
    assert plan.scope == PlanScope.subgraph
    assert "3" in plan.target_node_ids
    assert "6" in plan.target_node_ids


def test_subgraph_builder_includes_path_between_targets() -> None:
    summary = _summary()
    plan = Plan(scope=PlanScope.subgraph, target_node_ids=["3", "5"], confidence=0.8)
    subgraph = build_subgraph(summary, "", plan)
    assert "3" in subgraph.node_ids
    assert "5" in subgraph.node_ids
    assert "4" in subgraph.node_ids
