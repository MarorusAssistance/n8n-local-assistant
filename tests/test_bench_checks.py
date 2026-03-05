from __future__ import annotations

import json

from bench.checks import build_catalog_from_payload, run_checks
from bench.models import CaseLimits, CaseSpec, RequirementSpec


def _catalog():
    nodes_payload = [
        {"name": "n8n-nodes-base.webhook", "credentials": []},
        {
            "name": "n8n-nodes-base.googleSheets",
            "credentials": [{"name": "googleSheetsOAuth2Api", "required": True}],
        },
        {"name": "n8n-nodes-base.scheduleTrigger", "credentials": []},
        {"name": "n8n-nodes-base.slack", "credentials": [{"name": "slackApi", "required": True}]},
    ]
    credentials_payload = [
        {"name": "googleSheetsOAuth2Api", "supportedNodes": ["n8n-nodes-base.googleSheets"]},
        {"name": "slackApi", "supportedNodes": ["n8n-nodes-base.slack"]},
    ]
    return build_catalog_from_payload(nodes_payload, credentials_payload)


def test_run_checks_accepts_valid_workflow() -> None:
    workflow = {
        "name": "Demo",
        "nodes": [
            {
                "name": "Schedule Trigger",
                "type": "n8n-nodes-base.scheduleTrigger",
                "parameters": {"rule": {"interval": [{"hours": 8, "minutes": 0}]}},
                "position": [0, 0],
            },
            {
                "name": "Webhook",
                "type": "n8n-nodes-base.webhook",
                "parameters": {},
                "position": [250, 0],
            },
            {
                "name": "Google Sheets",
                "type": "n8n-nodes-base.googleSheets",
                "parameters": {"limit": 10},
                "credentials": {"googleSheetsOAuth2Api": {"id": "cred-1"}},
                "position": [500, 0],
            },
        ],
        "connections": {
            "Webhook": {
                "main": [[{"node": "Google Sheets", "type": "main", "index": 0}]]
            }
        },
    }
    case = CaseSpec(
        id="ok_case",
        user_message="x",
        requirements=[
            RequirementSpec(type="must_include_node_type", value="n8n-nodes-base.googleSheets"),
            RequirementSpec(
                type="must_include_keyword_in_node_params",
                value={"key": "limit", "equals": 10},
            ),
            RequirementSpec(type="must_have_schedule_daily_at", value="08:00"),
            RequirementSpec(
                type="must_have_connection",
                value={
                    "from_type": "n8n-nodes-base.webhook",
                    "to_type": "n8n-nodes-base.googleSheets",
                },
            ),
        ],
        limits=CaseLimits(max_nodes=10),
    )

    result = run_checks(json.dumps(workflow), case=case, catalog=_catalog())

    assert result["parse_ok"] is True
    assert result["compliance_score"] == 100.0
    assert result["failures"] == []


def test_run_checks_detects_incompatible_credential() -> None:
    workflow = {
        "name": "BadCred",
        "nodes": [
            {
                "name": "Google Sheets",
                "type": "n8n-nodes-base.googleSheets",
                "parameters": {},
                "credentials": {"slackApi": {"id": "cred-1"}},
                "position": [0, 0],
            }
        ],
        "connections": {},
    }
    case = CaseSpec(id="bad_cred", user_message="x")

    result = run_checks(json.dumps(workflow), case=case, catalog=_catalog())

    assert result["parse_ok"] is True
    assert result["compliance_score"] < 100.0
    compat_failures = result["trace"]["credential_compatibility"]["failures"]
    assert any("not compatible" in message for message in compat_failures)
