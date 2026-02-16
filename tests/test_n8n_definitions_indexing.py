from app.n8n_definitions_indexing import (
    build_credentials_chunks,
    build_nodes_chunks,
)


def _sample_node():
    return {
        "displayName": "Demo Node",
        "name": "n8n-nodes-base.demoNode",
        "group": ["transform"],
        "version": 2,
        "description": "Demo node for tests",
        "usableAsTool": True,
        "inputs": ["main"],
        "outputs": ["main"],
        "credentials": [{"name": "demoApi", "required": True}],
        "codex": {
            "resources": {
                "primaryDocumentation": [{"url": "https://docs.n8n.io/demo-node"}],
                "credentialDocumentation": [{"url": "https://docs.n8n.io/demo-credential"}],
            }
        },
        "properties": [
            {
                "displayName": "Resource",
                "name": "resource",
                "type": "options",
                "options": [
                    {"name": "User", "value": "user"},
                    {"name": "Team", "value": "team"},
                ],
                "default": "user",
            },
            {
                "displayName": "Operation",
                "name": "operation",
                "type": "options",
                "displayOptions": {"show": {"resource": ["user"]}},
                "options": [
                    {"name": "Create", "value": "create", "action": "Create user"},
                    {"name": "Get", "value": "get", "action": "Get user"},
                ],
                "default": "create",
            },
            {
                "displayName": "Operation",
                "name": "operation",
                "type": "options",
                "displayOptions": {"show": {"resource": ["team"]}},
                "options": [
                    {"name": "List", "value": "list", "action": "List teams"},
                ],
                "default": "list",
            },
            {
                "displayName": "Email",
                "name": "email",
                "type": "string",
                "required": True,
                "default": "",
                "displayOptions": {"show": {"resource": ["user"], "operation": ["create"]}},
            },
            {
                "displayName": "Return All",
                "name": "returnAll",
                "type": "boolean",
                "default": False,
                "displayOptions": {"show": {"resource": ["user"], "operation": ["get"]}},
            },
            {
                "displayName": "Limit",
                "name": "limit",
                "type": "number",
                "default": 50,
                "displayOptions": {"show": {"resource": ["user", "team"]}},
            },
            {
                "displayName": "Additional Fields",
                "name": "additionalFields",
                "type": "collection",
                "default": {},
                "displayOptions": {"show": {"resource": ["user"], "operation": ["create"]}},
                "options": [
                    {"displayName": "Timezone", "name": "timezone", "type": "string", "default": ""},
                    {
                        "displayName": "Location",
                        "name": "location",
                        "type": "fixedCollection",
                        "default": {},
                        "options": [
                            {
                                "displayName": "Geo",
                                "name": "geo",
                                "values": [
                                    {"displayName": "Lat", "name": "lat", "type": "number", "default": 0},
                                    {"displayName": "Lng", "name": "lng", "type": "number", "default": 0},
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "displayName": "Notes",
                "name": "notes",
                "type": "string",
                "default": "",
            },
        ],
    }


def _sample_credential():
    return {
        "name": "demoApi",
        "displayName": "Demo API",
        "documentationUrl": "demo-api",
        "supportedNodes": ["n8n-nodes-base.demoNode"],
        "test": {"request": {"baseURL": "https://api.demo.local", "url": "/me"}},
        "properties": [
            {
                "displayName": "API Key",
                "name": "apiKey",
                "type": "string",
                "default": "",
                "typeOptions": {"password": True},
            },
            {
                "displayName": "Mode",
                "name": "mode",
                "type": "options",
                "default": "full",
                "options": [
                    {"name": "Full", "value": "full"},
                    {"name": "Lite", "value": "lite"},
                ],
            },
            {
                "displayName": "Allowed Domains",
                "name": "allowedDomains",
                "type": "string",
                "default": "",
                "displayOptions": {"show": {"mode": ["full"]}},
            },
        ],
    }


def _sample_trigger_without_scopes():
    return {
        "displayName": "Demo Trigger",
        "name": "n8n-nodes-base.demoTrigger",
        "group": ["trigger"],
        "version": 1,
        "description": "Trigger without resource operation model",
        "inputs": [],
        "outputs": ["main"],
        "properties": [
            {"displayName": "Event", "name": "event", "type": "options", "options": [{"name": "A", "value": "a"}], "default": "a"},
            {"displayName": "Resolve Data", "name": "resolveData", "type": "boolean", "default": True},
        ],
    }


def _sample_operation_only_node():
    return {
        "displayName": "Operation Only Node",
        "name": "n8n-nodes-base.operationOnly",
        "group": ["transform"],
        "version": 1,
        "description": "Node with operation-only scopes",
        "inputs": ["main"],
        "outputs": ["main"],
        "properties": [
            {
                "displayName": "Operation",
                "name": "operation",
                "type": "options",
                "options": [
                    {"name": "Invoke", "value": "invoke"},
                    {"name": "List", "value": "list"},
                ],
                "default": "invoke",
            },
            {
                "displayName": "Function",
                "name": "fn",
                "type": "string",
                "default": "",
                "displayOptions": {"show": {"operation": ["invoke"]}},
            },
            {
                "displayName": "Limit",
                "name": "limit",
                "type": "number",
                "default": 10,
                "displayOptions": {"show": {"operation": ["list"]}},
            },
        ],
    }


def _sample_resource_only_node():
    return {
        "displayName": "Resource Only Node",
        "name": "n8n-nodes-base.resourceOnly",
        "group": ["trigger"],
        "version": 1,
        "description": "Node with resource-only scopes",
        "inputs": [],
        "outputs": ["main"],
        "properties": [
            {
                "displayName": "Resource",
                "name": "resource",
                "type": "options",
                "options": [
                    {"name": "Workspace", "value": "workspace"},
                    {"name": "Repository", "value": "repository"},
                ],
                "default": "workspace",
            },
            {
                "displayName": "Workspace",
                "name": "workspace",
                "type": "string",
                "default": "",
                "displayOptions": {"show": {"resource": ["workspace"]}},
            },
            {
                "displayName": "Repository",
                "name": "repository",
                "type": "string",
                "default": "",
                "displayOptions": {"show": {"resource": ["repository"]}},
            },
        ],
    }


def test_build_nodes_chunks_creates_scope_and_complex_kinds():
    chunks, metrics = build_nodes_chunks([_sample_node()], source="n8n-nodes")
    kinds = [chunk["kind"] for chunk in chunks]

    assert metrics.nodes_processed == 1
    assert metrics.scopes_detected == 3
    assert kinds.count("NODE_OVERVIEW") == 1
    assert kinds.count("NODE_RESOURCE") == 1
    assert kinds.count("NODE_RESOURCE_OPERATION") == 3
    assert kinds.count("NODE_PARAMS_FOR_RESOURCE_OPERATION") == 3
    assert kinds.count("NODE_GLOBAL_PARAMS") == 1
    assert kinds.count("NODE_COMPLEX_FIELD") == 1

    complex_chunk = next(chunk for chunk in chunks if chunk["kind"] == "NODE_COMPLEX_FIELD")
    assert "additionalFields.location.geo.lat" in complex_chunk["content"]
    assert complex_chunk["metadata"]["group_path"] == "additionalFields"


def test_trigger_node_without_scope_signals_does_not_warn():
    chunks, metrics = build_nodes_chunks([_sample_trigger_without_scopes()], source="n8n-nodes")
    assert metrics.warnings == []
    kinds = [chunk["kind"] for chunk in chunks]
    assert kinds.count("NODE_OVERVIEW") == 1
    assert kinds.count("NODE_GLOBAL_PARAMS") == 1


def test_operation_only_node_creates_scoped_chunks():
    chunks, metrics = build_nodes_chunks([_sample_operation_only_node()], source="n8n-nodes")
    kinds = [chunk["kind"] for chunk in chunks]

    assert any("operation-only scopes detected" in warning for warning in metrics.warnings)
    assert metrics.scopes_detected == 2
    assert kinds.count("NODE_RESOURCE_OPERATION") == 2
    assert kinds.count("NODE_PARAMS_FOR_RESOURCE_OPERATION") == 2

    scoped_ids = [chunk["id"] for chunk in chunks if chunk["kind"] == "NODE_RESOURCE_OPERATION"]
    assert any("resource_none" in chunk_id for chunk_id in scoped_ids)
    assert any("operation_invoke" in chunk_id for chunk_id in scoped_ids)


def test_resource_only_node_creates_scoped_chunks():
    chunks, metrics = build_nodes_chunks([_sample_resource_only_node()], source="n8n-nodes")
    kinds = [chunk["kind"] for chunk in chunks]

    assert any("resource-only scopes detected" in warning for warning in metrics.warnings)
    assert metrics.scopes_detected == 2
    assert kinds.count("NODE_RESOURCE") == 1
    assert kinds.count("NODE_RESOURCE_OPERATION") == 2
    assert kinds.count("NODE_PARAMS_FOR_RESOURCE_OPERATION") == 2

    scoped_chunks = [chunk for chunk in chunks if chunk["kind"] == "NODE_RESOURCE_OPERATION"]
    assert any(chunk["metadata"]["resource"] == "workspace" for chunk in scoped_chunks)
    assert all(chunk["metadata"]["operation"] is None for chunk in scoped_chunks)


def test_build_nodes_chunks_is_deterministic():
    first_chunks, _ = build_nodes_chunks([_sample_node()], source="n8n-nodes")
    second_chunks, _ = build_nodes_chunks([_sample_node()], source="n8n-nodes")
    assert [chunk["id"] for chunk in first_chunks] == [chunk["id"] for chunk in second_chunks]
    assert [chunk["metadata"]["contentHash"] for chunk in first_chunks] == [
        chunk["metadata"]["contentHash"] for chunk in second_chunks
    ]


def test_build_credentials_chunks_creates_overview_fields_and_detail():
    chunks, metrics = build_credentials_chunks(
        [_sample_credential()],
        source="n8n-credentials",
        options_preview_limit=2,
    )
    kinds = [chunk["kind"] for chunk in chunks]

    assert metrics.credentials_processed == 1
    assert kinds.count("CRED_OVERVIEW") == 1
    assert kinds.count("CRED_FIELDS") == 1
    assert kinds.count("CRED_FIELD_DETAIL") >= 1

    overview = next(chunk for chunk in chunks if chunk["kind"] == "CRED_OVERVIEW")
    assert overview["metadata"]["credentialType"] == "demoApi"
    assert overview["metadata"]["supportedNodes"] == ["n8n-nodes-base.demoNode"]
