from app.doc_links import (
    METHOD_DERIVED_FROM_NODE,
    METHOD_URL_EXACT,
    aggregate_doc_pages,
    build_credential_catalog,
    build_node_catalog,
    derive_doc_page_key,
    generate_credential_links,
    generate_node_links,
    normalize_reference,
    normalized_reference_variants,
)


def test_normalize_reference_handles_url_and_path_variants():
    value = "https://Docs.n8n.io/Integrations/Builtin/Credentials/ActionNetwork/#section"
    normalized = normalize_reference(value)
    assert normalized == "docs.n8n.io/integrations/builtin/credentials/actionnetwork"
    variants = normalized_reference_variants(value)
    assert "docs.n8n.io/integrations/builtin/credentials/actionnetwork" in variants
    assert "/integrations/builtin/credentials/actionnetwork" in variants


def test_derive_doc_page_key_uses_priority_page_id_over_url_path():
    metadata = {
        "page_id": "Docs:ActionNetwork",
        "source_url": "https://docs.n8n.io/integrations/builtin/credentials/actionnetwork/",
        "path": "/integrations/builtin/credentials/actionnetwork/",
    }
    key, refs, source_field = derive_doc_page_key(metadata)
    assert key == "docs:actionnetwork"
    assert source_field == "page_id"
    assert "/integrations/builtin/credentials/actionnetwork" in refs


def test_generate_node_links_url_exact():
    doc_rows = [
        {
            "text": "Action Network docs page",
            "metadata": {
                "source_url": "/integrations/builtin/app-nodes/n8n-nodes-base.actionnetwork/",
                "title": "Action Network",
            },
            "url": None,
        }
    ]
    node_rows = [
        {
            "metadata": {
                "nodeType": "n8n-nodes-base.actionNetwork",
                "displayName": "Action Network",
                "doc_urls": [
                    "https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.actionnetwork/"
                ],
                "credentialTypes_required": ["actionNetworkApi"],
            }
        }
    ]
    pages, warnings = aggregate_doc_pages(doc_rows)
    assert warnings == []
    nodes = build_node_catalog(node_rows)
    links = generate_node_links(pages, nodes, mention_limit_per_page=0, similarity_limit_per_page=0)
    assert len(links) == 1
    assert links[0].method == METHOD_URL_EXACT
    assert links[0].node_type == "n8n-nodes-base.actionNetwork"
    assert links[0].confidence == 1.0


def test_generate_credential_links_derived_from_node():
    doc_rows = [
        {
            "text": "Action Network docs page",
            "metadata": {
                "source_url": "/integrations/builtin/app-nodes/n8n-nodes-base.actionnetwork/",
                "title": "Action Network",
            },
            "url": None,
        }
    ]
    node_rows = [
        {
            "metadata": {
                "nodeType": "n8n-nodes-base.actionNetwork",
                "displayName": "Action Network",
                "doc_urls": [
                    "https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.actionnetwork/"
                ],
                "credentialTypes_required": ["actionNetworkApi"],
            }
        }
    ]
    credential_rows = [
        {
            "metadata": {
                "credentialType": "actionNetworkApi",
                "displayName": "Action Network API",
                "documentationKey": "not-matching-key",
                "supportedNodes": ["n8n-nodes-base.actionNetwork"],
            }
        }
    ]
    pages, _ = aggregate_doc_pages(doc_rows)
    nodes = build_node_catalog(node_rows)
    credentials = build_credential_catalog(credential_rows)
    node_links = generate_node_links(pages, nodes, mention_limit_per_page=0, similarity_limit_per_page=0)
    cred_links, warnings = generate_credential_links(
        pages,
        credentials,
        nodes,
        node_links,
        mention_limit_per_page=0,
        similarity_limit_per_page=0,
    )
    assert warnings == []
    assert len(cred_links) == 1
    assert cred_links[0].method == METHOD_DERIVED_FROM_NODE
    assert cred_links[0].credential_type == "actionNetworkApi"


def test_aggregate_doc_pages_warns_when_page_key_missing():
    rows = [{"text": "No key", "metadata": {}, "url": None}]
    pages, warnings = aggregate_doc_pages(rows)
    assert pages == {}
    assert len(warnings) == 1
