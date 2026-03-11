from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.config import settings
from app.features.reasoning.multi_agent_contracts import (
    ConsultantQueryAnalysis,
    ConsultantSource,
)
from app.graphs.nodes import consultant_agent as consultant


def _analysis(
    *,
    retrieval_needed: bool,
    selected_sources: List[ConsultantSource],
    conversation_history_sufficient: bool = False,
    needs_active_workflow_context: bool = False,
) -> ConsultantQueryAnalysis:
    return ConsultantQueryAnalysis(
        request_type="general_information",
        key_topics=[],
        needs_active_workflow_context=needs_active_workflow_context,
        retrieval_needed=retrieval_needed,
        conversation_history_sufficient=conversation_history_sufficient,
        source_limited=False,
        selected_sources=selected_sources,
        analysis_notes=[],
    )


def _state(query: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = {
        "user_query": query,
        "routing_signals": [],
        "runtime_context": {
            "model": None,
            "request_id": "req-consultant",
            "conversation_context": [
                {"role": "user", "content": "We already discussed webhook vs schedule."},
                {"role": "assistant", "content": "Webhook is event-based."},
            ],
        },
    }
    if extra:
        base.update(extra)
    return base


@pytest.mark.parametrize(
    "query",
    [
        "Explain again the difference between webhook and schedule trigger.",
        "Summarize what we already said about retry strategies.",
        "Based on our previous chat, what was the recommendation?",
    ],
)
def test_conversation_first_examples_do_not_call_tools(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=False,
            selected_sources=[ConsultantSource.conversation_history],
            conversation_history_sufficient=True,
        ),
    )
    monkeypatch.setattr(
        consultant,
        "_conversation_only_answer",
        lambda **kwargs: "Conversation-first answer.",
    )

    updates = consultant.consultant_agent_node(_state(query))

    assert updates["current_stage"] == "consultant_agent"
    assert updates["consultant_used_retrieval"] is False
    assert updates["consultant_tools_used"] == []
    assert updates["consultant_response"].text == "Conversation-first answer."


@pytest.mark.parametrize(
    "query",
    [
        "What nodes can capture inbound HTTP events?",
        "Compare webhook vs form trigger nodes.",
        "Which n8n nodes are best for branching logic?",
    ],
)
def test_nodes_index_dominant_examples_use_nodes_source(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    calls: List[str | None] = []

    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=True,
            selected_sources=[ConsultantSource.nodes_index],
        ),
    )
    monkeypatch.setattr(consultant, "create_agent", None)

    def fake_retrieve_context(*args, **kwargs):
        calls.append(kwargs.get("source_filter"))
        return [{"doc_id": "node-1", "title": "Webhook", "text": "Receives HTTP events."}]

    monkeypatch.setattr(consultant, "retrieve_context", fake_retrieve_context)
    updates = consultant.consultant_agent_node(_state(query))

    assert updates["consultant_used_retrieval"] is True
    assert updates["consultant_tools_used"][0].source == ConsultantSource.nodes_index
    assert calls and calls[0] == settings.LINKED_DEFS_NODES_SOURCE


@pytest.mark.parametrize(
    "query",
    [
        "What credential type is needed for Google Sheets node?",
        "Explain OAuth2 credential requirements in n8n.",
        "How should I map credential objects for API integrations?",
    ],
)
def test_credentials_index_dominant_examples_use_credentials_source(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    calls: List[str | None] = []
    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=True,
            selected_sources=[ConsultantSource.credentials_index],
        ),
    )
    monkeypatch.setattr(consultant, "create_agent", None)

    def fake_retrieve_context(*args, **kwargs):
        calls.append(kwargs.get("source_filter"))
        return [{"doc_id": "cred-1", "title": "OAuth2", "text": "Use OAuth2 for this integration."}]

    monkeypatch.setattr(consultant, "retrieve_context", fake_retrieve_context)
    updates = consultant.consultant_agent_node(_state(query))
    assert updates["consultant_tools_used"][0].source == ConsultantSource.credentials_index
    assert calls and calls[0] == settings.LINKED_DEFS_CREDENTIALS_SOURCE


@pytest.mark.parametrize(
    "query",
    [
        "What does the official documentation say about webhook retries?",
        "From docs, how does HTTP Request pagination work?",
        "Show API-doc behavior for wait node and timeouts.",
    ],
)
def test_api_docs_dominant_examples_use_api_docs_source(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    calls: List[str | None] = []
    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=True,
            selected_sources=[ConsultantSource.api_docs_index],
        ),
    )
    monkeypatch.setattr(consultant, "create_agent", None)

    def fake_retrieve_context(*args, **kwargs):
        calls.append(kwargs.get("source_filter"))
        return [{"doc_id": "doc-1", "url": "https://docs.n8n.io/x", "text": "Docs grounded snippet."}]

    monkeypatch.setattr(consultant, "retrieve_context", fake_retrieve_context)
    updates = consultant.consultant_agent_node(_state(query))
    assert updates["consultant_tools_used"][0].source == ConsultantSource.api_docs_index
    assert calls and calls[0] == "n8n-docs"


@pytest.mark.parametrize(
    "query",
    [
        "Explain what my current workflow is doing.",
        "Which part of this workflow handles outbound calls?",
    ],
)
def test_active_workflow_context_examples(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=True,
            selected_sources=[ConsultantSource.active_workflow],
            needs_active_workflow_context=True,
        ),
    )
    monkeypatch.setattr(consultant, "create_agent", None)
    updates = consultant.consultant_agent_node(
        _state(
            query,
            extra={
                "active_workflow_id": "wf_123",
                "final_workflow_json": {"nodes": [{"id": "1"}, {"id": "2"}]},
            },
        )
    )
    assert updates["consultant_tools_used"][0].source == ConsultantSource.active_workflow
    assert updates["consultant_retrieval_results"][0].result_count == 2


def test_multi_source_examples_preserve_tool_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=True,
            selected_sources=[
                ConsultantSource.nodes_index,
                ConsultantSource.credentials_index,
            ],
        ),
    )
    monkeypatch.setattr(consultant, "create_agent", None)
    monkeypatch.setattr(
        consultant,
        "retrieve_context",
        lambda *args, **kwargs: [{"doc_id": "x", "text": "snippet"}],
    )
    updates = consultant.consultant_agent_node(_state("Compare node + credentials requirements"))
    tools_used = updates["consultant_tools_used"]
    assert len(tools_used) == 2
    assert tools_used[0].call_order == 1
    assert tools_used[1].call_order == 2


@pytest.mark.parametrize(
    "query",
    [
        "Recommend a template for crm sync.",
        "Any starter template for onboarding automation?",
    ],
)
def test_templates_unavailable_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=True,
            selected_sources=[ConsultantSource.templates_index],
        ),
    )
    monkeypatch.setattr(consultant, "create_agent", None)
    updates = consultant.consultant_agent_node(_state(query))
    assert updates["consultant_tools_used"][0].status == "unavailable"
    assert updates["consultant_retrieval_results"][0].unavailable_reason


@pytest.mark.parametrize(
    "query",
    [
        "Explain the best approach to compare two node families.",
        "Give me guidance, not implementation.",
    ],
)
def test_consultant_output_is_informative_only_and_no_workflow_json(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    monkeypatch.setattr(
        consultant,
        "analyze_consultant_query",
        lambda **kwargs: _analysis(
            retrieval_needed=False,
            selected_sources=[ConsultantSource.conversation_history],
            conversation_history_sufficient=True,
        ),
    )
    monkeypatch.setattr(
        consultant,
        "_conversation_only_answer",
        lambda **kwargs: "Informational response only.",
    )
    updates = consultant.consultant_agent_node(_state(query))
    assert "target_stage" not in updates
    assert "final_workflow_json" not in updates
    assert updates["consultant_response"].text == "Informational response only."

