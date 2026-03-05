from __future__ import annotations

from bench.trace import (
    entries_for_request,
    extract_retrieval_views,
    parse_trace_text,
    redact_text_light,
)


def test_parse_trace_text_and_trace_event() -> None:
    text = """2026-03-05 10:00:00,000 INFO n8n-assistant.trace: TRACE REQUEST id=req-1
meta: conv=conv-1 wf=- stream=False messages=1 history=0 user_len=4
usuario:
  hola
2026-03-05 10:00:00,050 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_prompt","request_id":"req-1","stage":"docs_only.synthesis","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hola"}],"total_chars":7}
2026-03-05 10:00:00,100 INFO n8n-assistant.trace: TRACE EVENT {"event":"llm_output","request_id":"req-1","stage":"docs_only.synthesis","model":"x","latency_ms":12.3}
"""
    entries = parse_trace_text(text)
    assert len(entries) == 3
    assert entries[0].event_type == "trace_request"
    assert entries[0].request_id == "req-1"
    assert entries[1].event_type == "llm_prompt"
    assert entries[1].stage == "docs_only.synthesis"
    assert entries[1].request_id == "req-1"


def test_entries_for_request_fallback_to_conversation_id() -> None:
    text = """2026-03-05 10:00:00,000 INFO n8n-assistant.trace: TRACE REQUEST id=req-a
meta: conv=conv-a wf=- stream=False messages=1 history=0 user_len=4
usuario:
  hola
2026-03-05 10:00:01,000 INFO n8n-assistant.trace: other event without id
"""
    entries = parse_trace_text(text)
    by_req = entries_for_request(entries, request_id="req-a", conversation_id=None)
    assert len(by_req) == 1
    by_conv = entries_for_request(entries, request_id=None, conversation_id="conv-a")
    assert len(by_conv) == 1


def test_light_redaction_masks_secret_like_values() -> None:
    raw = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz token=super-secret-value"
    redacted = redact_text_light(raw)
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "super-secret-value" not in redacted
    assert "***REDACTED***" in redacted


def test_extract_retrieval_views_from_structured_event() -> None:
    text = """2026-03-05 10:00:00,000 INFO n8n-assistant.trace: TRACE EVENT {"event":"retrieval_final","request_id":"req-1","stage":"rag.retrieve_context","pre_rerank_chunks":[{"content":"doc pool","rrf_score":0.111}],"post_rerank_chunks":[{"content":"doc top","rerank_score":0.812}],"final_chunks":[{"content":"doc top","rerank_score":0.812},{"content":"node linked","rerank_score":0.801,"linked_def_type":"node"},{"content":"credential linked","rerank_score":0.755,"linked_def_type":"credential"}]}
"""
    entries = parse_trace_text(text)
    views = extract_retrieval_views(entries)
    assert len(views["pre_docs"]) == 1
    assert len(views["post_docs"]) == 1
    assert len(views["post_linked_node"]) == 1
    assert len(views["post_linked_credential"]) == 1
