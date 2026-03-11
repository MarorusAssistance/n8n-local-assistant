# Consultant Agent (Information Path)

## Scope
`consultant_agent` handles `information_request` in LangGraph.

- Informational output only.
- No workflow build/edit/fix/qa execution.
- Conversation-first behavior.
- Tool-calling only when retrieval is needed.

## Source boundaries
The consultant path exposes source-specific boundaries:

- `api_docs_index_tool` -> `retrieve_context(..., source_filter=\"n8n-docs\")`
- `nodes_index_tool` -> `retrieve_context(..., source_filter=LINKED_DEFS_NODES_SOURCE)`
- `credentials_index_tool` -> `retrieve_context(..., source_filter=LINKED_DEFS_CREDENTIALS_SOURCE)`
- `active_workflow_tool` -> read-only active workflow snapshot from runtime/state
- `templates_index_tool` -> explicit `unavailable` (future-compatible boundary)

`source_filter` is per-call and additive; legacy retrieval stays unchanged when `source_filter=None`.

## State fields
Consultant updates these typed fields in `MultiAgentGraphState` / `MultiAgentGraphResult`:

- `consultant_query_analysis`
- `consultant_selected_sources`
- `consultant_tools_used`
- `consultant_used_retrieval`
- `consultant_retrieval_results`
- `consultant_response`
- `consultant_notes`

## Chat behavior
- For consultant path (`current_stage=consultant_agent`), `message.content` is plain text.
- Streaming in this path is token-like incremental SSE.
- Non-consultant reasoning paths keep JSON envelope behavior.

## Observability
Consultant emits trace events with:

- retrieval decision (`conversation-first` vs retrieval)
- selected sources
- tool call order
- retrieval result counts
- final response length
