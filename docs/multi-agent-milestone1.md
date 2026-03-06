# Multi-Agent Milestone 1

## Scope
Milestone 1 introduces a LangGraph entry runtime for multi-agent routing with no real business logic execution in subagents.

Implemented scope:
- Typed entry intent classification.
- Typed shared state for future loops.
- Stub nodes for `commercial`, `consultant`, `product_manager`, `engineer`, and `qa`.
- Direct replacement of reasoning LangGraph runtime output with a routing envelope.

Not implemented in this milestone:
- Template retrieval/integration.
- Notion or external indexing integration.
- Internal PM/Engineer/QA replan loops.
- Workflow JSON generation.

## Runtime flow
1. `START -> entry_router`
2. Conditional route by `target_stage`:
   - `commercial_agent`
   - `consultant_agent`
   - `product_manager_agent`
   - `engineer_agent`
   - `qa_agent`
   - `unknown -> END`
3. Stub node updates `current_stage` and trace routing signal.
4. `END`

## Contracts

## `EntryIntent`
- `business_discovery_conversation`
- `workflow_build_request`
- `workflow_edit_request`
- `workflow_fix_request`
- `information_request`
- `unknown`

## `AgentStage`
- `commercial_agent`
- `consultant_agent`
- `product_manager_agent`
- `engineer_agent`
- `qa_agent`

## `EntryRouterDecision`
```json
{
  "entry_intent": "workflow_build_request",
  "target_stage": "product_manager_agent",
  "confidence": 0.84,
  "routing_signals": ["build_signals_detected"],
  "missing_user_inputs": []
}
```

## `MultiAgentGraphResult`
```json
{
  "user_query": "Create a workflow from webhook to Google Sheets",
  "entry_intent": "workflow_build_request",
  "target_stage": "product_manager_agent",
  "confidence": 0.84,
  "routing_signals": ["build_signals_detected", "entered_product_manager_agent"],
  "current_stage": "product_manager_agent",
  "missing_user_inputs": [],
  "qa_enabled": true,
  "needs_replan": false,
  "status": "stub_routed"
}
```

## Chat response shape (reasoning + langgraph)
`message.content` now returns a JSON routing envelope:
```json
{
  "entry_intent": "workflow_build_request",
  "target_stage": "product_manager_agent",
  "confidence": 0.84,
  "routing_signals": ["build_signals_detected", "entered_product_manager_agent"],
  "current_stage": "product_manager_agent",
  "missing_user_inputs": [],
  "status": "stub_routed"
}
```
