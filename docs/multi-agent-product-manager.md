# Multi-Agent Product Manager Stage

## Scope
This stage implements real behavior for `product_manager_agent` and keeps the rest of the reasoning runtime compatible with earlier stage contracts.

Included:
- Convert `selected_use_case` into a typed `ArchitecturePlan`.
- Retrieve API docs once per PM execution and derive `required_nodes` from explicit retrieval evidence only.
- Enforce strict evidence policy: no inferred node types when explicit node evidence is missing.
- Produce typed `workflow_context` for future engineer handoff.
- Keep response payload additive in reasoning LangGraph mode.

Out of scope:
- Workflow JSON generation.
- Node parameter configuration.
- Credentials value completion.
- Template retrieval or Notion integrations.
- Engineer/QA execution logic.

## Evidence policy for required nodes
`required_nodes` are extracted only from explicit node evidence in retrieved chunks:
- `context_kind=linked_def` and `linked_def_type=node`, or
- chunk metadata with `nodeType`, or
- chunk metadata `kind` starting with `NODE_`.

If explicit node evidence is missing:
- PM does not infer nodes from semantics.
- PM returns no actionable architecture plan.
- `missing_user_inputs` is populated with a focused clarification request.

## Product manager outputs
The PM node updates:
- `architecture_plan`
- `workflow_context`
- `planning_summary`
- `missing_user_inputs`
- `target_stage` (`engineer_agent` only when planning is actionable)

## Chat response additions (reasoning + langgraph)
`message.content` preserves existing envelope keys and now may include:
- `architecture_plan`
- `workflow_context`
- `planning_summary`

## Example payload fragment
```json
{
  "entry_intent": "business_discovery_conversation",
  "current_stage": "product_manager_agent",
  "target_stage": "engineer_agent",
  "architecture_plan": {
    "use_case_id": "uc_1",
    "required_nodes": [
      {
        "node_type": "n8n-nodes-base.webhook",
        "evidence_confidence": 0.87,
        "rerank_confidence": 0.74,
        "blended_confidence": 0.84
      }
    ]
  },
  "workflow_context": {
    "planning_ready": true,
    "required_node_types": ["n8n-nodes-base.webhook"]
  },
  "status": "stub_routed"
}
```
