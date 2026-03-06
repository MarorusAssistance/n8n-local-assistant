# Multi-Agent Milestone 2

## Scope
Milestone 2 adds real behavior to the `commercial_agent` path only.

Included:
- Extract business automation use-case candidates from discovery input.
- Rank candidates with business-value-first policy.
- Select exactly one primary use case.
- Preserve non-selected candidates as alternatives.
- End with handoff-ready state (`target_stage=product_manager_agent`) when selection exists.

Out of scope:
- Template retrieval (Notion or any external source).
- Technical architecture planning (PM/Engineer/QA).
- Workflow JSON generation.
- API-docs or credentials retrieval.

## Commercial output policy
The agent updates state with:
- `business_context_summary`
- `discovered_use_cases`
- `selected_use_case`
- `alternative_use_cases`
- `selection_reason`

If no actionable candidate is found:
- `selected_use_case = null`
- `alternative_use_cases = []`
- `selection_reason` explains why no actionable case was extracted.

## Ranking formula
Deterministic score:
```text
priority_score = ((value*0.55) + (clarity*0.20) + (feasibility*0.15) + (actionability*0.10)) * 20
```

Sort order:
1. `priority_score` desc
2. `value_score` desc
3. `clarity_score` desc
4. original extraction order

## Chat response addition (reasoning-langgraph)
`message.content` keeps the existing routing envelope and now can include:
- `business_context_summary`
- `discovered_use_cases`
- `selected_use_case`
- `alternative_use_cases`
- `selection_reason`

## Example payload fragment
```json
{
  "entry_intent": "business_discovery_conversation",
  "target_stage": "product_manager_agent",
  "current_stage": "commercial_agent",
  "selected_use_case": {
    "id": "uc_1",
    "title": "Support escalation automation",
    "priority_score": 88.0
  },
  "alternative_use_cases": [],
  "selection_reason": "Selected 'Support escalation automation' as primary because it has the highest value-first priority score (88.0) and clear business outcome.",
  "status": "stub_routed"
}
```
