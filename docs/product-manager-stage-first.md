# Product Manager Agent: Stage-First Planning

## Scope
The Product Manager agent now runs a stage-first planning pipeline:

1. Normalize input from either:
   - `selected_use_case` (commercial path), or
   - direct `workflow_build_request` query (router path).
2. Generate a typed stage plan (`PMStagePlan[]`).
3. For each stage, run adaptive retrieval/selection:
   - stage-specific docs query,
   - node evidence extraction from explicit API-doc signals,
   - candidate summary,
   - stage node selection,
   - hybrid acceptance gate.
4. If a stage cannot be accepted, request user clarification and pause.
5. Resume from saved PM state on the next user turn.
6. Build legacy bridge outputs (`architecture_plan`, `required_nodes`, `proposed_nodes`) for Engineer compatibility.

## PM state fields
New PM state fields in `MultiAgentGraphState`:

- `pm_status`
- `pm_stage_plan`
- `pm_stage_selections`
- `pm_stage_progress`
- `pm_clarification_state`
- `pm_stage_search_history`
- `pm_reasoning_trace_full` (internal trace only; not exposed in normal chat payload)

## Acceptance gate
Default gate policy (configurable via env):

- `PM_FIT_SCORE_THRESHOLD=0.70`
- `PM_RERANK_THRESHOLD=0.55`
- `PM_TOP_MARGIN_THRESHOLD=0.10`

A stage passes when:

- `pm_fit_score >= PM_FIT_SCORE_THRESHOLD`, and
- (`rerank_confidence >= PM_RERANK_THRESHOLD` or `top_margin >= PM_TOP_MARGIN_THRESHOLD`)

## Clarification and resume
- Max retrieval passes per stage: `PM_MAX_STAGE_RETRIEVAL_PASSES` (default `3`).
- Max clarification rounds: `PM_MAX_USER_CLARIFICATIONS` (default `2`).
- If blocked, PM returns `pm_blocked_waiting_user` with actionable question(s).
- On next turn (same conversation/thread), PM consumes user input and resumes from saved state.

## Output policy
- Public reasoning payload includes PM stage summaries (`pm_status`, stage plan/selections/progress/clarifications).
- Full PM reasoning trace remains internal (`pm_reasoning_trace_full`).
- Engineer bridge remains active through:
  - `architecture_plan`
  - `workflow_context`
  - `proposed_nodes`
