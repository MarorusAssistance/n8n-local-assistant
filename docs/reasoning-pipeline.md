# Reasoning Pipeline (Plan-Only)

## Overview
This pipeline adds a plan-only reasoning flow for docs-only chat mode:

1. `Router` (`create|fix|extend`)
2. `ContextPack` builder (compact retrieval context)
3. `Planner` (one LLM call that returns `PlanSpec`)
4. `Checker` (deterministic validation)
5. Optional one-shot `Planner` revision if checker has errors

The pipeline is local-runtime only and does not generate full n8n workflow JSON yet.

## Runtime behavior
- `REASONING_PIPELINE_ENABLED=false`: current legacy docs-only flow is unchanged.
- `REASONING_PIPELINE_ENABLED=true`: docs-only responses return plan JSON.
- Workflow-aware `/wf` flow remains unchanged.

## Data contracts

### RouterOutput
```json
{
  "intent": "create",
  "goal": "Create a workflow from webhook to Google Sheets",
  "constraints": {
    "services": ["Google Sheets"],
    "inputs": ["webhook payload"],
    "outputs": ["google sheets row(s)"],
    "nonFunctional": []
  },
  "missing_info": [],
  "complexity_score": 2
}
```

### ContextPack
```json
{
  "nodeCards": [
    {
      "type": "n8n-nodes-base.webhook",
      "displayName": "Webhook",
      "purpose": "Node overview and capabilities.",
      "requiredCredentials": [],
      "keyParams": []
    }
  ],
  "docChunks": [
    {
      "id": "doc-1",
      "title": "Webhook docs",
      "text": "Relevant excerpt...",
      "source": "docs"
    }
  ],
  "budget": {
    "maxNodeCards": 10,
    "maxDocChunks": 6,
    "maxContextTokens": 2500,
    "estimatedTokens": 720
  }
}
```

### PlanSpec
```json
{
  "summary": "Receive webhook and append payload to Google Sheets.",
  "steps": [
    {
      "id": "s1",
      "nodeType": "n8n-nodes-base.webhook",
      "purpose": "Trigger the workflow on inbound HTTP request.",
      "inputs": [],
      "outputs": ["json payload"],
      "credentialsNeeded": [],
      "notes": "Define path and method."
    },
    {
      "id": "s2",
      "nodeType": "n8n-nodes-base.googleSheets",
      "purpose": "Write webhook body into a sheet row.",
      "inputs": ["json payload"],
      "outputs": ["sheet row"],
      "credentialsNeeded": ["googleSheetsOAuth2Api"],
      "notes": "Map JSON fields to columns."
    }
  ],
  "dataFlowNotes": [
    "Webhook body is mapped directly into Google Sheets columns."
  ],
  "questionsForUser": []
}
```

### CheckerResult
```json
{
  "ok": true,
  "issues": []
}
```

If checker still has errors after one revision attempt, response content is:
```json
{
  "plan": { "summary": "...", "steps": [], "dataFlowNotes": [], "questionsForUser": ["..."] },
  "checker": { "ok": false, "issues": [{ "severity": "error", "code": "...", "msg": "..." }] }
}
```

## Environment flags
- `REASONING_PIPELINE_ENABLED` (default: `false`)
- `ROUTER_USE_LLM` (default: `false`)
- `MAX_NODE_CARDS` (default: `10`)
- `MAX_DOC_CHUNKS` (default: `6`)
- `MAX_CONTEXT_TOKENS` (default: `2500`)

`MAX_CONTEXT_CHARS` still applies to legacy RAG prompt assembly and is not used for `ContextPack` budget trimming.

## Logging
`n8n-assistant.trace` logs:
- intent
- node/doc counts
- `maxContextTokens` and `estimatedTokens`
- whether second iteration was used
- final checker issue counts
