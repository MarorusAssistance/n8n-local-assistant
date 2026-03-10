# Phase 1 Quality Harness (Router + Commercial + Product Manager)

Harness opt-in para evaluar end-to-end `POST /v1/chat/completions` con contrato JSON multi-agent (router/commercial/product_manager), comparar experimentos por `env_overrides` sin tocar `.env`, y puntuar compliance determinista.

## Ejecutar

Full:

```bash
python -m bench run --cases bench/cases.yaml --experiments bench/experiments.yaml --out bench/results
```

Quick (primeros `N` casos, `repetitions=1`, `warmup=0`):

```bash
python -m bench run --quick
```

Regenerar reportes:

```bash
python -m bench report --in bench/results/<timestamp>_<gitsha>
```

## Casos (`bench/cases.yaml`)

Formato:

```yaml
- id: router_fix_beats_edit
  stage: router|commercial|product_manager
  user_message: "prompt"
  json_only: true
  requirements:
    - type: must_equal_field
      value: { path: entry_intent, equals: workflow_fix_request }
  limits:
    max_missing_user_inputs: 3
```

`requirements.type` soportados:

- `must_equal_field`
- `must_not_equal_field`
- `must_be_null_field`
- `must_not_be_null_field`
- `must_include_routing_signal`
- `must_not_include_routing_signal`
- `must_have_required_nodes_min`
- `must_have_required_nodes_with_evidence`
- `must_not_include_workflow_json_keys`
- `must_have_planning_ready`
- `must_have_handoff_target`

`limits` soportados:

- `max_missing_user_inputs`

## Experimentos (`bench/experiments.yaml`)

Cada experimento corre aislado en subprocess:

```yaml
- name: baseline_langgraph
  env_overrides:
    AGENT_RUNTIME: langgraph
    REASONING_PIPELINE_ENABLED: "true"
  warmup: 1
  repetitions: 3
  adapter: direct
  json_only: true
```

No hace falta editar `.env`: `env_overrides` solo aplica al subprocess del experimento.

## Scoring (0..100)

Secciones:

- `json_parse_ok` (15)
- `router_routing_graph` (20)
- `commercial_selection` (20)
- `product_manager_planning` (20)
- `planning_safety_guardrails` (10)
- `retrieval_trace_checks` (15)

Si `json_parse_ok=false`, score final `0` (short-circuit).

## Artifacts

En `bench/results/<run_id>/`:

- `runs.jsonl`
- `summary.csv`
- `stage_metrics.csv`
- `report.html`
- `logs/report.html`
- `responses/<experiment>/<case>/<rep>.txt`
- `workflows/<experiment>/<case>/<rep>.json` (payload parseado del contrato)
- `checks/<experiment>/<case>/<rep>.json`
- `diffs/...` y `logs/diffs/...`
- `logs/raw/<experiment>.log`
- `logs/slices/...`
- `logs/events/...`
- `logs/prompts/...`
- `logs/html/...`

## Lectura rapida de resultados

- `summary.csv`: error rate, latencia p50/p95, compliance p50/p95 por experimento.
- `stage_metrics.csv`: metricas de router/commercial/product_manager y parse rate transversal.
- `report.html`: comparacion por experimento y por caso con links a artifacts.
- `logs/report.html`: metricas derivadas de trazas y links a logs por run.
- `logs/html/...`: vista de prompts por etapa y retrieval pre/post-rerank.
