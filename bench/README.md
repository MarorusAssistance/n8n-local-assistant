# Phase 1 Quality Harness (n8n JSON estricto)

Harness opt-in para evaluar la salida del asistente (`/v1/chat/completions`) contra checks deterministas de workflow n8n importable y comparar experimentos por metricas y logs.

## Archivos de entrada

- `bench/cases.yaml`: casos de benchmark.
- `bench/experiments.yaml`: variantes de configuracion con `env_overrides`.

Formato base de `cases.yaml`:

```yaml
- id: caso_unico
  user_message: "Prompt del usuario"
  json_only: true
  requirements:
    - type: must_include_node_type
      value: n8n-nodes-base.webhook
  limits:
    max_nodes: 30
```

`requirements.type` soportados:

- `must_include_node_type`
- `must_include_keyword_in_node_params`
- `must_have_schedule_daily_at`
- `must_have_connection`

Formato base de `experiments.yaml`:

```yaml
- name: baseline
  env_overrides: {}
  warmup: 1
  repetitions: 3
  adapter: direct
  json_only: false
```

## Ejecucion

Full:

```bash
python -m bench run --cases bench/cases.yaml --experiments bench/experiments.yaml --out bench/results
```

Quick (subset + 1 repetition):

```bash
python -m bench run --quick
```

Regenerar reportes:

```bash
python -m bench report --in bench/results/<timestamp>_<gitsha>
```

## Env overrides sin tocar `.env`

- Cada experimento se ejecuta en subprocess aislado.
- `env_overrides` se aplica solo al proceso del experimento.
- El `.env` real no se modifica.
- Para trazas, el runner aplica defaults por experimento (si no estan en overrides):
  - `TRACE_LOG_ENABLED=true`
  - `TRACE_LOG_LEVEL=INFO`
  - `TRACE_LOG_FILE=.../logs/raw/<experiment>.log`
  - `TRACE_LOG_MAX_BYTES=200000000`
  - `TRACE_LOG_BACKUPS=1`
  - `RETRIEVAL_DEBUG=true`

## Artifacts de salida

Cada run crea `results/<timestamp>_<gitsha>/` con:

- `runs.jsonl`
- `summary.csv`
- `report.html`
- `logs/report.html`
- `workflows/<experiment>/<case_id>/<rep>.json` (si parsea)
- `responses/<experiment>/<case_id>/<rep>.txt`
- `checks/<experiment>/<case_id>/<rep>.json`
- `diffs/<case_id>/<baseline>__vs__<experiment>__rep1.html`
- `logs/raw/<experiment>.log`
- `logs/slices/<experiment>/<case_id>/<rep>.log`
- `logs/events/<experiment>/<case_id>/<rep>.json`
- `logs/prompts/<experiment>/<case_id>/<rep>.json`
- `logs/html/<experiment>/<case_id>/<rep>.html`
- `logs/diffs/<case_id>/<baseline>__vs__<experiment>__rep1_prompts.html`

## Compliance score

Escala `0..100`:

- `json_parse_ok`: 20
- `workflow_min_schema`: 20
- `node_types_exist`: 20
- `credentials_shape_and_existence`: 15
- `credential_compatibility`: 10
- `requirements_and_limits`: 15

Si `json_parse_ok=false`, score final `0` y resto de checks marcado como `skipped_due_parse_failure`.

## Logs: que se captura

- Prompt/response del request (`TRACE REQUEST` / `TRACE RESPONSE`).
- Eventos estructurados `TRACE EVENT` para llamadas LLM por etapa (`llm_prompt`, `llm_output`).
- Retrieval final estructurado (`retrieval_final`) con chunks y linked defs.
- Eventos legacy de budget/fallback/reasoning para timeline.
- Redaccion ligera en artifacts de bench (tokens/secrets comunes enmascarados).

## Interpretacion rapida

- `report.html`: comparativa de calidad/latencia + links a artifacts.
- `logs/report.html`: comparativa de observabilidad (llm_calls, prompt_chars, retrieval_chunks, budget trims, etc.).
- `logs/html/...`: vista por run tipo pagina dedicada con:
  - prompts agrupados por `stage` y rol (`system/user/assistant`)
  - retrieval pre-rerank y post-rerank
  - linked defs de `nodes.json` y `credentials.json` separadas
  - solo `score + content` para evitar ruido visual.

## Nota importante

Con la configuracion actual del repo, el camino docs-only puede devolver `PlanSpec` en vez de workflow n8n importable. En contrato estricto, eso baja la compliance salvo experimentos que cambien runtime/prompt.
