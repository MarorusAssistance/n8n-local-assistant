# n8n Workflow Assistant (v1.1.0)

Local FastAPI backend that exposes an OpenAI-compatible API with RAG over your n8n docs in Postgres (pgvector). Designed to work with Open WebUI and a local LLM via LM Studio.

## Features
- OpenAI-compatible endpoints: `GET /v1/models`, `POST /v1/chat/completions`
- Streaming support via `stream=true` on chat completions
- RAG pipeline: embed question -> vector search -> prompt with context -> LLM response
- Workflow-awareness con selección por chat: `/wf <workflow_id>`
- Análisis por capas con scopes: `no_workflow`, `node_specific`, `subgraph`, `workflow_global`
- References appended at the end of each answer (solo en `dev`/`qa`)
- No secrets logged


## Reranking
Optional cross-encoder reranking to improve precision after retrieval.

How it works:
- Retrieval builds a larger pool (`RERANK_POOL_SIZE`).
- Reranker sorts candidates by relevance.
- Only the top K (`RERANK_TOP_K`) are sent to the LLM.

Relevant .env knobs:
- `ENABLE_RERANK=true|false`
- `RERANK_MODEL` (default: `BAAI/bge-reranker-v2.5-gemma2-lightweight`)
- `RERANK_FALLBACK_MODEL` (default: `BAAI/bge-reranker-v2-m3`)
- `RERANK_POOL_SIZE` (recommend 20-50)
- `RERANK_TOP_K` (recommend 6-10)
- `RERANK_MAX_CHARS` (truncate chunks before scoring)
- `RERANK_USE_FP16`
- `RERANK_BATCH_SIZE`
- `RERANK_DEVICE` (e.g. `cuda:0` or `cpu`)
- `RERANK_CUTOFF_LAYERS`, `RERANK_COMPRESS_LAYERS`, `RERANK_COMPRESS_RATIO`
- `RETRIEVAL_DEBUG=true` to log before/after rerank lists

Notes:
- For reranking to have impact, `RERANK_POOL_SIZE` must be > `RERANK_TOP_K`.
- If the lightweight model is incompatible with your `transformers` build, the code falls back to `RERANK_FALLBACK_MODEL` and logs a warning.

## Hybrid Retrieval
The backend can run hybrid retrieval (vector + Postgres FTS) before reranking.

Pipeline when `ENABLE_HYBRID=true`:
- Vector retrieval: top `N_VEC`
- FTS retrieval (strict): compact keyword query via `websearch_to_tsquery`
- FTS retrieval (relaxed fallback): OR query via `to_tsquery` if strict returns 0
- RRF fusion: `RRF_K`, `RRF_VECTOR_WEIGHT`, `RRF_FTS_WEIGHT`, output top `RRF_TOP_M`
- Existing reranker receives the merged pool and returns final top K

Relevant `.env` knobs:
- `ENABLE_HYBRID`
- `HYBRID_STRICT_MODE`
- `N_VEC`
- `N_FTS`
- `FTS_LANGUAGE`
- `FTS_TSVECTOR_COLUMN`
- `FTS_QUERY_MAX_CHARS`
- `FTS_KEYWORDS_MAX_TERMS`
- `FTS_KEYWORDS_MIN_TERM_LEN`
- `FTS_ENABLE_RELAXED_FALLBACK`
- `RRF_K`
- `RRF_TOP_M`
- `RRF_VECTOR_WEIGHT`
- `RRF_FTS_WEIGHT`
- `RETRIEVAL_DEBUG`

Failure behavior:
- If hybrid is enabled and FTS fails, default behavior is warning + fallback to vector-only.
- Set `HYBRID_STRICT_MODE=true` to fail fast instead of fallback.

## Requirements
- Python 3.11+
- Postgres with pgvector and your docs already indexed
- LM Studio running locally with OpenAI-compatible API

## Setup
1) Create a venv and install deps:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Configure environment:
```bash
copy .env.example .env
```
Edit `.env` to match your DB schema and LM Studio settings.

3) Run the API (standalone):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Run the API + Open WebUI together (PowerShell):
```bash
.\scripts\dev.ps1
```
If your Open WebUI lives elsewhere:
```bash
.\scripts\dev.ps1 -OpenWebUiDir "C:\Users\maror\Projects\openwebui"
```

## Open WebUI connection
- Base URL: `http://localhost:8000/v1`
- API key: any string (dummy is fine)

## Referencias (dev/qa vs prod)
El bloque "Contexto de documentacion (citado)" siempre se incluye en el prompt.
El bloque final "Referencias" ahora depende del entorno:
- `APP_ENV=prod`: se oculta el bloque "Referencias".
- `APP_ENV=dev` o `APP_ENV=qa`: se muestra "Referencias".
- Puedes forzarlo con `APP_SHOW_REFERENCES=true|false`.

## Antialucinaciones (parametros por defecto)
Para reducir invenciones del modelo, el backend usa estos defaults si el cliente no los manda:
- `DEFAULT_TEMPERATURE`
- `DEFAULT_TOP_P`
- `DEFAULT_FREQUENCY_PENALTY`
- `DEFAULT_PRESENCE_PENALTY`

Puedes ajustarlos en `.env`.

## Workflow-aware mode (`/wf`)
Puedes activar el modo “workflow-aware” desde el chat:

```text
/wf <workflow_id>
```

Comportamiento:
- Si no hay workflow activo, el asistente responde solo con RAG de documentación.
- Si hay workflow activo, el backend intenta obtener el workflow desde n8n local, construye un resumen compacto y decide el alcance (scope) antes de responder.
- Los mensajes `/wf ...` se usan como control y no se envían al modelo como parte del prompt.

Variables relevantes en `.env`:
- `N8N_BASE_URL`
- `N8N_API_KEY`
- `N8N_TIMEOUT_SECONDS`
- `N8N_WORKFLOW_ENDPOINT_TEMPLATE` (por defecto: `/api/v1/workflows/{workflow_id}`)

### Cómo se decide el scope
El planner decide uno de estos scopes:
- `no_workflow`: no hay workflow activo.
- `node_specific`: se detecta 1 nodo claro (o 1–2).
- `subgraph`: se detectan varios nodos relacionados o una zona del grafo.
- `workflow_global`: la pregunta es global (“mi flujo”, “qué falla”, “arquitectura”, “mejorar”).

Heurísticas actuales (simples, con hooks para mejorar luego):
- Matching por nombre de nodo (similaridad + contains).
- Matching por tipo de nodo con sinónimos comunes (HTTP Request, Webhook, Merge, IF, Code, DB).
- Señales: nodos con credenciales, expresiones o pivotes (IF/Merge/Switch) reciben más peso.

### Recortes y límites (token safety)
Nunca se inyecta el workflow JSON completo. Se usan:
- `workflow_summary` compacto (nodos + edges + señales mínimas).
- `micro_context` por nodo (nodo + vecinos cercanos).
- Top docs limitadas (`WORKFLOW_DOCS_TOP_K`).

Parámetros ajustables en `.env`:
- `WORKFLOW_SUMMARY_MAX_NODES`
- `WORKFLOW_MICRO_MAX_CHARS`
- `WORKFLOW_NEIGHBOR_DEPTH`
- `WORKFLOW_MAX_NODES_NODE_SPECIFIC`
- `WORKFLOW_MAX_NODES_SUBGRAPH`
- `WORKFLOW_MAX_NODES_GLOBAL`
- `WORKFLOW_DOCS_CANDIDATE_K`
- `WORKFLOW_DOCS_TOP_K`

## Conversation memory (server-side)
The API keeps a short history per conversation for context and can persist chats in Postgres.
- Enable/disable with `MEMORY_ENABLED`
- Limit context window with `MEMORY_MAX_MESSAGES`
- Hard prompt budget by estimated tokens with `CONVERSATION_MAX_TOKENS` (default: `16000`)
- `MEMORY_TTL_SECONDS` only applies to the in-memory backend (eviction)
- Provide a conversation key via `X-Conversation-Id` header, `conversation_id` in the request body, or `user`
- If your client doesn't send an ID, you can set `MEMORY_AUTO_CREATE_CONVERSATION_ID=true` to auto-generate one (creates a new chat per request unless the client reuses it)
- To avoid storing OpenWebUI task prompts, set `MEMORY_SKIP_PREFIXES` (default: `### Task`) and `MEMORY_SKIP_ASSISTANT_JSON_KEYS` (default: `follow_ups,title,tags`)
- To reduce duplicate messages when the UI sends multiple requests, set `MEMORY_DEDUP_WINDOW_SECONDS` (default: 8)
- Persist to Postgres with `MEMORY_BACKEND=postgres` (uses `chat_conversations` + `chat_messages`)
- Create tables with `scripts/chat_memory.sql`

## Chat history endpoints
- List chats: `GET /v1/chats`
- Get full history: `GET /v1/chats/{conversation_id}`

## Debug workflow endpoint
Use this to inspect how the assistant selects nodes/subgraphs and retrieves docs.
```bash
curl http://localhost:8000/v1/debug/workflow ^
  -H "Content-Type: application/json" ^
  -d "{\"workflow_id\":\"YOUR_ID\",\"question\":\"¿Qué hace este workflow?\"}"
```

## Quick test
```bash
curl http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"local-model\",\"messages\":[{\"role\":\"user\",\"content\":\"Como hago un workflow que escuche un webhook y escriba en Google Sheets?\"}]}"
```

## Health check
```bash
curl http://localhost:8000/health
```

## Database schema mapping
Defaults assume a table like:
- `docs_chunks`
  - `chunk_text` (text)
  - `embedding` (vector)
  - `url` or `path`
  - `title` and/or `section` (optional)
  - `source` (optional)

If your schema differs, map it via `.env`:
- `TABLE_NAME`
- `TEXT_COLUMN`
- `EMBEDDING_COLUMN`
- `URL_COLUMN`
- `TITLE_COLUMN` (optional)
- `SECTION_COLUMN` (optional)
- `SOURCE_COLUMN` (optional)
- `DOCS_SOURCE_FILTER` (optional)

If url/source/title live inside a JSON/JSONB column, configure:
- `METADATA_COLUMN`
- `URL_JSON_PATH` (dot notation, for example `metadata_docs.url`)
- `TITLE_JSON_PATH` (optional)
- `SECTION_JSON_PATH` (optional)
- `SOURCE_JSON_PATH` (optional)

Example JSON mapping:
```
METADATA_COLUMN=metadata
URL_JSON_PATH=metadata_docs.url
SOURCE_JSON_PATH=metadata_docs.source
```

Example for a table with `content` and `metadata` JSON:
```
TABLE_NAME=docs
TEXT_COLUMN=content
METADATA_COLUMN=metadata
URL_JSON_PATH=metadata_docs.url
SOURCE_JSON_PATH=metadata_docs.source
```

If you use the bundled indexer (`scripts/index_docs.py`), it stores metadata at the top level:
```
metadata: { source, url, title, section, ... }
```
In that case, set:
```
METADATA_COLUMN=metadata
URL_JSON_PATH=url
TITLE_JSON_PATH=title
SECTION_JSON_PATH=section
SOURCE_JSON_PATH=source
```

## Indexing markdown docs
This repo includes a local indexer that chunks Markdown by blocks (headings, lists, tables, code fences) and writes embeddings to Postgres.

Basic usage:
```bash
python scripts/index_docs.py --input-dir "C:\path\to\docs" --source n8n-docs
```

Options:
- `--base-url` for canonical links (default `https://docs.n8n.io`)
- `--docs-root` to trim a leading folder from URLs (default `docs`)
- `--purge-source` to delete existing rows for the source before indexing
- `--dry-run` to inspect block counts without embeddings/DB writes

## Indexing n8n definitions (nodes + credentials)
This repo includes a deterministic indexer for `nodes.json` and `credentials.json` that generates structured chunks with metadata and embeddings.

Dry run (no embeddings, no DB writes):
```bash
python scripts/index_n8n_definitions.py --dry-run
```

Full indexing (embeddings + upsert incremental in Postgres):
```bash
python scripts/index_n8n_definitions.py
```

Useful options:
- `--nodes-file` and `--credentials-file` to override input paths
- `--skip-nodes` or `--skip-credentials` to index only one source
- `--source-nodes` and `--source-credentials` to control source tags
- `--include-raw-json` to store per-chunk JSON payload in metadata
- `--nodes-limit` and `--credentials-limit` for quick test runs

## Linking docs -> defs
After docs + definitions are indexed in the same chunks table, run the linker to create page-level relations:
- `doc_page_node_link`
- `doc_page_credential_link`

Dry run:
```bash
python scripts/link_docs_defs.py --dry-run --docs-source n8n-docs --nodes-source n8n-nodes --credentials-source n8n-credentials
```

Apply changes (upsert + prune stale links for processed pages):
```bash
python scripts/link_docs_defs.py --docs-source n8n-docs --nodes-source n8n-nodes --credentials-source n8n-credentials
```

Useful options:
- `--doc-page-key` (repeatable) to process specific pages only
- `--mention-limit-per-page` to cap weak links (`mention`)
- `--similarity-limit-per-page` and `--similarity-threshold` for `similarity` fallback tuning

## Notes
- `EMBEDDING_MODEL` must match the model used to index your docs.
- If LM Studio does not support embeddings for that model, the API will return a clear error.
- `DISTANCE_OP` defaults to `<=>` (cosine distance). Change to `<->` if you use L2.
- Workflow node selection currently uses lightweight heuristics (name/type/signals). A future improvement is to replace or augment this with embeddings or a more robust semantic matcher.
