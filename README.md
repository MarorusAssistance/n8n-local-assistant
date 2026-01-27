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

## Notes
- `EMBEDDING_MODEL` must match the model used to index your docs.
- If LM Studio does not support embeddings for that model, the API will return a clear error.
- `DISTANCE_OP` defaults to `<=>` (cosine distance). Change to `<->` if you use L2.
- Workflow node selection currently uses lightweight heuristics (name/type/signals). A future improvement is to replace or augment this with embeddings or a more robust semantic matcher.
