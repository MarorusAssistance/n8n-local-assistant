# n8n Workflow Assistant (v1)

Local FastAPI backend that exposes an OpenAI-compatible API with RAG over your n8n docs in Postgres (pgvector). Designed to work with Open WebUI and a local LLM via LM Studio.

## Features
- OpenAI-compatible endpoints: `GET /v1/models`, `POST /v1/chat/completions`
- Streaming support via `stream=true` on chat completions
- RAG pipeline: embed question -> vector search -> prompt with context -> LLM response
- References appended at the end of each answer (urls/paths)
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

## Conversation memory (server-side)
The API keeps a short history per conversation for context and can persist chats in Postgres.
- Enable/disable with `MEMORY_ENABLED`
- Limit context window with `MEMORY_MAX_MESSAGES`
- `MEMORY_TTL_SECONDS` only applies to the in-memory backend (eviction)
- Provide a conversation key via `X-Conversation-Id` header, `conversation_id` in the request body, or `user`
- If your client doesn't send an ID, you can set `MEMORY_AUTO_CREATE_CONVERSATION_ID=true` to auto-generate one (creates a new chat per request unless the client reuses it)
- To avoid storing OpenWebUI task prompts, set `MEMORY_SKIP_PREFIXES` (default: `### Task`) and `MEMORY_SKIP_ASSISTANT_JSON_KEYS` (default: `follow_ups,title,tags`)
- Persist to Postgres with `MEMORY_BACKEND=postgres` (uses `chat_conversations` + `chat_messages`)
- Create tables with `scripts/chat_memory.sql`

## Chat history endpoints
- List chats: `GET /v1/chats`
- Get full history: `GET /v1/chats/{conversation_id}`

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
