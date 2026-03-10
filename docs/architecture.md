# API Architecture (Vertical Slice + LangGraph Runtime)

## Current target model
The backend uses a modular vertical-slice architecture:

- `app/api`: HTTP layer (FastAPI routers + request/response mapping)
- `app/features/chat`: application boundary for chat use case
- `app/features/reasoning`: reasoning slice exports/contracts
- `app/features/workflow`: workflow slice exports/contracts
- `app/features/retrieval`: retrieval slice exports/contracts
- `app/infrastructure`: adapters for LLM and DB health
- `app/graphs`: orchestration layer (LangGraph)
- `app/core`: shared errors and cross-cutting primitives

## Runtime orchestration
- Chat mode selection is handled by `ChatModePolicy` in `features/chat`.
- Reasoning and workflow execution are routed through `MasterGraphRuntime`.
- Reasoning runtime in LangGraph now uses a multi-agent entry graph:
  - `entry_router` (structured intent classification)
  - `commercial_agent` with discovery + value-first use-case prioritization
  - `product_manager_agent` with architecture planning and strict evidence-backed required node selection from retrieved API docs
  - stub stages: `consultant_agent`, `engineer_agent`, `qa_agent`
  - terminal `unknown` route without forced stage.
- Legacy reasoning fallback remains only for emergency mode when
  `LANGGRAPH_EMERGENCY_LEGACY_FALLBACK=true`.

## Public API compatibility
No endpoint contracts were changed:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`
- `GET /v1/chats`
- `GET /v1/chats/{conversation_id}`

## Key internal contracts
- `ChatUseCaseInput`
- `ChatUseCaseResult`
- `ChatRuntimeMode`
- `ChatModePolicyResult`
- `AppError` (`input`, `retrieval`, `llm`, `upstream`, `internal`)

## Migration principles
1. Keep external API stable.
2. Move mode routing and orchestration decisions out of transport layer.
3. Keep LangGraph as primary runtime.
4. Keep emergency fallback explicit and disabled by default.
