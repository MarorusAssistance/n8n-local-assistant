from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str = Field(..., description="Postgres connection string")

    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY: str = "dummy"
    LMSTUDIO_TIMEOUT_SECONDS: float = 30.0

    LLM_MODEL: Optional[str] = None
    EMBEDDING_MODEL: str = Field(..., description="Must match the index embedding model")

    DOCS_SOURCE_FILTER: Optional[str] = None
    TOP_K: int = 10
    MAX_CONTEXT_CHARS: int = 8000
    ENABLE_RERANK: bool = False
    RERANK_MODEL: str = "BAAI/bge-reranker-v2.5-gemma2-lightweight"
    RERANK_FALLBACK_MODEL: Optional[str] = "BAAI/bge-reranker-v2-m3"
    RERANK_POOL_SIZE: int = 30
    RERANK_TOP_K: int = 8
    RERANK_MAX_CHARS: int = 2000
    RERANK_USE_FP16: bool = True
    RERANK_BATCH_SIZE: Optional[int] = None
    RERANK_DEVICE: Optional[str] = None
    RERANK_CUTOFF_LAYERS: Optional[str] = None
    RERANK_COMPRESS_LAYERS: Optional[str] = None
    RERANK_COMPRESS_RATIO: Optional[int] = None
    RETRIEVAL_DEBUG: bool = False

    LOG_LEVEL: str = "INFO"
    TRACE_LOG_ENABLED: bool = False
    TRACE_LOG_LEVEL: str = "INFO"
    TRACE_MAX_TEXT_CHARS: int = 4000
    TRACE_MAX_CHUNK_CHARS: int = 320
    TRACE_MAX_CHUNKS: int = 12
    TRACE_LOG_FILE: str = "logs/trace.log"
    TRACE_LOG_MAX_BYTES: int = 5_000_000
    TRACE_LOG_BACKUPS: int = 3

    @field_validator("LOG_LEVEL", "TRACE_LOG_LEVEL", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        if value is None:
            return "INFO"
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized:
                return normalized
        return "INFO"
    APP_ENV: str = "dev"
    APP_SHOW_REFERENCES: Optional[bool] = None
    DEFAULT_TEMPERATURE: float = 0.1
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_FREQUENCY_PENALTY: float = 0.0
    DEFAULT_PRESENCE_PENALTY: float = 0.0

    N8N_BASE_URL: str = "http://localhost:5678"
    N8N_API_KEY: Optional[str] = None
    N8N_TIMEOUT_SECONDS: float = 15.0
    N8N_WORKFLOW_ENDPOINT_TEMPLATE: str = "/api/v1/workflows/{workflow_id}"

    WORKFLOW_SUMMARY_MAX_NODES: int = 200
    WORKFLOW_MICRO_MAX_CHARS: int = 1200
    WORKFLOW_NEIGHBOR_DEPTH: int = 1
    WORKFLOW_MAX_NODES_NODE_SPECIFIC: int = 6
    WORKFLOW_MAX_NODES_SUBGRAPH: int = 12
    WORKFLOW_MAX_NODES_GLOBAL: int = 15
    WORKFLOW_DOCS_CANDIDATE_K: int = 16
    WORKFLOW_DOCS_TOP_K: int = 8

    MEMORY_ENABLED: bool = True
    MEMORY_MAX_MESSAGES: int = 20
    CONVERSATION_MAX_TOKENS: int = 16000
    MEMORY_TTL_SECONDS: int = 3600
    MEMORY_HEADER: str = "x-conversation-id"
    MEMORY_AUTO_CREATE_CONVERSATION_ID: bool = False
    MEMORY_SKIP_PREFIXES: str = "### Task"
    MEMORY_SKIP_ASSISTANT_JSON_KEYS: str = "follow_ups,title,tags"
    MEMORY_DEDUP_WINDOW_SECONDS: int = 8
    MEMORY_BACKEND: str = "postgres"
    MEMORY_CONVERSATIONS_TABLE: str = "chat_conversations"
    MEMORY_MESSAGES_TABLE: str = "chat_messages"

    TABLE_NAME: str = "docs_chunks"
    TEXT_COLUMN: str = "chunk_text"
    EMBEDDING_COLUMN: str = "embedding"
    URL_COLUMN: str = "url"
    TITLE_COLUMN: Optional[str] = None
    SECTION_COLUMN: Optional[str] = None
    SOURCE_COLUMN: Optional[str] = "source"
    METADATA_COLUMN: Optional[str] = None
    URL_JSON_PATH: Optional[str] = None
    TITLE_JSON_PATH: Optional[str] = None
    SECTION_JSON_PATH: Optional[str] = None
    SOURCE_JSON_PATH: Optional[str] = None

    DISTANCE_OP: str = "<=>"

    @field_validator(
        "DOCS_SOURCE_FILTER",
        "TITLE_COLUMN",
        "SECTION_COLUMN",
        "SOURCE_COLUMN",
        "METADATA_COLUMN",
        "URL_JSON_PATH",
        "TITLE_JSON_PATH",
        "SECTION_JSON_PATH",
        "SOURCE_JSON_PATH",
        "LLM_MODEL",
        "N8N_API_KEY",
        "APP_ENV",
        "RERANK_FALLBACK_MODEL",
        "RERANK_DEVICE",
        "RERANK_CUTOFF_LAYERS",
        "RERANK_COMPRESS_LAYERS",
        "RERANK_BATCH_SIZE",
        "RERANK_COMPRESS_RATIO",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("APP_ENV", mode="after")
    @classmethod
    def _validate_app_env(cls, value: Optional[str]) -> str:
        env = (value or "dev").strip().lower()
        allowed = {"dev", "qa", "prod"}
        if env not in allowed:
            raise ValueError(f"APP_ENV must be one of {sorted(allowed)}")
        return env

    @field_validator("DISTANCE_OP")
    @classmethod
    def _validate_distance_op(cls, value: str) -> str:
        allowed = {"<=>", "<->", "<#>"}
        if value not in allowed:
            raise ValueError(f"DISTANCE_OP must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def _validate_json_mapping(self) -> "Settings":
        uses_json_path = any(
            [
                self.URL_JSON_PATH,
                self.TITLE_JSON_PATH,
                self.SECTION_JSON_PATH,
                self.SOURCE_JSON_PATH,
            ]
        )
        if uses_json_path and not self.METADATA_COLUMN:
            raise ValueError("METADATA_COLUMN is required when using *_JSON_PATH")
        if self.TITLE_JSON_PATH and not self.TITLE_COLUMN:
            self.TITLE_COLUMN = "title"
        if self.SECTION_JSON_PATH and not self.SECTION_COLUMN:
            self.SECTION_COLUMN = "section"
        return self

    def references_enabled(self) -> bool:
        if self.APP_SHOW_REFERENCES is not None:
            return bool(self.APP_SHOW_REFERENCES)
        return self.APP_ENV in {"dev", "qa"}


settings = Settings()
