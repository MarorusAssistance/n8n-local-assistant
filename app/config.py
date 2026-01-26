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

    LOG_LEVEL: str = "INFO"

    MEMORY_ENABLED: bool = True
    MEMORY_MAX_MESSAGES: int = 20
    MEMORY_TTL_SECONDS: int = 3600
    MEMORY_HEADER: str = "x-conversation-id"
    MEMORY_AUTO_CREATE_CONVERSATION_ID: bool = False
    MEMORY_SKIP_PREFIXES: str = "### Task"
    MEMORY_SKIP_ASSISTANT_JSON_KEYS: str = "follow_ups,title,tags"
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
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and not value.strip():
            return None
        return value

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


settings = Settings()
