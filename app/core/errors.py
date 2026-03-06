from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class AppErrorCategory(str, Enum):
    input = "input"
    retrieval = "retrieval"
    llm = "llm"
    upstream = "upstream"
    internal = "internal"


class AppError(Exception):
    def __init__(
        self,
        category: AppErrorCategory,
        message: str,
        *,
        status_code: int = 500,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.status_code = int(status_code)
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "status_code": self.status_code,
            "context": self.context,
        }
