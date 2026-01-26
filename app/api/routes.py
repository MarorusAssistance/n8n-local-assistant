from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query, Request, Response

from ..memory import get_memory_store
from ..schemas import ChatCompletionRequest
from ..services import ChatService


router = APIRouter()
chat_service = ChatService(get_memory_store())


@router.get("/health")
def health() -> Dict[str, Any]:
    return chat_service.health()


@router.get("/v1/models")
def get_models() -> Dict[str, Any]:
    return chat_service.list_models()


@router.post("/v1/chat/completions")
def create_chat_completions(
    request: ChatCompletionRequest, http_request: Request, response: Response
) -> Any:
    return chat_service.create_chat_completions(request, http_request, response)


@router.get("/v1/chats")
def list_chats(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    return chat_service.list_chats(limit, offset)


@router.get("/v1/chats/{conversation_id}")
def get_chat(conversation_id: str) -> Dict[str, Any]:
    return chat_service.get_chat(conversation_id)
