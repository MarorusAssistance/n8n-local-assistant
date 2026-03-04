from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from .config import settings

try:  # Optional at runtime until deps are installed.
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency fallback
    ChatOpenAI = None  # type: ignore[assignment]
    OpenAIEmbeddings = None  # type: ignore[assignment]

_client: Optional[OpenAI] = None
_http_client: Optional[httpx.Client] = None
_lc_chat_models: Dict[Tuple[str, float], Any] = {}
_lc_embeddings: Optional[Any] = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.LMSTUDIO_BASE_URL,
            api_key=settings.LMSTUDIO_API_KEY,
            timeout=settings.LMSTUDIO_TIMEOUT_SECONDS,
        )
    return _client


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        headers = {}
        if settings.LMSTUDIO_API_KEY:
            headers["Authorization"] = f"Bearer {settings.LMSTUDIO_API_KEY}"
        _http_client = httpx.Client(
            base_url=settings.LMSTUDIO_BASE_URL,
            headers=headers,
            timeout=settings.LMSTUDIO_TIMEOUT_SECONDS,
        )
    return _http_client


def list_models() -> Optional[Dict[str, Any]]:
    try:
        http_client = _get_http_client()
        response = http_client.get("models")
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def resolve_model(requested: Optional[str]) -> str:
    data = list_models()
    model_ids: List[str] = []
    if data and isinstance(data.get("data"), list):
        model_ids = [item.get("id") for item in data["data"] if item.get("id")]

    if requested and (not model_ids or requested in model_ids):
        return requested
    if settings.LLM_MODEL and (not model_ids or settings.LLM_MODEL in model_ids):
        return settings.LLM_MODEL
    if model_ids:
        return model_ids[0]
    return requested or settings.LLM_MODEL or "local-model"


def get_langchain_chat_model(model: Optional[str] = None, temperature: float = 0.0) -> Optional[Any]:
    """Return a cached ChatOpenAI model configured for LM Studio compatibility."""
    if ChatOpenAI is None:
        return None

    resolved_model = resolve_model(model)
    key = (resolved_model, float(temperature))
    cached = _lc_chat_models.get(key)
    if cached is not None:
        return cached

    chat_model = ChatOpenAI(
        model=resolved_model,
        api_key=settings.LMSTUDIO_API_KEY,
        base_url=settings.LMSTUDIO_BASE_URL,
        timeout=settings.LMSTUDIO_TIMEOUT_SECONDS,
        temperature=temperature,
    )
    _lc_chat_models[key] = chat_model
    return chat_model


def get_langchain_embeddings() -> Optional[Any]:
    """Return a cached OpenAIEmbeddings model configured for LM Studio compatibility."""
    global _lc_embeddings
    if OpenAIEmbeddings is None:
        return None
    if _lc_embeddings is None:
        _lc_embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.LMSTUDIO_API_KEY,
            base_url=settings.LMSTUDIO_BASE_URL,
            request_timeout=settings.LMSTUDIO_TIMEOUT_SECONDS,
        )
    return _lc_embeddings


def create_embedding(text: str) -> List[float]:
    embeddings = get_langchain_embeddings()
    if embeddings is not None:
        try:
            return embeddings.embed_query(text)
        except Exception:
            # Fallback to OpenAI client for compatibility with local providers.
            pass

    client = _get_openai_client()
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    **kwargs: Any,
) -> Any:
    """OpenAI-compatible completion used by legacy response formatting/streaming paths."""
    client = _get_openai_client()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
