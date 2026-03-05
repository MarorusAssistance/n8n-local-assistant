from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx

from .models import AdapterType


JSON_ONLY_INSTRUCTION = (
    "Responde SOLO con JSON valido de workflow n8n importable, sin markdown ni texto adicional."
)


@dataclass(frozen=True)
class AdapterResponse:
    success: bool
    assistant_text: str
    http_status: Optional[int]
    error: Optional[str]


class PipelineAdapter:
    def __init__(
        self,
        *,
        adapter: AdapterType,
        timeout_seconds: float = 90.0,
        http_base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._adapter = adapter
        self._timeout = timeout_seconds
        self._http_base_url = http_base_url or os.getenv("BENCH_HTTP_BASE_URL")
        self._model = model or os.getenv("BENCH_MODEL", "local-model")
        self._test_client = None
        self._http_client = None

    def close(self) -> None:
        if self._test_client is not None:
            self._test_client.close()
            self._test_client = None
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def invoke(
        self,
        *,
        user_message: str,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AdapterResponse:
        conv_id = conversation_id or f"bench-{uuid4().hex}"
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [{"role": "user", "content": user_message}],
            "conversation_id": conv_id,
        }
        headers: Dict[str, str] = {}
        if request_id:
            headers["x-request-id"] = request_id

        if self._adapter == "direct":
            return self._invoke_direct(payload, headers=headers)
        if self._adapter == "http":
            return self._invoke_http(payload, headers=headers)
        return AdapterResponse(
            success=False,
            assistant_text="",
            http_status=None,
            error=f"unsupported adapter: {self._adapter}",
        )

    @staticmethod
    def build_user_message(*, user_message: str, json_only: bool) -> str:
        prompt = user_message.strip()
        if not json_only:
            return prompt
        return f"{prompt}\n\n{JSON_ONLY_INSTRUCTION}"

    def _invoke_direct(
        self,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> AdapterResponse:
        try:
            if self._test_client is None:
                from fastapi.testclient import TestClient

                from app.main import app

                self._test_client = TestClient(app)

            response = self._test_client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers or None,
            )
        except Exception as exc:
            return AdapterResponse(
                success=False,
                assistant_text="",
                http_status=None,
                error=f"direct adapter error: {str(exc)}",
            )
        return _to_adapter_response(response.status_code, response.text)

    def _invoke_http(
        self,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> AdapterResponse:
        if not self._http_base_url:
            return AdapterResponse(
                success=False,
                assistant_text="",
                http_status=None,
                error="http adapter requires BENCH_HTTP_BASE_URL or --http-base-url",
            )
        try:
            if self._http_client is None:
                self._http_client = httpx.Client(
                    base_url=self._http_base_url.rstrip("/"),
                    timeout=self._timeout,
                )
            response = self._http_client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers or None,
            )
        except Exception as exc:
            return AdapterResponse(
                success=False,
                assistant_text="",
                http_status=None,
                error=f"http adapter error: {str(exc)}",
            )
        return _to_adapter_response(response.status_code, response.text)


def _to_adapter_response(status_code: int, raw_text: str) -> AdapterResponse:
    if status_code != 200:
        return AdapterResponse(
            success=False,
            assistant_text=raw_text,
            http_status=status_code,
            error=f"chat endpoint returned HTTP {status_code}",
        )

    assistant_text = ""
    try:
        payload = json.loads(raw_text)
        assistant_text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception:
        assistant_text = raw_text

    return AdapterResponse(
        success=True,
        assistant_text=str(assistant_text or ""),
        http_status=status_code,
        error=None,
    )
