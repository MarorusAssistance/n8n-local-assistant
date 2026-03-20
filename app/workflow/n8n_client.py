from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from ..config import settings


class N8NClientError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class N8NClient:
    base_url: str = settings.N8N_BASE_URL
    api_key: Optional[str] = settings.N8N_API_KEY
    timeout_seconds: float = settings.N8N_TIMEOUT_SECONDS
    endpoint_template: str = settings.N8N_WORKFLOW_ENDPOINT_TEMPLATE

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            # n8n commonly uses X-N8N-API-KEY but we also set Bearer for flexibility.
            headers["X-N8N-API-KEY"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _endpoint(self, workflow_id: str) -> str:
        try:
            return self.endpoint_template.format(workflow_id=workflow_id)
        except Exception as exc:  # pragma: no cover - defensive
            raise N8NClientError("Invalid N8N_WORKFLOW_ENDPOINT_TEMPLATE") from exc

    def _workflows_endpoint(self) -> str:
        token = "{workflow_id}"
        if token in self.endpoint_template:
            prefix = self.endpoint_template.split(token)[0]
            return prefix.rstrip("/")
        raise N8NClientError("Invalid N8N_WORKFLOW_ENDPOINT_TEMPLATE")

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            ) as client:
                response = client.request(method, endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise N8NClientError("Timeout contacting n8n", status_code=504) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise N8NClientError(
                f"n8n returned HTTP {status}", status_code=status
            ) from exc
        except Exception as exc:
            raise N8NClientError("Failed to contact n8n") from exc

        if not isinstance(data, dict):
            raise N8NClientError("Unexpected n8n response format")
        return data

    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        return self._request("GET", self._endpoint(workflow_id))

    def create_workflow(self, workflow_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", self._workflows_endpoint(), payload=workflow_payload)

    def update_workflow(self, workflow_id: str, workflow_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", self._endpoint(workflow_id), payload=workflow_payload)
