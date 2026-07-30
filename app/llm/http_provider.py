from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.llm.errors import LLMAuthenticationError, LLMProviderError

_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
Sleep = Callable[[float], Awaitable[None]]


class BaseHTTPProvider:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_seconds: float,
        max_retries: int,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 8.0,
        http_client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_base_delay_seconds < 0 or retry_max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")
        self._endpoint = endpoint
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        self._sleep = sleep
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            headers={"User-Agent": "company-research-agent/0.1"},
        )

    async def _post_json(
        self,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = await self._http_client.post(
                    self._endpoint,
                    headers=dict(headers),
                    json=dict(payload),
                )
                if response.status_code in _TRANSIENT_STATUS_CODES:
                    if attempt < self._max_retries:
                        await response.aread()
                        await self._sleep(self._retry_delay(attempt, response))
                        continue
                    raise LLMProviderError(
                        f"LLM provider remained unavailable (HTTP {response.status_code})",
                        retryable=True,
                    )
                if response.status_code in {401, 403}:
                    raise LLMAuthenticationError(
                        f"LLM provider rejected credentials (HTTP {response.status_code})"
                    )
                if response.is_error:
                    raise LLMProviderError(
                        f"LLM provider request failed (HTTP {response.status_code})"
                    )
                try:
                    body = response.json()
                except ValueError as exc:
                    raise LLMProviderError("LLM provider returned a non-JSON envelope") from exc
                if not isinstance(body, dict):
                    raise LLMProviderError("LLM provider returned an invalid JSON envelope")
                return body
            except LLMProviderError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if attempt >= self._max_retries:
                    raise LLMProviderError(
                        "LLM provider transport failed after retries", retryable=True
                    ) from exc
                await self._sleep(self._retry_delay(attempt, response))
        raise AssertionError("retry loop exited unexpectedly")

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None and (retry_after := response.headers.get("Retry-After")):
            if retry_after.isdigit():
                return min(float(retry_after), self._retry_max_delay_seconds)
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
                return min(seconds, self._retry_max_delay_seconds)
            except (TypeError, ValueError, OverflowError):
                pass
        exponential: float = self._retry_base_delay_seconds * (2.0**attempt)
        return float(min(exponential, self._retry_max_delay_seconds))

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()
