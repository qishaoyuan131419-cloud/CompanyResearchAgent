from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from app.core.protocols import LLMUsage
from app.llm.errors import LLMProviderResponseError
from app.llm.http_provider import BaseHTTPProvider, Sleep
from app.llm.types import JsonObject, ProviderResponse

_TOOL_NAME = "return_structured_output"


def _endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/messages") else f"{normalized}/messages"


def _required_usage_token(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LLMProviderResponseError(f"Anthropic-compatible response omitted valid usage.{key}")
    return value


class AnthropicCompatibleProvider(BaseHTTPProvider):
    """Strict structured output using the Anthropic-compatible Messages tool API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        anthropic_version: str = "2023-06-01",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 8.0,
        http_client: httpx.AsyncClient | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        super().__init__(
            endpoint=_endpoint(base_url),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            http_client=http_client,
            **({"sleep": sleep} if sleep is not None else {}),
        )
        self._api_key = api_key
        self._model = model
        self._anthropic_version = anthropic_version
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "provider": "anthropic-compatible",
            "endpoint": self._endpoint,
            "model": self._model,
            "anthropic_version": self._anthropic_version,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: JsonObject,
        schema_name: str,
    ) -> ProviderResponse:
        body = await self._post_json(
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self._anthropic_version,
                "Content-Type": "application/json",
            },
            payload={
                "model": self._model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": self._temperature,
                "max_tokens": self._max_output_tokens,
                "tools": [
                    {
                        "name": _TOOL_NAME,
                        "input_schema": json_schema,
                    }
                ],
                "tool_choice": {
                    "type": "tool",
                    "name": _TOOL_NAME,
                    "disable_parallel_tool_use": True,
                },
            },
        )
        if body.get("stop_reason") == "max_tokens":
            raise LLMProviderResponseError(
                "Anthropic-compatible response exceeded the output limit"
            )
        content = body.get("content")
        if not isinstance(content, list):
            raise LLMProviderResponseError("Anthropic-compatible response omitted content blocks")
        tool_blocks = [
            block
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == _TOOL_NAME
        ]
        if len(tool_blocks) != 1 or not isinstance(tool_blocks[0].get("input"), dict):
            raise LLMProviderResponseError(
                "Anthropic-compatible response did not contain one structured output tool call"
            )
        usage_data = body.get("usage")
        if not isinstance(usage_data, dict):
            raise LLMProviderResponseError("Anthropic-compatible response omitted token usage")
        usage = LLMUsage(
            input_tokens=_required_usage_token(usage_data, "input_tokens"),
            output_tokens=_required_usage_token(usage_data, "output_tokens"),
        )
        returned_model = body.get("model")
        return ProviderResponse(
            json_text=json.dumps(
                tool_blocks[0]["input"], ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ),
            usage=usage,
            model=returned_model if isinstance(returned_model, str) else self._model,
        )
