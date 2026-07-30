from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

import httpx

from app.core.protocols import LLMUsage
from app.llm.errors import LLMProviderResponseError
from app.llm.http_provider import BaseHTTPProvider, Sleep
from app.llm.types import JsonObject, ProviderResponse

_SCHEMA_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "default",
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "pattern",
        "title",
        "uniqueItems",
    }
)


def _endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return (
        normalized if normalized.endswith("/chat/completions") else f"{normalized}/chat/completions"
    )


def _required_usage_token(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LLMProviderResponseError(f"OpenAI-compatible response omitted valid usage.{key}")
    return value


def to_openai_strict_schema(schema: JsonObject) -> JsonObject:
    """Convert Pydantic JSON Schema to OpenAI's strict structured-output subset."""

    def normalize(value: Any) -> Any:
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if not isinstance(value, dict):
            return value
        normalized = {
            key: normalize(item)
            for key, item in value.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYWORDS
        }
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["required"] = list(properties)
            normalized["additionalProperties"] = False
        return normalized

    result = normalize(schema)
    if not isinstance(result, dict):
        raise TypeError("structured output schema must be an object")
    return result


class OpenAICompatibleProvider(BaseHTTPProvider):
    """Strict structured output over the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = (
            "max_completion_tokens"
        ),
        response_format: Literal["json_schema", "json_object"] = "json_schema",
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
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._max_tokens_field = max_tokens_field
        self._response_format = response_format

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "provider": "openai-compatible",
            "endpoint": self._endpoint,
            "model": self._model,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "max_tokens_field": self._max_tokens_field,
            "response_format": self._response_format,
        }

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: JsonObject,
        schema_name: str,
    ) -> ProviderResponse:
        safe_name = _SCHEMA_NAME.sub("_", schema_name).strip("_")[:64] or "structured_output"
        if self._response_format == "json_schema":
            strict_schema = to_openai_strict_schema(json_schema)
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": safe_name, "strict": True, "schema": strict_schema},
            }
        else:
            response_format = {"type": "json_object"}
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            self._max_tokens_field: self._max_output_tokens,
            "response_format": response_format,
        }
        body = await self._post_json(
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise LLMProviderResponseError("OpenAI-compatible response did not contain one choice")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise LLMProviderResponseError("OpenAI-compatible response exceeded the output limit")
        if finish_reason == "content_filter":
            raise LLMProviderResponseError(
                "OpenAI-compatible response was blocked by content filtering"
            )
        message = choice.get("message")
        refusal = message.get("refusal") if isinstance(message, dict) else None
        if isinstance(refusal, str) and refusal.strip():
            raise LLMProviderResponseError("OpenAI-compatible model refused the request")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderResponseError("OpenAI-compatible response omitted JSON content")
        usage_data = body.get("usage")
        if not isinstance(usage_data, dict):
            raise LLMProviderResponseError("OpenAI-compatible response omitted token usage")
        usage = LLMUsage(
            input_tokens=_required_usage_token(usage_data, "prompt_tokens"),
            output_tokens=_required_usage_token(usage_data, "completion_tokens"),
        )
        returned_model = body.get("model")
        return ProviderResponse(
            json_text=content,
            usage=usage,
            model=returned_model if isinstance(returned_model, str) else self._model,
        )
