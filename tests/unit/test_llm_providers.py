from __future__ import annotations

import json

import httpx
import pytest

from app.llm.anthropic_compatible import AnthropicCompatibleProvider
from app.llm.errors import LLMProviderError, LLMProviderResponseError
from app.llm.openai_compatible import OpenAICompatibleProvider


async def _no_sleep(_: float) -> None:
    return None


@pytest.mark.asyncio
async def test_openai_compatible_provider_forces_json_schema_and_retries_transient_status() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "returned-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"fact":"supported"}'},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = OpenAICompatibleProvider(
            api_key="secret",
            model="configured-model",
            base_url="https://provider.test/v1",
            max_retries=1,
            retry_base_delay_seconds=0,
            http_client=http_client,
            sleep=_no_sleep,
        )
        result = await provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            json_schema={"type": "object", "properties": {"fact": {"type": "string"}}},
            schema_name="Output",
        )

    assert len(requests) == 2
    payload = json.loads(requests[-1].content)
    assert requests[-1].url == "https://provider.test/v1/chat/completions"
    assert requests[-1].headers["authorization"] == "Bearer secret"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert result.json_text == '{"fact":"supported"}'
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 3
    assert result.model == "returned-model"


@pytest.mark.asyncio
async def test_openai_compatible_provider_does_not_retry_permanent_client_error() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, request=request, json={"error": {"message": "bad"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = OpenAICompatibleProvider(
            api_key="secret",
            model="model",
            base_url="https://provider.test/v1",
            max_retries=3,
            http_client=http_client,
            sleep=_no_sleep,
        )
        with pytest.raises(LLMProviderError, match="HTTP 400"):
            await provider.generate_json(
                system_prompt="system",
                user_prompt="user",
                json_schema={"type": "object"},
                schema_name="Output",
            )

    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choice", "message"),
    [
        (
            {"finish_reason": "content_filter", "message": {"content": None}},
            "content filtering",
        ),
        (
            {
                "finish_reason": "stop",
                "message": {"content": None, "refusal": "I cannot help with that."},
            },
            "refused",
        ),
    ],
)
async def test_openai_compatible_provider_rejects_filtered_or_refused_output(
    choice: dict[str, object],
    message: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "model",
                "choices": [choice],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = OpenAICompatibleProvider(
            api_key="secret",
            model="model",
            base_url="https://provider.test/v1",
            http_client=http_client,
        )
        with pytest.raises(LLMProviderResponseError, match=message):
            await provider.generate_json(
                system_prompt="system",
                user_prompt="user",
                json_schema={"type": "object"},
                schema_name="Output",
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": -1},
    ],
)
async def test_openai_compatible_provider_requires_valid_usage(
    usage: object,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"fact":"supported"}'},
                    }
                ],
                "usage": usage,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = OpenAICompatibleProvider(
            api_key="secret",
            model="model",
            base_url="https://provider.test/v1",
            http_client=http_client,
        )
        with pytest.raises(LLMProviderResponseError, match=r"usage|token usage"):
            await provider.generate_json(
                system_prompt="system",
                user_prompt="user",
                json_schema={"type": "object"},
                schema_name="Output",
            )


@pytest.mark.asyncio
async def test_anthropic_compatible_provider_forces_one_structured_tool_call() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "model": "claude-compatible",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "return_structured_output",
                        "id": "tool_1",
                        "input": {"fact": "supported", "source_ids": ["src_1"]},
                    }
                ],
                "usage": {"input_tokens": 21, "output_tokens": 5},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider = AnthropicCompatibleProvider(
            api_key="secret",
            model="configured-model",
            base_url="https://anthropic-compatible.test/v1",
            http_client=http_client,
        )
        result = await provider.generate_json(
            system_prompt="system",
            user_prompt="user",
            json_schema={"type": "object", "properties": {"fact": {"type": "string"}}},
            schema_name="Output",
        )

    payload = json.loads(requests[0].content)
    assert requests[0].url == "https://anthropic-compatible.test/v1/messages"
    assert requests[0].headers["x-api-key"] == "secret"
    assert payload["tool_choice"]["name"] == "return_structured_output"
    assert payload["tool_choice"]["disable_parallel_tool_use"] is True
    assert payload["tools"][0]["input_schema"]["type"] == "object"
    assert json.loads(result.json_text) == {"fact": "supported", "source_ids": ["src_1"]}
    assert result.usage.total_tokens == 26
