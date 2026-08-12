from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from app.config import Settings
from app.core.budget import BudgetLedger
from app.core.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    ResearchAgentError,
    StructuredOutputError,
)
from app.core.protocols import LLMUsage
from app.core.telemetry import RunTelemetry
from app.llm.client import BudgetedCachedLLMClient
from app.llm.errors import LLMProviderError
from app.llm.factory import build_llm_client
from app.llm.types import LLMPricing, ProviderResponse


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact: str
    source_ids: list[str]


class FakeProvider:
    def __init__(self, response: ProviderResponse) -> None:
        self.response = response
        self.calls = 0
        self.system_prompts: list[str] = []

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {"provider": "fake", "model": "fake-model"}

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> ProviderResponse:
        self.calls += 1
        self.system_prompts.append(system_prompt)
        assert user_prompt
        assert json_schema["type"] == "object"
        assert schema_name == "Output"
        return self.response

    async def aclose(self) -> None:
        return None


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], Any] = {}

    async def get(self, namespace: str, key: str) -> Any | None:
        return self.values.get((namespace, key))

    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: int) -> None:
        assert ttl_seconds == 60
        self.values[(namespace, key)] = value

    async def close(self) -> None:
        return None


def _budget() -> BudgetLedger:
    return BudgetLedger(token_limit=1_000, cost_limit_usd=10.0, query_limit=10)


def test_provider_errors_participate_in_application_fallbacks() -> None:
    assert issubclass(LLMProviderError, ResearchAgentError)


def test_factory_requires_an_explicit_provider_base_url() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_api_key="secret",
        llm_model="model",
        llm_base_url=None,
    )

    with pytest.raises(ConfigurationError, match="CRA_LLM_BASE_URL"):
        build_llm_client(settings, budget=_budget())


@pytest.mark.asyncio
async def test_client_caches_validated_output_and_charges_budget_once() -> None:
    provider = FakeProvider(
        ProviderResponse(
            json_text='{"fact":"supported","source_ids":["src_1"]}',
            usage=LLMUsage(input_tokens=100, output_tokens=20),
            model="fake-model-v1",
        )
    )
    cache = MemoryCache()
    budget = _budget()
    telemetry = RunTelemetry()
    client = BudgetedCachedLLMClient(
        provider=provider,
        budget=budget,
        cache=cache,
        cache_ttl_seconds=60,
        pricing=LLMPricing(input_cost_per_million=2.0, output_cost_per_million=10.0),
        telemetry=telemetry,
    )

    first = await client.generate_structured(
        system_prompt="Be accurate.",
        user_prompt="Return the fact.",
        response_model=Output,
        cache_namespace="extractor",
    )
    second = await client.generate_structured(
        system_prompt="Be accurate.",
        user_prompt="Return the fact.",
        response_model=Output,
        cache_namespace="extractor",
    )
    snapshot = await budget.snapshot()

    assert first.cached is False
    assert first.usage.estimated_cost_usd == pytest.approx(0.0004)
    assert second.cached is True
    assert second.usage == LLMUsage()
    assert second.value == first.value
    assert provider.calls == 1
    assert snapshot.total_tokens == 120
    assert snapshot.estimated_cost_usd == pytest.approx(0.0004)
    assert provider.system_prompts == ["Be accurate."]
    stage = (await telemetry.snapshot())["extractor"]
    assert stage.logical_calls == 2
    assert stage.provider_calls == 1
    assert stage.cache_hits == 1
    assert stage.failed_calls == 0
    assert stage.input_tokens == 100
    assert stage.output_tokens == 20
    assert stage.max_prompt_bytes > 0


@pytest.mark.asyncio
async def test_client_records_paid_usage_before_rejecting_invalid_output() -> None:
    provider = FakeProvider(
        ProviderResponse(
            json_text='{"fact":"https://fabricated.test","source_ids":[]}',
            usage=LLMUsage(input_tokens=12, output_tokens=4),
            model="fake-model-v1",
        )
    )
    budget = _budget()
    client = BudgetedCachedLLMClient(provider=provider, budget=budget)

    with pytest.raises(StructuredOutputError):
        await client.generate_structured(
            system_prompt="Be accurate.",
            user_prompt="Return the fact.",
            response_model=Output,
            cache_namespace="extractor",
        )

    assert (await budget.snapshot()).total_tokens == 16


@pytest.mark.asyncio
async def test_invalid_cache_entry_is_never_trusted() -> None:
    provider = FakeProvider(
        ProviderResponse(
            json_text='{"fact":"fresh","source_ids":["src_2"]}',
            usage=LLMUsage(),
            model="fake-model-v1",
        )
    )
    cache = MemoryCache()
    client = BudgetedCachedLLMClient(
        provider=provider,
        budget=_budget(),
        cache=cache,
        cache_ttl_seconds=60,
    )
    # Prime the deterministic key, then corrupt its validated payload.
    await client.generate_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=Output,
        cache_namespace="ns",
    )
    only_key = next(iter(cache.values))
    cache.values[only_key] = {"json_text": "not JSON", "model": "stale"}

    result = await client.generate_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=Output,
        cache_namespace="ns",
    )

    assert result.cached is False
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_call_is_not_started_when_conservative_reservation_exceeds_budget() -> None:
    provider = FakeProvider(
        ProviderResponse(json_text='{"fact":"x","source_ids":[]}', usage=LLMUsage(), model="fake")
    )
    client = BudgetedCachedLLMClient(
        provider=provider,
        budget=BudgetLedger(token_limit=10, cost_limit_usd=1, query_limit=1),
    )

    with pytest.raises(BudgetExceededError):
        await client.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=Output,
            cache_namespace="test",
        )

    assert provider.calls == 0


def test_maximum_usage_estimate_does_not_treat_every_utf8_byte_as_a_token() -> None:
    provider = FakeProvider(
        ProviderResponse(json_text='{"fact":"x","source_ids":[]}', usage=LLMUsage(), model="fake")
    )
    client = BudgetedCachedLLMClient(provider=provider, budget=_budget())

    estimate = client._estimate_maximum_usage(
        system_prompt="a" * 1_000,
        user_prompt="b" * 1_000,
        json_schema={"type": "object"},
    )

    assert 1_000 < estimate.input_tokens < 2_000


@pytest.mark.asyncio
async def test_failed_budget_reservation_is_absorbing_for_later_calls() -> None:
    budget = BudgetLedger(token_limit=10, cost_limit_usd=1, query_limit=1)

    with pytest.raises(BudgetExceededError):
        await budget.reserve_llm(LLMUsage(input_tokens=11))
    with pytest.raises(BudgetExceededError, match="already exhausted"):
        await budget.reserve_llm(LLMUsage(input_tokens=1))

    assert (await budget.snapshot()).llm_exhausted
