from __future__ import annotations

import httpx

from app.config import Settings
from app.core.budget import BudgetLedger
from app.core.exceptions import ConfigurationError
from app.core.protocols import AsyncCache
from app.core.telemetry import RunTelemetry
from app.llm.anthropic_compatible import AnthropicCompatibleProvider
from app.llm.client import BudgetedCachedLLMClient
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.types import LLMPricing, StructuredOutputProvider


def build_llm_client(
    settings: Settings,
    *,
    budget: BudgetLedger,
    cache: AsyncCache | None = None,
    telemetry: RunTelemetry | None = None,
    http_client: httpx.AsyncClient | None = None,
    retry_base_delay_seconds: float = 0.5,
    retry_max_delay_seconds: float = 8.0,
) -> BudgetedCachedLLMClient:
    """Construct the configured provider without exposing it to business logic."""

    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
        raise ConfigurationError("CRA_LLM_API_KEY is required to construct the LLM client")
    if not settings.llm_model.strip():
        raise ConfigurationError("CRA_LLM_MODEL is required to construct the LLM client")
    if not settings.llm_base_url:
        raise ConfigurationError("CRA_LLM_BASE_URL is required to construct the LLM client")
    api_key = settings.llm_api_key.get_secret_value().strip()
    provider: StructuredOutputProvider
    if settings.llm_provider == "openai":
        provider = OpenAICompatibleProvider(
            api_key=api_key,
            model=settings.llm_model.strip(),
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            max_tokens_field=settings.llm_openai_max_tokens_field,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            http_client=http_client,
        )
    elif settings.llm_provider == "anthropic":
        provider = AnthropicCompatibleProvider(
            api_key=api_key,
            model=settings.llm_model.strip(),
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            http_client=http_client,
        )
    else:
        raise ConfigurationError(f"unsupported LLM provider: {settings.llm_provider}")
    return BudgetedCachedLLMClient(
        provider=provider,
        budget=budget,
        cache=cache if settings.cache_enabled else None,
        cache_ttl_seconds=settings.llm_cache_ttl_seconds if settings.cache_enabled else 0,
        pricing=LLMPricing(
            input_cost_per_million=settings.llm_input_cost_per_million,
            output_cost_per_million=settings.llm_output_cost_per_million,
        ),
        telemetry=telemetry,
    )
