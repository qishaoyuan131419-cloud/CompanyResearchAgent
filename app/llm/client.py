from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.budget import BudgetLedger
from app.core.protocols import AsyncCache, LLMUsage, StructuredLLMResult
from app.core.telemetry import RunTelemetry
from app.llm.parsing import parse_structured_output
from app.llm.types import LLMPricing, StructuredOutputProvider
from app.utils.hashing import stable_hash

TModel = TypeVar("TModel", bound=BaseModel)


class BudgetedCachedLLMClient:
    """Provider-neutral structured generation with validation, caching, and accounting."""

    def __init__(
        self,
        *,
        provider: StructuredOutputProvider,
        budget: BudgetLedger,
        cache: AsyncCache | None = None,
        cache_ttl_seconds: int = 0,
        pricing: LLMPricing | None = None,
        telemetry: RunTelemetry | None = None,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        self._provider = provider
        self._budget = budget
        self._cache = cache
        self._cache_ttl_seconds = cache_ttl_seconds
        self._pricing = pricing or LLMPricing()
        self._telemetry = telemetry

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
        cache_namespace: str,
    ) -> StructuredLLMResult[TModel]:
        if not cache_namespace.strip():
            raise ValueError("cache_namespace cannot be empty")
        json_schema = response_model.model_json_schema(mode="validation")
        prompt_bytes = (
            len(system_prompt.encode("utf-8"))
            + len(user_prompt.encode("utf-8"))
            + len(json.dumps(json_schema, sort_keys=True).encode("utf-8"))
        )
        if self._telemetry is not None:
            await self._telemetry.record_logical_call(
                cache_namespace,
                prompt_bytes=prompt_bytes,
            )
        cache_key = stable_hash(
            {
                "version": 1,
                "provider": dict(self._provider.cache_identity),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_schema": json_schema,
            },
            prefix="llm_",
            length=40,
        )

        cached_result = await self._read_cache(
            namespace=cache_namespace,
            key=cache_key,
            response_model=response_model,
        )
        if cached_result is not None:
            if self._telemetry is not None:
                await self._telemetry.record_cache_hit(cache_namespace)
            return cached_result

        reservation = await self._budget.reserve_llm(
            self._estimate_maximum_usage(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
            )
        )
        try:
            if self._telemetry is not None:
                await self._telemetry.record_provider_started(cache_namespace)
            provider_response = await self._provider.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name=response_model.__name__,
            )
        except BaseException:
            await self._budget.commit_llm_reservation(reservation)
            if self._telemetry is not None:
                await self._telemetry.record_provider_finished(cache_namespace, failed=True)
            raise
        usage = LLMUsage(
            input_tokens=max(0, provider_response.usage.input_tokens),
            output_tokens=max(0, provider_response.usage.output_tokens),
            estimated_cost_usd=self._pricing.estimate(
                input_tokens=provider_response.usage.input_tokens,
                output_tokens=provider_response.usage.output_tokens,
            ),
        )
        # The request has already incurred usage even if validation below fails.
        await self._budget.settle_llm(reservation, usage)
        try:
            value = parse_structured_output(provider_response.json_text, response_model)
        except BaseException:
            if self._telemetry is not None:
                await self._telemetry.record_provider_finished(
                    cache_namespace,
                    usage=usage,
                    failed=True,
                )
            raise
        if self._telemetry is not None:
            await self._telemetry.record_provider_finished(cache_namespace, usage=usage)
        await self._write_cache(
            namespace=cache_namespace,
            key=cache_key,
            json_text=provider_response.json_text,
            model=provider_response.model,
        )
        return StructuredLLMResult(
            value=value,
            usage=usage,
            model=provider_response.model,
            cached=False,
        )

    def _estimate_maximum_usage(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> LLMUsage:
        input_bytes = len(system_prompt.encode("utf-8")) + len(user_prompt.encode("utf-8"))
        input_bytes += len(
            json.dumps(json_schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        # UTF-8 byte length is a safe tokenizer-independent boundary, but treating
        # every byte as a token over-reserves English-heavy prompts by roughly 4x
        # and can stop evidence extraction long before the configured run budget.
        # Two bytes per token remains conservative for mixed English/CJK research
        # payloads while provider-reported usage is still used for final accounting.
        conservative_input_tokens = (input_bytes + 1) // 2 + 256
        configured_output = self._provider.cache_identity.get("max_output_tokens", 0)
        max_output_tokens = (
            configured_output if isinstance(configured_output, int) and configured_output > 0 else 0
        )
        return LLMUsage(
            input_tokens=conservative_input_tokens,
            output_tokens=max_output_tokens,
            estimated_cost_usd=self._pricing.estimate(
                input_tokens=conservative_input_tokens,
                output_tokens=max_output_tokens,
            ),
        )

    async def _read_cache(
        self,
        *,
        namespace: str,
        key: str,
        response_model: type[TModel],
    ) -> StructuredLLMResult[TModel] | None:
        if self._cache is None or self._cache_ttl_seconds == 0:
            return None
        try:
            cached = await self._cache.get(namespace, key)
        except Exception:
            return None
        if not isinstance(cached, dict):
            return None
        json_text = cached.get("json_text")
        model = cached.get("model")
        if not isinstance(json_text, str) or not isinstance(model, str):
            return None
        try:
            value = parse_structured_output(json_text, response_model)
        except Exception:
            return None
        return StructuredLLMResult(
            value=value,
            usage=LLMUsage(),
            model=model,
            cached=True,
        )

    async def _write_cache(
        self,
        *,
        namespace: str,
        key: str,
        json_text: str,
        model: str,
    ) -> None:
        if self._cache is None or self._cache_ttl_seconds == 0:
            return
        value: dict[str, Any] = {"version": 1, "json_text": json_text, "model": model}
        try:
            await self._cache.set(namespace, key, value, self._cache_ttl_seconds)
        except Exception:
            # Caching is an optimization; a cache outage must not discard a paid,
            # already-validated provider response.
            return

    async def aclose(self) -> None:
        await self._provider.aclose()

    async def __aenter__(self) -> BudgetedCachedLLMClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()
