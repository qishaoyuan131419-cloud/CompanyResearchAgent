from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.protocols import LLMUsage

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Provider output before application-schema validation."""

    json_text: str
    usage: LLMUsage
    model: str


class StructuredOutputProvider(Protocol):
    @property
    def cache_identity(self) -> Mapping[str, Any]: ...

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: JsonObject,
        schema_name: str,
    ) -> ProviderResponse: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LLMPricing:
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    def estimate(self, *, input_tokens: int, output_tokens: int) -> float:
        return (
            max(0, input_tokens) * max(0.0, self.input_cost_per_million)
            + max(0, output_tokens) * max(0.0, self.output_cost_per_million)
        ) / 1_000_000
