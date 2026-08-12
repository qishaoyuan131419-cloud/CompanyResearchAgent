from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from app.schemas.planning import SearchQuery
from app.schemas.search import SearchResult

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class StructuredLLMResult(Generic[TModel]):
    value: TModel
    usage: LLMUsage
    model: str
    cached: bool = False


class LLMClient(Protocol):
    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
        cache_namespace: str,
    ) -> StructuredLLMResult[TModel]: ...


class SearchClient(Protocol):
    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]: ...


class AsyncCache(Protocol):
    async def get(self, namespace: str, key: str) -> Any | None: ...

    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: int) -> None: ...

    async def close(self) -> None: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class EventLogger(Protocol):
    def info(self, event: str, **fields: Any) -> None: ...

    def error(self, event: str, **fields: Any) -> None: ...


class PromptRepository(Protocol):
    def render(self, name: str, variables: Mapping[str, Any]) -> str: ...
