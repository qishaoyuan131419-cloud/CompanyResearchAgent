import asyncio
from dataclasses import dataclass, field

from app.core.protocols import LLMUsage


@dataclass(slots=True)
class LLMStageCounters:
    logical_calls: int = 0
    provider_calls: int = 0
    cache_hits: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    max_prompt_bytes: int = 0


@dataclass(slots=True)
class _TelemetryState:
    stages: dict[str, LLMStageCounters] = field(default_factory=dict)


class RunTelemetry:
    """Concurrency-safe, per-run provider telemetry."""

    def __init__(self) -> None:
        self._state = _TelemetryState()
        self._lock = asyncio.Lock()

    async def record_logical_call(self, stage: str, *, prompt_bytes: int) -> None:
        async with self._lock:
            counters = self._state.stages.setdefault(stage, LLMStageCounters())
            counters.logical_calls += 1
            counters.max_prompt_bytes = max(counters.max_prompt_bytes, max(0, prompt_bytes))

    async def record_cache_hit(self, stage: str) -> None:
        async with self._lock:
            self._state.stages.setdefault(stage, LLMStageCounters()).cache_hits += 1

    async def record_provider_started(self, stage: str) -> None:
        async with self._lock:
            self._state.stages.setdefault(stage, LLMStageCounters()).provider_calls += 1

    async def record_provider_finished(
        self,
        stage: str,
        *,
        usage: LLMUsage | None = None,
        failed: bool = False,
    ) -> None:
        async with self._lock:
            counters = self._state.stages.setdefault(stage, LLMStageCounters())
            if failed:
                counters.failed_calls += 1
            if usage is not None:
                counters.input_tokens += max(0, usage.input_tokens)
                counters.output_tokens += max(0, usage.output_tokens)

    async def snapshot(self) -> dict[str, LLMStageCounters]:
        async with self._lock:
            return {
                stage: LLMStageCounters(
                    logical_calls=counters.logical_calls,
                    provider_calls=counters.provider_calls,
                    cache_hits=counters.cache_hits,
                    failed_calls=counters.failed_calls,
                    input_tokens=counters.input_tokens,
                    output_tokens=counters.output_tokens,
                    max_prompt_bytes=counters.max_prompt_bytes,
                )
                for stage, counters in self._state.stages.items()
            }
