import asyncio

import pytest

from app.cache import MemoryTTLCache, SearchResultCache
from app.config import Settings
from app.core.exceptions import ConfigurationError, SearchError
from app.schemas.planning import SearchQuery
from app.schemas.search import SearchResult
from app.search.errors import RetryableSearchError, SearchResponseError
from app.search.executor import SearchExecutor


def _query(index: int) -> SearchQuery:
    return SearchQuery(
        query_id=f"q-{index}",
        query=f"Acme topic {index}",
        topic="Overview",
        priority=1,
        expected_evidence="Company facts",
    )


def _result(index: int = 1, *, url: str | None = None) -> SearchResult:
    return SearchResult(
        title=f"Result {index}",
        url=url or f"https://example.com/{index}",
        text="Evidence",
    )


class ConcurrencyClient:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del limit
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return [_result(int(query.query_id.removeprefix("q-")))]


async def test_executor_bounds_parallel_searches() -> None:
    client = ConcurrencyClient()
    executor = SearchExecutor(
        client,
        timeout_seconds=1,
        max_retries=0,
        backoff_base_seconds=0,
        max_concurrency=2,
        default_limit=5,
    )

    batch = await executor.execute([_query(index) for index in range(4)])

    assert client.maximum_active == 2
    assert batch.statistics.query_count == 4
    assert batch.statistics.successful_queries == 4
    assert batch.statistics.failed_queries == 0


async def test_from_settings_allows_execute_without_explicit_limit() -> None:
    client = CountingClient([_result()])
    settings = Settings(environment="test", exa_results_per_query=3)
    executor = SearchExecutor.from_settings(client, settings, cache=MemoryTTLCache())

    first = await executor.execute([_query(1)])
    second = await executor.execute([_query(1)])

    assert first.statistics.successful_queries == 1
    assert second.executions[0].cache_hit


class RetryOnceClient:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del query, limit
        self.calls += 1
        if self.calls == 1:
            raise RetryableSearchError("temporary outage")
        return [_result()]


async def test_executor_retries_with_injected_backoff_jitter() -> None:
    client = RetryOnceClient()
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    executor = SearchExecutor(
        client,
        timeout_seconds=1,
        max_retries=2,
        backoff_base_seconds=0.5,
        max_concurrency=1,
        default_limit=5,
        jitter=lambda delay: delay + 0.25,
        sleep=record_sleep,
    )

    batch = await executor.execute([_query(1)])

    assert client.calls == 2
    assert delays == [0.75]
    assert batch.executions[0].attempts == 2
    assert batch.statistics.retries == 1


class PartialFailureClient:
    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del limit
        if query.query_id == "q-2":
            raise SearchResponseError("invalid provider payload")
        return [_result(1)]


async def test_executor_preserves_partial_success() -> None:
    executor = SearchExecutor(
        PartialFailureClient(),
        timeout_seconds=1,
        max_retries=3,
        backoff_base_seconds=0,
        max_concurrency=2,
        default_limit=5,
    )

    batch = await executor.execute([_query(1), _query(2)])

    assert batch.statistics.successful_queries == 1
    assert batch.statistics.failed_queries == 1
    assert batch.executions[0].results
    assert batch.executions[1].results == []
    assert batch.executions[1].attempts == 1
    assert "SearchResponseError" in (batch.executions[1].error or "")


async def test_executor_propagates_batch_wide_permanent_provider_failure() -> None:
    executor = SearchExecutor(
        PartialFailureClient(),
        timeout_seconds=1,
        max_retries=3,
        backoff_base_seconds=0,
        max_concurrency=2,
        default_limit=5,
    )

    with pytest.raises(SearchError, match="permanently rejected every query"):
        await executor.execute([_query(2), _query(2)])


class MisconfiguredClient:
    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del query, limit
        raise ConfigurationError("MCP SDK unavailable")


async def test_executor_propagates_runtime_configuration_failure() -> None:
    executor = SearchExecutor(
        MisconfiguredClient(),
        timeout_seconds=1,
        max_retries=3,
        backoff_base_seconds=0,
        max_concurrency=2,
        default_limit=5,
    )

    with pytest.raises(ConfigurationError, match="MCP SDK unavailable"):
        await executor.execute([_query(1), _query(2)])


class MisconfiguredWithSlowSiblingsClient:
    def __init__(self) -> None:
        self.cancelled = 0
        self.completed = 0

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del limit
        if query.query_id == "q-1":
            await asyncio.sleep(0)
            raise ConfigurationError("MCP SDK unavailable")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.completed += 1
        return []


async def test_configuration_failure_cancels_and_awaits_sibling_queries() -> None:
    client = MisconfiguredWithSlowSiblingsClient()
    executor = SearchExecutor(
        client,
        timeout_seconds=20,
        max_retries=0,
        backoff_base_seconds=0,
        max_concurrency=3,
        default_limit=5,
    )

    with pytest.raises(ConfigurationError, match="MCP SDK unavailable"):
        await executor.execute([_query(1), _query(2), _query(3)])

    assert client.cancelled == 2
    assert client.completed == 0


class SlowClient:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del query, limit
        self.calls += 1
        await asyncio.sleep(0.05)
        return []


async def test_executor_times_out_each_attempt_and_returns_failure() -> None:
    client = SlowClient()
    executor = SearchExecutor(
        client,
        timeout_seconds=0.001,
        max_retries=1,
        backoff_base_seconds=0,
        max_concurrency=1,
        default_limit=5,
        jitter=lambda delay: delay,
    )

    batch = await executor.execute([_query(1)])

    assert client.calls == 2
    assert batch.statistics.failed_queries == 1
    assert batch.statistics.retries == 1
    assert "TimeoutError" in (batch.executions[0].error or "")


class CountingClient:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls = 0

    async def search(self, query: SearchQuery, *, limit: int) -> list[SearchResult]:
        del query, limit
        self.calls += 1
        return self.results


async def test_executor_uses_typed_search_cache() -> None:
    result_cache = SearchResultCache(MemoryTTLCache(), ttl_seconds=60)
    client = CountingClient([_result()])
    executor = SearchExecutor(
        client,
        timeout_seconds=1,
        max_retries=0,
        backoff_base_seconds=0,
        max_concurrency=1,
        default_limit=5,
        result_cache=result_cache,
    )

    first = await executor.execute([_query(1)])
    second = await executor.execute([_query(1)])

    assert not first.executions[0].cache_hit
    assert second.executions[0].cache_hit
    assert second.statistics.cache_hits == 1
    assert client.calls == 1


async def test_executor_deduplicates_provider_urls_without_rewriting_them() -> None:
    first_url = "https://example.com/page?utm_source=test"
    client = CountingClient(
        [
            _result(1, url=first_url),
            _result(2, url="https://example.com/page"),
            _result(3, url="https://example.com/other"),
        ]
    )
    executor = SearchExecutor(
        client,
        timeout_seconds=1,
        max_retries=0,
        backoff_base_seconds=0,
        max_concurrency=1,
        default_limit=5,
    )

    batch = await executor.execute([_query(1)])

    assert len(batch.executions[0].results) == 2
    assert str(batch.executions[0].results[0].url) == first_url
    assert batch.statistics.deduplicated_results == 1


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


async def test_concurrent_runs_have_distinct_search_log_correlation() -> None:
    logger = RecordingLogger()
    executor = SearchExecutor(
        ConcurrencyClient(),
        timeout_seconds=1,
        max_retries=0,
        backoff_base_seconds=0,
        max_concurrency=2,
        default_limit=5,
        logger=logger,
    )

    await asyncio.gather(
        executor.execute([_query(1)], run_id="run-one"),
        executor.execute([_query(1)], run_id="run-two"),
    )

    search_events = [fields for event, fields in logger.events if event.startswith("search.")]
    assert search_events
    assert {fields["run_id"] for fields in search_events} == {"run-one", "run-two"}
    assert all(fields.get("run_id") in {"run-one", "run-two"} for fields in search_events)
