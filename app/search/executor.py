import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from app.cache.search_results import SearchResultCache
from app.config import Settings
from app.core.clock import SystemClock
from app.core.exceptions import ConfigurationError, SearchError
from app.core.protocols import AsyncCache, Clock, EventLogger, SearchClient
from app.schemas.planning import SearchQuery
from app.schemas.search import QueryExecution, SearchBatch, SearchResult, SearchStatistics
from app.search.errors import SearchClientError
from app.utils.urls import canonicalize_url

Sleep: TypeAlias = Callable[[float], Awaitable[None]]
Jitter: TypeAlias = Callable[[float], float]
RetryPredicate: TypeAlias = Callable[[BaseException], bool]


@dataclass(frozen=True, slots=True)
class _ExecutionOutcome:
    execution: QueryExecution
    duplicates_removed: int = 0
    permanent_failure: bool = False


class SearchExecutor:
    """Execute independent search queries concurrently without failing the batch."""

    def __init__(
        self,
        client: SearchClient,
        *,
        timeout_seconds: float,
        max_retries: int,
        backoff_base_seconds: float,
        max_concurrency: int,
        default_limit: int | None = None,
        result_cache: SearchResultCache | None = None,
        clock: Clock | None = None,
        logger: EventLogger | None = None,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter | None = None,
        retry_predicate: RetryPredicate | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds cannot be negative")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if default_limit is not None and default_limit < 1:
            raise ValueError("default_limit must be at least 1")

        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._backoff_base_seconds = backoff_base_seconds
        self._default_limit = default_limit
        self._result_cache = result_cache
        self._clock = clock or SystemClock()
        self._logger = logger
        self._sleep = sleep
        self._jitter = jitter or _full_jitter
        self._retry_predicate = retry_predicate or _is_retryable
        self._monotonic = monotonic
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @classmethod
    def from_settings(
        cls,
        client: SearchClient,
        settings: Settings,
        *,
        cache: AsyncCache | None = None,
        result_cache: SearchResultCache | None = None,
        clock: Clock | None = None,
        logger: EventLogger | None = None,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter | None = None,
    ) -> "SearchExecutor":
        if cache is not None and result_cache is not None:
            raise ValueError("provide cache or result_cache, not both")
        if cache is not None:
            result_cache = SearchResultCache(
                cache,
                ttl_seconds=settings.search_cache_ttl_seconds if settings.cache_enabled else 0,
                page_ttl_seconds=(settings.page_cache_ttl_seconds if settings.cache_enabled else 0),
            )
        return cls(
            client,
            timeout_seconds=settings.search_timeout_seconds,
            max_retries=settings.search_max_retries,
            backoff_base_seconds=settings.search_backoff_base_seconds,
            max_concurrency=settings.search_max_concurrency,
            default_limit=settings.exa_results_per_query,
            result_cache=result_cache,
            clock=clock,
            logger=logger,
            sleep=sleep,
            jitter=jitter,
        )

    async def execute(
        self,
        queries: Sequence[SearchQuery],
        *,
        limit: int | None = None,
        run_id: str | None = None,
    ) -> SearchBatch:
        limit = limit if limit is not None else self._default_limit
        if limit is None:
            raise ValueError("limit is required when no default_limit is configured")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        started = self._monotonic()
        tasks = [
            asyncio.create_task(self._execute_one(query, limit=limit, run_id=run_id))
            for query in queries
        ]
        try:
            outcomes = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if (
            outcomes
            and all(outcome.execution.error is not None for outcome in outcomes)
            and all(outcome.permanent_failure for outcome in outcomes)
        ):
            raise SearchError("search provider permanently rejected every query in the batch")
        executions = [outcome.execution for outcome in outcomes]
        successful = sum(execution.error is None for execution in executions)
        failed = len(executions) - successful
        statistics = SearchStatistics(
            query_count=len(executions),
            successful_queries=successful,
            failed_queries=failed,
            cache_hits=sum(execution.cache_hit for execution in executions),
            total_results=sum(len(execution.results) for execution in executions),
            deduplicated_results=sum(outcome.duplicates_removed for outcome in outcomes),
            total_duration_ms=max((self._monotonic() - started) * 1_000, 0.0),
            retries=sum(max(execution.attempts - 1, 0) for execution in executions),
        )
        self._log_info(
            "search.batch_completed",
            run_id=run_id,
            round=queries[0].round
            if queries and len({query.round for query in queries}) == 1
            else None,
            duration_ms=statistics.total_duration_ms,
            queries=statistics.query_count,
            results=statistics.total_results,
            errors=statistics.failed_queries,
            retry=statistics.retries,
        )
        return SearchBatch(executions=executions, statistics=statistics)

    async def execute_batch(
        self,
        queries: Sequence[SearchQuery],
        *,
        limit: int | None = None,
        run_id: str | None = None,
    ) -> SearchBatch:
        """Compatibility alias for orchestration code that names the returned unit."""

        return await self.execute(queries, limit=limit, run_id=run_id)

    async def _execute_one(
        self,
        query: SearchQuery,
        *,
        limit: int,
        run_id: str | None,
    ) -> _ExecutionOutcome:
        started_at = self._clock.now()
        started = self._monotonic()

        cached = await self._read_cache(query, limit=limit, run_id=run_id)
        if cached is not None:
            completed_at = self._clock.now()
            self._log_info(
                "search.cache_hit",
                run_id=run_id,
                round=query.round,
                query_id=query.query_id,
                query=query.query,
                results=len(cached),
            )
            return _ExecutionOutcome(
                QueryExecution(
                    query_id=query.query_id,
                    query=query.query,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=max((self._monotonic() - started) * 1_000, 0.0),
                    attempts=1,
                    cache_hit=True,
                    results=cached[:limit],
                )
            )

        last_error: BaseException | None = None
        attempts = 0
        for attempt in range(1, self._max_retries + 2):
            attempts = attempt
            self._log_info(
                "search.attempt_started",
                run_id=run_id,
                round=query.round,
                query_id=query.query_id,
                query=query.query,
                attempt=attempt,
            )
            try:
                async with self._semaphore:
                    results = await asyncio.wait_for(
                        self._client.search(query, limit=limit),
                        timeout=self._timeout_seconds,
                    )
                deduplicated, removed = _deduplicate_results(results)
                deduplicated = deduplicated[:limit]
                await self._write_cache(query, deduplicated, limit=limit, run_id=run_id)
                completed_at = self._clock.now()
                duration_ms = max((self._monotonic() - started) * 1_000, 0.0)
                self._log_info(
                    "search.query_completed",
                    run_id=run_id,
                    round=query.round,
                    query_id=query.query_id,
                    query=query.query,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    results=len(deduplicated),
                )
                return _ExecutionOutcome(
                    QueryExecution(
                        query_id=query.query_id,
                        query=query.query,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        attempts=attempt,
                        results=deduplicated,
                    ),
                    duplicates_removed=removed,
                )
            except asyncio.CancelledError:
                raise
            except ConfigurationError:
                raise
            except TimeoutError:
                last_error = TimeoutError(
                    f"search attempt exceeded {self._timeout_seconds:.3f} seconds"
                )
            except Exception as exc:
                last_error = exc

            if attempt > self._max_retries or not self._retry_predicate(last_error):
                break

            unjittered_delay = self._backoff_base_seconds * (2 ** (attempt - 1))
            delay = max(self._jitter(unjittered_delay), 0.0)
            self._log_info(
                "search.retry_scheduled",
                run_id=run_id,
                round=query.round,
                query_id=query.query_id,
                query=query.query,
                attempt=attempt,
                retry=attempt,
                delay_seconds=delay,
                error=_safe_error(last_error),
            )
            await self._sleep(delay)

        completed_at = self._clock.now()
        duration_ms = max((self._monotonic() - started) * 1_000, 0.0)
        error = _safe_error(last_error)
        self._log_error(
            "search.query_failed",
            run_id=run_id,
            round=query.round,
            query_id=query.query_id,
            query=query.query,
            attempts=attempts,
            duration_ms=duration_ms,
            error=error,
        )
        return _ExecutionOutcome(
            QueryExecution(
                query_id=query.query_id,
                query=query.query,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                attempts=max(attempts, 1),
                error=error,
            ),
            permanent_failure=(
                isinstance(last_error, SearchClientError) and not last_error.retryable
            ),
        )

    async def _read_cache(
        self,
        query: SearchQuery,
        *,
        limit: int,
        run_id: str | None,
    ) -> list[SearchResult] | None:
        if self._result_cache is None:
            return None
        try:
            return await self._result_cache.get(query, limit=limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_error(
                "search.cache_read_failed",
                run_id=run_id,
                round=query.round,
                query_id=query.query_id,
                query=query.query,
                error=_safe_error(exc),
            )
            return None

    async def _write_cache(
        self,
        query: SearchQuery,
        results: list[SearchResult],
        *,
        limit: int,
        run_id: str | None,
    ) -> None:
        if self._result_cache is None:
            return
        try:
            await self._result_cache.set(query, results, limit=limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_error(
                "search.cache_write_failed",
                run_id=run_id,
                round=query.round,
                query_id=query.query_id,
                query=query.query,
                error=_safe_error(exc),
            )

    def _log_info(self, event: str, **fields: object) -> None:
        if self._logger is None:
            return
        try:
            self._logger.info(event, **fields)
        except Exception:
            return

    def _log_error(self, event: str, **fields: object) -> None:
        if self._logger is None:
            return
        try:
            self._logger.error(event, **fields)
        except Exception:
            return


def _full_jitter(maximum_delay: float) -> float:
    return random.uniform(0.0, maximum_delay)


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, SearchClientError):
        return error.retryable
    return isinstance(error, (TimeoutError, ConnectionError, SearchError))


def _deduplicate_results(results: Sequence[SearchResult]) -> tuple[list[SearchResult], int]:
    unique: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        key = canonicalize_url(str(result.url))
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique, len(results) - len(unique)


def _safe_error(error: BaseException | None) -> str:
    if error is None:
        return "UnknownSearchError: search failed without an exception"
    message = " ".join(str(error).split())[:500]
    if not message:
        message = "no details provided"
    return f"{type(error).__name__}: {message}"
