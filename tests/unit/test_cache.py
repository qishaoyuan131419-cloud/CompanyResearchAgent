from pathlib import Path

import pytest

from app.cache import MemoryTTLCache, NullCache, SearchResultCache, SQLiteTTLCache
from app.schemas.planning import SearchQuery
from app.schemas.search import SearchResult


class MutableTimer:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _query(*, query_id: str = "q-1", text: str = "Acme pipeline") -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        query=text,
        topic="Pipeline",
        priority=1,
        expected_evidence="Pipeline assets",
    )


async def test_null_cache_never_stores_values() -> None:
    cache = NullCache()
    await cache.set("search", "key", {"value": 1}, ttl_seconds=60)
    assert await cache.get("search", "key") is None
    await cache.close()


async def test_memory_cache_expires_and_isolates_mutation() -> None:
    timer = MutableTimer()
    cache = MemoryTTLCache(timer=timer)
    original = {"items": [1]}

    await cache.set("ns", "key", original, ttl_seconds=10)
    original["items"].append(2)
    cached = await cache.get("ns", "key")
    assert cached == {"items": [1]}

    cached["items"].append(3)
    assert await cache.get("ns", "key") == {"items": [1]}

    timer.value += 10
    assert await cache.get("ns", "key") is None


async def test_memory_cache_zero_ttl_deletes_and_close_is_terminal() -> None:
    cache = MemoryTTLCache()
    await cache.set("ns", "key", "value", ttl_seconds=10)
    await cache.set("ns", "key", object(), ttl_seconds=0)
    assert await cache.get("ns", "key") is None
    await cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        await cache.get("ns", "key")


async def test_sqlite_cache_persists_json_and_enforces_ttl(tmp_path: Path) -> None:
    timer = MutableTimer()
    path = tmp_path / "cache.sqlite3"
    cache = SQLiteTTLCache(path, timer=timer)
    await cache.set("ns", "key", {"value": [1, 2]}, ttl_seconds=10)
    assert await cache.get("ns", "key") == {"value": [1, 2]}
    await cache.close()

    reopened = SQLiteTTLCache(path, timer=timer)
    assert await reopened.get("ns", "key") == {"value": [1, 2]}
    timer.value += 10
    assert await reopened.get("ns", "key") is None
    await reopened.close()


async def test_sqlite_cache_zero_ttl_can_delete_unserializable_value(tmp_path: Path) -> None:
    cache = SQLiteTTLCache(tmp_path / "cache.sqlite3")
    await cache.set("ns", "key", "value", ttl_seconds=10)
    await cache.set("ns", "key", object(), ttl_seconds=0)
    assert await cache.get("ns", "key") is None
    await cache.close()


async def test_search_result_cache_round_trips_typed_results() -> None:
    backend = MemoryTTLCache()
    cache = SearchResultCache(backend, ttl_seconds=60)
    query = _query()
    result = SearchResult(title="Acme", url="https://example.com/acme", text="Evidence")

    await cache.set(query, [result], limit=5)
    cached = await cache.get(
        _query(query_id="another-id", text="  ACME   PIPELINE "),
        limit=5,
    )

    assert cached == [result]
    assert isinstance(cached[0], SearchResult)


async def test_search_result_cache_treats_invalid_payload_as_miss() -> None:
    backend = MemoryTTLCache()
    cache = SearchResultCache(backend, ttl_seconds=60)
    query = _query()
    key = cache.key_for(query, limit=5)
    await backend.set("search-results-v1", key, [{"title": "missing URL"}], ttl_seconds=60)

    assert await cache.get(query, limit=5) is None
    assert await backend.get("search-results-v1", key) is None


async def test_page_content_expiry_invalidates_search_hit() -> None:
    timer = MutableTimer()
    backend = MemoryTTLCache(timer=timer)
    cache = SearchResultCache(backend, ttl_seconds=60, page_ttl_seconds=10)
    query = _query()
    await cache.set(
        query,
        [SearchResult(title="Acme", url="https://example.com/page", text="Evidence")],
        limit=5,
    )

    assert await cache.get(query, limit=5) is not None
    timer.value += 10
    assert await cache.get(query, limit=5) is None
