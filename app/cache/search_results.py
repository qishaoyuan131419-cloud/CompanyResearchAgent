from pydantic import TypeAdapter, ValidationError

from app.core.protocols import AsyncCache
from app.schemas.planning import SearchQuery
from app.schemas.search import SearchResult
from app.utils.hashing import stable_hash
from app.utils.text import normalized_fingerprint_text
from app.utils.urls import canonicalize_url

_SEARCH_RESULTS_ADAPTER = TypeAdapter(list[SearchResult])


class SearchResultCache:
    """Typed cache facade for provider search results."""

    def __init__(
        self,
        cache: AsyncCache,
        *,
        ttl_seconds: int,
        page_ttl_seconds: int | None = None,
        provider: str = "exa-mcp",
        namespace: str = "search-results-v1",
        page_namespace: str = "page-contents-v1",
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds cannot be negative")
        if page_ttl_seconds is not None and page_ttl_seconds < 0:
            raise ValueError("page_ttl_seconds cannot be negative")
        self._cache = cache
        self._ttl_seconds = ttl_seconds
        self._page_ttl_seconds = ttl_seconds if page_ttl_seconds is None else page_ttl_seconds
        self._provider = provider
        self._namespace = namespace
        self._page_namespace = page_namespace

    def key_for(self, query: SearchQuery, *, limit: int) -> str:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return stable_hash(
            {
                "version": 1,
                "provider": self._provider,
                "query": normalized_fingerprint_text(query.query),
                "language": query.language.casefold(),
                "source_preference": sorted(
                    {preference.casefold() for preference in query.source_preference}
                ),
                "limit": limit,
            },
            prefix="search_",
        )

    async def get(self, query: SearchQuery, *, limit: int) -> list[SearchResult] | None:
        if self._ttl_seconds == 0:
            return None
        key = self.key_for(query, limit=limit)
        value = await self._cache.get(self._namespace, key)
        if value is None:
            return None
        if isinstance(value, dict):
            return await self._restore_results_with_pages(value)
        try:
            return _SEARCH_RESULTS_ADAPTER.validate_python(value)
        except (TypeError, ValidationError, ValueError):
            await self._cache.set(self._namespace, key, None, ttl_seconds=0)
            return None

    async def set(
        self,
        query: SearchQuery,
        results: list[SearchResult],
        *,
        limit: int,
    ) -> None:
        if self._ttl_seconds == 0:
            return
        key = self.key_for(query, limit=limit)
        metadata: list[dict[str, object]] = []
        page_keys: list[str] = []
        for result in results:
            page_key = self._page_key(result)
            page_keys.append(page_key)
            metadata.append(result.model_copy(update={"text": ""}).model_dump(mode="json"))
            await self._cache.set(
                self._page_namespace,
                page_key,
                {"text": result.text},
                ttl_seconds=self._page_ttl_seconds,
            )
        value = {"version": 2, "results": metadata, "page_keys": page_keys}
        await self._cache.set(
            self._namespace,
            key,
            value,
            ttl_seconds=self._ttl_seconds,
        )

    async def _restore_results_with_pages(
        self,
        value: dict[object, object],
    ) -> list[SearchResult] | None:
        metadata = value.get("results")
        page_keys = value.get("page_keys")
        if not isinstance(page_keys, list) or not all(
            isinstance(page_key, str) for page_key in page_keys
        ):
            return None
        try:
            results = _SEARCH_RESULTS_ADAPTER.validate_python(metadata)
        except (TypeError, ValidationError, ValueError):
            return None
        if len(results) != len(page_keys):
            return None
        restored: list[SearchResult] = []
        for result, page_key in zip(results, page_keys, strict=True):
            page = await self._cache.get(self._page_namespace, page_key)
            if not isinstance(page, dict) or not isinstance(page.get("text"), str):
                return None
            restored.append(result.model_copy(update={"text": page["text"]}))
        return restored

    def _page_key(self, result: SearchResult) -> str:
        return stable_hash(
            {
                "version": 1,
                "provider": self._provider,
                "url": canonicalize_url(str(result.url)),
            },
            prefix="page_",
            length=40,
        )

    async def close(self) -> None:
        await self._cache.close()
