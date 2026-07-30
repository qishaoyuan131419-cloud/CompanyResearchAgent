import asyncio
import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float


class MemoryTTLCache:
    """Process-local TTL cache with bounded storage and mutation isolation."""

    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._timer = timer
        self._entries: dict[tuple[str, str], _CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def get(self, namespace: str, key: str) -> Any | None:
        cache_key = (namespace, key)
        async with self._lock:
            self._ensure_open()
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            if entry.expires_at <= self._timer():
                self._entries.pop(cache_key, None)
                return None
            return copy.deepcopy(entry.value)

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int,
    ) -> None:
        cache_key = (namespace, key)
        async with self._lock:
            self._ensure_open()
            if ttl_seconds <= 0:
                self._entries.pop(cache_key, None)
                return

            now = self._timer()
            self._remove_expired(now)
            if cache_key not in self._entries and len(self._entries) >= self._max_entries:
                oldest_key = min(
                    self._entries,
                    key=lambda candidate: self._entries[candidate].expires_at,
                )
                self._entries.pop(oldest_key, None)
            self._entries[cache_key] = _CacheEntry(
                value=copy.deepcopy(value),
                expires_at=now + ttl_seconds,
            )

    async def close(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._closed = True

    def _remove_expired(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("cache is closed")
