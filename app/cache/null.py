from typing import Any


class NullCache:
    """An ``AsyncCache`` implementation that deliberately never stores values."""

    async def get(self, namespace: str, key: str) -> Any | None:
        del namespace, key
        return None

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int,
    ) -> None:
        del namespace, key, value, ttl_seconds

    async def close(self) -> None:
        return None
