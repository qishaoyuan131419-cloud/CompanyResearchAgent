import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class SQLiteTTLCache:
    """JSON-only persistent TTL cache backed by SQLite.

    A single connection is serialized behind an async lock and all blocking SQLite
    operations run in a worker thread. Cache data is treated as untrusted: malformed
    JSON is evicted and returned as a miss rather than deserialized as executable data.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        timer: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._timer = timer
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def get(self, namespace: str, key: str) -> Any | None:
        async with self._lock:
            connection = await self._connection_or_raise()
            return await asyncio.to_thread(
                self._get_sync,
                connection,
                namespace,
                key,
                self._timer(),
            )

    async def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            async with self._lock:
                connection = await self._connection_or_raise()
                await asyncio.to_thread(self._delete_sync, connection, namespace, key)
            return
        serialized = json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._lock:
            connection = await self._connection_or_raise()
            await asyncio.to_thread(
                self._set_sync,
                connection,
                namespace,
                key,
                serialized,
                self._timer() + ttl_seconds,
            )

    async def purge_expired(self) -> int:
        async with self._lock:
            connection = await self._connection_or_raise()
            return await asyncio.to_thread(self._purge_sync, connection, self._timer())

    async def close(self) -> None:
        async with self._lock:
            if self._connection is not None:
                await asyncio.to_thread(self._connection.close)
                self._connection = None
            self._closed = True

    async def _connection_or_raise(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("cache is closed")
        if self._connection is None:
            self._connection = await asyncio.to_thread(self._open_sync)
        return self._connection

    def _open_sync(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path,
            timeout=5.0,
            check_same_thread=False,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (namespace, cache_key)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries(expires_at)"
        )
        connection.commit()
        return connection

    @staticmethod
    def _get_sync(
        connection: sqlite3.Connection,
        namespace: str,
        key: str,
        now: float,
    ) -> Any | None:
        row = connection.execute(
            "SELECT value_json, expires_at FROM cache_entries "
            "WHERE namespace = ? AND cache_key = ?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        value_json, expires_at = row
        if float(expires_at) <= now:
            SQLiteTTLCache._delete_sync(connection, namespace, key)
            return None
        try:
            return json.loads(str(value_json))
        except (json.JSONDecodeError, TypeError, ValueError):
            SQLiteTTLCache._delete_sync(connection, namespace, key)
            return None

    @staticmethod
    def _set_sync(
        connection: sqlite3.Connection,
        namespace: str,
        key: str,
        value_json: str,
        expires_at: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO cache_entries(namespace, cache_key, value_json, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, cache_key) DO UPDATE SET
                value_json = excluded.value_json,
                expires_at = excluded.expires_at
            """,
            (namespace, key, value_json, expires_at),
        )
        connection.commit()

    @staticmethod
    def _delete_sync(connection: sqlite3.Connection, namespace: str, key: str) -> None:
        connection.execute(
            "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
            (namespace, key),
        )
        connection.commit()

    @staticmethod
    def _purge_sync(connection: sqlite3.Connection, now: float) -> int:
        cursor = connection.execute(
            "DELETE FROM cache_entries WHERE expires_at <= ?",
            (now,),
        )
        connection.commit()
        return max(cursor.rowcount, 0)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"cache value of type {type(value).__name__} is not JSON serializable")
