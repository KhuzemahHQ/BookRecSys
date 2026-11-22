"""Persistent thread-safe SQLite cache for online API and search lookups."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from bookrecsys.config import ONLINE_CACHE_DB, ensure_data_dirs


class OnlineCache:
    """Lightweight SQLite key-value JSON cache with TTL expiration."""

    def __init__(self, db_path: Path = ONLINE_CACHE_DB, ttl_seconds: int = 7 * 86400) -> None:
        ensure_data_dirs()
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM kv_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            payload, created_at = row
            if now - float(created_at) > self.ttl_seconds:
                return None
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None

    def set(self, key: str, value: Any) -> None:
        serialized = json.dumps(value, ensure_ascii=False)
        now = time.time()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO kv_cache (cache_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (key, serialized, now),
            )
            conn.commit()
