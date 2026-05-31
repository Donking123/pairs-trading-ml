"""Redis hot-state cache: latest spreads, z-scores, rolling stats, positions.

A :class:`HotCache` protocol with an in-memory implementation (tested) and a
:class:`RedisHotCache` backend via ``redis.asyncio`` (lazy import). Holds the
volatile state that is rebuilt on restart (rolling stats) or reloaded from
PostgreSQL (positions) — never the source of truth, just low-latency access.

Imports only the standard library.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = ["HotCache", "InMemoryHotCache", "RedisHotCache"]


@runtime_checkable
class HotCache(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def set_spread(self, pair_id: str, spread: Decimal, z_score: Decimal) -> None: ...
    async def get_spread(self, pair_id: str) -> tuple[Decimal, Decimal] | None: ...
    async def set_rolling_stats(self, pair_id: str, mean: Decimal, std: Decimal, count: int) -> None: ...
    async def get_rolling_stats(self, pair_id: str) -> dict | None: ...
    async def set_position(self, pair_id: str, snapshot: dict) -> None: ...
    async def get_position(self, pair_id: str) -> dict | None: ...


class InMemoryHotCache:
    """Dict-backed hot cache (tests, local dev)."""

    def __init__(self) -> None:
        self._spread: dict[str, tuple[Decimal, Decimal]] = {}
        self._stats: dict[str, dict] = {}
        self._position: dict[str, dict] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def set_spread(self, pair_id: str, spread: Decimal, z_score: Decimal) -> None:
        self._spread[pair_id] = (spread, z_score)

    async def get_spread(self, pair_id: str) -> tuple[Decimal, Decimal] | None:
        return self._spread.get(pair_id)

    async def set_rolling_stats(self, pair_id: str, mean: Decimal, std: Decimal, count: int) -> None:
        self._stats[pair_id] = {"mean": mean, "std": std, "count": count}

    async def get_rolling_stats(self, pair_id: str) -> dict | None:
        return self._stats.get(pair_id)

    async def set_position(self, pair_id: str, snapshot: dict) -> None:
        self._position[pair_id] = dict(snapshot)

    async def get_position(self, pair_id: str) -> dict | None:
        return self._position.get(pair_id)


class RedisHotCache:
    """Redis 7 backend via ``redis.asyncio`` (lazy import). Values are JSON."""

    def __init__(self, url: str = "redis://localhost:6379/0", *, prefix: str = "adr:") -> None:
        self._url = url
        self._prefix = prefix
        self._redis = None

    async def connect(self) -> None:  # pragma: no cover - requires redis
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise RuntimeError("RedisHotCache requires 'redis' (uv add redis).") from exc
        self._redis = aioredis.from_url(self._url, decode_responses=True)

    async def close(self) -> None:  # pragma: no cover
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def _key(self, kind: str, pair_id: str) -> str:
        return f"{self._prefix}{kind}:{pair_id}"

    async def set_spread(self, pair_id, spread, z_score) -> None:  # pragma: no cover
        await self._redis.set(
            self._key("spread", pair_id),
            json.dumps({"spread": str(spread), "z": str(z_score)}),
        )

    async def get_spread(self, pair_id):  # pragma: no cover
        raw = await self._redis.get(self._key("spread", pair_id))
        if raw is None:
            return None
        d = json.loads(raw)
        return Decimal(d["spread"]), Decimal(d["z"])

    async def set_rolling_stats(self, pair_id, mean, std, count) -> None:  # pragma: no cover
        await self._redis.set(
            self._key("stats", pair_id),
            json.dumps({"mean": str(mean), "std": str(std), "count": count}),
        )

    async def get_rolling_stats(self, pair_id):  # pragma: no cover
        raw = await self._redis.get(self._key("stats", pair_id))
        return json.loads(raw) if raw else None

    async def set_position(self, pair_id, snapshot: dict) -> None:  # pragma: no cover
        await self._redis.set(
            self._key("pos", pair_id), json.dumps(snapshot, default=str)
        )

    async def get_position(self, pair_id):  # pragma: no cover
        raw = await self._redis.get(self._key("pos", pair_id))
        return json.loads(raw) if raw else None
