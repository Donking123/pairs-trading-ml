"""PostgreSQL store: pair registry, orders, fills, positions, audit log.

Provides a :class:`RelationalStore` protocol with two implementations:

* :class:`InMemoryRelationalStore` — dict/list-backed, fully functional for
  tests and local dev (and the default behind :class:`PersistenceRecorder`).
* :class:`PostgresStore` — real PostgreSQL 16 backend via ``asyncpg`` (imported
  lazily, so this module is import-safe without the driver). Events are stored
  as their pydantic JSON in a ``payload`` column alongside indexed key columns.

:class:`PersistenceRecorder` subscribes to the bus and writes orders, fills, and
positions (plus an optional audit trail) — the durable counterpart to the
in-memory engines, supporting the §1.6 state-recovery sequence via the
``load_*`` methods.

Imports only ``core`` and ``event_bus``.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, Protocol, runtime_checkable

from asian_adr.core import (
    AsianADRApprovedPair,
    BaseEvent,
    FillEvent,
    OrderRequest,
    PositionUpdateEvent,
)
from asian_adr.event_bus import AbstractEventBus, Topic

logger = logging.getLogger(__name__)

__all__ = [
    "RelationalStore",
    "InMemoryRelationalStore",
    "PostgresStore",
    "PersistenceRecorder",
    "SCHEMA_DDL",
]

# Reference DDL (also the basis for Alembic migrations, Phase 4).
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS pairs (
    pair_id TEXT PRIMARY KEY,
    adr_ticker TEXT NOT NULL,
    underlying_ticker TEXT NOT NULL,
    approved_date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    is_active BOOLEAN NOT NULL,
    payload JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    event_id UUID PRIMARY KEY,
    pair_id TEXT, ticker TEXT, side TEXT,
    ts_exchange TIMESTAMPTZ, payload JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    client_order_id UUID, pair_id TEXT, ticker TEXT, side TEXT,
    ts_exchange TIMESTAMPTZ, payload JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    pair_id TEXT, ticker TEXT, net_quantity NUMERIC,
    ts_exchange TIMESTAMPTZ, payload JSONB NOT NULL,
    PRIMARY KEY (pair_id, ticker)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL, event_type TEXT NOT NULL, payload JSONB NOT NULL
);
"""


@runtime_checkable
class RelationalStore(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def save_pair(self, pair: AsianADRApprovedPair) -> None: ...
    async def load_active_pairs(self, as_of: date) -> list[AsianADRApprovedPair]: ...
    async def save_order(self, order: OrderRequest) -> None: ...
    async def save_fill(self, fill: FillEvent) -> None: ...
    async def save_position(self, position: PositionUpdateEvent) -> None: ...
    async def load_open_positions(self) -> list[PositionUpdateEvent]: ...
    async def append_audit(self, event: BaseEvent) -> None: ...


class InMemoryRelationalStore:
    """Dict/list-backed relational store (tests, local dev, default recorder sink)."""

    def __init__(self) -> None:
        self._pairs: dict[str, AsianADRApprovedPair] = {}
        self.orders: list[OrderRequest] = []
        self.fills: list[FillEvent] = []
        self._positions: dict[tuple[str, str], PositionUpdateEvent] = {}
        self.audit: list[BaseEvent] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def save_pair(self, pair: AsianADRApprovedPair) -> None:
        self._pairs[pair.pair_id] = pair

    async def load_active_pairs(self, as_of: date) -> list[AsianADRApprovedPair]:
        return [p for p in self._pairs.values() if p.is_active_as_of(as_of)]

    async def save_order(self, order: OrderRequest) -> None:
        self.orders.append(order)

    async def save_fill(self, fill: FillEvent) -> None:
        self.fills.append(fill)

    async def save_position(self, position: PositionUpdateEvent) -> None:
        self._positions[(position.pair_id, position.ticker)] = position

    async def load_open_positions(self) -> list[PositionUpdateEvent]:
        return [p for p in self._positions.values() if p.net_quantity != 0]

    async def append_audit(self, event: BaseEvent) -> None:
        self.audit.append(event)


class PostgresStore:
    """PostgreSQL 16 backend via ``asyncpg`` (lazy import)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None

    async def connect(self) -> None:  # pragma: no cover - requires postgres
        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError("PostgresStore requires 'asyncpg' (uv add asyncpg).") from exc
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_DDL)

    async def close(self) -> None:  # pragma: no cover - requires postgres
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def save_pair(self, pair: AsianADRApprovedPair) -> None:  # pragma: no cover
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO pairs (pair_id, adr_ticker, underlying_ticker,
                       approved_date, expiry_date, is_active, payload)
                   VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb)
                   ON CONFLICT (pair_id) DO UPDATE SET payload=EXCLUDED.payload,
                       is_active=EXCLUDED.is_active""",
                pair.pair_id, pair.adr_ticker, pair.underlying_ticker,
                pair.approved_date, pair.expiry_date, pair.is_active,
                pair.model_dump_json(),
            )

    async def load_active_pairs(self, as_of: date) -> list[AsianADRApprovedPair]:  # pragma: no cover
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT payload FROM pairs WHERE is_active
                       AND approved_date <= $1 AND expiry_date >= $1""",
                as_of,
            )
        return [AsianADRApprovedPair.model_validate_json(r["payload"]) for r in rows]

    async def save_order(self, order: OrderRequest) -> None:  # pragma: no cover
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO orders (event_id, pair_id, ticker, side, ts_exchange, payload)
                   VALUES ($1,$2,$3,$4,$5,$6::jsonb) ON CONFLICT (event_id) DO NOTHING""",
                order.event_id, order.pair_id, order.ticker, order.side,
                order.timestamp_exchange, order.model_dump_json(),
            )

    async def save_fill(self, fill: FillEvent) -> None:  # pragma: no cover
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO fills (fill_id, client_order_id, pair_id, ticker, side,
                       ts_exchange, payload) VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb)
                   ON CONFLICT (fill_id) DO NOTHING""",
                fill.fill_id, fill.client_order_id, fill.pair_id, fill.ticker,
                fill.side, fill.timestamp_exchange, fill.model_dump_json(),
            )

    async def save_position(self, position: PositionUpdateEvent) -> None:  # pragma: no cover
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO positions (pair_id, ticker, net_quantity, ts_exchange, payload)
                   VALUES ($1,$2,$3,$4,$5::jsonb)
                   ON CONFLICT (pair_id, ticker) DO UPDATE SET
                       net_quantity=EXCLUDED.net_quantity, payload=EXCLUDED.payload,
                       ts_exchange=EXCLUDED.ts_exchange""",
                position.pair_id, position.ticker, position.net_quantity,
                position.timestamp_exchange, position.model_dump_json(),
            )

    async def load_open_positions(self) -> list[PositionUpdateEvent]:  # pragma: no cover
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT payload FROM positions WHERE net_quantity <> 0")
        return [PositionUpdateEvent.model_validate_json(r["payload"]) for r in rows]

    async def append_audit(self, event: BaseEvent) -> None:  # pragma: no cover
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO audit_log (ts, event_type, payload) VALUES ($1,$2,$3::jsonb)",
                event.timestamp_received, event.event_type, event.model_dump_json(),
            )


class PersistenceRecorder:
    """Subscribes to the bus and durably records events to a relational store."""

    def __init__(
        self,
        store: RelationalStore,
        *,
        audit: bool = True,
        audit_topics: Iterable[str] = (
            Topic.ORDERS, Topic.FILLS, Topic.RISK_DECISIONS,
            Topic.POSITIONS, Topic.ALERTS,
        ),
    ) -> None:
        self._store = store
        self._audit = audit
        self._audit_topics = tuple(audit_topics)

    async def attach(self, bus: AbstractEventBus) -> None:
        await bus.subscribe(Topic.ORDERS, self._on_order)
        await bus.subscribe(Topic.FILLS, self._on_fill)
        await bus.subscribe(Topic.POSITIONS, self._on_position)
        if self._audit:
            for topic in self._audit_topics:
                await bus.subscribe(topic, self._on_audit)

    async def _on_order(self, event: OrderRequest) -> None:
        await self._store.save_order(event)

    async def _on_fill(self, event: FillEvent) -> None:
        await self._store.save_fill(event)

    async def _on_position(self, event: PositionUpdateEvent) -> None:
        await self._store.save_position(event)

    async def _on_audit(self, event: BaseEvent) -> None:
        await self._store.append_audit(event)
