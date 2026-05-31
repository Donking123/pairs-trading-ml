"""TimescaleDB store for OHLCV + FX time-series.

A :class:`TimeSeriesStore` protocol with an in-memory implementation (tested)
and a :class:`TimescaleStore` hypertable backend via ``asyncpg`` (lazy import).
:meth:`last_bars` supports the §1.6 recovery step "rebuild rolling spread stats
from the last T bars".

Imports only ``core``.
"""

from __future__ import annotations

import bisect
import logging
from datetime import datetime
from typing import Iterable, Protocol, runtime_checkable

from asian_adr.core import BarEvent, FXRateEvent

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesStore", "InMemoryTimeSeriesStore", "TimescaleStore", "HYPERTABLE_DDL"]

HYPERTABLE_DDL = """
CREATE TABLE IF NOT EXISTS bars (
    ticker TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
    volume BIGINT, exchange TEXT, currency TEXT,
    PRIMARY KEY (ticker, ts)
);
SELECT create_hypertable('bars', 'ts', if_not_exists => TRUE);
CREATE TABLE IF NOT EXISTS fx_rates (
    base_currency TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL,
    mid NUMERIC, provider TEXT, is_stale BOOLEAN,
    PRIMARY KEY (base_currency, ts)
);
SELECT create_hypertable('fx_rates', 'ts', if_not_exists => TRUE);
"""


@runtime_checkable
class TimeSeriesStore(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def save_bar(self, bar: BarEvent) -> None: ...
    async def save_bars(self, bars: Iterable[BarEvent]) -> None: ...
    async def load_bars(self, ticker: str, start: datetime, end: datetime) -> list[BarEvent]: ...
    async def last_bars(self, ticker: str, n: int) -> list[BarEvent]: ...
    async def save_fx(self, event: FXRateEvent) -> None: ...
    async def load_fx(self, base_currency: str, start: datetime, end: datetime) -> list[FXRateEvent]: ...


class InMemoryTimeSeriesStore:
    """Sorted-in-memory OHLCV/FX store (tests, local dev)."""

    def __init__(self) -> None:
        self._bars: dict[str, list[BarEvent]] = {}
        self._bar_ts: dict[str, list[datetime]] = {}
        self._fx: dict[str, list[FXRateEvent]] = {}
        self._fx_ts: dict[str, list[datetime]] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def save_bar(self, bar: BarEvent) -> None:
        keys = self._bar_ts.setdefault(bar.ticker, [])
        vals = self._bars.setdefault(bar.ticker, [])
        i = bisect.bisect_left(keys, bar.timestamp_exchange)
        if i < len(keys) and keys[i] == bar.timestamp_exchange:
            vals[i] = bar  # upsert same timestamp
        else:
            keys.insert(i, bar.timestamp_exchange)
            vals.insert(i, bar)

    async def save_bars(self, bars: Iterable[BarEvent]) -> None:
        for bar in bars:
            await self.save_bar(bar)

    async def load_bars(self, ticker: str, start: datetime, end: datetime) -> list[BarEvent]:
        keys = self._bar_ts.get(ticker, [])
        vals = self._bars.get(ticker, [])
        lo = bisect.bisect_left(keys, start)
        hi = bisect.bisect_right(keys, end)
        return vals[lo:hi]

    async def last_bars(self, ticker: str, n: int) -> list[BarEvent]:
        return self._bars.get(ticker, [])[-n:]

    async def save_fx(self, event: FXRateEvent) -> None:
        keys = self._fx_ts.setdefault(event.base_currency, [])
        vals = self._fx.setdefault(event.base_currency, [])
        i = bisect.bisect_left(keys, event.timestamp_exchange)
        if i < len(keys) and keys[i] == event.timestamp_exchange:
            vals[i] = event
        else:
            keys.insert(i, event.timestamp_exchange)
            vals.insert(i, event)

    async def load_fx(self, base_currency: str, start: datetime, end: datetime) -> list[FXRateEvent]:
        keys = self._fx_ts.get(base_currency, [])
        vals = self._fx.get(base_currency, [])
        lo = bisect.bisect_left(keys, start)
        hi = bisect.bisect_right(keys, end)
        return vals[lo:hi]


class TimescaleStore:
    """TimescaleDB hypertable backend via ``asyncpg`` (lazy import)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None

    async def connect(self) -> None:  # pragma: no cover - requires timescale
        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError("TimescaleStore requires 'asyncpg' (uv add asyncpg).") from exc
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(HYPERTABLE_DDL)

    async def close(self) -> None:  # pragma: no cover
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def save_bar(self, bar: BarEvent) -> None:  # pragma: no cover
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO bars (ticker, ts, open, high, low, close, volume, exchange, currency)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                   ON CONFLICT (ticker, ts) DO UPDATE SET close=EXCLUDED.close""",
                bar.ticker, bar.timestamp_exchange, bar.open, bar.high, bar.low,
                bar.close, bar.volume, bar.exchange, bar.currency,
            )

    async def save_bars(self, bars: Iterable[BarEvent]) -> None:  # pragma: no cover
        for bar in bars:
            await self.save_bar(bar)

    async def load_bars(self, ticker, start, end):  # pragma: no cover
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM bars WHERE ticker=$1 AND ts BETWEEN $2 AND $3 ORDER BY ts",
                ticker, start, end,
            )
        return [self._row_to_bar(r) for r in rows]

    async def last_bars(self, ticker, n):  # pragma: no cover
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM bars WHERE ticker=$1 ORDER BY ts DESC LIMIT $2", ticker, n
            )
        return [self._row_to_bar(r) for r in reversed(rows)]

    async def save_fx(self, event: FXRateEvent) -> None:  # pragma: no cover
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO fx_rates (base_currency, ts, mid, provider, is_stale)
                   VALUES ($1,$2,$3,$4,$5) ON CONFLICT (base_currency, ts)
                   DO UPDATE SET mid=EXCLUDED.mid""",
                event.base_currency, event.timestamp_exchange, event.mid,
                event.provider, event.is_stale,
            )

    async def load_fx(self, base_currency, start, end):  # pragma: no cover
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM fx_rates WHERE base_currency=$1 AND ts BETWEEN $2 AND $3
                   ORDER BY ts""",
                base_currency, start, end,
            )
        return [self._row_to_fx(r) for r in rows]

    @staticmethod
    def _row_to_bar(r) -> BarEvent:  # pragma: no cover
        return BarEvent(
            timestamp_exchange=r["ts"], timestamp_received=r["ts"], ticker=r["ticker"],
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], is_adjusted=True, exchange=r["exchange"], currency=r["currency"],
        )

    @staticmethod
    def _row_to_fx(r) -> FXRateEvent:  # pragma: no cover
        return FXRateEvent(
            timestamp_exchange=r["ts"], timestamp_received=r["ts"],
            currency_pair=f"{r['base_currency']}_USD", base_currency=r["base_currency"],
            quote_currency="USD", mid=r["mid"], provider=r["provider"], is_stale=r["is_stale"],
        )
