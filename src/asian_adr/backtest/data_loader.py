"""U.S. ADR price loader — streams ``BarEvent``s from the Datastream Parquet cache.

Reads ``adr_prices.parquet`` (``ticker``, ``marketdate``, OHLCV, ...), filters
to the requested tickers and date range, and yields one :class:`BarEvent` per
row in chronological order. A synthetic intraday hour (default 21:00 UTC ≈ U.S.
close) orders ADR bars after the same-day Asian and FX bars in the merged
stream, so the engine processes FX → Asia → U.S. within each calendar day.

Uses pandas (the backtest layer may depend on it; ``core`` may not).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from asian_adr.core import BarEvent

logger = logging.getLogger(__name__)

__all__ = ["ADRDataLoader", "US_BAR_HOUR"]

US_BAR_HOUR = 21  # UTC hour assigned to U.S. close bars


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except OSError:  # pragma: no cover - engine fallback
        return pd.read_parquet(path, engine="fastparquet")


def _build_bar(row, ts_time, *, exchange: str, currency: str) -> BarEvent | None:
    """Build a BarEvent from a price row, skipping rows with no valid close.

    Real Datastream rows can carry NaN OHLC for non-trading or missing days;
    such rows are dropped (no spread can be computed from them). When the close
    is valid but an O/H/L field is NaN, it falls back to the close.
    """
    close = row.close
    if pd.isna(close):
        return None
    c = Decimal(str(close))
    o = Decimal(str(row.open)) if not pd.isna(row.open) else c
    h = Decimal(str(row.high)) if not pd.isna(row.high) else c
    low = Decimal(str(row.low)) if not pd.isna(row.low) else c
    exch = exchange if exchange is not None else str(getattr(row, "exchange", ""))
    ccy = currency if currency is not None else str(getattr(row, "currency", ""))
    ts = datetime.combine(row.marketdate.date(), ts_time)
    return BarEvent(
        timestamp_exchange=ts,
        timestamp_received=ts,
        ticker=str(row.ticker),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=int(row.volume) if not pd.isna(row.volume) else 0,
        is_adjusted=True,
        exchange=exch,
        currency=ccy,
    )


class ADRDataLoader:
    """Streams U.S. ADR daily bars from a Parquet cache."""

    def __init__(self, path: str | Path, *, hour: int = US_BAR_HOUR, exchange: str = "US") -> None:
        self._path = Path(path)
        self._hour = hour
        self._exchange = exchange

    def stream(
        self, start: date, end: date, tickers: Iterable[str]
    ) -> Iterator[BarEvent]:
        df = _read_parquet(self._path)
        df = df[df["ticker"].isin(set(tickers))].copy()
        df["marketdate"] = pd.to_datetime(df["marketdate"]).dt.normalize()
        mask = (df["marketdate"] >= pd.Timestamp(start)) & (
            df["marketdate"] <= pd.Timestamp(end)
        )
        df = df.loc[mask].sort_values("marketdate")
        ts_time = time(self._hour, 0, tzinfo=timezone.utc)
        for row in df.itertuples(index=False):
            bar = _build_bar(
                row, ts_time, exchange=self._exchange, currency="USD"
            )
            if bar is not None:
                yield bar
