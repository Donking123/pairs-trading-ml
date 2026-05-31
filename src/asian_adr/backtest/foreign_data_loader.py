"""Asian underlying price loader — streams ``BarEvent``s from the global Parquet cache.

Reads ``global_prices.parquet`` (``ticker``, ``marketdate``, OHLCV,
``exchange``, ``currency``) and yields one :class:`BarEvent` per row in
chronological order, carrying the native exchange and currency so the position
engine and foreign exchange can convert to USD. A synthetic intraday hour
(default 08:00 UTC ≈ Asian session) orders these bars before the same-day U.S.
bars.

Uses pandas (backtest layer dependency).
"""

from __future__ import annotations

import logging
from datetime import date, time, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from asian_adr.core import BarEvent

from .data_loader import _build_bar, _read_parquet

logger = logging.getLogger(__name__)

__all__ = ["GlobalDataLoader", "FOREIGN_BAR_HOUR"]

FOREIGN_BAR_HOUR = 8  # UTC hour assigned to Asian session bars


class GlobalDataLoader:
    """Streams Asian underlying daily bars from a Parquet cache."""

    def __init__(self, path: str | Path, *, hour: int = FOREIGN_BAR_HOUR) -> None:
        self._path = Path(path)
        self._hour = hour

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
            # exchange/currency=None → read them from the row.
            bar = _build_bar(row, ts_time, exchange=None, currency=None)
            if bar is not None:
                yield bar
