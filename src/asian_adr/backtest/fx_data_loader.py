"""FX rate loader — streams ``FXRateEvent``s from the Datastream FX Parquet cache.

Reads ``fx_rates.parquet`` (``date``, ``base_currency``, ``quote_currency``,
``currency_pair``, ``mid``, ``provider``); ``mid`` is already USD per 1 unit of
the base currency (inverted on fetch). A synthetic intraday hour of 00:00 UTC
orders FX before the same-day price bars, so a fresh rate is cached before any
spread is computed.

Uses pandas (backtest layer dependency).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from asian_adr.core import FXRateEvent

from .data_loader import _read_parquet

logger = logging.getLogger(__name__)

__all__ = ["FXDataLoader", "FX_BAR_HOUR"]

FX_BAR_HOUR = 0  # UTC hour assigned to FX rates (before the day's bars)


class FXDataLoader:
    """Streams daily FX close rates from a Parquet cache."""

    def __init__(self, path: str | Path, *, hour: int = FX_BAR_HOUR) -> None:
        self._path = Path(path)
        self._hour = hour

    def stream(
        self, start: date, end: date, currencies: Iterable[str]
    ) -> Iterator[FXRateEvent]:
        df = _read_parquet(self._path)
        df = df[df["base_currency"].isin(set(currencies))].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        df = df.loc[mask].sort_values("date")
        ts_time = time(self._hour, 0, tzinfo=timezone.utc)
        for row in df.itertuples(index=False):
            if pd.isna(row.mid):
                continue  # skip missing FX days
            ts = datetime.combine(row.date.date(), ts_time)
            yield FXRateEvent(
                timestamp_exchange=ts,
                timestamp_received=ts,
                currency_pair=str(getattr(row, "currency_pair", f"{row.base_currency}_USD")),
                base_currency=str(row.base_currency),
                quote_currency=str(getattr(row, "quote_currency", "USD")),
                mid=Decimal(str(row.mid)),
                provider=str(getattr(row, "provider", "datastream")),
            )
