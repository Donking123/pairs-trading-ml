"""Persistence layer: relational, time-series, hot-cache, and object stores.

Each store exposes a Protocol with an in-memory / local implementation
(functional for tests and dev) and a real backend imported lazily:

* :mod:`postgres`   — pair registry, orders, fills, positions, audit log
                      (PostgreSQL 16) + :class:`PersistenceRecorder` (bus → DB).
* :mod:`timescale`  — OHLCV + FX time-series (TimescaleDB hypertables).
* :mod:`redis_cache`— hot state: spreads, z-scores, rolling stats, positions.
* :mod:`s3_store`   — Parquet archive + backtest results (S3 / MinIO).

Supports the §1.6 state-recovery sequence (load active pairs, reload open
positions, rebuild rolling stats from the last T bars). Imports only ``core``
and ``event_bus``; DB/SDK drivers are lazy.
"""

from __future__ import annotations

from .postgres import (
    InMemoryRelationalStore,
    PersistenceRecorder,
    PostgresStore,
    RelationalStore,
    SCHEMA_DDL,
)
from .redis_cache import HotCache, InMemoryHotCache, RedisHotCache
from .s3_store import LocalObjectStore, ObjectStore, S3Store
from .timescale import (
    HYPERTABLE_DDL,
    InMemoryTimeSeriesStore,
    TimescaleStore,
    TimeSeriesStore,
)

__all__ = [
    # relational
    "RelationalStore",
    "InMemoryRelationalStore",
    "PostgresStore",
    "PersistenceRecorder",
    "SCHEMA_DDL",
    # time-series
    "TimeSeriesStore",
    "InMemoryTimeSeriesStore",
    "TimescaleStore",
    "HYPERTABLE_DDL",
    # hot cache
    "HotCache",
    "InMemoryHotCache",
    "RedisHotCache",
    # object store
    "ObjectStore",
    "LocalObjectStore",
    "S3Store",
]
