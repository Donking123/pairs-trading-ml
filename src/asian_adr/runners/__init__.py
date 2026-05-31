"""Runners: the executable entry points that wire the system together.

* ``backtest_runner`` — CLI driving ``BacktestEngine`` over the Parquet cache.
* ``live_runner`` — wires the production components on an ``AsyncioBus`` with
  injected feed/gateway producers (or a ``ReplayProducer`` for paper mode).
* ``data_recorder`` — record a live session to JSONL and replay it.

These are the top of the dependency graph: they orchestrate every other layer.
"""

from __future__ import annotations

from .config import RunnerConfig
from .data_recorder import DataRecorder, ReplayProducer
from .live_runner import LiveRunner

__all__ = ["RunnerConfig", "LiveRunner", "DataRecorder", "ReplayProducer"]
