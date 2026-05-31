"""Backtest clock.

The simulated clock already lives in ``core`` (so engines can depend on it
without reaching into the backtest layer). This module re-exports it under the
name the architecture's project structure uses, plus a ``BacktestClock`` alias.
"""

from __future__ import annotations

from asian_adr.core import SimulatedClock

BacktestClock = SimulatedClock

__all__ = ["SimulatedClock", "BacktestClock"]
