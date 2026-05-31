"""Backtest layer: deterministic multi-feed replay over the real components.

The :class:`BacktestEngine` drives the production ``HongSusmelEngine``,
``RiskEngine``, ``PositionEngine``, and ``AsianExecutionSequencer`` through a
``MemoryBus``, with simulated exchanges, cost/slippage models, a point-in-time
registry loader, the ROCE/RUCE calculator, and an HTML tearsheet — so a run that
reproduces the paper validates the same code that trades live.
"""

from __future__ import annotations

from .clock import BacktestClock, SimulatedClock
from .cost_model import Costs, EquitiesCostModel
from .data_loader import ADRDataLoader
from .engine import BacktestConfig, BacktestEngine
from .foreign_data_loader import GlobalDataLoader
from .fx_data_loader import FXDataLoader
from .pair_registry_loader import PairRegistryLoader, PairRegistryUpdateEvent
from .roce_ruce_calculator import RoceRuceCalculator, distribution_stats
from .report import build_distributions, build_summary, write_report
from .simulated_foreign_exchange import SimulatedForeignExchange
from .simulated_us_exchange import SimulatedUSExchange
from .slippage_models import (
    HalfSpreadSlippageModel,
    SlippageModel,
    SqrtImpactSlippageModel,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "SimulatedClock",
    "BacktestClock",
    "EquitiesCostModel",
    "Costs",
    "HalfSpreadSlippageModel",
    "SqrtImpactSlippageModel",
    "SlippageModel",
    "ADRDataLoader",
    "GlobalDataLoader",
    "FXDataLoader",
    "PairRegistryLoader",
    "PairRegistryUpdateEvent",
    "SimulatedUSExchange",
    "SimulatedForeignExchange",
    "RoceRuceCalculator",
    "distribution_stats",
    "build_summary",
    "build_distributions",
    "write_report",
]
