"""Backtest replay engine (architecture §3.12, §9.1).

Drives the **real** production components — ``HongSusmelEngine``, ``RiskEngine``,
``PositionEngine``, ``AsianExecutionSequencer`` — through a deterministic
``MemoryBus``, replaying merged U.S. / foreign / FX daily streams in
chronological order. Because the same components run here and live, a backtest
that reproduces the paper is evidence the *production* code does too — the
research/production-parity principle.

Fill timing follows the paper: the ADR short fills at the U.S. close that
generated the signal; the local leg fills on the next Asian bar (enforced by the
sequencer's ``AWAITING_ASIA_OPEN`` recheck). Point-in-time correctness is
enforced by reloading the registry on each date rollover.

A small in-engine router bridges the one ambiguity in the spec's wiring: the
risk engine emits ``RiskDecision``s, but the sequencer needs the originating
*signal*. The router caches signals and forwards approved ones to the sequencer,
so risk gating stays in the path without changing any event contract.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Iterable, Iterator, Protocol

from asian_adr.core import (
    AsianADRApprovedPair,
    BarEvent,
    BaseEvent,
    FXRateEvent,
    RiskDecisionStatus,
    SimulatedClock,
)
from asian_adr.event_bus import MemoryBus, Topic
from asian_adr.position import PositionEngine
from asian_adr.risk import RiskConfig, RiskEngine
from asian_adr.strategy.hong_susmel import AsianExecutionSequencer, HongSusmelEngine

from .cost_model import EquitiesCostModel
from .pair_registry_loader import PairRegistryLoader, PairRegistryUpdateEvent
from .roce_ruce_calculator import RoceRuceCalculator
from .report import write_report
from .simulated_foreign_exchange import SimulatedForeignExchange
from .simulated_us_exchange import SimulatedUSExchange
from .slippage_models import SlippageModel

logger = logging.getLogger(__name__)

__all__ = ["BacktestConfig", "BacktestEngine", "PriceLoader", "FXLoader"]


class PriceLoader(Protocol):
    def stream(self, start: date, end: date, tickers: Iterable[str]) -> Iterator[BarEvent]: ...


class FXLoader(Protocol):
    def stream(self, start: date, end: date, currencies: Iterable[str]) -> Iterator[FXRateEvent]: ...


@dataclass
class BacktestConfig:
    """Knobs for a backtest run."""

    risk_config: RiskConfig = field(default_factory=RiskConfig)
    cost_model: EquitiesCostModel = field(default_factory=EquitiesCostModel)
    us_slippage: SlippageModel | None = None
    foreign_slippage: SlippageModel | None = None
    notional_per_leg: Decimal = Decimal("100000")
    # Optional strategy-parameter overrides applied to every pair (k0, kc, T, H).
    param_overrides: dict = field(default_factory=dict)


class BacktestEngine:
    """Deterministic multi-feed replay over the real components."""

    def __init__(
        self,
        config: BacktestConfig,
        us_loader: PriceLoader,
        foreign_loader: PriceLoader,
        fx_loader: FXLoader,
        registry_loader: PairRegistryLoader,
    ) -> None:
        self._config = config
        self._us_loader = us_loader
        self._foreign_loader = foreign_loader
        self._fx_loader = fx_loader
        self._registry_loader = registry_loader

    @classmethod
    def from_parquet(
        cls,
        config: BacktestConfig,
        *,
        adr_prices: str,
        global_prices: str,
        fx_rates: str,
        pairs_json: str,
    ) -> "BacktestEngine":
        from .data_loader import ADRDataLoader
        from .foreign_data_loader import GlobalDataLoader
        from .fx_data_loader import FXDataLoader

        return cls(
            config,
            ADRDataLoader(adr_prices),
            GlobalDataLoader(global_prices),
            FXDataLoader(fx_rates),
            PairRegistryLoader.from_json(pairs_json),
        )

    # -- helpers -------------------------------------------------------------

    def _apply_overrides(
        self, pairs: list[AsianADRApprovedPair]
    ) -> list[AsianADRApprovedPair]:
        """Apply per-pair parameter overrides.

        Besides the strategy grid (``k0``, ``kc``, ``estimation_days``,
        ``holding_days``), ``approved_date`` / ``expiry_date`` may be overridden
        to backtest a selected pair-set over its full sample history (a
        reproduction run) rather than only after its production selection date.
        """
        ov = self._config.param_overrides
        if not ov:
            return pairs
        update: dict = {}
        for k, v in ov.items():
            if k in ("k0", "kc"):
                update[k] = Decimal(str(v))
            elif k in ("estimation_days", "holding_days"):
                update[k] = int(v)
            elif k in ("approved_date", "expiry_date"):
                update[k] = date.fromisoformat(v) if isinstance(v, str) else v
        return [p.model_copy(update=update) for p in pairs]

    @staticmethod
    def _route(event: BaseEvent) -> str:
        return Topic.FX_RATES if isinstance(event, FXRateEvent) else Topic.MARKET_DATA

    def _merge(self, start: date, end: date) -> Iterator[BaseEvent]:
        adr = self._registry_loader.all_adr_tickers()
        und = self._registry_loader.all_underlying_tickers()
        ccy = self._registry_loader.all_currencies()
        return heapq.merge(
            self._fx_loader.stream(start, end, ccy),
            self._foreign_loader.stream(start, end, und),
            self._us_loader.stream(start, end, adr),
            key=lambda e: e.timestamp_exchange,
        )

    # -- run -----------------------------------------------------------------

    async def run(self, start: date, end: date) -> list:
        """Replay ``[start, end]`` and return the round-trip results."""
        bus = MemoryBus()
        clock = SimulatedClock(start)

        all_pairs = self._apply_overrides(self._registry_loader.all_pairs)
        active = [p for p in all_pairs if p.is_active_as_of(start)]

        hs = HongSusmelEngine(bus, clock, active)

        def zero_return(pid: str) -> Decimal:
            st = hs.state_for(pid)
            return st.rolling_zero_return_pct() if st else Decimal(0)

        risk = RiskEngine(
            bus, clock, self._config.risk_config, all_pairs,
            zero_return_provider=zero_return,
        )
        position = PositionEngine(bus, clock, all_pairs)
        sequencer = AsianExecutionSequencer(
            bus, clock, all_pairs, notional_per_leg=self._config.notional_per_leg
        )
        us_exch = SimulatedUSExchange(
            bus, clock, self._config.cost_model, self._config.us_slippage
        )
        foreign_exch = SimulatedForeignExchange(
            bus, clock, self._config.cost_model, self._config.foreign_slippage
        )
        roce = RoceRuceCalculator(bus, clock, all_pairs)

        # Approved-signal router: forward APPROVED decisions' signals to the OMS.
        pending_signals: dict = {}

        def cache_signal(sig: BaseEvent) -> None:
            pending_signals[sig.event_id] = sig

        async def on_decision(dec: BaseEvent) -> None:
            sig = pending_signals.pop(dec.signal_id, None)  # type: ignore[attr-defined]
            if sig is not None and dec.status == RiskDecisionStatus.APPROVED:  # type: ignore[attr-defined]
                await sequencer.on_signal(sig)

        # -- wiring (market-data: exchanges first so last_bar is fresh) ------
        await bus.subscribe(Topic.MARKET_DATA, us_exch.on_bar)
        await bus.subscribe(Topic.MARKET_DATA, foreign_exch.on_bar)
        await bus.subscribe(Topic.MARKET_DATA, hs.on_daily_bar)
        await bus.subscribe(Topic.MARKET_DATA, sequencer.on_bar)
        await bus.subscribe(Topic.MARKET_DATA, position.on_bar)

        await bus.subscribe(Topic.FX_RATES, hs.on_fx_rate)
        await bus.subscribe(Topic.FX_RATES, sequencer.on_fx_rate)
        await bus.subscribe(Topic.FX_RATES, position.on_fx_rate)
        await bus.subscribe(Topic.FX_RATES, foreign_exch.on_fx_rate)

        await bus.subscribe(Topic.SIGNALS, risk.on_signal)
        await bus.subscribe(Topic.SIGNALS, roce.on_signal)
        await bus.subscribe(Topic.SIGNALS, cache_signal)
        await bus.subscribe(Topic.RISK_DECISIONS, on_decision)

        await bus.subscribe(Topic.ORDERS, us_exch.on_order)
        await bus.subscribe(Topic.ORDERS, foreign_exch.on_order)

        await bus.subscribe(Topic.FILLS, sequencer.on_fill)
        await bus.subscribe(Topic.FILLS, position.on_fill)
        await bus.subscribe(Topic.FILLS, roce.on_fill)

        await bus.subscribe(Topic.POSITIONS, risk.on_position_update)
        await bus.subscribe(Topic.PAIR_REGISTRY, hs.on_registry_update)

        # -- replay loop -----------------------------------------------------
        active_ids = {p.pair_id for p in active}
        for event in self._merge(start, end):
            clock.advance_to(event.timestamp_exchange)
            if clock.date_changed:
                new_active = [p for p in all_pairs if p.is_active_as_of(clock.today())]
                new_ids = {p.pair_id for p in new_active}
                if new_ids != active_ids:
                    active_ids = new_ids
                    await bus.publish(
                        Topic.PAIR_REGISTRY,
                        PairRegistryUpdateEvent(
                            timestamp_exchange=event.timestamp_exchange,
                            timestamp_received=clock.now(),
                            registry=new_active,
                        ),
                    )
            await bus.publish(self._route(event), event)
            await bus.flush()

        logger.info("backtest complete: %d round-trips", len(roce.results))
        return roce.results

    async def run_and_report(self, start: date, end: date, out_dir: str) -> list:
        """Run the backtest and write the tearsheet/JSON artefacts."""
        results = await self.run(start, end)
        cfg = {
            "start": str(start), "end": str(end),
            **{k: str(v) for k, v in self._config.param_overrides.items()},
        }
        write_report(results, out_dir, cfg)
        return results
