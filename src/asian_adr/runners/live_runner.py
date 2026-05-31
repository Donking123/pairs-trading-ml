"""Live / paper trading runner (architecture §8.1, Phase 3).

Wires the production trading components — ``HongSusmelEngine``, ``RiskEngine``,
``PositionEngine``, ``AsianExecutionSequencer``, and the ROCE/RUCE tracker — onto
an ``AsyncioBus`` driven by a ``LiveClock``, with the same subscription topology
and approved-signal router the backtest uses (research/production parity).

Data sources (feed handlers, FX handler) and the broker gateway are **injected**
producers, so this module imports none of the not-yet-built ``feed_handler`` /
``fx_handler`` / ``gateways`` packages and stays import-safe today. A producer is
any object exposing ``async run()`` (and, optionally, bus handlers it wires
itself). For paper runs with no live connectivity, a
:class:`~asian_adr.runners.data_recorder.ReplayProducer` can stand in as the feed.

Run:  ``python -m asian_adr.runners.live_runner --config config/development.toml``
      ``python -m asian_adr.runners.live_runner --replay session.jsonl``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from decimal import Decimal
from typing import Protocol

from asian_adr.core import Clock, LiveClock, RiskDecisionStatus
from asian_adr.event_bus import AbstractEventBus, AsyncioBus, Topic
from asian_adr.position import PositionEngine
from asian_adr.risk import RiskEngine
from asian_adr.strategy.hong_susmel import AsianExecutionSequencer, HongSusmelEngine
from asian_adr.backtest import PairRegistryLoader, RoceRuceCalculator

from .config import RunnerConfig

logger = logging.getLogger(__name__)

__all__ = ["Producer", "LiveRunner", "main"]


class Producer(Protocol):
    """Anything that drives the bus: a feed handler, FX handler, or gateway."""

    async def run(self) -> None: ...


class LiveRunner:
    """Wires and runs the live trading pipeline."""

    def __init__(
        self,
        config: RunnerConfig,
        bus: AbstractEventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config = config
        self._bus = bus or AsyncioBus()
        self._clock = clock or LiveClock()

        loader = PairRegistryLoader.from_json(config.pairs_json)
        self._pairs = loader.all_pairs
        active = [p for p in self._pairs if p.is_active_as_of(self._clock.today())]

        self._hs = HongSusmelEngine(self._bus, self._clock, active)
        self._risk = RiskEngine(
            self._bus, self._clock, config.risk_config(), self._pairs,
            zero_return_provider=self._zero_return,
        )
        self._position = PositionEngine(self._bus, self._clock, self._pairs)
        self._sequencer = AsianExecutionSequencer(
            self._bus, self._clock, self._pairs,
            notional_per_leg=Decimal(str(config.notional_per_leg)),
        )
        self._roce = RoceRuceCalculator(self._bus, self._clock, self._pairs)

        self._producers: list[Producer] = []
        self._gateway: Producer | None = None
        self._pending_signals: dict = {}
        self._us_exch = None
        self._foreign_exch = None

    # -- dependency injection ------------------------------------------------

    def add_producer(self, producer: Producer) -> None:
        """Register a feed/FX producer that publishes onto the bus."""
        self._producers.append(producer)

    def set_gateway(self, gateway: Producer) -> None:
        """Register the U.S. broker gateway (consumes orders, emits fills)."""
        self._gateway = gateway
        self._sequencer.register_gateway("us_equity", gateway)

    def enable_simulated_execution(self, cost_model=None) -> None:
        """Wire the backtest simulated exchanges as the fill source (paper mode).

        Lets the runner execute end-to-end without a live broker: the simulated
        U.S. and foreign exchanges consume ``orders`` and emit ``fills``, marking
        against the replayed/live bars. Call before :meth:`run`.
        """
        from asian_adr.backtest import (
            EquitiesCostModel,
            SimulatedForeignExchange,
            SimulatedUSExchange,
        )

        cm = cost_model or EquitiesCostModel()
        self._us_exch = SimulatedUSExchange(self._bus, self._clock, cm)
        self._foreign_exch = SimulatedForeignExchange(self._bus, self._clock, cm)

    @property
    def bus(self) -> AsyncioBus:
        return self._bus

    def _zero_return(self, pair_id: str) -> Decimal:
        st = self._hs.state_for(pair_id)
        return st.rolling_zero_return_pct() if st else Decimal(0)

    # -- wiring (mirrors backtest topology for research/production parity) ----

    async def setup(self) -> None:
        bus = self._bus

        async def on_decision(dec) -> None:
            sig = self._pending_signals.pop(dec.signal_id, None)
            if sig is not None and dec.status == RiskDecisionStatus.APPROVED:
                await self._sequencer.on_signal(sig)

        def cache_signal(sig) -> None:
            self._pending_signals[sig.event_id] = sig

        # Simulated exchanges / gateway (if present) subscribe to market-data
        # FIRST so their last-seen bar is fresh when the sequencer emits an order.
        if self._us_exch is not None:
            await bus.subscribe(Topic.MARKET_DATA, self._us_exch.on_bar)
        if self._foreign_exch is not None:
            await bus.subscribe(Topic.MARKET_DATA, self._foreign_exch.on_bar)
        if self._gateway is not None and hasattr(self._gateway, "on_bar"):
            await bus.subscribe(Topic.MARKET_DATA, self._gateway.on_bar)

        await bus.subscribe(Topic.MARKET_DATA, self._hs.on_daily_bar)
        await bus.subscribe(Topic.MARKET_DATA, self._sequencer.on_bar)
        await bus.subscribe(Topic.MARKET_DATA, self._position.on_bar)

        await bus.subscribe(Topic.FX_RATES, self._hs.on_fx_rate)
        await bus.subscribe(Topic.FX_RATES, self._sequencer.on_fx_rate)
        await bus.subscribe(Topic.FX_RATES, self._position.on_fx_rate)
        if self._foreign_exch is not None:
            await bus.subscribe(Topic.FX_RATES, self._foreign_exch.on_fx_rate)

        await bus.subscribe(Topic.SIGNALS, self._risk.on_signal)
        await bus.subscribe(Topic.SIGNALS, self._roce.on_signal)
        await bus.subscribe(Topic.SIGNALS, cache_signal)
        await bus.subscribe(Topic.RISK_DECISIONS, on_decision)

        if self._us_exch is not None:
            await bus.subscribe(Topic.ORDERS, self._us_exch.on_order)
        if self._foreign_exch is not None:
            await bus.subscribe(Topic.ORDERS, self._foreign_exch.on_order)
        if self._gateway is not None and hasattr(self._gateway, "on_order"):
            await bus.subscribe(Topic.ORDERS, self._gateway.on_order)

        await bus.subscribe(Topic.FILLS, self._sequencer.on_fill)
        await bus.subscribe(Topic.FILLS, self._position.on_fill)
        await bus.subscribe(Topic.FILLS, self._roce.on_fill)

        await bus.subscribe(Topic.POSITIONS, self._risk.on_position_update)
        await bus.subscribe(Topic.PAIR_REGISTRY, self._hs.on_registry_update)

        logger.info(
            "live pipeline wired: %d pairs, %d producers, gateway=%s",
            len(self._pairs), len(self._producers), self._gateway is not None,
        )

    # -- run -----------------------------------------------------------------

    async def run(self) -> None:
        await self.setup()
        if not self._producers:
            logger.warning(
                "no producers registered — no market data will arrive. "
                "Inject a feed handler or a ReplayProducer to drive the runner."
            )
        try:
            async with asyncio.TaskGroup() as tg:
                for producer in self._producers:
                    tg.create_task(producer.run())
                if self._gateway is not None:
                    tg.create_task(self._gateway.run())
        finally:
            await self._bus.flush()
            await self._bus.close()
        logger.info("live run finished: %d round-trips tracked", len(self._roce.results))

    @property
    def results(self) -> list:
        return self._roce.results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Asian ADR live / paper runner.")
    parser.add_argument("--config", default=None, help="TOML config path")
    parser.add_argument(
        "--replay", default=None,
        help="JSONL recording to replay as the feed (paper mode, no live data)",
    )
    parser.add_argument(
        "--replay-delay", type=float, default=0.0,
        help="seconds between replayed events (0 = as fast as possible)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")

    config = RunnerConfig.from_toml(args.config)
    runner = LiveRunner(config)

    if args.replay:
        from .data_recorder import ReplayProducer

        # Replay only the feed topics; the pipeline regenerates signals/fills.
        runner.enable_simulated_execution()
        runner.add_producer(
            ReplayProducer(
                runner.bus, args.replay,
                topics=(Topic.MARKET_DATA, Topic.FX_RATES),
                delay_seconds=args.replay_delay,
            )
        )
    else:
        logger.warning(
            "no --replay and no live connectors built yet (Phase 3/5). "
            "Wire a feed handler / gateway via add_producer()/set_gateway()."
        )

    asyncio.run(runner.run())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
