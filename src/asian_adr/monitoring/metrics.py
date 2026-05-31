"""Prometheus metrics (architecture §11.2).

Defines the strategy metrics and a :class:`MetricsCollector` that updates them
from bus events. If ``prometheus_client`` is unavailable, no-op metric stubs are
used so the rest of the system runs unchanged (metrics simply aren't exported).

Imports only ``core`` and ``event_bus``; ``prometheus_client`` is optional.
"""

from __future__ import annotations

import logging
from typing import Any

from asian_adr.core import HSSignal
from asian_adr.event_bus import AbstractEventBus, Topic

logger = logging.getLogger(__name__)

__all__ = ["MetricsCollector", "start_metrics_server", "HAVE_PROMETHEUS"]

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover - env-dependent
    HAVE_PROMETHEUS = False

    class _NoOpMetric:
        def __init__(self, *a, **k) -> None: ...
        def labels(self, *a, **k) -> "_NoOpMetric":
            return self
        def inc(self, *a, **k) -> None: ...
        def set(self, *a, **k) -> None: ...
        def observe(self, *a, **k) -> None: ...

    Counter = Gauge = Histogram = _NoOpMetric  # type: ignore

    def start_http_server(*a, **k) -> None:  # type: ignore
        logger.warning("prometheus_client not installed; metrics server disabled")


# -- metric definitions (architecture §11.2) --------------------------------
spread_z_score = Gauge("hs_spread_z_score", "Current z-score of dollar spread",
                       ["pair_id", "adr_ticker"])
position_days_held = Gauge("hs_position_days_held", "Days current position has been open",
                           ["pair_id"])
overnight_aborts_total = Counter("hs_overnight_aborts_total",
                                 "Positions aborted due to overnight spread reversal", ["pair_id"])
force_closes_total = Counter("hs_force_closes_total",
                             "Positions closed due to holding period expiry", ["pair_id"])
roce_per_trade = Histogram("hs_roce_per_trade", "ROCE per closed round-trip",
                           ["pair_id", "liquidity_bucket"],
                           buckets=[-.10, -.05, -.02, 0, .01, .02, .03, .05, .08, .10, .15, .20])
adr_zero_return_pct = Gauge("hs_adr_zero_return_pct", "Rolling zero-return-day percentage for ADR",
                            ["pair_id", "adr_ticker"])
fills_total = Counter("hs_fills_total", "Fills received", ["venue", "side"])
gross_notional_usd = Gauge("hs_gross_notional_usd", "Gross portfolio notional (USD)")


def start_metrics_server(port: int = 9090) -> None:
    """Expose the Prometheus ``/metrics`` endpoint (no-op without the client)."""
    start_http_server(port)
    if HAVE_PROMETHEUS:
        logger.info("prometheus metrics server on :%d", port)


class MetricsCollector:
    """Updates Prometheus metrics from bus events.

    Wire ``on_signal`` to ``signals``, ``on_fill`` to ``fills``, and ``on_alert``
    to ``alerts``; call :meth:`record_roce` from the ROCE/RUCE calculator.
    """

    async def attach(self, bus: AbstractEventBus) -> None:
        await bus.subscribe(Topic.SIGNALS, self.on_signal)
        await bus.subscribe(Topic.FILLS, self.on_fill)
        await bus.subscribe(Topic.ALERTS, self.on_alert)

    async def on_signal(self, signal: Any) -> None:
        spread_z_score.labels(pair_id=signal.pair_id, adr_ticker=signal.adr_ticker).set(
            float(signal.z_score)
        )
        position_days_held.labels(pair_id=signal.pair_id).set(signal.days_held)
        if signal.signal == HSSignal.FORCE_CLOSE:
            force_closes_total.labels(pair_id=signal.pair_id).inc()

    async def on_fill(self, fill: Any) -> None:
        fills_total.labels(venue=fill.venue, side=fill.side).inc()

    async def on_alert(self, event: Any) -> None:
        etype = getattr(event, "event_type", "")
        if etype == "adr_overnight_abort":
            overnight_aborts_total.labels(pair_id=event.pair_id).inc()
        elif etype == "zero_return":
            adr_zero_return_pct.labels(
                pair_id=getattr(event, "pair_id", event.ticker), adr_ticker=event.ticker
            ).set(float(event.zero_return_pct))

    def record_roce(self, result: Any) -> None:
        roce_per_trade.labels(
            pair_id=result.pair_id, liquidity_bucket=result.liquidity_bucket.value
        ).observe(float(result.roce))
