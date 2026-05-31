"""Alert rule engine and dispatcher (architecture §11.4).

:class:`AlertEngine` consumes bus events and applies the architecture's alert
rules — kill switch, FX staleness, overnight-abort cluster, force-close cluster,
zero-return-day spike, feed gap, order rejection — emitting :class:`Alert`s to a
:class:`AlertDispatcher`, which routes them to channels (log / Slack / PagerDuty)
by severity.

Events are matched by their ``event_type`` string (duck-typed), so this module
stays decoupled from the strategy/risk/feed packages — it imports only ``core``
and ``event_bus``. Slack/PagerDuty use ``httpx`` lazily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

from asian_adr.event_bus import AbstractEventBus, Topic

logger = logging.getLogger(__name__)

__all__ = [
    "AlertSeverity",
    "Alert",
    "AlertChannel",
    "LogChannel",
    "SlackChannel",
    "PagerDutyChannel",
    "AlertDispatcher",
    "AlertEngine",
]


class AlertSeverity(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass(frozen=True)
class Alert:
    severity: AlertSeverity
    title: str
    detail: str
    source: str = "monitoring"


@runtime_checkable
class AlertChannel(Protocol):
    async def send(self, alert: Alert) -> None: ...


class LogChannel:
    """Always-on channel that logs alerts (default)."""

    async def send(self, alert: Alert) -> None:
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }[alert.severity]
        logger.log(level, "ALERT [%s] %s — %s", alert.severity.name, alert.title, alert.detail)


class SlackChannel:
    """Posts alerts to a Slack incoming webhook (``httpx`` lazy)."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    async def send(self, alert: Alert) -> None:  # pragma: no cover - network
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; Slack alert dropped")
            return
        text = f"[{alert.severity.name}] {alert.title}\n{alert.detail}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._url, json={"text": text})
        except Exception as exc:
            logger.warning("Slack alert failed: %s", exc)


class PagerDutyChannel:
    """Triggers a PagerDuty Events API v2 incident (``httpx`` lazy)."""

    def __init__(self, routing_key: str) -> None:
        self._key = routing_key
        self._url = "https://events.pagerduty.com/v2/enqueue"

    async def send(self, alert: Alert) -> None:  # pragma: no cover - network
        if alert.severity < AlertSeverity.CRITICAL:
            return
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; PagerDuty alert dropped")
            return
        payload = {
            "routing_key": self._key,
            "event_action": "trigger",
            "payload": {
                "summary": f"{alert.title}: {alert.detail}",
                "severity": "critical",
                "source": alert.source,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._url, json=payload)
        except Exception as exc:
            logger.warning("PagerDuty alert failed: %s", exc)


class AlertDispatcher:
    """Fans an alert out to channels filtered by a minimum severity."""

    def __init__(self) -> None:
        self._channels: list[tuple[AlertChannel, AlertSeverity]] = [
            (LogChannel(), AlertSeverity.INFO)
        ]

    def add_channel(self, channel: AlertChannel, min_severity: AlertSeverity = AlertSeverity.WARNING) -> None:
        self._channels.append((channel, min_severity))

    async def dispatch(self, alert: Alert) -> None:
        for channel, min_sev in self._channels:
            if alert.severity >= min_sev:
                await channel.send(alert)


@dataclass
class _BarCounters:
    """Per-bar (per-date) event tallies for cluster rules."""

    aborts: int = 0
    force_closes: int = 0
    zero_returns: int = 0
    extra: dict = field(default_factory=dict)


class AlertEngine:
    """Applies alert rules to bus events and dispatches resulting alerts.

    Cluster rules (force-close, zero-return spike, abort cluster) aggregate
    counts within a calendar bar and fire once a threshold is crossed.
    """

    def __init__(
        self,
        dispatcher: AlertDispatcher | None = None,
        *,
        force_close_cluster: int = 5,
        abort_cluster: int = 4,
        zero_return_cluster: int = 10,
    ) -> None:
        self.dispatcher = dispatcher or AlertDispatcher()
        self._force_close_cluster = force_close_cluster
        self._abort_cluster = abort_cluster
        self._zero_return_cluster = zero_return_cluster
        self._bar: date | None = None
        self._counters = _BarCounters()
        self.alerts_fired: list[Alert] = []

    async def attach(self, bus: AbstractEventBus) -> None:
        await bus.subscribe(Topic.ALERTS, self.on_alert)
        await bus.subscribe(Topic.SIGNALS, self.on_signal)

    def _roll_bar(self, when: date) -> None:
        if when != self._bar:
            self._bar = when
            self._counters = _BarCounters()

    async def _fire(self, alert: Alert) -> None:
        self.alerts_fired.append(alert)
        await self.dispatcher.dispatch(alert)

    async def on_signal(self, signal: Any) -> None:
        if getattr(signal, "signal", None) is None:
            return
        if signal.signal.value == "force_close":
            self._roll_bar(signal.timestamp_exchange.date())
            self._counters.force_closes += 1
            if self._counters.force_closes == self._force_close_cluster:
                await self._fire(Alert(
                    AlertSeverity.WARNING, "Force-close cluster",
                    f"{self._counters.force_closes} force-closes in one bar",
                    source="risk"))

    async def on_alert(self, event: Any) -> None:
        etype = getattr(event, "event_type", "")
        when = event.timestamp_exchange.date()

        if etype == "kill_switch":
            await self._fire(Alert(AlertSeverity.CRITICAL, "Kill switch triggered",
                                   getattr(event, "reason", ""), source="risk"))
        elif etype == "fx_stale" and getattr(event, "suspended", False):
            await self._fire(Alert(AlertSeverity.CRITICAL, "FX staleness",
                                   f"{event.base_currency} stale {event.age_business_days} business days",
                                   source="fx_handler"))
        elif etype == "feed_gap":
            await self._fire(Alert(AlertSeverity.WARNING, "Feed gap",
                                   f"{event.ticker} gap {event.gap_days} days", source="feed_handler"))
        elif etype == "order_rejected":
            await self._fire(Alert(AlertSeverity.WARNING, "Order rejected",
                                   f"{event.ticker} {event.side}: {event.reason}", source="gateway"))
        elif etype == "adr_overnight_abort":
            self._roll_bar(when)
            self._counters.aborts += 1
            if self._counters.aborts == self._abort_cluster:
                await self._fire(Alert(AlertSeverity.CRITICAL, "Overnight abort cluster",
                                       f"{self._counters.aborts} aborts in one morning", source="oms"))
        elif etype == "zero_return":
            self._roll_bar(when)
            self._counters.zero_returns += 1
            if self._counters.zero_returns == self._zero_return_cluster:
                await self._fire(Alert(AlertSeverity.WARNING, "Zero-return-day spike",
                                       f"{self._counters.zero_returns} zero-return flags in one bar",
                                       source="feed_handler"))
