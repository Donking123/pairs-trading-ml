"""Global kill switch: state object, event, and the pre-trade gate rule.

The :class:`KillSwitch` is a one-way latch. Once :meth:`trigger` fires it
publishes a :class:`KillSwitchEvent`, runs any registered shutdown callbacks
(e.g. close-all / cancel-all on the sequencer), and stays latched. The
:class:`KillSwitchRule` blocks every new entry while the switch is active.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Literal

from asian_adr.core import BaseEvent, HongSusmelSignalEvent
from asian_adr.event_bus import AbstractEventBus, Topic

from .base import AbstractRiskRule, RiskContext, RiskRuleResult, Severity

logger = logging.getLogger(__name__)

__all__ = ["KillSwitchEvent", "KillSwitch", "KillSwitchRule"]


class KillSwitchEvent(BaseEvent):
    event_type: Literal["kill_switch"] = "kill_switch"
    reason: str
    operator: str = "system"


ShutdownCallback = Callable[[], Awaitable[None]]


class KillSwitch:
    """One-way latch that halts all new risk and runs shutdown hooks."""

    def __init__(self, bus: AbstractEventBus | None = None) -> None:
        self._bus = bus
        self._triggered = False
        self._reason: str | None = None
        self._callbacks: list[ShutdownCallback] = []

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def reason(self) -> str | None:
        return self._reason

    def on_shutdown(self, callback: ShutdownCallback) -> None:
        """Register a coroutine to run when the switch trips (close-all etc.)."""
        self._callbacks.append(callback)

    async def trigger(self, reason: str, operator: str = "system") -> None:
        if self._triggered:
            return
        self._triggered = True
        self._reason = reason
        logger.critical("KILL_SWITCH_TRIGGERED", extra={"reason": reason, "operator": operator})
        if self._bus is not None:
            await self._bus.publish(
                Topic.ALERTS,
                KillSwitchEvent(
                    timestamp_exchange=datetime.now(timezone.utc),
                    timestamp_received=datetime.now(timezone.utc),
                    reason=reason,
                    operator=operator,
                ),
            )
        for callback in self._callbacks:
            await callback()


class KillSwitchRule(AbstractRiskRule):
    """Blocks all entries while the kill switch is active."""

    name = "kill_switch"

    def __init__(self, kill_switch: KillSwitch) -> None:
        self._kill_switch = kill_switch

    def evaluate(
        self, signal: HongSusmelSignalEvent | None, ctx: RiskContext
    ) -> RiskRuleResult:
        if self._kill_switch.triggered or ctx.kill_switch_active:
            return RiskRuleResult.block(
                self.name,
                f"kill switch active: {self._kill_switch.reason or 'triggered'}",
                severity=Severity.KILL,
            )
        return RiskRuleResult.ok(self.name)
