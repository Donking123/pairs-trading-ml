"""Risk management engine (architecture §3.7).

Subscribes to ``signals`` and ``positions``. For every entry signal it runs the
pre-trade rule pipeline against a :class:`RiskContext` snapshot and publishes a
:class:`~asian_adr.core.RiskDecision` to ``risk-decisions``. Close signals
(``EXIT`` / ``FORCE_CLOSE``) are always approved — risk never blocks a position
*reduction*. A ``KILL``-severity result escalates to the kill switch.

The zero-return-day percentage and short-locate availability are supplied via
injected providers so the engine stays decoupled from the strategy and broker
layers (it imports neither ``strategy`` nor ``gateways``).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Callable, Iterable

from asian_adr.core import (
    AsianADRApprovedPair,
    Clock,
    HongSusmelSignalEvent,
    HSSignal,
    PositionUpdateEvent,
    RiskDecision,
    RiskDecisionStatus,
)
from asian_adr.event_bus import AbstractEventBus, Topic

from .rules.base import AbstractRiskRule, RiskConfig, RiskRuleResult, Severity
from .rules.country_concentration import CountryConcentrationRule
from .rules.drawdown_limits import DrawdownLimitsRule
from .rules.kill_switch import KillSwitch, KillSwitchRule
from .rules.notional_limits import MaxOpenPairsRule, NotionalLimitsRule
from .rules.rate_of_loss import RateOfLossRule
from .rules.short_locate import ShortLocateRule
from .rules.zero_return_day_filter import ZeroReturnDayFilter
from .state import RiskState

logger = logging.getLogger(__name__)

__all__ = ["RiskEngine"]

ZeroReturnProvider = Callable[[str], Decimal]
ShortLocateProvider = Callable[[str], bool]


class RiskEngine:
    """Pre-trade gatekeeper between the strategy and the execution sequencer.

    Parameters
    ----------
    bus, clock:
        Event bus and time source.
    config:
        Risk limits.
    pairs:
        Active pair registry (for country mapping and holding limits).
    kill_switch:
        Shared kill switch; created if not supplied.
    rules:
        Override the default pre-trade pipeline (mainly for testing).
    zero_return_provider:
        ``pair_id -> Decimal`` ADR zero-return-day pct (e.g. wired to the HS
        engine's state). Defaults to ``0`` (filter always passes).
    short_locate_provider:
        ``adr_ticker -> bool`` borrow availability. Defaults to always-available.
    """

    def __init__(
        self,
        bus: AbstractEventBus,
        clock: Clock,
        config: RiskConfig,
        pairs: Iterable[AsianADRApprovedPair] = (),
        *,
        kill_switch: KillSwitch | None = None,
        rules: list[AbstractRiskRule] | None = None,
        zero_return_provider: ZeroReturnProvider | None = None,
        short_locate_provider: ShortLocateProvider | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._config = config
        self._pairs: dict[str, AsianADRApprovedPair] = {p.pair_id: p for p in pairs}
        self.kill_switch = kill_switch or KillSwitch(bus)
        self._state = RiskState(config, list(self._pairs.values()))
        self._zero_return_provider = zero_return_provider
        self._short_locate_provider = short_locate_provider
        self._rules: list[AbstractRiskRule] = rules or self._default_rules()

    def _default_rules(self) -> list[AbstractRiskRule]:
        # Order: cheapest / hardest stops first.
        return [
            KillSwitchRule(self.kill_switch),
            ShortLocateRule(),
            ZeroReturnDayFilter(),
            MaxOpenPairsRule(),
            NotionalLimitsRule(),
            CountryConcentrationRule(),
            DrawdownLimitsRule(),
            RateOfLossRule(),
        ]

    # -- position feed -------------------------------------------------------

    async def on_position_update(self, event: PositionUpdateEvent) -> None:
        self._state.update_position(event)
        # A drawdown breach can fire the kill switch outside the signal flow.
        if self._state.drawdown_pct >= self._config.max_drawdown_pct:
            await self.kill_switch.trigger(
                f"drawdown {self._state.drawdown_pct:.1%} ≥ "
                f"{self._config.max_drawdown_pct:.1%}"
            )

    # -- signal gating -------------------------------------------------------

    async def on_signal(self, signal: HongSusmelSignalEvent) -> None:
        decision, results = self._evaluate(signal)
        await self._bus.publish(Topic.RISK_DECISIONS, decision)
        if any(not r.passed and r.severity >= Severity.KILL for r in results):
            await self.kill_switch.trigger("kill-severity risk rule breached")

    def evaluate(self, signal: HongSusmelSignalEvent) -> RiskDecision:
        """Pure core: run the pipeline and build a :class:`RiskDecision`."""
        return self._evaluate(signal)[0]

    def _evaluate(
        self, signal: HongSusmelSignalEvent
    ) -> tuple[RiskDecision, list[RiskRuleResult]]:
        # Closing trades are always approved — never block a reduction.
        if signal.signal in (HSSignal.EXIT, HSSignal.FORCE_CLOSE):
            return self._decision(signal, RiskDecisionStatus.APPROVED, None, None, []), []

        pair = self._pairs.get(signal.pair_id)
        zero_return_pct = (
            self._zero_return_provider(signal.pair_id)
            if self._zero_return_provider
            else Decimal(0)
        )
        if self._short_locate_provider is not None:
            self._state.set_short_locate(
                signal.adr_ticker, self._short_locate_provider(signal.adr_ticker)
            )
        ctx = self._state.build_context(
            signal, pair, zero_return_pct=zero_return_pct,
            kill_switch_active=self.kill_switch.triggered,
        )

        results = [rule.evaluate(signal, ctx) for rule in self._rules]
        blocking = [r for r in results if r.is_blocking]

        if blocking:
            reason = "; ".join(r.reason for r in blocking)
            return (
                self._decision(signal, RiskDecisionStatus.REJECTED, None, reason, results),
                results,
            )
        return (
            self._decision(
                signal, RiskDecisionStatus.APPROVED,
                self._config.entry_notional_usd, None, results,
            ),
            results,
        )

    def _decision(
        self,
        signal: HongSusmelSignalEvent,
        status: RiskDecisionStatus,
        approved_notional: Decimal | None,
        rejected_reason: str | None,
        results: list[RiskRuleResult],
    ) -> RiskDecision:
        return RiskDecision(
            timestamp_exchange=signal.timestamp_exchange,
            timestamp_received=self._clock.now(),
            signal_id=signal.event_id,
            pair_id=signal.pair_id,
            status=status,
            approved_notional=approved_notional,
            rejected_reason=rejected_reason,
            risk_rule_results=[r.model_dump() for r in results],
        )

    # -- introspection -------------------------------------------------------

    @property
    def state(self) -> RiskState:
        return self._state

    async def run(self) -> None:  # pragma: no cover - live entrypoint
        return None
