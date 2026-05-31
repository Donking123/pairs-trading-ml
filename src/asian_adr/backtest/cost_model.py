"""Transaction-cost model for simulated fills (architecture §3.12).

Splits costs by leg:

* **U.S. ADR leg** — commission ``max($1, $0.005×shares)``, SEC fee on sells
  (``notional × 0.0000278``), and a one-day short-borrow proxy
  (``notional × borrow_rate / 252``) on the short sale.
* **Asian underlying leg** — commission ``max(min, rate×shares)``, stamp duty
  (exchange-specific, e.g. HK 0.10%, AU 0.00%), and a local transaction levy.

All figures are returned in the leg's native currency; USD conversion happens
in the simulated exchange via the FX rate.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from asian_adr.core import Side

__all__ = ["Costs", "EquitiesCostModel"]


@dataclass(frozen=True)
class Costs:
    """Itemised costs for one fill, in the leg's native currency."""

    commission: Decimal = Decimal("0")
    sec_fee: Decimal = Decimal("0")
    stamp_duty: Decimal = Decimal("0")
    short_borrow_fee: Decimal = Decimal("0")
    local_levy: Decimal = Decimal("0")

    @property
    def total(self) -> Decimal:
        return (
            self.commission + self.sec_fee + self.stamp_duty
            + self.short_borrow_fee + self.local_levy
        )


# Stamp duty by Datastream exchange mnemonic (approximate, paper era).
_STAMP_RATES: dict[str, Decimal] = {
    "HKG": Decimal("0.0010"),   # Hong Kong ~0.10%
    "ASX": Decimal("0.0000"),   # Australia — none
    "TSE": Decimal("0.0000"), "TKS": Decimal("0.0000"),  # Japan — none
    "KRX": Decimal("0.0023"),   # Korea securities transaction tax (sell)
    "SES": Decimal("0.0000"),   # Singapore — none on equities
    "BOM": Decimal("0.00015"), "BSE": Decimal("0.00015"), "NSE": Decimal("0.00015"),  # India
    "TAI": Decimal("0.0030"),   # Taiwan transaction tax (sell)
    "IDX": Decimal("0.0010"),
    "PHS": Decimal("0.0060"),
    "KLS": Decimal("0.0010"),
}
_DEFAULT_STAMP = Decimal("0.0005")


class EquitiesCostModel:
    """Per-leg cost calculator with sensible defaults."""

    def __init__(
        self,
        *,
        us_commission_per_share: Decimal = Decimal("0.005"),
        us_min_commission: Decimal = Decimal("1.00"),
        sec_fee_rate: Decimal = Decimal("0.0000278"),
        borrow_rate_annual: Decimal = Decimal("0.005"),
        foreign_commission_per_share: Decimal = Decimal("0.001"),
        foreign_min_commission: Decimal = Decimal("1.00"),
        local_levy_rate: Decimal = Decimal("0.00005"),
        stamp_rates: dict[str, Decimal] | None = None,
    ) -> None:
        self.us_commission_per_share = us_commission_per_share
        self.us_min_commission = us_min_commission
        self.sec_fee_rate = sec_fee_rate
        self.borrow_rate_annual = borrow_rate_annual
        self.foreign_commission_per_share = foreign_commission_per_share
        self.foreign_min_commission = foreign_min_commission
        self.local_levy_rate = local_levy_rate
        self._stamp_rates = stamp_rates or _STAMP_RATES

    def us_leg(
        self, side: Side, quantity: Decimal, price: Decimal, is_short_sale: bool
    ) -> Costs:
        notional = quantity * price
        commission = max(self.us_min_commission, self.us_commission_per_share * quantity)
        sec_fee = notional * self.sec_fee_rate if side == "sell" else Decimal("0")
        # One-day short-borrow proxy charged on the short sale.
        short_borrow = (
            notional * self.borrow_rate_annual / Decimal("252")
            if (side == "sell" and is_short_sale)
            else Decimal("0")
        )
        return Costs(commission=commission, sec_fee=sec_fee, short_borrow_fee=short_borrow)

    def foreign_leg(
        self, exchange: str, side: Side, quantity: Decimal, price: Decimal
    ) -> Costs:
        notional = quantity * price
        commission = max(
            self.foreign_min_commission, self.foreign_commission_per_share * quantity
        )
        stamp_rate = self._stamp_rates.get(exchange, _DEFAULT_STAMP)
        stamp_duty = notional * stamp_rate
        local_levy = notional * self.local_levy_rate
        return Costs(commission=commission, stamp_duty=stamp_duty, local_levy=local_levy)
