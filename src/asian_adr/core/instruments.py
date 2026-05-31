"""The approved-pair instrument and Asian-exchange reference constants.

An :class:`AsianADRApprovedPair` is the frozen registry record the screening
pipeline writes and every online component reads. It carries the structural
``adr_ratio`` (β, never OLS-estimated) plus the strategy parameters
(``estimation_days`` T, ``holding_days`` H, ``k0``, ``kc``) and the liquidity /
cost metadata used for risk gating and attribution.

This module depends only on the standard library and pydantic.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "AsianADRApprovedPair",
    "AsianADRPairSpec",
    "ASIAN_EXCHANGES",
    "STANDARD_ADR_RATIOS",
]


# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------

ASIAN_EXCHANGES: frozenset[str] = frozenset(
    {
        "TKS", "TSE",   # Tokyo Stock Exchange (Japan) — both Datastream mnemonics
        "HKG",          # Hong Kong Exchange (HKEX)
        "KRX",          # Korea Exchange
        "BOM", "BSE",   # Bombay Stock Exchange (India) — two mnemonics in use
        "NSE",          # National Stock Exchange (India)
        "ASX",          # Australian Securities Exchange
        "SES",          # Singapore Exchange (SGX)
        "TAI",          # Taiwan Stock Exchange (TWSE)
        "IDX",          # Indonesia Stock Exchange
        "PHS",          # Philippine Stock Exchange (PSE)
        "KLS",          # Bursa Malaysia
    }
)
"""Datastream exchange mnemonics whose listings are eligible Asian underlyings."""

STANDARD_ADR_RATIOS: tuple[Decimal, ...] = tuple(
    Decimal(r)
    for r in ("0.01", "0.1", "0.2", "0.25", "0.5", "1", "2", "2.5", "5", "10", "20", "50", "100")
)
"""Conventional ADR conversion ratios the screener snaps a price-median estimate to."""


# ---------------------------------------------------------------------------
# Approved pair
# ---------------------------------------------------------------------------


class AsianADRApprovedPair(BaseModel):
    """An approved, tradable ADR / Asian-underlying pair (registry record).

    Frozen and self-validating: ``adr_ratio`` must be strictly positive (an
    unresolved ratio means the pair is rejected, not traded with a guessed
    hedge) and ``expiry_date`` must not precede ``approved_date``.
    """

    model_config = ConfigDict(frozen=True)

    pair_id: str
    adr_ticker: str
    underlying_ticker: str
    underlying_exchange: str           # "TSE", "HKEX", "KRX", "ASX", …
    underlying_currency: str           # "JPY", "HKD", "KRW", "AUD", …
    adr_ratio: Decimal                 # local shares per 1 ADR; structural β

    # Strategy parameters (paper grid; platform defaults)
    estimation_days: int = 60          # T: rolling window for µ/σ
    holding_days: int = 90             # H: max holding period before force-close
    k0: Decimal = Decimal("2.0")       # entry multiplier
    kc: Decimal = Decimal("0.0")       # exit multiplier

    # Liquidity / cost metadata
    zero_return_pct_adr: Decimal = Decimal("0")   # Bekaert et al. (2007) metric
    roll_spread_local: Decimal = Decimal("0")     # Roll (1984) spread, local leg
    roll_spread_adr: Decimal = Decimal("0")       # Roll (1984) spread, ADR leg

    fx_hedge_required: bool = False    # always False — FX is conversion-only
    withholding_tax_rate: Decimal = Decimal("0.0")

    approved_date: date
    expiry_date: date
    is_active: bool = True

    # -- validation ----------------------------------------------------------

    @field_validator("adr_ratio")
    @classmethod
    def _ratio_must_be_positive(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("adr_ratio must be strictly positive")
        return value

    @field_validator("estimation_days", "holding_days")
    @classmethod
    def _windows_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("estimation_days and holding_days must be positive")
        return value

    def model_post_init(self, __context: object) -> None:
        if self.expiry_date < self.approved_date:
            raise ValueError(
                f"expiry_date {self.expiry_date} precedes approved_date {self.approved_date}"
            )

    # -- convenience ---------------------------------------------------------

    def is_active_as_of(self, as_of: date) -> bool:
        """True if the pair is active and ``as_of`` falls within its validity window."""
        return self.is_active and self.approved_date <= as_of <= self.expiry_date


# Architecture alias: the project structure labels this file's model
# ``AsianADRPairSpec``; the data-contract names it ``AsianADRApprovedPair``.
# Keep both names pointing at the same type.
AsianADRPairSpec = AsianADRApprovedPair
