"""Liquidity-bucket assignment for post-hoc ROCE/RUCE attribution (paper §2.5).

The paper documents *higher* median profit in less-liquid buckets (a
limits-to-arbitrage premium), so every closed trade is tagged with a bucket for
distribution breakdowns. Buckets are defined by two interchangeable criteria;
the engine tracks the ADR zero-return-day percentage, so that is the primary
classifier, with an ADV-based variant provided for completeness.

Depends only on the standard library and ``asian_adr.core``.
"""

from __future__ import annotations

from decimal import Decimal

from asian_adr.core import LiquidityBucket

__all__ = [
    "assign_bucket",
    "assign_by_zero_return",
    "assign_by_adv",
    "ZERO_RETURN_CUTOFFS",
    "ADV_CUTOFFS",
]

# Zero-return-day percentage cut-points (paper Table, §2.5).
ZERO_RETURN_CUTOFFS: tuple[Decimal, Decimal, Decimal] = (
    Decimal("0.0621"),   # < 6.21%   → HIGH
    Decimal("0.1457"),   # < 14.57%  → HIGH_MEDIUM
    Decimal("0.2975"),   # < 29.75%  → MEDIUM_LOW   (else LOW)
)

# Average-daily-volume cut-points in shares (paper Table, §2.5).
ADV_CUTOFFS: tuple[Decimal, Decimal, Decimal] = (
    Decimal("91332"),    # > 91,332         → HIGH
    Decimal("17222"),    # 17,222–91,332    → HIGH_MEDIUM
    Decimal("3531"),     # 3,531–17,222     → MEDIUM_LOW  (else LOW)
)


def assign_by_zero_return(zero_return_pct: Decimal) -> LiquidityBucket:
    """Bucket by ADR zero-return-day percentage (rises with illiquidity)."""
    low, mid, high = ZERO_RETURN_CUTOFFS
    if zero_return_pct < low:
        return LiquidityBucket.HIGH
    if zero_return_pct < mid:
        return LiquidityBucket.HIGH_MEDIUM
    if zero_return_pct < high:
        return LiquidityBucket.MEDIUM_LOW
    return LiquidityBucket.LOW


def assign_by_adv(adv_shares: Decimal) -> LiquidityBucket:
    """Bucket by average daily volume in shares (falls with illiquidity)."""
    high, mid, low = ADV_CUTOFFS
    if adv_shares > high:
        return LiquidityBucket.HIGH
    if adv_shares > mid:
        return LiquidityBucket.HIGH_MEDIUM
    if adv_shares > low:
        return LiquidityBucket.MEDIUM_LOW
    return LiquidityBucket.LOW


def assign_bucket(zero_return_pct: Decimal) -> LiquidityBucket:
    """Default classifier — by ADR zero-return-day percentage."""
    return assign_by_zero_return(zero_return_pct)
