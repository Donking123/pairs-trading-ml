"""
Risk-factor panel for factor-beta clustering  [Phase 2.5 — QF621 contribution].

Builds an 18-column daily factor panel, entirely from data already pulled in Phase 0
(no external ETF / commodity pull — see phases/phase2_5/decisions.md D2.5.1):

  * 6 STYLE factors  — Fama-French 5-factor + momentum, from data/ff_factors.parquet:
        mktrf, smb, hml, rmw, cma, umd      (rf excluded: it's the risk-free rate)

  * 12 INDUSTRY factors — equal-weight daily return of the universe stocks in each
        Fama-French 12-industry bucket, mapped from CRSP `siccd`. (Some buckets may be
        dropped for a given window if they hold < `min_stocks_per_industry` names.)

The factor panel is consumed by `distances.factor_beta_distance()`, which ridge-regresses
each stock's returns on these factors and clusters on the resulting beta vectors.

SIC -> FF12 ranges are Ken French's canonical definitions (Siccodes12.txt). The ranges
are disjoint; first match wins, anything unmatched falls into "Other".
"""
from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR

# ────────────────────────────────────────────────────────────────────────────────
# Style factors
# ────────────────────────────────────────────────────────────────────────────────

STYLE_FACTORS = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]


def load_style_factors(data_dir=DATA_DIR) -> pd.DataFrame:
    """Load the 6 Fama-French style factors as a date-indexed DataFrame."""
    ff = pd.read_parquet(data_dir / "ff_factors.parquet")
    ff = ff.set_index("date").sort_index()
    return ff[STYLE_FACTORS]


# ────────────────────────────────────────────────────────────────────────────────
# Fama-French 12-industry classification (canonical Siccodes12.txt ranges)
# ────────────────────────────────────────────────────────────────────────────────

# Ordered: first matching range wins. Ranges are disjoint by construction.
FF12_DEFINITIONS: list[tuple[str, list[tuple[int, int]]]] = [
    ("NoDur", [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799),
               (3100, 3199), (3940, 3989)]),
    ("Durbl", [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711),
               (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792),
               (3900, 3939), (3990, 3999)]),
    ("Manuf", [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099),
               (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713),
               (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
               (3830, 3839), (3860, 3899)]),
    ("Enrgy", [(1200, 1399), (2900, 2999)]),
    ("Chems", [(2800, 2829), (2840, 2899)]),
    ("BusEq", [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829),
               (7370, 7379)]),
    ("Telcm", [(4800, 4899)]),
    ("Utils", [(4900, 4949)]),
    ("Shops", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Hlth",  [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ("Money", [(6000, 6999)]),
]

# Canonical industry order (12th = residual bucket).
FF12_INDUSTRIES = [name for name, _ in FF12_DEFINITIONS] + ["Other"]


def sic_to_ff12(siccd) -> str:
    """Map a 4-digit SIC code to one of the 12 Fama-French industries.

    Returns "Other" for unmatched / unparseable codes (the FF12 residual bucket).
    """
    try:
        code = int(siccd)
    except (TypeError, ValueError):
        return "Other"
    for name, ranges in FF12_DEFINITIONS:
        for lo, hi in ranges:
            if lo <= code <= hi:
                return name
    return "Other"


# ────────────────────────────────────────────────────────────────────────────────
# Industry factors (equal-weight return per FF12 bucket, built from the universe)
# ────────────────────────────────────────────────────────────────────────────────


def build_industry_factors(
    returns_panel: pd.DataFrame,
    sic_map: pd.Series,
    min_stocks_per_industry: int = 3,
) -> pd.DataFrame:
    """Equal-weight daily industry-return factors from the universe's own returns.

    Parameters
    ----------
    returns_panel : DataFrame
        Daily returns, rows = dates, columns = permno. (Typically
        `formation_panel.pct_change().iloc[1:]`.)
    sic_map : Series
        permno -> SIC code, covering every column of `returns_panel`.
    min_stocks_per_industry : int
        Industries with fewer names than this are dropped — a 1- or 2-stock bucket
        makes a degenerate factor (a stock would load ~1.0 on its own industry).

    Returns
    -------
    DataFrame
        Rows = dates of `returns_panel`; columns = FF12 industry names that cleared
        the `min_stocks_per_industry` threshold. Each value = equal-weight mean return
        of that industry's stocks on that day.

    Note (D2.5.5): a stock is included in its own industry factor. With dozens of
    names per bucket the self-contribution is negligible; accepted for the core.
    """
    ff12 = sic_map.reindex(returns_panel.columns).map(sic_to_ff12)
    cols = {}
    for industry in FF12_INDUSTRIES:
        members = ff12.index[ff12 == industry]
        if len(members) >= min_stocks_per_industry:
            cols[industry] = returns_panel[members].mean(axis=1)
    if not cols:
        raise ValueError(
            "no FF12 industry cleared min_stocks_per_industry="
            f"{min_stocks_per_industry}"
        )
    return pd.DataFrame(cols, index=returns_panel.index)


def build_factor_panel(
    prices_panel: pd.DataFrame,
    sic_map: pd.Series,
    style_factors: pd.DataFrame | None = None,
    min_stocks_per_industry: int = 3,
) -> pd.DataFrame:
    """Assemble the full factor panel (style + industry) for a formation window.

    Parameters
    ----------
    prices_panel : DataFrame
        Formation-window prices, rows = dates, columns = permno (no NaNs) — the same
        panel handed to `distances.*`.
    sic_map : Series
        permno -> SIC code for every column of `prices_panel`.
    style_factors : DataFrame, optional
        Pre-loaded FF style factors (date-indexed). Loaded from disk if None.
    min_stocks_per_industry : int
        Passed to `build_industry_factors`.

    Returns
    -------
    DataFrame
        Rows = formation-window return dates; columns = 6 style + up-to-12 industry
        factors. Aligned (inner join) so there are no NaNs.
    """
    returns = prices_panel.pct_change().iloc[1:]
    industry = build_industry_factors(returns, sic_map, min_stocks_per_industry)

    if style_factors is None:
        style_factors = load_style_factors()
    style = style_factors.reindex(returns.index)

    panel = pd.concat([style, industry], axis=1).dropna(how="any")
    return panel
