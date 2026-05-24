"""
Cointegration filter — Engle-Granger ADF test + half-life filter  [Phase 2].

Inserts between Phase 1's `clusters_to_pairs()` and `fit_hedge_ratio()` in the
pipeline. Given a list of candidate pairs, drops pairs whose spread fails the
Engle-Granger stationarity test or whose mean-reversion is too fast (< 5 days,
i.e. noise) or too slow (> 60 days, i.e. won't revert within trading window).

Paper reference: Rotondi & Russo (2025), §3.3. Our convention (per
`phases/phase2/decisions.md` D2.4 and D2.5):

  * Half-life estimated via AR(1) on the spread:
        s_t = c + ρ · s_{t-1} + ε_t
        τ_{1/2} = -ln(2) / ln(ρ),  defined only for 0 < ρ < 1.

  * Engle-Granger direction: we run BOTH A-on-B and B-on-A regressions and take
    the direction with the lower (more cointegrated) ADF p-value. Small
    acceptance bias is documented and accepted.

  * Default thresholds (paper):
        p-value < 0.05
        5 ≤ half-life ≤ 60 trading days
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from src.config import HALF_LIFE_BOUNDS, COINTEGRATION_P_THRESHOLD


# ────────────────────────────────────────────────────────────────────────────────
# data classes
# ────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CointegrationResult:
    """Engle-Granger test outcome for one pair."""

    adf_pvalue: float          # min p-value across both regression directions
    gamma: float               # OLS γ from the WINNING direction (lower p-value)
    alpha: float               # OLS α from the winning direction
    direction: str             # "A_on_B" or "B_on_A" — which gave the lower p-value
    half_life: float           # AR(1) half-life of the residual spread (in days)
    n_obs: int

    @property
    def is_stationary(self) -> bool:
        """Spread residual rejects unit root at the configured threshold."""
        return self.adf_pvalue < COINTEGRATION_P_THRESHOLD

    @property
    def has_tradeable_half_life(self) -> bool:
        """Half-life within [HALF_LIFE_BOUNDS[0], HALF_LIFE_BOUNDS[1]] days."""
        lo, hi = HALF_LIFE_BOUNDS
        return (
            not np.isnan(self.half_life)
            and lo <= self.half_life <= hi
        )

    @property
    def passes_filter(self) -> bool:
        """Combined gate: stationary AND tradeable half-life."""
        return self.is_stationary and self.has_tradeable_half_life


# ────────────────────────────────────────────────────────────────────────────────
# half-life estimation (AR(1) on the spread)
# ────────────────────────────────────────────────────────────────────────────────


def half_life_ar1(spread: pd.Series) -> float:
    """Estimate the mean-reversion half-life of a spread via AR(1).

    Fit:        s_t = c + ρ · s_{t-1} + ε_t  (OLS)
    Half-life:  τ_{1/2} = -ln(2) / ln(ρ)

    Returns np.nan when ρ is outside (0, 1) — i.e. the spread is either
    non-stationary (ρ >= 1) or over-shooting (ρ <= 0), in which case the
    half-life is undefined.

    Parameters
    ----------
    spread : Series
        Spread series (typically OLS residual from Engle-Granger step 1).

    Returns
    -------
    float
        Half-life in the same time units as the spread index (typically days).
    """
    s = spread.dropna()
    if len(s) < 10:
        return float("nan")
    # OLS: s_t = c + ρ · s_{t-1} + ε
    y = s.iloc[1:].to_numpy()
    x = s.iloc[:-1].to_numpy()
    x_with_const = sm.add_constant(x)
    try:
        model = sm.OLS(y, x_with_const).fit()
        rho = float(model.params[1])
    except Exception:
        return float("nan")
    if rho <= 0 or rho >= 1:
        return float("nan")
    return -np.log(2.0) / np.log(rho)


# ────────────────────────────────────────────────────────────────────────────────
# Engle-Granger two-step test
# ────────────────────────────────────────────────────────────────────────────────


def _ols_residuals(y: pd.Series, x: pd.Series) -> tuple[pd.Series, float, float]:
    """OLS regress y on x → (residuals, gamma, alpha)."""
    yy, xx = y.align(x, join="inner")
    xx_with_const = sm.add_constant(xx.to_numpy())
    model = sm.OLS(yy.to_numpy(), xx_with_const).fit()
    alpha = float(model.params[0])
    gamma = float(model.params[1])
    residuals = yy - (alpha + gamma * xx)
    return residuals, gamma, alpha


def engle_granger(
    prices_a: pd.Series,
    prices_b: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
) -> CointegrationResult:
    """Two-step Engle-Granger test of cointegration between A and B.

    Step 1: OLS regression for both directions (A on B, B on A) → residuals.
    Step 2: ADF test on each residual; reject the unit-root null = stationary.
    Decision: take the direction with the lower (better) ADF p-value (D2.5).

    Half-life estimation uses the winning direction's residual.

    Parameters
    ----------
    prices_a, prices_b : Series
        Aligned formation-window price series. Same length, no NaNs.
    name_a, name_b : str
        Optional labels used only in the `direction` field of the result.

    Returns
    -------
    CointegrationResult
    """
    if len(prices_a) != len(prices_b):
        raise ValueError(
            f"prices_a and prices_b length mismatch: {len(prices_a)} vs {len(prices_b)}"
        )
    if prices_a.isna().any() or prices_b.isna().any():
        raise ValueError("price series contain NaNs - clean upstream")
    n = len(prices_a)
    if n < 30:
        raise ValueError(f"need >= 30 obs for ADF test, got {n}")

    # Both directions
    residuals_AB, gamma_AB, alpha_AB = _ols_residuals(prices_a, prices_b)
    residuals_BA, gamma_BA, alpha_BA = _ols_residuals(prices_b, prices_a)

    # ADF test on each — autolag="AIC" picks the lag order automatically.
    # `regression="c"` includes a constant (paper convention).
    p_AB = float(adfuller(residuals_AB.dropna(), autolag="AIC", regression="c")[1])
    p_BA = float(adfuller(residuals_BA.dropna(), autolag="AIC", regression="c")[1])

    if p_AB <= p_BA:
        winning_residuals = residuals_AB
        winning_p = p_AB
        winning_gamma = gamma_AB
        winning_alpha = alpha_AB
        winning_direction = f"{name_a}_on_{name_b}"
    else:
        winning_residuals = residuals_BA
        winning_p = p_BA
        winning_gamma = gamma_BA
        winning_alpha = alpha_BA
        winning_direction = f"{name_b}_on_{name_a}"

    hl = half_life_ar1(winning_residuals)

    return CointegrationResult(
        adf_pvalue=winning_p,
        gamma=winning_gamma,
        alpha=winning_alpha,
        direction=winning_direction,
        half_life=hl,
        n_obs=n,
    )


# ────────────────────────────────────────────────────────────────────────────────
# pair filter
# ────────────────────────────────────────────────────────────────────────────────


def filter_cointegrated_pairs(
    pairs: list[tuple[int, int]],
    prices_panel: pd.DataFrame,
    p_threshold: float = COINTEGRATION_P_THRESHOLD,
    half_life_bounds: tuple[float, float] = HALF_LIFE_BOUNDS,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], CointegrationResult]]:
    """Apply Engle-Granger + half-life filter to a list of candidate pairs.

    Parameters
    ----------
    pairs : list of (permno_a, permno_b)
        Output of `clustering.clusters_to_pairs()`.
    prices_panel : DataFrame
        Formation-window prices: rows = dates, columns = permno. Same panel used
        to compute distances.
    p_threshold : float
        ADF p-value threshold; pairs with p < threshold are deemed stationary.
        Default from `src/config.COINTEGRATION_P_THRESHOLD` (=0.05).
    half_life_bounds : (float, float)
        (lower, upper) bounds in days; both inclusive. Default from
        `src/config.HALF_LIFE_BOUNDS` (= (5.0, 60.0)).

    Returns
    -------
    (kept_pairs, results)
        kept_pairs : list of pairs that passed both filters
        results : dict {pair -> CointegrationResult} for ALL input pairs
                  (useful for diagnostics — see which pairs were dropped and why)
    """
    kept: list[tuple[int, int]] = []
    results: dict[tuple[int, int], CointegrationResult] = {}
    lo, hi = half_life_bounds

    for a, b in pairs:
        if a not in prices_panel.columns or b not in prices_panel.columns:
            continue
        try:
            res = engle_granger(
                prices_panel[a], prices_panel[b],
                name_a=str(a), name_b=str(b),
            )
        except (ValueError, Exception):
            continue
        results[(a, b)] = res
        if (
            res.adf_pvalue < p_threshold
            and not np.isnan(res.half_life)
            and lo <= res.half_life <= hi
        ):
            kept.append((a, b))

    return kept, results
