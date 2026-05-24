"""
Spread construction & z-score signals  [Pipeline Stage 3].

Given a candidate pair (A, B), this module turns two price series into the trading
signal the backtest acts on:

   prices_a, prices_b  ──fit_hedge_ratio──>  γ, α
                       ──spread_series────>  spread_t = A_t - γ·B_t
                       ──rolling_zscore──>   z_t = (spread_t - μ_{t-1}) / σ_{t-1}

The hedge ratio γ is estimated on the formation window and **frozen** for the
trading month — that's the paper's choice. The RLM-Tukey-biweight robust variant
is a Phase-3 robustness cell.

Look-ahead protection: rolling_zscore uses `.shift(1)` on the rolling mean/std,
so the z-score at day t never includes spread_t in its own normalisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeRatio:
    """Result of fit_hedge_ratio — α, γ, residual diagnostics."""

    gamma: float           # slope: A = α + γ·B
    alpha: float           # intercept
    residual_std: float    # std of OLS residuals (formation-window spread σ)
    n_obs: int             # number of days fit


def fit_hedge_ratio(prices_a: pd.Series, prices_b: pd.Series) -> HedgeRatio:
    """Univariate OLS: prices_a = α + γ · prices_b + ε.

    Closed-form (no statsmodels needed for a single regressor):
        γ = Cov(A, B) / Var(B)
        α = mean(A) - γ · mean(B)

    Parameters
    ----------
    prices_a, prices_b : Series
        Aligned price series over the formation window. Same length, no NaNs.

    Returns
    -------
    HedgeRatio
    """
    if len(prices_a) != len(prices_b):
        raise ValueError(
            f"prices_a and prices_b length mismatch: {len(prices_a)} vs {len(prices_b)}"
        )
    if prices_a.isna().any() or prices_b.isna().any():
        raise ValueError("price series contain NaNs — clean upstream")
    n = len(prices_a)
    if n < 2:
        raise ValueError(f"need >= 2 obs to fit a regression, got {n}")

    a = prices_a.to_numpy(dtype=float)
    b = prices_b.to_numpy(dtype=float)
    a_mean = a.mean()
    b_mean = b.mean()
    a_dev = a - a_mean
    b_dev = b - b_mean

    b_var = float((b_dev * b_dev).sum())
    if b_var == 0.0:
        raise ValueError("regressor B has zero variance — cannot fit slope")

    gamma = float((a_dev * b_dev).sum() / b_var)
    alpha = float(a_mean - gamma * b_mean)
    residuals = a - (alpha + gamma * b)
    # OLS residual std with n-2 dof (two estimated params)
    residual_std = float(np.sqrt((residuals * residuals).sum() / max(n - 2, 1)))

    return HedgeRatio(gamma=gamma, alpha=alpha, residual_std=residual_std, n_obs=n)


def spread_series(
    prices_a: pd.Series, prices_b: pd.Series, hedge_ratio: HedgeRatio | float
) -> pd.Series:
    """Compute spread_t = A_t - γ·B_t.

    We do NOT subtract α here. The intercept gets absorbed into the rolling-window
    mean μ_{t-1} in rolling_zscore, which is what we actually use to z-score.

    Parameters
    ----------
    prices_a, prices_b : Series
        Aligned price series. May extend beyond the formation window into the
        trading window — this function just applies γ uniformly.
    hedge_ratio : HedgeRatio or float
        γ from fit_hedge_ratio (or a raw float for unit testing).

    Returns
    -------
    Series
        Spread, same index as prices_a.
    """
    if isinstance(hedge_ratio, HedgeRatio):
        gamma = hedge_ratio.gamma
    else:
        gamma = float(hedge_ratio)
    aligned_a, aligned_b = prices_a.align(prices_b, join="inner")
    return aligned_a - gamma * aligned_b


def rolling_zscore(spread: pd.Series, window: int = 126) -> pd.Series:
    """Rolling z-score with strict look-ahead protection.

    z_t = (spread_t - μ_{t-1}) / σ_{t-1}

    Where μ_{t-1}, σ_{t-1} are the mean/std of spread over the `window` days
    ending on (and including) day t-1 — never day t. This is enforced via
    `.shift(1)`.

    Parameters
    ----------
    spread : Series
        Output of spread_series(); single pair's spread time series.
    window : int
        Rolling window in trading days. Paper default = 126 (~6 months).

    Returns
    -------
    Series
        z-scores; first `window` entries are NaN (insufficient history).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    # shift(1) -> at day t we only see spread up to day t-1 in the rolling window
    past = spread.shift(1)
    mu = past.rolling(window=window, min_periods=window).mean()
    sigma = past.rolling(window=window, min_periods=window).std(ddof=1)
    z = (spread - mu) / sigma
    return z
