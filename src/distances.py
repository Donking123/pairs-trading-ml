"""
Pair-selection distance metrics  [Pipeline Stage 2].

A distance metric turns "how alike are two stocks?" into a single number; the
clustering step (clustering.py) then groups stocks by these numbers. This module is
the pluggable heart of the project - every metric returns a square distance matrix of
the same shape, so the clustering / trading / backtest code downstream never changes.

Metrics, built phase by phase:
    ssd_distance    Phase 1   sum of squared deviations of z-normalized prices
    pc_distance     Phase 2   partial correlation of returns, market-controlled
    beta_distance   Phase 3   Euclidean distance on factor-beta vectors  [extension]
    pca_distance    Bonus     Euclidean distance on PCA-reconstructed returns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def ssd_distance(prices: pd.DataFrame) -> pd.DataFrame:
    """SSD distance - Gatev et al. (2006); Rotondi & Russo (2025), Eq. (1).

    For each pair (X, Y) the SSD distance is the sum of squared differences between
    their *z-normalized* price series over the formation window:

        SSD(X, Y) = sum_t [ z(S^X_t) - z(S^Y_t) ]^2
        where     z(S_t) = (S_t - mean(S)) / std(S)

    Equivalently, the squared Euclidean distance between the two standardized price
    vectors. Standardizing first strips out each stock's price level and volatility,
    so the metric compares the *shape* of the price paths - not their dollar prices.

    Note: this is the paper's Eq. (1), which differs from the classic Gatev SSD on
    the price *ratio*. We follow the paper for replication fidelity.

    Parameters
    ----------
    prices : DataFrame
        Formation-window prices: rows = dates, columns = stocks (permno). Every
        column must be fully priced over the window (no NaNs) - screen upstream so
        only stocks continuously in the index/priced are passed in.

    Returns
    -------
    DataFrame
        Square SSD distance matrix; index and columns are the stock identifiers,
        diagonal is zero.
    """
    if prices.isna().any().any():
        raise ValueError(
            "prices contains NaNs - pass only stocks fully priced over the window"
        )
    if prices.shape[1] < 2:
        raise ValueError("need at least two stocks to form a distance matrix")

    # z-normalize each stock's price series over the formation window
    normed = (prices - prices.mean()) / prices.std()

    # pairwise squared Euclidean distance between the standardized price vectors
    condensed = pdist(normed.to_numpy().T, metric="sqeuclidean")
    matrix = squareform(condensed)

    return pd.DataFrame(matrix, index=prices.columns, columns=prices.columns)


def pc_distance(
    prices: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.DataFrame:
    """Partial-correlation distance - Rotondi & Russo (2025), Eq. (3) (Phase 2).

    For each pair (X, Y), market-adjust both stocks' returns by OLS-regressing each
    stock's return on the market return, then take 1 minus the correlation of the
    residuals:

        r_i,t = alpha_i + beta_i * r_market,t + tilde_r_i,t          (per-stock OLS)
        d_PC(X, Y) = 1 - corr(tilde_r_X, tilde_r_Y)

    The residual tilde_r is the *idiosyncratic* return - the part not explained by
    market beta. Pairs whose idiosyncratic returns co-move share something economically
    meaningful (industry, business model, regulatory environment), not just generic
    equity exposure. This is exactly the kind of structure that supports pairs trading.

    Why this matters vs SSD: SSD on prices groups stocks that follow the market with
    similar betas (their *trajectories* look alike). PC on residuals only groups
    stocks whose residuals co-move. So a high-beta stock and a low-beta stock with no
    real relationship - which SSD would call similar (both follow market) - get
    correctly separated by PC.

    Parameters
    ----------
    prices : DataFrame
        Formation-window prices: rows = dates, columns = stocks (permno). No NaNs.
    market_returns : Series
        Daily market return (e.g. S&P 500 total return) over the same date range.
        Must be aligned to the same DatetimeIndex as `prices` (or a superset).

    Returns
    -------
    DataFrame
        Square distance matrix in [0, 2]; index and columns are stock identifiers,
        diagonal is exactly zero. Values < 1 = positive idiosyncratic co-movement
        (candidate pairs); >= 1 = uncorrelated or negatively correlated residuals.

    Notes
    -----
    - Returns are computed as simple percentage changes from `prices`.
    - We use sample variance (ddof=1) for both market and stock returns - matches
      standard practice and the unit test's planted-noise structure.
    """
    if prices.isna().any().any():
        raise ValueError(
            "prices contains NaNs - pass only stocks fully priced over the window"
        )
    if prices.shape[1] < 2:
        raise ValueError("need at least two stocks to form a distance matrix")

    # 1. Convert prices to daily returns; drop the first row (NaN from pct_change).
    returns = prices.pct_change().iloc[1:]

    # 2. Align market returns to the returns' date index; drop any unmatched dates.
    mkt = market_returns.reindex(returns.index).dropna()
    if len(mkt) < 2:
        raise ValueError(
            "market_returns has fewer than 2 dates aligned with `prices`"
        )
    returns = returns.loc[mkt.index]

    var_mkt = float(mkt.var(ddof=1))
    if var_mkt <= 0:
        raise ValueError("market_returns has zero variance - cannot regress")
    mean_mkt = float(mkt.mean())

    # 3. Vectorised OLS market adjustment.
    #    Cov(r_i, r_mkt) for every column i = sum((r_i - mean_i) * (r_mkt - mean_mkt)) / (n-1)
    centered_mkt = mkt - mean_mkt                          # Series, length n
    mean_stocks = returns.mean()                            # Series, length k
    centered_stocks = returns.subtract(mean_stocks, axis=1)  # DataFrame n x k
    n = len(mkt)
    cov_with_mkt = centered_stocks.multiply(centered_mkt, axis=0).sum() / (n - 1)
    betas = cov_with_mkt / var_mkt                          # Series, length k
    alphas = mean_stocks - betas * mean_mkt                 # Series, length k

    # residual_i,t = r_i,t - alpha_i - beta_i * r_mkt,t
    # vectorised: predicted = alphas[None, :] + outer(mkt, betas)
    predicted_arr = (
        alphas.to_numpy()[None, :]
        + np.outer(mkt.to_numpy(), betas.to_numpy())
    )
    residuals = returns - predicted_arr

    # 4. Pairwise correlation of residuals; distance = 1 - corr.
    corr = residuals.corr()
    distance = 1.0 - corr
    # Force the diagonal to exactly zero (guards against tiny float drift).
    np.fill_diagonal(distance.values, 0.0)

    return distance


def market_adjusted_returns(
    prices: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.DataFrame:
    """Helper - return the market-adjusted residuals used inside pc_distance.

    Exposed separately so the cointegration filter (Phase 2) and any later analytics
    can reuse the same residual definition without recomputing.
    """
    if prices.isna().any().any():
        raise ValueError("prices contains NaNs")
    returns = prices.pct_change().iloc[1:]
    mkt = market_returns.reindex(returns.index).dropna()
    returns = returns.loc[mkt.index]
    var_mkt = float(mkt.var(ddof=1))
    mean_mkt = float(mkt.mean())
    mean_stocks = returns.mean()
    centered_stocks = returns.subtract(mean_stocks, axis=1)
    cov_with_mkt = centered_stocks.multiply(mkt - mean_mkt, axis=0).sum() / (len(mkt) - 1)
    betas = cov_with_mkt / var_mkt
    alphas = mean_stocks - betas * mean_mkt
    predicted_arr = (
        alphas.to_numpy()[None, :]
        + np.outer(mkt.to_numpy(), betas.to_numpy())
    )
    return returns - predicted_arr
