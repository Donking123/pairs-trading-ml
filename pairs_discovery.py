"""
pairs_discovery.py — Candidate pair selection within dynamic clusters.

Inputs
  - Russell close/return matrices from Sem1 Data
  - Cluster labels from pairs_clustering_features.build_and_cluster()

Outputs
  - Ranked candidate pairs per cluster with diagnostics:
      * 63d corr
      * hedge beta (OLS on log prices)
      * ADF p-value on spread (Engle–Granger step 2, if statsmodels present)
      * Half-life estimate from AR(1) on spread

Filters (default, aligned with proposal)
  - 63d corr >= 0.7
  - Half-life in [5, 30] trading days
  - ADF p-value < 0.05 (if available; otherwise not enforced)

Usage (CLI)
  python pairs_discovery.py --n-clusters 10 --corr-window 126 --recluster-freq 63 \
         --formation 252 --per-cluster 40 --top-k 5 --save out_pairs.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Russell 2000 path (local CSV, optional)
try:
    from pairs_clustering_features import (
        build_and_cluster,
        load_russell_close_returns,
    )
except Exception:
    build_and_cluster = None
    load_russell_close_returns = None

# SPX path (yfinance cache)
try:
    from spx_data import load_all as spx_load_all
    from pairs_feature_matrix import (
        cluster_method_a as spx_cluster_a,
        cluster_method_b as spx_cluster_b,
        cluster_fused as spx_cluster_fused,
    )
except Exception:
    spx_load_all = None


def _residualize(
    ret_series: pd.Series,
    factor_rets: pd.DataFrame,
    ridge_alpha: float = 10.0,
    use_ols: bool = False,
) -> pd.Series:
    """Remove common factor exposure; return idiosyncratic residuals.

    use_ols=True: plain OLS via lstsq (remote PC-core approach).
    use_ols=False: Ridge(alpha) for regularized stability (default).
    """
    aligned = factor_rets.reindex(ret_series.index).fillna(0.0)
    X = np.c_[np.ones(len(aligned)), aligned.values]
    y = ret_series.fillna(0.0).values
    if use_ols:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coef
    else:
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
        ridge.fit(X[:, 1:], y)
        residuals = y - ridge.predict(X[:, 1:])
    return pd.Series(residuals, index=ret_series.index)


def _ols_beta_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return slope and intercept from OLS of y on x.

    x, y: 1D arrays (or 2D with single column). Handles shape cleanup.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    X = np.c_[np.ones((x.shape[0], 1)), x]
    coef = np.linalg.lstsq(X, y, rcond=None)[0].ravel()
    return float(coef[1]), float(coef[0])


def _adf_pvalue(series: pd.Series) -> float | np.nan:
    try:
        from statsmodels.tsa.stattools import adfuller

        res = adfuller(series.dropna().values, maxlag=None, regression="c")
        return float(res[1])
    except Exception:
        return float("nan")


def _half_life(spread: pd.Series) -> float | np.nan:
    s = spread.dropna()
    if len(s) < 20:
        return float("nan")
    ds = s.diff().dropna().values
    s_lag = s.shift(1).dropna().values
    if len(ds) != len(s_lag) or len(ds) < 5:
        return float("nan")
    # Δs_t = γ * s_{t-1} + ε_t
    gamma, _ = _ols_beta_alpha(s_lag.reshape(-1, 1), ds.reshape(-1, 1))
    if gamma >= 0:
        return float("inf")
    try:
        hl = -np.log(2) / gamma
        return float(hl)
    except Exception:
        return float("nan")


@dataclass
class Filters:
    # ── Formation thresholds (new pairs) ────────────────────────────────
    min_corr63: float = 0.70
    min_hl: float = 5.0
    max_hl: float = 30.0
    max_beta: float = 4.0
    min_beta: float = 0.25
    adf_alpha: float = 0.05

    # ── Maintenance thresholds (pairs that passed last segment) ──────────
    # Looser to allow pairs through a temporary weak-cointegration window
    # without full re-qualification. Proposal §4: "p<0.10 for maintenance."
    maint_adf_alpha: float = 0.10
    maint_max_hl: float = 45.0
    maint_min_corr63: float = 0.65
    maint_passes_required: int = 2    # must have passed in ≥N of last 3 segments

    # ── Avellaneda-Lee residual spread mode ──────────────────────────────
    neutralize_factors: bool = False
    neutralize_ridge_alpha: float = 10.0
    neutralize_use_ols: bool = False  # OLS instead of Ridge for residualization


def discover_pairs(
    close_df: pd.DataFrame,
    labels: pd.Series,
    formation_window: int = 252,
    top_k_per_cluster: int = 5,
    per_cluster_cap: int = 40,
    filt: Filters = Filters(),
    factor_rets: Optional[pd.DataFrame] = None,
    prev_pairs: Optional[set] = None,
) -> pd.DataFrame:
    """Return a DataFrame of candidate pairs per cluster with diagnostics.

    When filt.neutralize_factors=True and factor_rets is provided:
      - Residualize each stock's returns against factor_rets (ridge regression)
      - Compute corr63 on RESIDUAL returns (idiosyncratic co-movement only)
      - Construct spread from CUMULATIVE RESIDUAL returns (Avellaneda-Lee §3)
      - Run ADF and HL on the residual spread
    This removes common-factor contamination from the cointegration test,
    making pair selection more stable quarter-to-quarter.
    """
    logp = np.log(close_df)
    ret_df = close_df.pct_change()
    dates = close_df.index
    end_idx = len(dates) - 1
    start_idx = max(0, end_idx - formation_window)
    L = labels.dropna().astype(int)

    use_resid = filt.neutralize_factors and factor_rets is not None

    records = []
    for cl in sorted(L.unique()):
        names = L.index[L == cl]
        nn = close_df[names].notna().sum().sort_values(ascending=False)
        names = nn.index[:per_cluster_cap]
        if len(names) < 2:
            continue

        sub_logp = logp.iloc[start_idx:end_idx + 1][names]
        sub_rets  = ret_df.iloc[start_idx:end_idx + 1][names]

        if use_resid:
            # Residualize returns for all cluster stocks against the factor set
            fret_window = factor_rets.reindex(sub_rets.index).fillna(0.0)
            resid: dict[str, pd.Series] = {}
            for nm in names:
                r = sub_rets[nm].fillna(0.0)
                resid[nm] = _residualize(r, fret_window, filt.neutralize_ridge_alpha, use_ols=filt.neutralize_use_ols)
            resid_df = pd.DataFrame(resid)
            # Cumulative residual returns → acts as "residual price" for spread
            cumresid_df = resid_df.cumsum()
            # 63d corr on RESIDUAL returns (not raw log-price changes)
            corr63 = resid_df.iloc[-63:].corr()
        else:
            corr63 = sub_logp.iloc[-63:].corr()

        for a, b in combinations(names, 2):
            # Determine if this pair is a maintenance candidate (passed last segment)
            pair_key = f"{a}|{b}"
            pair_key_rev = f"{b}|{a}"
            is_maint = prev_pairs is not None and (
                pair_key in prev_pairs or pair_key_rev in prev_pairs
            )

            # Use looser thresholds for maintenance pairs (Proposal §4)
            adf_thresh  = filt.maint_adf_alpha  if is_maint else filt.adf_alpha
            max_hl_th   = filt.maint_max_hl     if is_maint else filt.max_hl
            min_corr_th = filt.maint_min_corr63 if is_maint else filt.min_corr63

            c = corr63.loc[a, b]
            if np.isnan(c) or c < min_corr_th:
                continue

            if use_resid:
                pa = cumresid_df[a].dropna()
                pb = cumresid_df[b].dropna()
            else:
                pa = sub_logp[a].dropna()
                pb = sub_logp[b].dropna()

            idx = pa.index.intersection(pb.index)
            pa, pb = pa.loc[idx], pb.loc[idx]
            if len(idx) < 126:
                continue

            beta, alpha = _ols_beta_alpha(pb.values, pa.values)
            beta = float(beta)
            if not np.isfinite(beta) or beta < filt.min_beta or beta > filt.max_beta:
                continue
            spread = pa - (beta * pb + alpha)
            pval = _adf_pvalue(spread)
            hl = _half_life(spread)
            if not np.isnan(hl) and (hl < filt.min_hl or hl > max_hl_th):
                continue
            if not np.isnan(pval) and pval >= adf_thresh:
                continue
            records.append(
                {
                    "cluster":    int(cl),
                    "pair":       pair_key,
                    "A":          a,
                    "B":          b,
                    "corr63":     float(c),
                    "beta":       beta,
                    "adf_p":      float(pval) if np.isfinite(pval) else np.nan,
                    "half_life":  float(hl)   if np.isfinite(hl)   else np.nan,
                    "neutralized": use_resid,
                    "maintenance": is_maint,
                }
            )

    if not records:
        return pd.DataFrame(columns=["cluster", "pair", "A", "B", "corr63", "beta", "adf_p", "half_life", "neutralized", "maintenance"]).set_index("cluster")

    df = pd.DataFrame(records)
    # Rank by ADF (asc), then by half-life closeness to 10d, then by corr desc
    df["hl_score"] = (df["half_life"] - 10).abs()
    df = df.sort_values(["cluster", "adf_p", "hl_score", "corr63"], ascending=[True, True, True, False])
    # Top-K per cluster
    out = df.groupby("cluster").head(top_k_per_cluster).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["russell", "spx"], default="russell")
    ap.add_argument(
        "--method",
        choices=["a", "b", "fused", "a_coint", "c", "c_optics", "c_dbscan"],
        default="a",
        help="Clustering method: A (returns), B (beta), fused, A+cointegration, or C=partial-corr variants",
    )
    ap.add_argument("--n-clusters", type=int, default=10, help="Only for russell corr/beta KMeans path")
    ap.add_argument("--corr-window", type=int, default=126)
    ap.add_argument("--pca", type=int, default=20)
    ap.add_argument("--beta-window", type=int, default=252)
    ap.add_argument("--formation", type=int, default=252)
    ap.add_argument("--per-cluster", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--save", type=str, default="")
    # Formation filter knobs for quick iteration
    ap.add_argument("--min-corr", type=float, default=0.7, help="Minimum 63d corr filter (default 0.7)")
    ap.add_argument("--min-hl",   type=float, default=5.0, help="Minimum half-life in days (default 5)")
    ap.add_argument("--max-hl",   type=float, default=30.0, help="Maximum half-life in days (default 30)")
    ap.add_argument("--adf-alpha",type=float, default=0.05, help="ADF p-value threshold (default 0.05)")
    args = ap.parse_args()

    if args.source == "russell":
        close_df, ret_df = load_russell_close_returns()
        beta_window = None if (args.beta_window is None or args.beta_window <= 0) else args.beta_window
        feats, labs = build_and_cluster(
            end_date=None,
            corr_window=args.corr_window,
            pca_components=args.pca,
            beta_window=beta_window,
            factor_list=None,
            n_clusters=args.n_clusters,
            w_corr=0.5,
            save_features=False,
        )
    else:
        if spx_load_all is None:
            raise RuntimeError("SPX path unavailable (spx_data/pairs_feature_matrix not importable)")
        close_df, volume_df, ret_df, fclose, frets = spx_load_all(refresh=False)
        end_date = ret_df.index[-1]
        if args.method in {"a", "b", "fused", "a_coint"}:
            if args.method == "a":
                labs, k = spx_cluster_a(ret_df, end_date=end_date, window=args.corr_window)
            elif args.method == "b":
                labs_a, k = spx_cluster_a(ret_df, end_date=end_date, window=args.corr_window)
                labs = spx_cluster_b(ret_df, frets, end_date=end_date, k=k)
            elif args.method == "fused":
                labs, k = spx_cluster_fused(
                    ret_df, frets, end_date=end_date,
                    corr_window=args.corr_window, beta_window=max(args.beta_window, args.corr_window), w_ret=0.5,
                )
            else:  # a_coint
                from pairs_feature_matrix import cluster_a_coint
                labs, k = cluster_a_coint(
                    ret_df, close_df, end_date=end_date,
                    corr_window=args.corr_window, coint_window=args.formation,
                    w_coint=0.5, min_corr_prefilter=0.70,
                )
        else:
            # Method C family: partial-correlation with spectral/OPTICS/DBSCAN
            from pairs_feature_matrix import cluster_method_c_partialcorr
            algo = {"c": "spectral", "c_optics": "optics", "c_dbscan": "dbscan"}[args.method]
            labs, k = cluster_method_c_partialcorr(
                ret_df, frets, end_date=end_date, window=args.corr_window, algo=algo
            )

    filt = Filters(min_corr63=args.min_corr, min_hl=args.min_hl, max_hl=args.max_hl, adf_alpha=args.adf_alpha)
    pairs = discover_pairs(
        close_df,
        labs,
        formation_window=args.formation,
        top_k_per_cluster=args.top_k,
        per_cluster_cap=args.per_cluster,
        filt=filt,
    )
    print(pairs)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.save, index=False)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
