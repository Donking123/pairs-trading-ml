"""
02_rolling_betas.py
───────────────────
For each rolling formation window, regresses every stock's daily returns
against all factor ETF returns using Ridge regression (RidgeCV with
cross-validated λ). Ridge is used instead of OLS because the 15-30 factor
ETFs are highly correlated (XLF, XLK, SPY all load on the same broad market
factor), making OLS betas unstable. Ridge's L2 penalty shrinks and stabilises
the beta estimates — this is the "Ridge regression" approach the professor
described.

Output: one parquet per window saved to data/processed/betas/
        rows = permnos, columns = factor names

Also writes data/processed/window_index.parquet

Runtime: ~5-10 min for S&P 500 with 15 factors, parallelised.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from config import (
    DATA_RAW, DATA_PROC,
    FORMATION_DAYS, TRADING_DAYS, ROLL_STEP_DAYS,
    MIN_OBS_FRAC, FACTOR_NAMES,
    N_JOBS,
)

BETAS_DIR = DATA_PROC / "betas"
BETAS_DIR.mkdir(parents=True, exist_ok=True)

# Ridge regularisation candidates — cross-validated per stock per window
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# ── Helpers ────────────────────────────────────────────────────────────────────
def strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Force DatetimeIndex to tz-naive regardless of what parquet stored."""
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_convert(None) if idx.tz is not None else idx
    return df


# ── Window index ───────────────────────────────────────────────────────────────
def build_window_index(all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    windows = []
    i = 0
    while True:
        f_start   = all_dates[i]
        f_end_idx = i + FORMATION_DAYS - 1
        if f_end_idx >= len(all_dates):
            break
        f_end = all_dates[f_end_idx]

        t_start_idx = f_end_idx + 1
        t_end_idx   = t_start_idx + TRADING_DAYS - 1
        if t_end_idx >= len(all_dates):
            break

        windows.append({
            "formation_start": f_start,
            "formation_end":   f_end,
            "trading_start":   all_dates[t_start_idx],
            "trading_end":     all_dates[t_end_idx],
        })
        i += ROLL_STEP_DAYS

    df  = pd.DataFrame(windows)
    out = DATA_PROC / "window_index.parquet"
    df.to_parquet(out, index=False)
    print(f"  → {len(df)} rolling windows  |  saved to {out}")
    return df


# ── One full window (joblib worker) ───────────────────────────────────────────
def compute_betas_one_window(
    f_start: pd.Timestamp,
    f_end:   pd.Timestamp,
    stock_returns:  pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> str:
    """
    Fits RidgeCV for every stock in the formation window.
    Ridge is instantiated inside the worker to avoid Mac joblib pickling issues.
    Returns the saved parquet path, or "" if nothing computed.
    """
    # Integer-position slicing avoids tz/type issues with .loc[]
    dates   = stock_returns.index
    i_start = int(dates.searchsorted(f_start))
    i_end   = int(dates.searchsorted(f_end, side="right"))

    sw = stock_returns.iloc[i_start:i_end].copy()
    fw = factor_returns.iloc[i_start:i_end].copy()

    # Ensure both slices have tz-naive index for pd.concat alignment
    sw.index = sw.index.tz_convert(None) if sw.index.tz is not None else sw.index
    fw.index = fw.index.tz_convert(None) if fw.index.tz is not None else fw.index

    # Keep only factors present in this window; drop any that are entirely NaN
    # (e.g. XLC launched 2018, XLRE launched 2015 — all NaN in earlier windows)
    available = [c for c in FACTOR_NAMES if c in fw.columns]
    if not available:
        return ""
    fw = fw[available].dropna(axis=1, how="all")
    if fw.empty:
        return ""
    fw_cols = fw.columns.tolist()

    min_obs = int(len(sw) * MIN_OBS_FRAC)
    results = {}

    for permno in sw.columns:
        # Align stock and factor returns, drop any NaN rows jointly
        combined = pd.concat([sw[permno], fw], axis=1).dropna()
        if len(combined) < min_obs:
            continue

        # to_numpy(dtype=float) handles pandas nullable Float64Dtype
        # which .values alone cannot convert for sklearn/numpy operations
        y = combined.iloc[:, 0].to_numpy(dtype=float)
        X = combined.iloc[:, 1:].to_numpy(dtype=float)

        # Basic sanity checks
        if not (np.isfinite(y).all() and np.isfinite(X).all()):
            continue
        if np.std(y) < 1e-10:
            continue

        # Ridge regression with cross-validated λ
        # StandardScaler on X so all factor betas are on comparable scale
        # (sector ETFs and SPY have different return magnitudes)
        try:
            scaler  = StandardScaler()
            X_sc    = scaler.fit_transform(X)
            ridge   = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)
            ridge.fit(X_sc, y)
            # Unscale betas back to original factor return units
            betas = ridge.coef_ / scaler.scale_
            results[permno] = betas
        except Exception:
            continue

    if not results:
        return ""

    beta_df = pd.DataFrame.from_dict(results, orient="index", columns=fw_cols)
    beta_df.index.name = "permno"

    fname = f"betas_{f_start.strftime('%Y%m%d')}_{f_end.strftime('%Y%m%d')}.parquet"
    out   = BETAS_DIR / fname
    beta_df.to_parquet(out)
    return str(out)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data…")
    stock_returns  = strip_tz(pd.read_parquet(DATA_RAW / "stock_returns.parquet"))
    factor_returns = strip_tz(pd.read_parquet(DATA_RAW / "factor_returns.parquet"))
    # Cast to float64 — parquet nullable Float64Dtype causes TypeError in sklearn
    stock_returns  = stock_returns.astype("float64")
    factor_returns = factor_returns.astype("float64")

    common_dates   = stock_returns.index.intersection(factor_returns.index)
    stock_returns  = stock_returns.loc[common_dates]
    factor_returns = factor_returns.loc[common_dates]

    print(f"  Stocks : {stock_returns.shape[1]:,} permnos  ×  {stock_returns.shape[0]:,} days")
    print(f"  Factors: {factor_returns.shape[1]:,} factors ×  {factor_returns.shape[0]:,} days\n")

    print("Building rolling window index…")
    windows = build_window_index(common_dates)
    print()

    # ── Sanity check: run window 0 serially before parallelising ──────────
    print("Sanity-checking window 0…")
    w0 = windows.iloc[0]
    test_path = compute_betas_one_window(
        w0["formation_start"], w0["formation_end"],
        stock_returns, factor_returns
    )
    if not test_path:
        raise RuntimeError(
            "Window 0 returned no betas.\n"
            "Check: stock_returns and factor_returns share dates, "
            "and MIN_OBS_FRAC in config.py is not too high."
        )
    sample = pd.read_parquet(test_path)
    print(f"  OK — {sample.shape[0]:,} stocks × {sample.shape[1]} factors\n")

    print(f"Computing all {len(windows)} windows  "
          f"({stock_returns.shape[1]:,} stocks each)…")
    print(f"  Ridge regression (RidgeCV, λ cross-validated)  |  n_jobs={N_JOBS}\n")

    paths = Parallel(n_jobs=N_JOBS)(
        delayed(compute_betas_one_window)(
            row["formation_start"], row["formation_end"],
            stock_returns, factor_returns
        )
        for _, row in tqdm(windows.iterrows(), total=len(windows), desc="  Windows")
    )

    n_ok = sum(1 for p in paths if p)
    print(f"\n  → {n_ok}/{len(windows)} windows completed.")
    print(f"  → Beta files: {BETAS_DIR}")
    print("\nNext: python 03_clustering.py")
