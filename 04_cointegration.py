"""
04_cointegration.py
───────────────────
For each formation window:
  1. Loads within-cluster stock pairs (from 03_clustering.py)
  2. Tests each pair for cointegration (Engle-Granger ADF on price residuals)
  3. Estimates the hedge ratio β via Ridge regression (consistent with 02)
  4. Computes spread mean and std for z-score normalisation in backtesting

Output: data/processed/pairs/pairs_YYYYMMDD_YYYYMMDD.parquet
  columns: permno_a, permno_b, cluster_id, pvalue, half_life,
           hedge_ratio, spread_mean, spread_std

Also writes data/processed/pairs_summary.csv

Runtime: ~10-20 min for S&P 500 (serial execution for Mac reliability).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from statsmodels.tsa.stattools import coint
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import warnings

from config import (
    DATA_RAW, DATA_PROC,
    COINT_PVALUE_THRESHOLD,
    MIN_OBS_FRAC, FORMATION_DAYS,
    HALFLIFE_MIN_DAYS, HALFLIFE_MAX_DAYS,
    MAX_CLUSTER_SIZE,
)

CLUSTER_DIR = DATA_PROC / "clusters"
PAIRS_DIR   = DATA_PROC / "pairs"
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# ── Helpers ────────────────────────────────────────────────────────────────────
def strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_convert(None) if idx.tz is not None else idx
    return df


# ── Test one pair ─────────────────────────────────────────────────────────────
def test_pair(
    permno_a: int,
    permno_b: int,
    cluster_id: int,
    price_a: pd.Series,
    price_b: pd.Series,
) -> dict | None:
    """
    Engle-Granger cointegration test on one pair.

    Step 1: align price series, require minimum joint observations.
    Step 2: statsmodels coint() — OLS regression then ADF on residuals.
    Step 3: if p-value passes, re-estimate hedge ratio β via Ridge regression
            (consistent with the factor-beta estimation in 02_rolling_betas.py).
    Step 4: compute spread statistics for z-score normalisation.

    Returns a dict if the pair passes, None otherwise.
    """
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    combined.columns = ["a", "b"]

    min_obs = int(FORMATION_DAYS * MIN_OBS_FRAC)
    if len(combined) < max(min_obs, 60):
        return None

    # Use .to_numpy(dtype=float) not .values — parquet-loaded data may use
    # pandas nullable Float64Dtype which sklearn cannot interpret
    pa = combined["a"].to_numpy(dtype=float)
    pb = combined["b"].to_numpy(dtype=float)

    # Engle-Granger cointegration test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, pvalue, _ = coint(pa, pb)
        except Exception:
            return None

    if pvalue > COINT_PVALUE_THRESHOLD:
        return None

    # Half-life filter: estimate how fast the spread mean-reverts using OLS.
    # Run AR(1) on spread differences: Δspread_t = α + β·spread_{t-1} + ε
    # half-life = -log(2) / log(1 + β)
    # Reject pairs where mean-reversion is too slow (>60 days, hits stop-loss
    # before converging) or too fast (<5 days, microstructure noise).
    try:
        # Quick OLS spread for half-life check (use coint's implicit OLS hedge)
        from sklearn.linear_model import LinearRegression
        _ols = LinearRegression(fit_intercept=True)
        _ols.fit(pb.reshape(-1, 1), pa)
        _spread_hl = pa - float(_ols.coef_[0]) * pb
        _delta  = np.diff(_spread_hl)
        _lagged = _spread_hl[:-1]
        _ar = LinearRegression(fit_intercept=True)
        _ar.fit(_lagged.reshape(-1, 1), _delta)
        _beta = float(_ar.coef_[0])
        if _beta >= 0:           # no mean-reversion at all
            return None
        half_life = -np.log(2) / np.log(1 + _beta)
        if not (HALFLIFE_MIN_DAYS <= half_life <= HALFLIFE_MAX_DAYS):
            return None
    except Exception:
        return None
    # Regress price_a on price_b: price_a = intercept + β * price_b
    # Ridge instantiated here (inside function) — avoids joblib pickling issues on Mac
    try:
        # Extra guard: cumprod can produce inf/nan if any return = -1.0 (stock → 0)
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            return None
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)
        ridge.fit(pb.reshape(-1, 1), pa)
        hedge_ratio = float(ridge.coef_[0])
    except Exception:
        return None

    spread      = pa - hedge_ratio * pb
    spread_mean = float(spread.mean())
    spread_std  = float(spread.std())

    if spread_std < 1e-8:
        return None

    return {
        "permno_a":    int(permno_a),
        "permno_b":    int(permno_b),
        "cluster_id":  int(cluster_id),
        "pvalue":      float(pvalue),
        "half_life":   round(float(half_life), 1),
        "hedge_ratio": hedge_ratio,
        "spread_mean": spread_mean,
        "spread_std":  spread_std,
    }


# ── One window ────────────────────────────────────────────────────────────────
def process_one_window(
    cluster_file: Path,
    sr: pd.DataFrame,
) -> tuple[str, int, int]:
    """
    Tests all within-cluster pairs for one formation window.
    Returns (output_path, n_candidates, n_passing).

    stock_returns is passed in (loaded once in __main__) rather than re-read
    from disk each call — avoids 131x redundant parquet reads.
    """
    date_str = cluster_file.stem.replace("clusters_", "")
    f_start, f_end = date_str[:8], date_str[9:]

    try:
        clusters = pd.read_parquet(cluster_file)
    except Exception as e:
        print(f"  ⚠  Skipping corrupted cluster file: {cluster_file.name} ({e})")
        return ("", 0, 0)
    clusters  = clusters[clusters["cluster_id"] >= 0]
    if clusters.empty:
        return ("", 0, 0)

    ret_window = sr.loc[pd.Timestamp(f_start):pd.Timestamp(f_end)]
    if ret_window.empty:
        return ("", 0, 0)

    # Reconstruct price series from returns (no fillna — keep NaNs for alignment)
    price_window = (1 + ret_window).cumprod()
    price_window.columns = price_window.columns.astype(int)
    valid_permnos = set(price_window.columns.tolist())

    results      = []
    n_candidates = 0

    for cid, grp in clusters.groupby("cluster_id"):
        in_cluster = [
            int(p) for p in grp["permno"].tolist()
            if int(p) in valid_permnos
        ]
        if len(in_cluster) < 2:
            continue
        if len(in_cluster) > MAX_CLUSTER_SIZE:
            continue   # skip giant catch-all clusters

        for pa, pb in combinations(in_cluster, 2):
            n_candidates += 1
            rec = test_pair(pa, pb, cid, price_window[pa], price_window[pb])
            if rec is not None:
                results.append(rec)

    if not results:
        return ("", n_candidates, 0)

    pairs_df = pd.DataFrame(results)
    out      = PAIRS_DIR / f"pairs_{date_str}.parquet"
    pairs_df.to_parquet(out, index=False)
    return (str(out), n_candidates, len(results))


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cluster_files = sorted(CLUSTER_DIR.glob("clusters_*.parquet"))
    if not cluster_files:
        raise FileNotFoundError(
            f"No cluster files in {CLUSTER_DIR}. Run 03_clustering.py first."
        )

    print(f"Testing cointegration for {len(cluster_files)} windows  "
          f"(p < {COINT_PVALUE_THRESHOLD}, max cluster size = {MAX_CLUSTER_SIZE})…")
    print(f"  Hedge ratio estimation: RidgeCV\n")

    # Load stock returns once — passed into each window call rather than
    # re-read from disk 131 times.
    print("Loading stock returns…")
    sr = pd.read_parquet(DATA_RAW / "stock_returns.parquet").astype("float64")
    sr = strip_tz(sr)
    sr.columns = sr.columns.astype(int)
    print(f"  → {sr.shape[1]} stocks × {sr.shape[0]} days loaded\n")

    outputs = []
    for cf in tqdm(cluster_files, desc="  Windows"):
        outputs.append(process_one_window(cf, sr))

    summary_rows = []
    for (path, n_cand, n_pass), cf in zip(outputs, cluster_files):
        date_str  = cf.stem.replace("clusters_", "")
        pass_rate = (n_pass / n_cand * 100) if n_cand > 0 else 0
        summary_rows.append({
            "window":        date_str,
            "n_candidates":  n_cand,
            "n_pairs":       n_pass,
            "pass_rate_pct": round(pass_rate, 1),
        })

    summary_df  = pd.DataFrame(summary_rows).set_index("window")
    summary_out = DATA_PROC / "pairs_summary.csv"
    summary_df.to_csv(summary_out)

    total_cand = summary_df["n_candidates"].sum()
    total_pass = summary_df["n_pairs"].sum()

    print(f"\n── Cointegration summary ──────────────────────────────")
    print(f"  Total candidate pairs : {total_cand:,}")
    print(f"  Total passing pairs   : {total_pass:,}  "
          f"({'N/A' if total_cand == 0 else f'{total_pass/total_cand*100:.1f}%'} pass rate)")
    print(f"  Avg pairs per window  : {summary_df['n_pairs'].mean():.1f}")
    print(f"\n  Summary saved → {summary_out}")
    print("\nNext: python 05_backtest.py")
