"""
HMM vol-regime classifier for pairs trading overlay.

Features (only CRSP/SPY data — no external credit data needed):
  - log_rvol:  SPY 20-day realized vol, annualized  → overall vol regime
  - log_disp:  cross-sectional dispersion of CRSP constituent daily returns,
               20-day rolling mean  → hard/easy pairs environment

Two features instead of three allows coverage from 2003 to 2025 (HY OAS from
FRED only runs from 2014 in the local data, so using it would truncate the IS
backtest).  Dispersion + vol capture the same calm/stressed/crisis structure
because they co-move with credit stress (2008-09, 2020-03 clearly visible).

Outputs daily state posteriors and Viterbi argmax to data/hmm_regimes.parquet.
States are sorted by ascending mean log_rvol so labels are stable across refits:
  0 = calm, 1 = stressed, 2 = crisis

Usage:
  python regime_hmm.py            # fit + save
  python regime_hmm.py --plot     # fit + save + sanity plot
  python regime_hmm.py --n-states 2  # 2-state (calm/crisis) variant
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

DATA_DIR         = Path(__file__).resolve().parent / "data"
HMM_FEATURES_PATH = DATA_DIR / "hmm_features.parquet"   # lightweight (from wrds_pull step 5/6)
CRSP_PATH        = DATA_DIR / "crsp_close.parquet"      # legacy fallback (large)
FACTOR_PATH      = DATA_DIR / "factor_close.parquet"    # legacy fallback (large)
OUT_PATH         = DATA_DIR / "hmm_regimes.parquet"

SEED = 42
EWMA_SPAN = 5        # 5-day EWMA smoothing of raw features
RVOL_WINDOW = 20     # realized vol lookback in trading days
MIN_TRAIN_YEARS = 2  # years of history before first refit


def _load_features(
    features_path: Path = HMM_FEATURES_PATH,
    factor_path: Path = FACTOR_PATH,
    crsp_path: Path = CRSP_PATH,
) -> pd.DataFrame:
    """Build 2-feature DataFrame: log_rvol (SPY realized vol) + log_disp (cross-sectional dispersion).

    Preferred source: hmm_features.parquet (tiny — two scalars per day, pulled via
    wrds_pull get_hmm_features). Falls back to the legacy large parquet files if
    hmm_features.parquet does not exist.
    """
    if features_path.exists():
        raw = pd.read_parquet(features_path)
        # sp500_ret is already a daily return; convert to realized vol
        log_ret = np.log1p(raw["sp500_ret"].clip(-0.99))
        rvol = log_ret.rolling(RVOL_WINDOW).std() * np.sqrt(252)
        disp_smooth = raw["disp"].rolling(RVOL_WINDOW).mean()
    else:
        # Legacy fallback: reconstruct from wide price matrices
        fc = pd.read_parquet(factor_path)
        px = fc["SPY"].dropna()
        log_ret_spy = np.log(px / px.shift(1)).dropna()
        rvol = log_ret_spy.rolling(RVOL_WINDOW).std() * np.sqrt(252)

        crsp = pd.read_parquet(crsp_path)
        stock_rets = crsp.pct_change()
        disp = stock_rets.std(axis=1)
        disp_smooth = disp.rolling(RVOL_WINDOW).mean()

    idx = rvol.dropna().index.intersection(disp_smooth.dropna().index)
    df = pd.DataFrame({
        "rvol": rvol.loc[idx],
        "disp": disp_smooth.loc[idx],
    }).dropna().ffill().dropna()

    df_s = df.ewm(span=EWMA_SPAN, adjust=False).mean()

    feat = pd.DataFrame(index=df_s.index)
    feat["log_rvol"] = np.log(df_s["rvol"].clip(1e-4))
    feat["log_disp"] = np.log(df_s["disp"].clip(1e-6))

    return feat


def _anchor_states(model: GaussianHMM) -> np.ndarray:
    """Permutation that sorts states by ascending mean log_rvol."""
    return np.argsort(model.means_[:, 0])


def fit_hmm_expanding(
    feat: pd.DataFrame,
    n_states: int = 3,
    seed: int = SEED,
) -> pd.DataFrame:
    """Annual expanding-window HMM fits — no look-ahead bias.

    For each year Y (starting MIN_TRAIN_YEARS after data start),
    fit on all data through Dec 31 of Y-1, assign states to year Y.
    """
    years = sorted(feat.index.year.unique())
    first_year = years[MIN_TRAIN_YEARS]

    all_rows: list[pd.DataFrame] = []

    for y in years:
        if y < first_year:
            continue
        train_end = pd.Timestamp(f"{y - 1}-12-31")
        test_idx = feat.index[feat.index.year == y]
        if test_idx.empty:
            continue

        train = feat.loc[:train_end].dropna()
        if len(train) < 60:
            continue

        mu_tr  = train.mean()
        sig_tr = train.std().replace(0, 1.0)
        X_train = ((train - mu_tr) / sig_tr).values
        X_test  = ((feat.loc[test_idx] - mu_tr) / sig_tr).values

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            init_params="kmeans",
            random_state=seed,
        )
        model.fit(X_train)

        perm = _anchor_states(model)
        inv_perm = np.argsort(perm)

        X_full = np.vstack([X_train, X_test])
        posteriors_full = model.predict_proba(X_full)
        viterbi_full    = model.predict(X_full)

        n_train = len(X_train)
        posteriors = posteriors_full[n_train:][:, perm]
        viterbi    = inv_perm[viterbi_full[n_train:]]

        if n_states == 3:
            chunk = pd.DataFrame({
                "p_calm":     posteriors[:, 0],
                "p_stressed": posteriors[:, 1],
                "p_crisis":   posteriors[:, 2],
                "state":      viterbi,
            }, index=test_idx)
        else:
            chunk = pd.DataFrame({
                "p_calm":   posteriors[:, 0],
                "p_crisis": posteriors[:, -1],
                "state":    viterbi,
            }, index=test_idx)

        all_rows.append(chunk)

    return pd.concat(all_rows).sort_index() if all_rows else pd.DataFrame()


def build_regime_scale(
    regimes: pd.DataFrame,
    w_calm: float = 1.0,
    w_stressed: float = 0.5,
    w_crisis: float = 0.0,
) -> pd.Series:
    if "p_stressed" in regimes.columns:
        return (
            w_calm     * regimes["p_calm"]
            + w_stressed * regimes["p_stressed"]
            + w_crisis   * regimes["p_crisis"]
        )
    return w_calm * regimes["p_calm"] + w_crisis * regimes["p_crisis"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit HMM vol-regime classifier (2-feature)")
    ap.add_argument("--n-states", type=int, default=3, choices=[2, 3])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    print("Loading features (SPY realized vol + CRSP cross-sectional dispersion)...")
    feat = _load_features()
    print(f"  {len(feat)} trading days: {feat.index[0].date()} to {feat.index[-1].date()}")

    print(f"Fitting {args.n_states}-state HMM (annual expanding-window refits)...")
    regimes = fit_hmm_expanding(feat, n_states=args.n_states)

    if regimes.empty:
        print("ERROR: no regimes produced — check data availability.")
        return

    regimes["regime_scale"] = build_regime_scale(regimes)

    out_path = Path(args.out)
    regimes.to_parquet(out_path)
    print(f"Saved {len(regimes)} rows → {out_path}")
    print(f"  Date range: {regimes.index.min().date()} to {regimes.index.max().date()}")

    state_names = {0: "calm", 1: "stressed", 2: "crisis"}
    print("\nState distribution:")
    for s, cnt in regimes["state"].value_counts().sort_index().items():
        print(f"  State {s} ({state_names.get(s, s)}): {cnt} days ({cnt / len(regimes) * 100:.1f}%)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        feat_a = feat.loc[regimes.index]
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        colors = {0: "forestgreen", 1: "orange", 2: "crimson"}

        for s in sorted(regimes["state"].unique()):
            mask = regimes["state"] == s
            for ax in axes:
                ax.fill_between(regimes.index, 0, 1, where=mask,
                                color=colors.get(s, "grey"), alpha=0.25,
                                label=state_names.get(s, str(s)) if ax == axes[0] else None)

        axes[0].plot(feat_a.index, np.exp(feat_a["log_rvol"]) * 100, "k-", lw=0.7)
        axes[0].set_ylabel("SPY Realized Vol (%)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title("HMM Regime States (2-feature: vol + dispersion)")

        axes[1].plot(feat_a.index, np.exp(feat_a["log_disp"]) * 100, "navy", lw=0.7)
        axes[1].set_ylabel("CRSP Dispersion (%)")

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = DATA_DIR / "hmm_regime_plot.png"
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved → {fig_path}")


if __name__ == "__main__":
    main()
