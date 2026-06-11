"""
Phase 6 — quantify the regime/dispersion dependence (descriptive, no backtest).

CONCLUSIONS.md recommends deploying with "a regime/dispersion filter (trade only when
dispersion/volatility is elevated)" — but that recommendation currently rests on
eyeballing the 2007-09 P&L concentration. This script quantifies it from data already
on disk:

  * cross-sectional dispersion = monthly mean of the daily cross-sectional std of
    returns across the CRSP cache universe (the strategy's raw material: how far
    stocks are flying apart each day)
  * market volatility = monthly realized vol of the S&P 500 (annualized)

against the pc_core monthly returns, two ways:
  * CONTEMPORANEOUS — mechanism: do returns coincide with high-dispersion months?
  * LAGGED (prior month) — deployability: could you have SCALED EXPOSURE using only
    information available at the month's start? (Not a tuned rule — a monotonicity
    check across quartiles.)

Run (from pairs-trading-ml/):
  python phases/phase6/notebooks/04_regime_dispersion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# walk up to project root
_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.config import PHASES_DIR
from src.panel import load_crsp_daily, load_market_returns

MONTHS = 12


def sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(MONTHS)) if sd > 1e-12 else float("nan")


def main() -> None:
    pc = pd.read_parquet(PHASES_DIR / "phase2/results/pc_core_monthly.parquet")
    ret = pc["monthly_return"]
    ret.index = pd.DatetimeIndex(ret.index).to_period("M")

    print("computing daily cross-sectional dispersion from crsp_daily ...")
    crsp = load_crsp_daily()
    disp_daily = crsp.groupby("date")["ret"].std()
    dispersion = disp_daily.groupby(disp_daily.index.to_period("M")).mean()

    mkt = load_market_returns()
    mvol = mkt.groupby(mkt.index.to_period("M")).std() * np.sqrt(252)

    df = pd.DataFrame({
        "ret": ret,
        "disp": dispersion.reindex(ret.index),
        "mvol": mvol.reindex(ret.index),
    }).dropna()
    df["disp_lag"] = df["disp"].shift(1)
    df["mvol_lag"] = df["mvol"].shift(1)

    print(f"\n{len(df)} months. Correlations with pc_core monthly return:")
    for col, label in [("disp", "dispersion (same month)"),
                       ("mvol", "mkt vol (same month)"),
                       ("disp_lag", "dispersion (PRIOR month)"),
                       ("mvol_lag", "mkt vol (PRIOR month)")]:
        sub = df[["ret", col]].dropna()
        print(f"  {label:<28}: corr = {sub['ret'].corr(sub[col]):+.3f}")

    for col, label in [("disp", "CONTEMPORANEOUS dispersion (mechanism)"),
                       ("disp_lag", "PRIOR-MONTH dispersion (deployable signal)")]:
        sub = df.dropna(subset=[col]).copy()
        sub["q"] = pd.qcut(sub[col], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
        g = sub.groupby("q", observed=True)["ret"]
        out = pd.DataFrame({
            "months": g.size(),
            "mean_ret": (g.mean() * 1e4).round(1),
            "sharpe": g.apply(sharpe).round(2),
            "hit": (g.apply(lambda x: (x > 0).mean()) * 100).round(0),
        })
        out.columns = ["months", "mean (bps/m)", "Sharpe", "hit %"]
        print(f"\nBy {label} quartile:")
        print(out.to_string())

    # the headline regime claim: P&L share from the GFC window
    gfc = df.loc["2007-07":"2009-06", "ret"]
    print(
        f"\nGFC window 2007-07..2009-06: {len(gfc)} months ({len(gfc)/len(df):.0%} of sample) "
        f"contributed {gfc.sum() / df['ret'].sum():.0%} of total arithmetic return"
    )


if __name__ == "__main__":
    main()
