"""
Phase 2.5 — P&L attribution for the factor-beta cells.

Applies the same diagnostic battery used in Phase 1/2 (bimodal duration finding,
regime concentration, sector/direction attribution) to the factor-beta backtest,
with PC core alongside for contrast. Answers: WHERE does factor-beta's P&L come
from, and HOW does it differ from PC?

Sections:
  1. Holding-duration stats
  2. Per-trade P&L distribution (mean/median bps, skew, kurtosis, tails)
  3. The BIMODAL pattern by exit reason  (the load-bearing lever)
  4. Duration buckets (where trades flip profitable → losing)
  5. Regime concentration (crisis vs calm)
  6. Direction balance (long vs short spread)
  7. FF12 sector-pair concentration

Run AFTER 01_run_factor_backtest.py:
  python phases/phase2_5/notebooks/03_factor_attribution.py

Cells without results are skipped (safe to run mid-backtest).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.config import PHASE2_5_DIR, PHASE2_DIR
from src.factors import sic_to_ff12
from src.panel import load_crsp_daily, siccd_lookup

# cell_name -> results dir
CELLS = [
    ("pc_core",         PHASE2_DIR),    # contrast baseline
    ("factor_core",     PHASE2_5_DIR),
    ("factor_filtered", PHASE2_5_DIR),
]


def load_trades(cell: str, rdir: Path) -> pd.DataFrame | None:
    path = rdir / "results" / f"{cell}_trades.parquet"
    if not path.exists():
        return None
    t = pd.read_parquet(path)
    t["hold_days"] = (t["exit_date"] - t["entry_date"]).dt.days
    return t


def section_duration(t: pd.DataFrame) -> None:
    d = t["hold_days"]
    print(f"  duration: mean {d.mean():.1f}d  median {d.median():.0f}d  "
          f"p25/p75 {d.quantile(.25):.0f}/{d.quantile(.75):.0f}d  max {d.max():.0f}d")


def section_pnl_dist(t: pd.DataFrame) -> None:
    r = t["round_trip_return"].to_numpy()
    print(f"  per-trade P&L: mean {np.nanmean(r)*1e4:+.0f}bps  median {np.nanmedian(r)*1e4:+.0f}bps  "
          f"std {np.nanstd(r, ddof=1)*100:.2f}%  skew {skew(r, nan_policy='omit'):+.2f}  "
          f"exkurt {kurtosis(r, nan_policy='omit'):+.1f}")
    print(f"  best {np.nanmax(r)*100:+.1f}%   worst {np.nanmin(r)*100:+.1f}%")


def section_bimodal(t: pd.DataFrame) -> None:
    print("  exit reason       n      %    mean_dur  mean_bps  win%    Σreturn")
    for reason in ["reversion", "force_close", "delisting"]:
        sub = t.loc[t["exit_reason"] == reason]
        if len(sub) == 0:
            continue
        r = sub["round_trip_return"]
        print(f"    {reason:12s} {len(sub):6d}  {100*len(sub)/len(t):4.1f}%  "
              f"{sub['hold_days'].mean():6.1f}d  {r.mean()*1e4:+7.0f}  "
              f"{100*(r>0).mean():4.0f}%  {r.sum():+8.2f}")
    r = t["round_trip_return"]
    outliers = int((r.abs() > 0.50).sum())
    print(f"    {'NET':12s} {len(t):6d}  100.0%  {t['hold_days'].mean():6.1f}d  "
          f"{r.mean()*1e4:+7.0f}  {100*(r>0).mean():4.0f}%  {r.sum():+8.2f}")
    print(f"  outlier trades |return| > 50%: {outliers}")


def section_duration_buckets(t: pd.DataFrame) -> None:
    bins = [0, 3, 7, 14, 21, 1000]
    labels = ["1-3d", "4-7d", "8-14d", "15-21d", "22d+"]
    t = t.assign(bucket=pd.cut(t["hold_days"], bins=bins, labels=labels, right=True))
    print("  bucket    n     share   mean_bps  win%")
    for lab in labels:
        sub = t.loc[t["bucket"] == lab]
        if len(sub) == 0:
            continue
        r = sub["round_trip_return"]
        print(f"    {lab:7s} {len(sub):6d}  {100*len(sub)/len(t):4.1f}%  "
              f"{r.mean()*1e4:+7.0f}  {100*(r>0).mean():4.0f}%")


def section_regime(t: pd.DataFrame) -> None:
    regimes = [
        ("2003-06 pre-crisis", "2003-01-01", "2007-06-30"),
        ("2007-09 GFC",        "2007-07-01", "2009-12-31"),
        ("2010-19 expansion",  "2010-01-01", "2019-12-31"),
        ("2020+ covid/after",  "2020-01-01", "2023-12-31"),
    ]
    total = t["round_trip_return"].sum()
    print("  regime                 n      %trades   Σreturn   %P&L")
    for name, lo, hi in regimes:
        sub = t.loc[(t["entry_date"] >= lo) & (t["entry_date"] <= hi)]
        if len(sub) == 0:
            continue
        s = sub["round_trip_return"].sum()
        print(f"    {name:20s} {len(sub):6d}  {100*len(sub)/len(t):5.1f}%  "
              f"{s:+8.2f}  {100*s/total if total else 0:5.1f}%")


def section_direction(t: pd.DataFrame) -> None:
    for d, lbl in [(1, "long_spread"), (-1, "short_spread")]:
        sub = t.loc[t["direction"] == d]
        if len(sub) == 0:
            continue
        print(f"  {lbl:13s} n {len(sub):6d}  Σreturn {sub['round_trip_return'].sum():+8.2f}  "
              f"win {100*(sub['round_trip_return']>0).mean():4.0f}%")


def section_sector(t: pd.DataFrame, sic_map: pd.Series) -> None:
    ff = sic_map.map(sic_to_ff12)
    a = t["permno_a"].map(ff).fillna("Other")
    b = t["permno_b"].map(ff).fillna("Other")
    pair_ind = pd.Series(
        ["/".join(sorted([x, y])) for x, y in zip(a, b)], index=t.index
    )
    grp = t.assign(pair_ind=pair_ind).groupby("pair_ind")["round_trip_return"]
    contrib = grp.sum().sort_values(ascending=False)
    total = contrib.sum()
    print("  top FF12 industry-pairs by Σreturn:")
    for name, val in contrib.head(5).items():
        print(f"    {name:16s} {val:+8.2f}  ({100*val/total if total else 0:4.1f}% of P&L)")


def main() -> None:
    crsp = load_crsp_daily()
    sic_map = siccd_lookup(crsp=crsp)  # permno -> latest SIC

    any_found = False
    for cell, rdir in CELLS:
        t = load_trades(cell, rdir)
        if t is None:
            print(f"\n[skip] {cell} — results not found yet")
            continue
        any_found = True
        print("\n" + "=" * 84)
        print(f"CELL: {cell}   ({len(t):,} trades)")
        print("=" * 84)
        print("\n[1] Holding duration");          section_duration(t)
        print("\n[2] Per-trade P&L distribution"); section_pnl_dist(t)
        print("\n[3] Bimodal pattern by exit reason"); section_bimodal(t)
        print("\n[4] Duration buckets");          section_duration_buckets(t)
        print("\n[5] Regime concentration");      section_regime(t)
        print("\n[6] Direction balance");         section_direction(t)
        print("\n[7] FF12 sector-pair concentration"); section_sector(t, sic_map)

    if not any_found:
        print("\nNo factor results found. Run 01_run_factor_backtest.py first.")


if __name__ == "__main__":
    main()
