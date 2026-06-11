"""
Phase 4d — true out-of-sample forward test (2024-2025).

The only genuine generalisation test: run the FROZEN strategies on data that played no
role in any design or tuning decision — i.e. AFTER the 2003-2023 development sample. (Even
2023 isn't a clean holdout, since OPTICS xi / ridge-α were tuned on Dec-2023 windows.)

Prerequisite — pull the extended data with your WRDS login (keeps the validated 2003-2023
cache pristine by writing to a separate directory):
    cd pairs-trading-ml
    python src/wrds_pull.py --start 2000-01-01 --end 2025-12-31 --data-dir data_through_2025

Then run this (frozen — NO re-tuning):
    python phases/phase4/notebooks/05_forward_test.py --data-dir data_through_2025

It runs PC and factor-beta over 2024-01 → 2025-12 and compares the forward Sharpe to the
in-sample headlines (PC 1.028, factor 1.013). A forward Sharpe in the same ballpark =
generalises; a collapse = the edge was period-specific. (Note: ~24 monthly returns, so the
forward Sharpe is noisier than the 251-month in-sample figure — report it as indicative.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        ROOT = _p
        break
    _p = _p.parent
del _p

from src.backtest import Trade, load_delisting, run_backtest
from src.config import PHASE4_DIR
from src.factors import load_style_factors
from src.panel import (
    load_crsp_daily,
    load_market_returns,
    load_sp500_constituents,
)
from src.performance import compute_metrics, format_metrics

INSAMPLE = {"pc": 1.028, "factor": 1.013}
FWD_START = "2024-01-01"
FWD_END = "2025-12-31"


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def main(data_dir: Path) -> None:
    if not (data_dir / "crsp_daily.parquet").exists():
        raise SystemExit(
            f"No data in {data_dir}. Pull it first:\n"
            f"  python src/wrds_pull.py --start 2000-01-01 --end 2025-12-31 "
            f"--data-dir {data_dir}")

    print(f"Loading extended data from {data_dir} ...")
    crsp = load_crsp_daily(data_dir)
    cons = load_sp500_constituents(data_dir)
    dlst = load_delisting(data_dir)
    mkt = load_market_returns(data_dir)
    style = load_style_factors(data_dir)
    last = crsp["date"].max()
    print(f"  data through {last.date()}  ({crsp['date'].nunique():,} trading days)")
    if last < pd.Timestamp("2024-06-30"):
        print("  ⚠️  data does not extend past mid-2024 — forward window will be short/empty.")

    results_dir = PHASE4_DIR / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    rows = []
    for metric in ["pc", "factor"]:
        print(f"\n=== FORWARD {metric.upper()}  {FWD_START} → {FWD_END} (frozen) ===")
        monthly, trades = run_backtest(
            start=FWD_START, end=FWD_END, crsp=crsp, constituents=cons,
            delisting_df=dlst, market_returns=mkt, style_factors=style,
            metric=metric, verbose=True)
        monthly.to_parquet(results_dir / f"forward_{metric}_monthly.parquet")
        trades_to_df(trades).to_parquet(results_dir / f"forward_{metric}_trades.parquet")
        m = compute_metrics(monthly["monthly_return"].astype(float))
        print(format_metrics(m))
        rows.append({"metric": metric, "n_months": m.n_months,
                     "forward_sharpe": round(m.sharpe, 3),
                     "insample_sharpe": INSAMPLE[metric],
                     "ann_ret": round(m.ann_return, 4), "ann_vol": round(m.ann_vol, 4),
                     "mdd": round(m.max_drawdown, 4), "n_trades": len(trades)})

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "forward_summary.csv", index=False)
    print("\n" + "=" * 74)
    print("FORWARD (2024-2025, frozen) vs IN-SAMPLE (2003-2023)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data_through_2025"),
                    help="directory holding the extended (through-2025) parquet cache")
    args = ap.parse_args()
    main(Path(args.data_dir))
