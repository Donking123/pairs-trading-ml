"""
Phase 5 — true out-of-sample forward test of the carry-over strategies (2024-2025).

Runs the FROZEN carry cells on data that played no part in any design/tuning decision —
the only genuine generalisation test. Mirrors phase4/05_forward_test.py but with
carry_over=True:

  E  carry + frictionless     → vs Phase-4 frictionless OOS (PC 0.858, factor 0.117)
  D  carry + passive (no stop)→ the realistic carry strategy, out of sample

NO re-tuning: MAX_CARRY_MONTHS and every other knob are the values locked for the in-sample
grid. ~23 monthly returns → indicative and noisier than the 251-month figure.

Prerequisite — the extended cache (same one Phase 4 used):
    python src/wrds_pull.py --start 2000-01-01 --end 2025-12-31 --data-dir data_through_2025

Run (frozen):
    python phases/phase5/notebooks/03_forward_test_carry.py --data-dir data_through_2025
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
from src.config import PHASE5_DIR
from src.costs import REALISM_PASSIVE
from src.factors import load_style_factors
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents
from src.performance import compute_metrics, format_metrics

RESULTS_DIR = PHASE5_DIR / "results"
FWD_START, FWD_END = "2024-01-01", "2025-12-31"

# Phase-4 frictionless no-carry OOS, for reference (did carry change the picture?)
PHASE4_OOS = {"pc": 0.858, "factor": 0.117, "ssd": float("nan")}

# forward cells: name -> extra run_backtest kwargs (carry always on)
FWD_CELLS = {
    "E": dict(carry_over=True),                            # frictionless
    "D": dict(carry_over=True, realism=REALISM_PASSIVE),  # passive, no stop
}


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def _insample_sharpe(stem: Path) -> float:
    p = Path(f"{stem}_monthly.parquet")
    if not p.exists():
        return float("nan")
    return compute_metrics(pd.read_parquet(p)["monthly_return"].astype(float)).sharpe


def main(data_dir: Path, metrics: list[str], cells: list[str]) -> None:
    if not (data_dir / "crsp_daily.parquet").exists():
        raise SystemExit(
            f"No data in {data_dir}. Pull it first:\n"
            f"  python src/wrds_pull.py --start 2000-01-01 --end 2025-12-31 "
            f"--data-dir {data_dir}")

    print(f"Loading extended data from {data_dir} ...")
    crsp = load_crsp_daily(data_dir); cons = load_sp500_constituents(data_dir)
    dlst = load_delisting(data_dir); mkt = load_market_returns(data_dir)
    style = load_style_factors(data_dir)
    last = crsp["date"].max()
    print(f"  data through {last.date()}  ({crsp['date'].nunique():,} trading days)")
    if last < pd.Timestamp("2024-06-30"):
        print("  ⚠️  data does not extend past mid-2024 — forward window will be short/empty.")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    rows = []
    for metric in metrics:
        for cell in cells:
            kwargs = FWD_CELLS[cell]
            tag = f"forward_{cell}_{metric}"
            print(f"\n=== {tag}  {FWD_START} → {FWD_END} (frozen, carry on) ===")
            monthly, trades = run_backtest(
                start=FWD_START, end=FWD_END, crsp=crsp, constituents=cons,
                delisting_df=dlst, market_returns=mkt, style_factors=style,
                metric=metric, verbose=True, **kwargs)
            monthly.to_parquet(RESULTS_DIR / f"{tag}_monthly.parquet")
            trades_to_df(trades).to_parquet(RESULTS_DIR / f"{tag}_trades.parquet")
            m = compute_metrics(monthly["monthly_return"].astype(float))
            print(format_metrics(m))
            rows.append({
                "cell": cell, "metric": metric, "n_months": m.n_months,
                "forward_sharpe": round(m.sharpe, 3),
                "insample_sharpe": round(_insample_sharpe(RESULTS_DIR / f"{cell}_{metric}"), 3),
                "phase4_nocarry_oos": PHASE4_OOS.get(metric, float("nan")),
                "ann_ret": round(m.ann_return, 4), "mdd": round(m.max_drawdown, 4),
                "n_trades": len(trades)})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "forward_carry_summary.csv", index=False)
    print("\n" + "=" * 78)
    print("FORWARD carry (2024-2025, frozen) vs in-sample carry vs Phase-4 no-carry OOS")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data_through_2025"))
    ap.add_argument("--metrics", default="pc,factor,ssd")
    ap.add_argument("--cells", default="E,D")
    args = ap.parse_args()
    main(Path(args.data_dir),
         [m.strip() for m in args.metrics.split(",") if m.strip()],
         [c.strip() for c in args.cells.split(",") if c.strip()])
