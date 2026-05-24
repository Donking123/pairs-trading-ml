"""
CP1 evaluator — compute and print performance metrics for the SSD backtest.

Reads:
  results/ssd_core_monthly.parquet     (from notebook 05)

Prints:
  - Full performance battery (Sharpe, Sortino, Calmar, drawdown, …)
  - CP1 verdict (Sharpe within 0.88 ± 0.15 of paper)

Usage:
  python notebooks/06_evaluate_cp1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config import RESULTS_DIR
from src.performance import compute_metrics, format_metrics


CP1_SHARPE_TARGET = 0.88
CP1_SHARPE_TOL = 0.15


def main() -> None:
    monthly_path = RESULTS_DIR / "ssd_core_monthly.parquet"
    if not monthly_path.exists():
        print(f"❌ Monthly returns not found at {monthly_path}")
        print("   Run notebooks/05_run_full_backtest.py first.")
        sys.exit(1)

    monthly = pd.read_parquet(monthly_path)
    rets = monthly["monthly_return"]

    print("=" * 80)
    print("CP1 evaluation — SSD core backtest (2003-2023)")
    print("=" * 80)
    print()
    print(f"Source : {monthly_path}")
    print(f"Period : {monthly.index.min().date()} → {monthly.index.max().date()}")
    print(f"Months : {len(rets)}")
    print()

    metrics = compute_metrics(rets)
    print("Performance metrics:")
    print(format_metrics(metrics))
    print()

    # CP1 verdict
    print("─" * 80)
    print("CP1 verdict")
    print("─" * 80)
    delta = metrics.sharpe - CP1_SHARPE_TARGET
    ok = abs(delta) <= CP1_SHARPE_TOL
    print(f"  Sharpe          : {metrics.sharpe:.3f}")
    print(f"  Paper target    : {CP1_SHARPE_TARGET} ± {CP1_SHARPE_TOL}")
    print(f"  Distance        : {delta:+.3f}")
    if ok:
        print(f"\n  ✅ CP1 PASSED — Phase 1 (SSD vertical slice) complete.")
        print(f"     Ready to start Phase 2 (PC distance + cointegration).")
    elif metrics.sharpe > CP1_SHARPE_TARGET + CP1_SHARPE_TOL:
        print(f"\n  ⚠ Sharpe HIGHER than paper — likely no-cost effect.")
        print(f"     Compare against the realism variant (with bid/ask + 35bps borrow).")
    else:
        print(f"\n  ❌ Sharpe LOWER than paper tolerance — investigate:")
        print(f"     - mostly force-closed trades? (need cointegration filter Phase 2)")
        print(f"     - delisting handling too punitive?")
        print(f"     - z-score window timing bug?")
    print()

    # diagnostic: best and worst years
    monthly = monthly.copy()
    monthly["year"] = monthly.index.year
    yearly = monthly.groupby("year")["monthly_return"].apply(
        lambda x: (1 + x).prod() - 1
    ).sort_values()
    print("Best 3 years:")
    for yr, ret in yearly.tail(3).items():
        print(f"  {yr}: {ret:+.2%}")
    print("Worst 3 years:")
    for yr, ret in yearly.head(3).items():
        print(f"  {yr}: {ret:+.2%}")
    print()

    # diagnostic: monthly trade counts
    print("Trade activity:")
    print(f"  avg candidate pairs/month : {monthly['n_candidate_pairs'].mean():.0f}")
    print(f"  avg pairs traded/month    : {monthly['n_pairs_traded'].mean():.0f}")
    print(f"  avg trades/month          : {monthly['n_trades'].mean():.1f}")
    print(f"  avg open pairs/day        : {monthly['avg_pairs_open'].mean():.1f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
