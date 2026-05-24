"""
Inspect the backtest run — wall-clock time, warnings, anomalies.

Reads:
  results/ssd_core_log.txt         (if run with log redirect)
  results/ssd_core_monthly.parquet
  results/ssd_core_trades.parquet

Reports:
  * Wall-clock total + per-month average (from log)
  * Skipped months / errors / warnings
  * Months with anomalous returns (|ret| > 10%)
  * Months with no trades or very few trades
  * Trade outliers (round-trip return > 50% or < -50%)
  * Continuity check — every month present 2003-01 → 2023-12?

Usage:
  python notebooks/07_inspect_backtest.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.config import RESULTS_DIR


def inspect_log(log_path: Path) -> None:
    print("── Log file inspection ──")
    if not log_path.exists():
        print(f"  ⚠ log file not found at {log_path}")
        print(f"     (re-run with `python notebook05.py > results/ssd_core_log.txt 2>&1` to capture)")
        return

    log = log_path.read_text()
    n_lines = log.count("\n")
    print(f"  log file        : {log_path}")
    print(f"  total lines     : {n_lines:,}")

    # wall-clock
    wc_match = re.search(r"Wall-clock:\s*([\d.]+)\s*min", log)
    if wc_match:
        wc = float(wc_match.group(1))
        print(f"  wall-clock      : {wc:.1f} min  (~{wc * 60:.0f} s)")
    else:
        print(f"  wall-clock      : not found in log")

    # skipped months / errors / warnings
    skipped = re.findall(r"^\s*(\d{4}-\d{2}-\d{2}):\s*skipped\s*\((.*?)\)", log, flags=re.MULTILINE)
    errors = re.findall(r"(Error|Traceback|ERROR|FAIL)", log)
    warnings = re.findall(r"(Warning|UserWarning|FutureWarning)", log)

    print(f"\n  skipped months  : {len(skipped)}")
    for date, reason in skipped[:10]:
        print(f"    {date}: {reason}")
    if len(skipped) > 10:
        print(f"    ... and {len(skipped) - 10} more")

    print(f"\n  errors detected : {len(errors)}")
    if errors:
        print(f"    (categories: {dict(pd.Series(errors).value_counts())})")
    print(f"  warnings        : {len(warnings)}")
    if warnings:
        # show a sample
        for line in log.splitlines():
            if re.search(r"Warning|UserWarning|FutureWarning", line):
                print(f"    {line.strip()}")
                break


def inspect_monthly(monthly: pd.DataFrame) -> None:
    print("\n── Monthly returns inspection ──")
    print(f"  rows                 : {len(monthly)}")
    print(f"  first month          : {monthly.index.min().date()}")
    print(f"  last month           : {monthly.index.max().date()}")

    # Continuity check — every month present?
    expected = pd.date_range(monthly.index.min(), monthly.index.max(), freq="ME")
    missing = expected.difference(monthly.index)
    print(f"  continuity gaps      : {len(missing)} missing months")
    for d in missing[:5]:
        print(f"    missing: {d.date()}")
    if len(missing) > 5:
        print(f"    ... and {len(missing) - 5} more")

    # NaN / inf returns
    rets = monthly["monthly_return"]
    nan_count = rets.isna().sum()
    inf_count = (~rets.isna() & ~rets.between(-1, 10)).sum()  # outside [-100%, +1000%]
    print(f"  NaN returns          : {nan_count}")
    print(f"  inf / outlier returns: {inf_count}")

    # Extreme months (|ret| > 10%)
    extreme = rets[rets.abs() > 0.10].sort_values()
    print(f"  |return| > 10% months: {len(extreme)}")
    for d, r in extreme.items():
        print(f"    {d.date()}: {r:+.2%}")

    # Zero-trade months
    if "n_trades" in monthly.columns:
        zero_trades = monthly[monthly["n_trades"] == 0]
        few_trades = monthly[monthly["n_trades"].between(1, 5)]
        print(f"  zero-trade months    : {len(zero_trades)}")
        for d, row in zero_trades.iterrows():
            print(f"    {d.date()}: candidate_pairs={row['n_candidate_pairs']}")
        print(f"  <=5-trade months     : {len(few_trades)}")


def inspect_trades(trades: pd.DataFrame) -> None:
    print("\n── Trade inspection ──")
    print(f"  total round-trips    : {len(trades):,}")

    # round-trip return distribution
    rt = trades["round_trip_return"]
    print(f"  mean rt return       : {rt.mean():+.4%}")
    print(f"  median rt return     : {rt.median():+.4%}")
    print(f"  std rt return        : {rt.std():.4%}")
    print(f"  best rt              : {rt.max():+.2%}")
    print(f"  worst rt             : {rt.min():+.2%}")

    # outlier trades
    outliers = trades[rt.abs() > 0.5]
    print(f"  |rt| > 50% trades    : {len(outliers)}")
    for _, t in outliers.head(10).iterrows():
        print(
            f"    ({t['permno_a']}, {t['permno_b']})  "
            f"{t['direction']:+d}  "
            f"{t['entry_date']:%Y-%m-%d} → {t['exit_date']:%Y-%m-%d}  "
            f"[{t['exit_reason']}]  rt={t['round_trip_return']:+.2%}"
        )

    # exit reason breakdown
    print(f"\n  exit reasons:")
    reason_counts = trades["exit_reason"].value_counts()
    for k, v in reason_counts.items():
        print(f"    {k:<12} : {v:>6,}  ({v / len(trades):.1%})")


def main() -> None:
    print("=" * 80)
    print("Backtest inspection — SSD core")
    print("=" * 80)

    log_path = RESULTS_DIR / "ssd_core_log.txt"
    monthly_path = RESULTS_DIR / "ssd_core_monthly.parquet"
    trades_path = RESULTS_DIR / "ssd_core_trades.parquet"

    # 1. Log inspection (warnings, skipped months, wall-clock)
    inspect_log(log_path)

    # 2. Monthly returns inspection
    if monthly_path.exists():
        monthly = pd.read_parquet(monthly_path)
        inspect_monthly(monthly)
    else:
        print(f"\n  ⚠ monthly parquet not found at {monthly_path}")

    # 3. Trade inspection
    if trades_path.exists():
        trades = pd.read_parquet(trades_path)
        inspect_trades(trades)
    else:
        print(f"\n  ⚠ trades parquet not found at {trades_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
