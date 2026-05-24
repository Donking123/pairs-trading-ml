"""
Phase 1c — single-month backtest sanity check.

Runs run_one_month on the Dec-2023 formation → Jan-2024 trading slice to verify
the full pipeline plumbing (formation panel → cluster → fit γ → simulate → P&L)
end-to-end on real data before the 252-month loop runs.

Output: monthly return, trade summary, top winners/losers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.backtest import load_delisting, run_one_month
from src.panel import load_crsp_daily, load_sp500_constituents, ticker_lookup


FORMATION_END = pd.Timestamp("2023-12-29")   # last day of formation window
TRADING_MONTH_END = pd.Timestamp("2024-01-31")  # but our data ends 2023-12-29...


def main() -> None:
    print("=" * 80)
    print("Phase 1c — single-month sanity check")
    print("=" * 80)

    print("\nLoading panels …")
    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    dlst = load_delisting()

    # we only have data through 2023-12-29, so pick a trading month INSIDE our window.
    # Use Nov-2023 trading window (formation_end = end of Oct 2023).
    all_dates = pd.DatetimeIndex(crsp["date"].drop_duplicates().sort_values())
    oct_dates = all_dates[(all_dates.year == 2023) & (all_dates.month == 10)]
    nov_dates = all_dates[(all_dates.year == 2023) & (all_dates.month == 11)]

    formation_end = oct_dates[-1]
    trading_dates = nov_dates

    print(f"\nFormation ends : {formation_end.date()}")
    print(f"Trading days   : {len(trading_dates)}  "
          f"({trading_dates[0].date()} → {trading_dates[-1].date()})")

    print("\nRunning month …  (this may take ~30s)")
    result = run_one_month(
        formation_end=formation_end,
        trading_dates=trading_dates,
        crsp=crsp,
        constituents=cons,
        delisting_df=dlst,
    )

    print(f"\n── Month summary ──")
    print(f"  Candidate pairs     : {result.n_candidate_pairs}")
    print(f"  Pairs that traded   : {result.n_pairs_traded}")
    print(f"  Total round-trips   : {result.n_trades}")
    print(f"  Avg open per day    : {result.avg_pairs_open:.1f}")
    print(f"  Monthly return      : {result.monthly_return:+.4f}  "
          f"(annualised: {((1 + result.monthly_return) ** 12 - 1):+.2%})")

    if result.trades:
        # Top winners & losers
        ticker_map = ticker_lookup(
            list({t.permno_a for t in result.trades} | {t.permno_b for t in result.trades}),
            crsp=crsp, as_of=trading_dates[-1],
        )

        sorted_trades = sorted(result.trades, key=lambda t: t.round_trip_return)
        print("\n── Worst 5 trades ──")
        for t in sorted_trades[:5]:
            ta = ticker_map.get(t.permno_a, t.permno_a)
            tb = ticker_map.get(t.permno_b, t.permno_b)
            print(f"  ({ta:>5}, {tb:>5})  {t.direction:+d}  "
                  f"entry z={t.entry_z:+.2f} on {t.entry_date.date()} → "
                  f"exit z={t.exit_z:+.2f} on {t.exit_date.date()} "
                  f"[{t.exit_reason:>11}]   ret={t.round_trip_return:+.4f}")
        print("\n── Best 5 trades ──")
        for t in sorted_trades[-5:]:
            ta = ticker_map.get(t.permno_a, t.permno_a)
            tb = ticker_map.get(t.permno_b, t.permno_b)
            print(f"  ({ta:>5}, {tb:>5})  {t.direction:+d}  "
                  f"entry z={t.entry_z:+.2f} on {t.entry_date.date()} → "
                  f"exit z={t.exit_z:+.2f} on {t.exit_date.date()} "
                  f"[{t.exit_reason:>11}]   ret={t.round_trip_return:+.4f}")

        # Exit-reason breakdown
        reasons = pd.Series([t.exit_reason for t in result.trades]).value_counts()
        print("\n── Exit reasons ──")
        for k, v in reasons.items():
            print(f"  {k:<12} : {v}")

    print("\n── Daily returns (first 10 / last 5) ──")
    print(result.daily_returns.head(10).to_string())
    print("...")
    print(result.daily_returns.tail(5).to_string())
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
