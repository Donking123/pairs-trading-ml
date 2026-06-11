"""
Phase 6 — run the corrections ladder, one cell at a time.

Each cell = one full 2003-2023 backtest (~1-2h laptop wall-clock) with exactly ONE
correction flag flipped (plus an "all fixes" cell per group), so any Sharpe change is
attributable to a specific fix. See phases/phase6/decisions.md for what each flag does.

Groups (all PC metric — the headline cells):
  a*  frictionless core            → compare vs phases/phase2/results/pc_core_*
  b*  + cointegration filter       → compare vs phases/phase2/results/pc_filtered_*
  c*  realism (REALISM_FULL + 3.5σ stop, no filter) → compare vs c0 baseline

Cells:
  a0_core_baseline    no flags (sanity re-run; optional — phase2 parquet is the baseline)
  a1_core_delist      delisting_fix
  a2_core_delay       execution_delay=1          ← the big honesty test
  a3_core_allfix      delisting_fix + execution_delay + block_last_day_entry
  b0_filt_baseline    no flags (optional)
  b1_filt_mackinnon   coint_pvalue="mackinnon"   ← the statistical fix
  b2_filt_gamma       use_coint_gamma
  b3_filt_allfix      mackinnon + gamma + delisting + delay + last-day
  c0_real_baseline    realism, no flags (run once — phase4 used slightly different cells)
  c1_real_cooldown    stop_cooldown              ← fair re-test of "the stop is bad"
  c2_real_lastday     block_last_day_entry
  c3_real_allfix      cooldown + last-day + delisting + delay

Recommended order (highest information first):
  b1, a1, c0, c1, a2, then the rest.

Run (from pairs-trading-ml/):
  python phases/phase6/notebooks/01_run_corrections_ladder.py --list
  python phases/phase6/notebooks/01_run_corrections_ladder.py --cells b1,a1
  nohup python phases/phase6/notebooks/01_run_corrections_ladder.py --all \\
    > phases/phase6/results/phase6_ladder_log.txt 2>&1 &

Each cell writes {cell}_monthly.parquet + {cell}_trades.parquet into
phases/phase6/results/ and is skipped if its monthly parquet already exists
(use --force to re-run). Evaluate with 02_evaluate_corrections.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# walk up to project root
_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.backtest import Trade, load_delisting, run_backtest
from src.config import PHASE6_DIR
from src.costs import REALISM_FULL
from src.panel import load_crsp_daily, load_market_returns, load_sp500_constituents
from src.performance import compute_metrics, format_metrics

RESULTS_DIR = PHASE6_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ── cell definitions: name -> run_backtest kwargs (beyond metric="pc") ──────────
_REAL = dict(realism=REALISM_FULL, stop_sigma=3.5)
CELLS: dict[str, dict] = {
    # A — frictionless PC core
    "a0_core_baseline": {},
    "a1_core_delist":   dict(delisting_fix=True),
    "a2_core_delay":    dict(execution_delay=1),
    "a3_core_allfix":   dict(delisting_fix=True, execution_delay=1,
                             block_last_day_entry=True),
    # B — PC + cointegration filter
    "b0_filt_baseline": dict(cointegration_filter=True),
    "b1_filt_mackinnon": dict(cointegration_filter=True, coint_pvalue="mackinnon"),
    "b2_filt_gamma":    dict(cointegration_filter=True, use_coint_gamma=True),
    "b3_filt_allfix":   dict(cointegration_filter=True, coint_pvalue="mackinnon",
                             use_coint_gamma=True, delisting_fix=True,
                             execution_delay=1, block_last_day_entry=True),
    # C — realism (full bid/ask + 35bps borrow + 3.5σ stop)
    "c0_real_baseline": dict(**_REAL),
    "c1_real_cooldown": dict(**_REAL, stop_cooldown=True),
    "c2_real_lastday":  dict(**_REAL, block_last_day_entry=True),
    "c3_real_allfix":   dict(**_REAL, stop_cooldown=True, block_last_day_entry=True,
                             delisting_fix=True, execution_delay=1),
}
RECOMMENDED_ORDER = ["b1_filt_mackinnon", "a1_core_delist", "c0_real_baseline",
                     "c1_real_cooldown", "a2_core_delay"]


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "permno_a": t.permno_a, "permno_b": t.permno_b, "direction": t.direction,
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "entry_z": t.entry_z, "exit_z": t.exit_z,
        "round_trip_return": t.round_trip_return, "exit_reason": t.exit_reason,
    } for t in trades])


def run_cell(name: str, kwargs: dict, data: dict, force: bool) -> None:
    out_monthly = RESULTS_DIR / f"{name}_monthly.parquet"
    out_trades = RESULTS_DIR / f"{name}_trades.parquet"
    if out_monthly.exists() and not force:
        print(f"\n=== {name}: SKIP (exists — use --force to re-run) ===")
        return

    print(f"\n{'=' * 70}\n=== CELL {name}  kwargs={kwargs}\n{'=' * 70}")
    t0 = time.time()
    monthly, trades = run_backtest(metric="pc", verbose=True, **data, **kwargs)
    elapsed = (time.time() - t0) / 60

    monthly.to_parquet(out_monthly)
    trades_to_df(trades).to_parquet(out_trades)
    print(f"\n--- {name} done in {elapsed:.1f} min → {out_monthly.name} ---")
    print(format_metrics(compute_metrics(monthly["monthly_return"])))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 6 correction cells.")
    ap.add_argument("--cells", help="comma-separated cell names (see --list)")
    ap.add_argument("--all", action="store_true", help="run every cell")
    ap.add_argument("--list", action="store_true", help="list cells and exit")
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()

    if args.list or (not args.cells and not args.all):
        print("Available cells:")
        for n, kw in CELLS.items():
            print(f"  {n:<20} {kw}")
        print(f"\nRecommended first: {', '.join(RECOMMENDED_ORDER)}")
        print("\nUsage: --cells b1_filt_mackinnon,a1_core_delist   or   --all")
        return

    if args.all:
        selected = list(CELLS)
    else:
        selected = [c.strip() for c in args.cells.split(",")]
        unknown = [c for c in selected if c not in CELLS]
        if unknown:
            sys.exit(f"unknown cell(s): {unknown} — see --list")

    print("Loading data (once for all cells)...")
    data = dict(
        crsp=load_crsp_daily(),
        constituents=load_sp500_constituents(),
        delisting_df=load_delisting(),
        market_returns=load_market_returns(),
    )
    print(f"Running {len(selected)} cell(s): {selected}")
    for name in selected:
        run_cell(name, CELLS[name], data, force=args.force)
    print("\nAll requested cells finished. Evaluate with 02_evaluate_corrections.py")


if __name__ == "__main__":
    main()
