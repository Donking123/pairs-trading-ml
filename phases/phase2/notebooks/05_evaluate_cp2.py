"""
CP2 evaluator — full performance scorecard for the 4-way Phase 2 backtest grid.

Reads:
  phases/phase2/results/ssd_core_monthly.parquet
  phases/phase2/results/ssd_filtered_monthly.parquet
  phases/phase2/results/pc_core_monthly.parquet
  phases/phase2/results/pc_filtered_monthly.parquet

Prints:
  * Full performance battery (Sharpe / Sortino / Calmar / MDD / hit rate / etc.) for each cell.
  * 2x2 scorecard with side-by-side comparison.
  * Paper-vs-ours verdict per cell.
  * Did the force-close drag actually shrink? (the Phase 2 lever)
  * Phase 1 invariant check: SSD core Sharpe should match Phase 1's 0.589.

Usage: python phases/phase2/notebooks/05_evaluate_cp2.py
"""
from __future__ import annotations

import sys
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

from src.config import PHASE2_DIR
from src.performance import compute_metrics, format_metrics


PAPER_TARGETS = {
    "ssd_core":     {"sharpe": 0.88, "tol": 0.15, "label": "SSD core"},
    "ssd_filtered": {"sharpe": 0.75, "tol": 0.15, "label": "SSD + cointegration filter"},
    "pc_core":      {"sharpe": 1.01, "tol": 0.15, "label": "PC core"},
    "pc_filtered":  {"sharpe": 0.80, "tol": 0.15, "label": "PC + cointegration filter"},
}

PHASE_1_SSD_SHARPE = 0.589  # for invariant check


def load_cell(cell_name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (monthly, trades) DataFrames for one cell, or None if files missing."""
    monthly_path = PHASE2_DIR / "results" / f"{cell_name}_monthly.parquet"
    trades_path = PHASE2_DIR / "results" / f"{cell_name}_trades.parquet"
    if not monthly_path.exists():
        return None
    monthly = pd.read_parquet(monthly_path)
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    return monthly, trades


def evaluate_cell(cell_name: str) -> dict | None:
    loaded = load_cell(cell_name)
    if loaded is None:
        print(f"⚠ {cell_name}: not found — run the grid first")
        return None
    monthly, trades = loaded
    rets = monthly["monthly_return"]
    metrics = compute_metrics(rets)

    print()
    print("=" * 80)
    print(f"{PAPER_TARGETS[cell_name]['label']} ({cell_name})")
    print("=" * 80)
    print(format_metrics(metrics))

    # Force-close drag (the Phase 2 lever check)
    if not trades.empty and "exit_reason" in trades.columns:
        rev = trades.loc[trades["exit_reason"] == "reversion"]
        fc  = trades.loc[trades["exit_reason"] == "force_close"]
        dl  = trades.loc[trades["exit_reason"] == "delisting"]
        rev_total = rev["round_trip_return"].sum() if not rev.empty else 0.0
        fc_total  = fc["round_trip_return"].sum() if not fc.empty else 0.0
        dl_total  = dl["round_trip_return"].sum() if not dl.empty else 0.0
        net = rev_total + fc_total + dl_total
        print()
        print(f"  Force-close drag decomposition (Phase 2 lever check):")
        print(f"    reversion  : n={len(rev):>5,}  total={rev_total:+.2f}  mean={rev['round_trip_return'].mean()*10000 if len(rev) else 0:+.0f} bps")
        print(f"    force_close: n={len(fc):>5,}  total={fc_total:+.2f}  mean={fc['round_trip_return'].mean()*10000 if len(fc) else 0:+.0f} bps")
        print(f"    delisting  : n={len(dl):>5,}  total={dl_total:+.2f}")
        print(f"    NET (sum of per-trade) : {net:+.2f}")

    # Paper-vs-ours verdict for this cell
    paper_sharpe = PAPER_TARGETS[cell_name]["sharpe"]
    paper_tol = PAPER_TARGETS[cell_name]["tol"]
    delta = metrics.sharpe - paper_sharpe
    verdict = "✅" if abs(delta) <= paper_tol else ("⚠ above" if delta > 0 else "❌ below")
    print()
    print(f"  vs paper: ours {metrics.sharpe:.3f} vs paper {paper_sharpe} ±{paper_tol}  ({delta:+.3f})  {verdict}")

    return {
        "cell_name": cell_name,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "calmar": metrics.calmar,
        "ann_return": metrics.ann_return,
        "ann_vol": metrics.ann_vol,
        "max_drawdown": metrics.max_drawdown,
        "hit_rate": metrics.hit_rate,
        "n_months": metrics.n_months,
        "n_trades": len(trades),
        "delta_vs_paper": delta,
    }


def main() -> None:
    print("=" * 80)
    print("CP2 evaluation — Phase 2 2×2 grid")
    print("=" * 80)

    cells = ["ssd_core", "ssd_filtered", "pc_core", "pc_filtered"]
    summaries: list[dict] = []
    for cell in cells:
        s = evaluate_cell(cell)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("\nNo cells found. Run notebooks/04_run_full_backtest_grid.py first.")
        return

    # Build the 2×2 scorecard
    print()
    print("=" * 80)
    print("PHASE 2 SCORECARD — 2×2 grid")
    print("=" * 80)
    print()
    print(f"{'':<32}  {'no filter':>14}  {'with filter':>14}")
    print(f"{'-' * 32}  {'-' * 14}  {'-' * 14}")
    sharpe_map = {s["cell_name"]: s["sharpe"] for s in summaries}
    ann_ret_map = {s["cell_name"]: s["ann_return"] for s in summaries}
    mdd_map = {s["cell_name"]: s["max_drawdown"] for s in summaries}
    for metric in ["ssd", "pc"]:
        print(f"  {metric.upper()} — Sharpe                  {sharpe_map.get(f'{metric}_core', float('nan')):>14.3f}  {sharpe_map.get(f'{metric}_filtered', float('nan')):>14.3f}")
        print(f"  {metric.upper()} — Annualised return        {ann_ret_map.get(f'{metric}_core', float('nan')):>13.2%}   {ann_ret_map.get(f'{metric}_filtered', float('nan')):>13.2%}")
        print(f"  {metric.upper()} — Max drawdown             {mdd_map.get(f'{metric}_core', float('nan')):>13.1%}   {mdd_map.get(f'{metric}_filtered', float('nan')):>13.1%}")
        print()

    # Phase 1 invariant check
    print("-" * 80)
    print("Phase 1 invariant check (SSD core should reproduce Phase 1's 0.589 Sharpe)")
    print("-" * 80)
    ssd_core_sharpe = sharpe_map.get("ssd_core")
    if ssd_core_sharpe is not None:
        diff = ssd_core_sharpe - PHASE_1_SSD_SHARPE
        if abs(diff) < 0.01:
            print(f"  ssd_core Sharpe = {ssd_core_sharpe:.3f}  (Phase 1: {PHASE_1_SSD_SHARPE})  Δ={diff:+.4f}  ✅ matches")
        else:
            print(f"  ssd_core Sharpe = {ssd_core_sharpe:.3f}  (Phase 1: {PHASE_1_SSD_SHARPE})  Δ={diff:+.4f}  ⚠ DRIFT — investigate")

    # Final CP2 verdict
    print()
    print("=" * 80)
    print("CP2 VERDICT")
    print("=" * 80)
    pc_core_delta = next((s["delta_vs_paper"] for s in summaries if s["cell_name"] == "pc_core"), None)
    pc_filtered_delta = next((s["delta_vs_paper"] for s in summaries if s["cell_name"] == "pc_filtered"), None)
    if pc_core_delta is not None:
        ok_core = abs(pc_core_delta) <= 0.15
        print(f"  PC core         : Δ vs paper 1.01 = {pc_core_delta:+.3f}  → {'✅ within tolerance' if ok_core else '❌ outside tolerance'}")
    if pc_filtered_delta is not None:
        ok_filt = abs(pc_filtered_delta) <= 0.15
        print(f"  PC + filter     : Δ vs paper 0.80 = {pc_filtered_delta:+.3f}  → {'✅ within tolerance' if ok_filt else '❌ outside tolerance'}")
    print()


if __name__ == "__main__":
    main()
