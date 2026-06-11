"""
Phase 6 — evaluate the corrections ladder.

Reads every {cell}_monthly.parquet in phases/phase6/results/, plus the Phase 2
baselines (pc_core / pc_filtered), and prints one comparison table:

  * Sharpe (rf=0)        — the project's headline convention (paper-faithful)
  * Sharpe (excess rf)   — NEW reporting column: monthly return minus the compounded
                           FF daily risk-free rate (review improvement; engine untouched)
  * ann return / vol, max drawdown, hit rate, n_trades
  * ΔSharpe vs the group's baseline:
        a* → a0 if run, else phase2 pc_core
        b* → b0 if run, else phase2 pc_filtered
        c* → c0 (realism cells have no exact phase2/4 counterpart)

Run (from pairs-trading-ml/):
  python phases/phase6/notebooks/02_evaluate_corrections.py
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

from src.config import DATA_DIR, PHASE2_DIR, PHASE6_DIR
from src.performance import compute_metrics

RESULTS_DIR = PHASE6_DIR / "results"
MONTHS_PER_YEAR = 12


def monthly_rf() -> pd.Series:
    """Compound the FF daily risk-free rate into a month-end-indexed series."""
    ff = pd.read_parquet(DATA_DIR / "ff_factors.parquet")
    ff = ff.set_index(pd.DatetimeIndex(ff["date"])).sort_index()
    rf = ff["rf"].astype("float64")
    by_month = (1.0 + rf).groupby(rf.index.to_period("M")).prod() - 1.0
    return by_month


def excess_sharpe(monthly_returns: pd.Series, rf_m: pd.Series) -> float:
    rf_aligned = rf_m.reindex(monthly_returns.index.to_period("M")).fillna(0.0)
    excess = monthly_returns.to_numpy() - rf_aligned.to_numpy()
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(MONTHS_PER_YEAR)) if sd > 1e-12 else float("nan")


def load_cells() -> dict[str, tuple[pd.DataFrame, Path | None]]:
    """{label: (monthly_df, trades_path)} for phase6 cells + phase2 baselines."""
    cells: dict[str, tuple[pd.DataFrame, Path | None]] = {}
    for base_label, fname in [("phase2:pc_core", "pc_core_monthly.parquet"),
                              ("phase2:pc_filtered", "pc_filtered_monthly.parquet")]:
        f = PHASE2_DIR / "results" / fname
        if f.exists():
            cells[base_label] = (pd.read_parquet(f), None)
    for f in sorted(RESULTS_DIR.glob("*_monthly.parquet")):
        name = f.name.replace("_monthly.parquet", "")
        tp = RESULTS_DIR / f"{name}_trades.parquet"
        cells[name] = (pd.read_parquet(f), tp if tp.exists() else None)
    return cells


def baseline_for(name: str, cells: dict) -> str | None:
    if name.startswith("a") and name != "a0_core_baseline":
        return "a0_core_baseline" if "a0_core_baseline" in cells else "phase2:pc_core"
    if name.startswith("b") and name != "b0_filt_baseline":
        return "b0_filt_baseline" if "b0_filt_baseline" in cells else "phase2:pc_filtered"
    if name.startswith("c") and name != "c0_real_baseline":
        return "c0_real_baseline" if "c0_real_baseline" in cells else None
    return None


def main() -> None:
    cells = load_cells()
    if not cells:
        sys.exit("no results found — run 01_run_corrections_ladder.py first")
    rf_m = monthly_rf()

    rows = []
    sharpes: dict[str, float] = {}
    for name, (monthly, trades_path) in cells.items():
        ret = monthly["monthly_return"]
        m = compute_metrics(ret)
        sharpes[name] = m.sharpe
        n_trades = None
        if trades_path is not None:
            n_trades = len(pd.read_parquet(trades_path))
        rows.append({
            "cell": name,
            "sharpe(rf=0)": round(m.sharpe, 3),
            "sharpe(xs-rf)": round(excess_sharpe(ret, rf_m), 3),
            "ann_ret": f"{m.ann_return:+.2%}",
            "ann_vol": f"{m.ann_vol:.2%}",
            "mdd": f"{m.max_drawdown:.1%}",
            "hit": f"{m.hit_rate:.0%}",
            "months": m.n_months,
            "trades": n_trades if n_trades is not None else "-",
        })

    df = pd.DataFrame(rows).set_index("cell")
    df["Δsharpe_vs_base"] = [
        round(sharpes[n] - sharpes[b], 3)
        if (b := baseline_for(n, cells)) and b in sharpes else ""
        for n in df.index
    ]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nPhase 6 corrections ladder — comparison\n")
        print(df.to_string())
        print(
            "\nNotes: sharpe(rf=0) is the project headline convention; sharpe(xs-rf)"
            "\nsubtracts the compounded FF daily risk-free rate (reporting only)."
            "\n24-month OOS Sharpes carry a standard error of roughly ±0.45 — judge"
            "\nΔs against in-sample N=251 (se ≈ ±0.22) accordingly."
        )


if __name__ == "__main__":
    main()
