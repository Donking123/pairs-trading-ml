"""
Phase 2.5 head-to-head evaluation — factor-beta vs SSD / PC.

Reads the new factor cells (phases/phase2_5/results/) alongside the FROZEN Phase 2
baselines (phases/phase2/results/), runs the full performance battery on each, and
prints a single scorecard so the QF621 writeup can compare the extension directly to
the replication.

It also recomputes the "bimodal lever" diagnostic (force-close drag + outlier count)
for every cell — the same mechanism that explained Phase 2's Sharpe lift — so we can
see HOW factor-beta clustering moves the needle, not just whether.

Run AFTER 01_run_factor_backtest.py finishes:
  cd pairs-trading-ml
  python phases/phase2_5/notebooks/02_compare_to_pc.py

Missing cells are skipped with a note (so this is safe to run mid-backtest).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.config import PHASE2_5_DIR, PHASE2_DIR
from src.performance import compute_metrics

# cell_name -> (results_dir, label, paper/reference note)
CELLS = [
    ("ssd_core",        PHASE2_DIR,   "SSD core (Phase 1 baseline)",      "0.589"),
    ("pc_core",         PHASE2_DIR,   "PC core (paper headline)",         "1.028"),
    ("pc_filtered",     PHASE2_DIR,   "PC + cointegration filter",        "0.752"),
    ("factor_core",     PHASE2_5_DIR, "Factor-beta core (EXTENSION)",     "—"),
    ("factor_filtered", PHASE2_5_DIR, "Factor-beta + cointeg. filter",    "—"),
]


def load_monthly(cell: str, results_dir: Path) -> pd.Series | None:
    path = results_dir / "results" / f"{cell}_monthly.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df["monthly_return"].astype(float)


def load_trades(cell: str, results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "results" / f"{cell}_trades.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def bimodal_lever(trades: pd.DataFrame) -> dict:
    """force-close drag (mean round-trip return of force_close trades, in bps) and
    count of |round_trip_return| > 50% outlier trades — the Phase 1/2 diagnostic."""
    fc = trades.loc[trades["exit_reason"] == "force_close", "round_trip_return"]
    rev = trades.loc[trades["exit_reason"] == "reversion", "round_trip_return"]
    outliers = int((trades["round_trip_return"].abs() > 0.50).sum())
    return {
        "n_trades": len(trades),
        "pct_reversion": 100.0 * len(rev) / max(len(trades), 1),
        "reversion_mean_bps": 1e4 * rev.mean() if len(rev) else float("nan"),
        "force_close_mean_bps": 1e4 * fc.mean() if len(fc) else float("nan"),
        "outliers_gt_50pct": outliers,
    }


def main() -> None:
    print("=" * 92)
    print("Phase 2.5 — Factor-beta vs SSD / PC  (head-to-head scorecard)")
    print("=" * 92)

    rows = []
    levers = []
    for cell, rdir, label, ref in CELLS:
        rets = load_monthly(cell, rdir)
        if rets is None:
            print(f"  [skip] {label:34s} — results not found yet ({cell})")
            continue
        m = compute_metrics(rets)
        rows.append({
            "cell": cell, "label": label, "ref_sharpe": ref,
            "n": m.n_months, "Sharpe": m.sharpe, "Sortino": m.sortino,
            "Calmar": m.calmar, "ann_ret": m.ann_return, "ann_vol": m.ann_vol,
            "MDD": m.max_drawdown, "hit": m.hit_rate, "win_loss": m.win_loss_ratio,
        })
        trades = load_trades(cell, rdir)
        if trades is not None:
            lev = bimodal_lever(trades)
            lev["cell"] = cell
            levers.append(lev)

    if not rows:
        print("\nNo result cells found. Run 01_run_factor_backtest.py first.")
        return

    tbl = pd.DataFrame(rows).set_index("cell")
    print("\n── Performance battery " + "─" * 68)
    show = tbl[["label", "ref_sharpe", "n", "Sharpe", "Sortino", "Calmar",
                "ann_ret", "ann_vol", "MDD", "hit", "win_loss"]].copy()
    for c in ["Sharpe", "Sortino", "Calmar", "win_loss"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    for c in ["ann_ret", "ann_vol", "MDD", "hit"]:
        show[c] = show[c].map(lambda x: f"{x*100:+.2f}%")
    print(show.to_string())

    if levers:
        print("\n── Bimodal lever (how the metric moves P&L) " + "─" * 47)
        lv = pd.DataFrame(levers).set_index("cell")
        lv_show = lv.copy()
        lv_show["pct_reversion"] = lv_show["pct_reversion"].map(lambda x: f"{x:.1f}%")
        for c in ["reversion_mean_bps", "force_close_mean_bps"]:
            lv_show[c] = lv_show[c].map(lambda x: f"{x:+.0f}")
        print(lv_show.to_string())

    # Save the scorecard for the README / notebook.
    out = PHASE2_5_DIR / "results" / "phase2_5_scorecard.csv"
    tbl.to_csv(out)
    print(f"\nSaved scorecard → {out}")

    # Headline comparison line.
    if "factor_core" in tbl.index and "pc_core" in tbl.index:
        fc_s = float(tbl.loc["factor_core", "Sharpe"])
        pc_s = float(tbl.loc["pc_core", "Sharpe"])
        ssd_s = float(tbl.loc["ssd_core", "Sharpe"]) if "ssd_core" in tbl.index else float("nan")
        print("\n" + "=" * 92)
        print(f"HEADLINE: factor-beta core Sharpe {fc_s:.3f}  "
              f"vs PC core {pc_s:.3f}  vs SSD core {ssd_s:.3f}")
        verdict = "beats" if fc_s > pc_s else ("matches" if abs(fc_s - pc_s) < 0.15 else "below")
        print(f"          → factor-beta {verdict} PC core (Δ {fc_s - pc_s:+.3f})")
        print("=" * 92)


if __name__ == "__main__":
    main()
