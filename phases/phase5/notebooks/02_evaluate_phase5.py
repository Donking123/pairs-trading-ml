"""
Phase 5 — evaluate the carry-over grid.

Reads the cell parquets written by 01_run_carryover_grid.py plus the stored frictionless
no-carry *core* baselines (cell F: phase1/2/2.5), and prints the two comparisons that
answer the Phase-5 questions:

  Table 1  E vs F  (frictionless)        → does carry-over improve the SIGNAL?
  Table 2  D vs B  (passive, no stop)    → does it help at the realistic operating point?
  Table 3  PC sensitivity D / Dstop / Dmkt vs B

For each cell: Sharpe / Sortino / Calmar / max-DD / hit-rate, and the trade diagnostics
that show the carry mechanism working — force-close %, reversion %, mean hold (calendar
days), and trade count. Carry should DROP force-close %, RAISE mean hold, and LOWER trade
count.

Run (any subset of cells that exist on disk will be shown):
    python phases/phase5/notebooks/02_evaluate_phase5.py
"""
from __future__ import annotations

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

from src.config import PHASE1_DIR, PHASE2_5_DIR, PHASE2_DIR, PHASE5_DIR
from src.performance import compute_metrics

P5 = PHASE5_DIR / "results"

# cell F (frictionless, no-carry) lives in the earlier phases
F_BASELINE = {
    "pc": PHASE2_DIR / "results" / "pc_core",
    "factor": PHASE2_5_DIR / "results" / "factor_core",
    "ssd": PHASE1_DIR / "results" / "ssd_core",
}


def _stats(stem: Path) -> dict | None:
    """Compute headline metrics + trade diagnostics for one cell, given the parquet stem
    (`<stem>_monthly.parquet` and `<stem>_trades.parquet`). Returns None if missing."""
    mp, tp = Path(f"{stem}_monthly.parquet"), Path(f"{stem}_trades.parquet")
    if not mp.exists():
        return None
    monthly = pd.read_parquet(mp)
    m = compute_metrics(monthly["monthly_return"].astype(float))
    out = {"sharpe": m.sharpe, "sortino": m.sortino, "calmar": m.calmar,
           "mdd": m.max_drawdown, "hit": m.hit_rate, "n_trades": 0,
           "force_close_%": float("nan"), "reversion_%": float("nan"),
           "mean_hold_d": float("nan")}
    if tp.exists():
        tr = pd.read_parquet(tp)
        n = len(tr)
        out["n_trades"] = n
        if n:
            reasons = tr["exit_reason"].value_counts()
            out["force_close_%"] = 100.0 * reasons.get("force_close", 0) / n
            out["reversion_%"] = 100.0 * reasons.get("reversion", 0) / n
            hold = (pd.to_datetime(tr["exit_date"]) - pd.to_datetime(tr["entry_date"])).dt.days
            out["mean_hold_d"] = float(hold.mean())
    return out


def _row(label: str, stem: Path) -> dict | None:
    s = _stats(stem)
    if s is None:
        print(f"  (skip {label}: {stem.name}_monthly.parquet not found)")
        return None
    return {"cell": label, **s}


def _table(title: str, rows: list[dict]) -> None:
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"\n### {title}\n  (no cells available yet)")
        return
    df = pd.DataFrame(rows).set_index("cell")
    fmt = df.copy()
    for c in ("sharpe", "sortino", "calmar"):
        fmt[c] = fmt[c].map(lambda x: f"{x:.3f}")
    for c in ("mdd", "hit"):
        fmt[c] = fmt[c].map(lambda x: f"{x:.1%}")
    for c in ("force_close_%", "reversion_%", "mean_hold_d"):
        fmt[c] = fmt[c].map(lambda x: f"{x:.1f}")
    print(f"\n### {title}")
    print(fmt.to_string())


def main() -> None:
    for metric in ("pc", "factor", "ssd"):
        print("\n" + "=" * 78)
        print(f"METRIC: {metric.upper()}")
        print("=" * 78)

        # Table 1 — signal effect: E (carry, frictionless) vs F (no-carry, frictionless)
        _table(f"{metric} — Table 1: carry vs no-carry, FRICTIONLESS  (signal effect)", [
            _row("F  no-carry", F_BASELINE[metric]),
            _row("E  carry", P5 / f"E_{metric}"),
        ])

        # Table 2 — operating point: D (carry, passive) vs B (no-carry, passive)
        _table(f"{metric} — Table 2: carry vs no-carry, PASSIVE no-stop  (operating point)", [
            _row("B  no-carry", P5 / f"B_{metric}"),
            _row("D  carry", P5 / f"D_{metric}"),
        ])

        # Table 3 — PC-only stop / execution sensitivity
        if metric == "pc":
            _table("pc — Table 3: carry sensitivity (passive vs +stop vs marketable)", [
                _row("B  no-carry passive", P5 / "B_pc"),
                _row("D  carry passive", P5 / "D_pc"),
                _row("Dstop carry passive+3.5σ", P5 / "Dstop_pc"),
                _row("Dmkt  carry marketable", P5 / "Dmkt_pc"),
            ])

    print("\n" + "=" * 78)
    print("Read: carry should LOWER force-close %, RAISE mean hold, LOWER trade count, and")
    print("(thesis) RAISE Sharpe. E>F confirms the signal effect; D>B the realistic gain.")


if __name__ == "__main__":
    main()
