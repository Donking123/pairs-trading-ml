"""
Phase 3 — robustness comparison. Reads every robustness cell alongside its frozen
baseline and prints, per headline metric, the Sharpe across all perturbations — i.e.
the confidence band around the headline.

  PC baseline     : phase2/results/pc_core
  factor baseline : phase2_5/results/factor_core
  robustness cells: phase3/results/{metric}_{hdbscan,hierarchical,rlm,zweight}

Run after 01_run_robustness_grid.py (skips cells not yet produced):
  python phases/phase3/notebooks/02_compare_robustness.py
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

from src.config import PHASE2_5_DIR, PHASE2_DIR, PHASE3_DIR
from src.performance import compute_metrics

# metric -> (baseline_label, baseline_path, [variant cells in phase3])
GROUPS = {
    "PC": ("pc_core (baseline)", PHASE2_DIR / "results" / "pc_core_monthly.parquet",
           ["pc_hdbscan", "pc_hierarchical", "pc_rlm", "pc_zweight"]),
    "Factor-beta": ("factor_core (baseline)",
                    PHASE2_5_DIR / "results" / "factor_core_monthly.parquet",
                    ["factor_hdbscan", "factor_hierarchical", "factor_rlm", "factor_zweight"]),
}
VARIANT_LABEL = {"hdbscan": "3a HDBSCAN", "hierarchical": "3a hierarchical",
                 "rlm": "3b RLM hedge", "zweight": "3c z-weighted"}


def sharpe_of(path: Path):
    if not path.exists():
        return None
    return compute_metrics(pd.read_parquet(path)["monthly_return"].astype(float))


def main():
    print("=" * 78)
    print("Phase 3 — robustness band around each headline")
    print("=" * 78)
    for metric, (base_label, base_path, cells) in GROUPS.items():
        print(f"\n### {metric}")
        rows = []
        base = sharpe_of(base_path)
        if base is not None:
            rows.append((base_label, base.sharpe, base.ann_return, base.ann_vol, base.max_drawdown))
        sharpes = [base.sharpe] if base is not None else []
        for cell in cells:
            variant = cell.split("_", 1)[1]
            m = sharpe_of(PHASE3_DIR / "results" / f"{cell}_monthly.parquet")
            if m is None:
                print(f"  [pending] {VARIANT_LABEL.get(variant, variant)}")
                continue
            rows.append((VARIANT_LABEL.get(variant, variant), m.sharpe,
                         m.ann_return, m.ann_vol, m.max_drawdown))
            sharpes.append(m.sharpe)
        if rows:
            df = pd.DataFrame(rows, columns=["variant", "Sharpe", "AnnRet", "AnnVol", "MDD"])
            df["Sharpe"] = df["Sharpe"].map(lambda x: f"{x:.3f}")
            for c in ["AnnRet", "AnnVol", "MDD"]:
                df[c] = df[c].map(lambda x: f"{x*100:+.2f}%")
            print(df.to_string(index=False))
        if len(sharpes) >= 2:
            print(f"  → Sharpe band: {min(sharpes):.3f} – {max(sharpes):.3f}  "
                  f"(spread {max(sharpes)-min(sharpes):.3f}, n={len(sharpes)})")


if __name__ == "__main__":
    main()
