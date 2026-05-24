"""
Phase 2 — disciplined xi tuning for PC distance.

Goal: pick a single xi_pc value that
  (a) lands Dec-2023 cluster count near the paper's 109, AND
  (b) produces sensible cluster counts on Dec 2010 + Dec 2015 too (so the choice
      isn't over-fit to a single date).

Mirrors the discipline of phases/phase1/notebooks/02_xi_tuning_sweep.py — same
3 validation dates, lock-once policy, paper-comparison framing.

Run: python phases/phase2/notebooks/02_xi_tuning_pc.py
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

from src.clustering import (
    cluster_optics,
    cluster_summary,
    purity_index,
    sic_division,
)
from src.distances import pc_distance
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_market_returns,
    load_sp500_constituents,
    siccd_lookup,
)


DATES = ["2010-12-31", "2015-12-31", "2023-12-29"]
XI_VALUES = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
MIN_SAMPLES = 2
MIN_CLUSTER_SIZE = 2


def main() -> None:
    print("=" * 82)
    print("OPTICS xi tuning sweep — PC distance — Dec 2010 / Dec 2015 / Dec 2023")
    print("=" * 82)

    print("\nLoading panels …")
    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    mkt_ret = load_market_returns()
    print(f"  crsp_daily   : {crsp.shape[0]:,} rows")
    print(f"  constituents : {cons.shape[0]:,} intervals")
    print(f"  market ret   : {mkt_ret.shape[0]:,} days")

    cache: dict[str, tuple] = {}
    for d in DATES:
        print(f"\nBuilding panel + PC matrix for {d} …")
        panel = formation_window_panel(d, crsp=crsp, constituents=cons)
        dmat = pc_distance(panel, mkt_ret)
        siccds = siccd_lookup(panel.columns.tolist(), crsp=crsp, as_of=pd.Timestamp(d))
        sectors = siccds.apply(sic_division)
        cache[d] = (panel, dmat, sectors)
        print(f"  panel: {panel.shape}, distance matrix: {dmat.shape}")

    print("\n" + "=" * 82)
    print("Results grid")
    print("=" * 82)
    print(f"\n{'xi':>6} | {'date':<11} | {'n_clusters':>10} | {'clustered':>9} | "
          f"{'outliers':>8} | {'purity':>6}")
    print(f"{'-' * 6}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 6}")
    for xi in XI_VALUES:
        for d in DATES:
            _, dmat, sectors = cache[d]
            labels = cluster_optics(
                dmat,
                min_samples=MIN_SAMPLES,
                xi=xi,
                min_cluster_size=MIN_CLUSTER_SIZE,
            )
            summary = cluster_summary(labels)
            purity = purity_index(labels, sectors)
            print(
                f"{xi:>6.3f} | {d:<11} | "
                f"{summary['n_clusters']:>10} | "
                f"{summary['n_clustered_stocks']:>9} | "
                f"{summary['n_outliers']:>8} | "
                f"{purity:>6.3f}"
            )
        print(f"{'-' * 6}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 6}")

    print("\nPaper target on Dec 2023: 109 PC clusters (±10), purity 0.84 (±0.05)")
    print("\nInterpretation guide:")
    print("  - Pick the xi whose Dec 2023 count is closest to 109 AND whose")
    print("    counts on Dec 2010 / Dec 2015 are 'in the same ballpark'")
    print("    (no extreme value like 5 or 200).")
    print("  - Smaller xi = looser boundaries = more clusters. We need more than 66.")
    print("=" * 82)


if __name__ == "__main__":
    main()
