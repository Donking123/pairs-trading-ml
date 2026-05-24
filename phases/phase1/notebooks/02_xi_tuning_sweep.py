"""
OPTICS xi tuning sweep — discipline against in-sample bias.

Goal: pick a single xi value that
  (a) lands cluster count near the paper's 48 on Dec 2023, AND
  (b) produces sensible cluster counts on Dec 2010 + Dec 2015 too (so the choice
      isn't over-fit to a single date).

Output: a small grid (xi × date) of (n_clusters, purity) so we can eyeball which
xi is the most stable across windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.clustering import cluster_optics, cluster_summary, purity_index, sic_division
from src.distances import ssd_distance
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_sp500_constituents,
    siccd_lookup,
)


DATES = ["2010-12-31", "2015-12-31", "2023-12-29"]
XI_VALUES = [0.05, 0.10, 0.15]
MIN_SAMPLES = 2
MIN_CLUSTER_SIZE = 2


def main() -> None:
    print("=" * 78)
    print("OPTICS xi tuning sweep — Dec 2010 / Dec 2015 / Dec 2023")
    print("=" * 78)

    print("\nLoading CRSP daily + constituents …")
    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    print(f"  crsp_daily         : {crsp.shape[0]:,} rows")
    print(f"  sp500_constituents : {cons.shape[0]:,} intervals")

    # build & cache the per-date panel + distance matrix once (reused across xi)
    cache: dict[str, tuple] = {}
    for d in DATES:
        print(f"\nBuilding panel + SSD matrix for {d} …")
        panel = formation_window_panel(d, crsp=crsp, constituents=cons)
        dmat = ssd_distance(panel)
        # sectors at as-of date for purity computation
        siccds = siccd_lookup(panel.columns.tolist(), crsp=crsp, as_of=pd.Timestamp(d))
        sectors = siccds.apply(sic_division)
        cache[d] = (panel, dmat, sectors)
        print(f"  panel: {panel.shape}, distance matrix: {dmat.shape}")

    # sweep
    print("\n" + "=" * 78)
    print("Results grid")
    print("=" * 78)
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
                f"{xi:>6.2f} | {d:<11} | "
                f"{summary['n_clusters']:>10} | "
                f"{summary['n_clustered_stocks']:>9} | "
                f"{summary['n_outliers']:>8} | "
                f"{purity:>6.3f}"
            )
        print(f"{'-' * 6}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 6}")

    # interpretation
    print("\nPaper target on Dec 2023: 48 SSD clusters (±5), purity 0.81 (±0.05)")
    print("\nInterpretation guide:")
    print("  - Pick the xi whose Dec 2023 count is closest to 48 AND whose")
    print("    counts on Dec 2010 / Dec 2015 are 'in the same ballpark' (no")
    print("    extreme value like 5 or 200).")
    print("  - If no xi is good across all three dates, the paper's settings")
    print("    might use a different min_samples or min_cluster_size.")
    print("=" * 78)


if __name__ == "__main__":
    main()
