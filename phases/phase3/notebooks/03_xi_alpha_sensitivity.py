"""
Phase 3d — hyperparameter sensitivity (cheap, no full backtest).

Shows the locked constants sit on a STABLE plateau, not a cliff edge: we sweep
  * OPTICS xi   around OPTICS_XI_PC (PC metric), and
  * ridge alpha around RIDGE_ALPHA (factor metric)
on Dec 2015 + Dec 2023 and report cluster count, candidate-pair count, and purity.

A flat region around the locked value = the headline isn't an artefact of a knife-edge
hyperparameter. For the *Sharpe* sensitivity band, re-run 01_run_robustness_grid.py with
the config value changed (a few points is enough to anchor the band).

Run: python phases/phase3/notebooks/03_xi_alpha_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.config import OPTICS_XI_PC, RIDGE_ALPHA, OPTICS_XI_FACTOR
from src.panel import (formation_window_panel, load_crsp_daily,
                       load_market_returns, load_sp500_constituents, siccd_lookup)
from src.factors import build_factor_panel
from src.distances import pc_distance, factor_beta_distance
from src.clustering import cluster_optics, clusters_to_pairs, purity_index, sic_division

DATES = ["2015-12-31", "2023-12-29"]


def main():
    crsp = load_crsp_daily(); cons = load_sp500_constituents(); mkt = load_market_returns()
    panels = {}
    for d in DATES:
        p = formation_window_panel(d, crsp=crsp, constituents=cons)
        sic = siccd_lookup(list(p.columns), crsp=crsp, as_of=pd.Timestamp(d))
        div = sic.reindex(p.columns).map(sic_division)
        panels[d] = (p, sic, div)

    print("=" * 76)
    print(f"PC — OPTICS xi sensitivity (locked xi = {OPTICS_XI_PC})")
    print("=" * 76)
    for d in DATES:
        p, sic, div = panels[d]
        dmat = pc_distance(p, mkt)
        print(f"  {d}:")
        for xi in [0.02, 0.03, 0.04, 0.05, 0.06]:
            lab = cluster_optics(dmat, min_samples=2, xi=xi, min_cluster_size=2)
            mark = " <-- locked" if abs(xi - OPTICS_XI_PC) < 1e-9 else ""
            print(f"    xi={xi:.2f}: {len(set(lab[lab>=0])):3d} clusters, "
                  f"{len(clusters_to_pairs(lab)):4d} pairs, purity {purity_index(lab,div):.3f}{mark}")

    print("\n" + "=" * 76)
    print(f"Factor — ridge-alpha sensitivity (locked alpha = {RIDGE_ALPHA}, xi = {OPTICS_XI_FACTOR})")
    print("=" * 76)
    for d in DATES:
        p, sic, div = panels[d]
        fp = build_factor_panel(p, sic)
        print(f"  {d}:")
        for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
            dmat = factor_beta_distance(p, fp, ridge_alpha=alpha)
            lab = cluster_optics(dmat, min_samples=2, xi=OPTICS_XI_FACTOR, min_cluster_size=2)
            mark = " <-- locked" if abs(alpha - RIDGE_ALPHA) < 1e-9 else ""
            print(f"    alpha={alpha:.2f}: {len(set(lab[lab>=0])):3d} clusters, "
                  f"{len(clusters_to_pairs(lab)):4d} pairs, purity {purity_index(lab,div):.3f}{mark}")


if __name__ == "__main__":
    main()
