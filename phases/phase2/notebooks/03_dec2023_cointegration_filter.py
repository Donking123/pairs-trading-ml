"""
Phase 2 — apply Engle-Granger cointegration filter on real Dec-2023 candidate pairs.

Sanity check: with our locked thresholds (p < 0.05, half-life in [5, 60] days),
what fraction of SSD and PC candidate pairs pass? Paper reports ~50% pass rate
for PC (109 → 78 clusters). We expect similar for our pairs.

Run: python phases/phase2/notebooks/03_dec2023_cointegration_filter.py
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

from src.clustering import (
    cluster_optics,
    clusters_to_pairs,
    sic_division,
)
from src.cointegration import filter_cointegrated_pairs
from src.config import (
    COINTEGRATION_P_THRESHOLD,
    HALF_LIFE_BOUNDS,
    OPTICS_MIN_CLUSTER_SIZE,
    OPTICS_MIN_SAMPLES,
    OPTICS_XI,
    OPTICS_XI_PC,
)
from src.distances import pc_distance, ssd_distance
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_market_returns,
    load_sp500_constituents,
    ticker_lookup,
)


AS_OF = "2023-12-29"


def run_one_metric(metric_name: str, panel: pd.DataFrame, dmat: pd.DataFrame, xi: float, ticker_map: pd.Series) -> None:
    print(f"\n{'=' * 80}")
    print(f"Cointegration filter on {metric_name} candidate pairs (Dec 2023)")
    print(f"  thresholds: p < {COINTEGRATION_P_THRESHOLD}, half-life ∈ {HALF_LIFE_BOUNDS}")
    print(f"{'=' * 80}")

    labels = cluster_optics(dmat, min_samples=OPTICS_MIN_SAMPLES, xi=xi,
                            min_cluster_size=OPTICS_MIN_CLUSTER_SIZE)
    candidate_pairs = clusters_to_pairs(labels)
    print(f"\n[1] Candidate pairs (from clustering): {len(candidate_pairs):,}")

    print(f"\n[2] Running Engle-Granger + half-life filter on each candidate "
          f"(~{len(candidate_pairs)} regressions × 2 directions) …")
    kept, results = filter_cointegrated_pairs(candidate_pairs, panel)
    print(f"    kept: {len(kept):,} / {len(candidate_pairs):,}  ({len(kept) / max(len(candidate_pairs), 1):.1%})")

    # Diagnostic breakdown of rejection reasons
    stationary = sum(1 for r in results.values() if r.is_stationary)
    tradeable_hl = sum(1 for r in results.values() if r.has_tradeable_half_life)
    print(f"\n[3] Rejection breakdown:")
    print(f"    ADF rejects unit root (p < {COINTEGRATION_P_THRESHOLD}) : {stationary:,} / {len(results):,}  "
          f"({stationary / max(len(results), 1):.1%})")
    print(f"    half-life ∈ [5, 60] days                             : {tradeable_hl:,} / {len(results):,}  "
          f"({tradeable_hl / max(len(results), 1):.1%})")
    print(f"    BOTH (passes filter)                                  : {len(kept):,} / {len(results):,}  "
          f"({len(kept) / max(len(results), 1):.1%})")

    # P-value distribution
    pvalues = pd.Series([r.adf_pvalue for r in results.values()])
    half_lives = pd.Series([r.half_life for r in results.values()]).dropna()
    print(f"\n[4] ADF p-value distribution across all candidates:")
    print(f"    min / 10% / 25% / 50% / 75% / 90% / max =")
    print(f"    {pvalues.min():.4f} / {pvalues.quantile(0.10):.4f} / {pvalues.quantile(0.25):.4f} / "
          f"{pvalues.quantile(0.50):.4f} / {pvalues.quantile(0.75):.4f} / {pvalues.quantile(0.90):.4f} / "
          f"{pvalues.max():.4f}")
    print(f"\n[5] Half-life distribution (only defined values):")
    print(f"    min / 10% / 25% / 50% / 75% / 90% / max =")
    print(f"    {half_lives.min():.1f} / {half_lives.quantile(0.10):.1f} / {half_lives.quantile(0.25):.1f} / "
          f"{half_lives.quantile(0.50):.1f} / {half_lives.quantile(0.75):.1f} / {half_lives.quantile(0.90):.1f} / "
          f"{half_lives.max():.1f}")

    # Show some example pairs (best 5 by ADF p-value)
    print(f"\n[6] Top-5 most cointegrated pairs (lowest p-value):")
    sorted_pairs = sorted(results.items(), key=lambda kv: kv[1].adf_pvalue)
    for (a, b), r in sorted_pairs[:5]:
        ta = ticker_map.get(a, str(a))
        tb = ticker_map.get(b, str(b))
        print(f"    ({ta:>6}, {tb:>6})  p={r.adf_pvalue:.2e}, γ={r.gamma:.3f}, "
              f"half_life={r.half_life:.1f}d, passes={r.passes_filter}")


def main() -> None:
    print("=" * 80)
    print(f"Phase 2 — Dec-2023 cointegration filter sanity check")
    print(f"  metric: SSD and PC distance, filter applied to each")
    print(f"  date  : {AS_OF}")
    print("=" * 80)

    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    mkt = load_market_returns()
    panel = formation_window_panel(AS_OF, crsp=crsp, constituents=cons)
    ticker_map = ticker_lookup(panel.columns.tolist(), crsp=crsp, as_of=pd.Timestamp(AS_OF))

    # SSD candidates
    dmat_ssd = ssd_distance(panel)
    run_one_metric("SSD", panel, dmat_ssd, OPTICS_XI, ticker_map)

    # PC candidates
    dmat_pc = pc_distance(panel, mkt)
    run_one_metric("PC", panel, dmat_pc, OPTICS_XI_PC, ticker_map)

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
