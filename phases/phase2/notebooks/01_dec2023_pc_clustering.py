"""
Phase 2 — PC distance + OPTICS on real Dec-2023 formation window.

Sanity check that pc_distance works on real CRSP data and produces a sensible
cluster count vs the paper's reported 109 PC clusters.

Mirrors phases/phase1/notebooks/01_dec2023_ssd_clustering.py but uses the new
pc_distance metric instead of ssd_distance.

Run:  python phases/phase2/notebooks/01_dec2023_pc_clustering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# walk up to project root (contains src/config.py) — works from any depth
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
    clusters_to_pairs,
    purity_index,
    sic_division,
)
from src.config import OPTICS_MIN_CLUSTER_SIZE, OPTICS_MIN_SAMPLES, OPTICS_XI_PC
from src.distances import pc_distance
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_market_returns,
    load_sp500_constituents,
    siccd_lookup,
    ticker_lookup,
)


AS_OF = "2023-12-29"


def main() -> None:
    print("=" * 72)
    print(f"Phase 2 — real Dec-2023 PC + OPTICS run")
    print("=" * 72)

    print("\n[1] Loading CRSP daily + constituents + market returns …")
    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    mkt_ret = load_market_returns()
    print(f"    crsp_daily     : {crsp.shape[0]:>10,} rows × {crsp.shape[1]} cols")
    print(f"    constituents   : {cons.shape[0]:>10,} intervals")
    print(f"    market returns : {mkt_ret.shape[0]:>10,} days")

    print(f"\n[2] Building formation-window panel ending {AS_OF} …")
    panel = formation_window_panel(AS_OF, crsp=crsp, constituents=cons)
    print(f"    panel: {panel.shape}  (days × stocks)")
    print(f"    window: {panel.index.min().date()} → {panel.index.max().date()}")

    print("\n[3] Computing PC distance matrix …")
    dmat = pc_distance(panel, mkt_ret)
    print(f"    distance matrix: {dmat.shape}  ({dmat.shape[0] * (dmat.shape[0] - 1) // 2:,} unique pairs)")
    upper = dmat.where(~pd.DataFrame(
        [[i >= j for j in range(dmat.shape[1])] for i in range(dmat.shape[0])],
        index=dmat.index, columns=dmat.columns,
    )).stack()
    print(f"    PC distance distribution:")
    print(f"      min / 25th / median / 75th / max = "
          f"{upper.min():.4f} / {upper.quantile(0.25):.4f} / {upper.median():.4f} / "
          f"{upper.quantile(0.75):.4f} / {upper.max():.4f}")
    print(f"      fraction < 0.5  : {(upper < 0.5).mean():.3%}   (tight pairs)")
    print(f"      fraction < 1.0  : {(upper < 1.0).mean():.3%}   (positively correlated residuals)")

    print(f"\n[4] Running OPTICS (xi={OPTICS_XI_PC}, min_samples={OPTICS_MIN_SAMPLES}, "
          f"min_cluster_size={OPTICS_MIN_CLUSTER_SIZE}) — PC-specific xi locked in config …")
    labels = cluster_optics(
        dmat,
        min_samples=OPTICS_MIN_SAMPLES,
        xi=OPTICS_XI_PC,
        min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
    )
    summary = cluster_summary(labels)
    print(f"    {summary}")

    # Decorate with tickers + sectors
    as_of_ts = pd.Timestamp(AS_OF)
    tickers = ticker_lookup(panel.columns.tolist(), crsp=crsp, as_of=as_of_ts)
    siccds = siccd_lookup(panel.columns.tolist(), crsp=crsp, as_of=as_of_ts)
    sectors = siccds.apply(sic_division)

    # Per-cluster table (just first 15 to keep output tidy)
    print(f"\n[5] First 15 clusters (of {summary['n_clusters']}):")
    print(f"    {'cluster':>7} | {'size':>4} | {'dominant SIC':<22} | tickers")
    print(f"    {'-' * 7}-+-{'-' * 4}-+-{'-' * 22}-+-{'-' * 50}")
    clustered = labels[labels != -1]
    for cluster_id, group in sorted(clustered.groupby(clustered), key=lambda kv: kv[0])[:15]:
        members = group.index
        member_tickers = tickers.reindex(members).fillna("?").tolist()
        member_sectors = sectors.reindex(members)
        dominant = member_sectors.value_counts().idxmax()
        ticker_str = ", ".join(sorted(map(str, member_tickers)))
        if len(ticker_str) > 60:
            ticker_str = ticker_str[:57] + "..."
        print(f"    {cluster_id:>7} | {len(members):>4} | {dominant:<22} | {ticker_str}")

    # GOOG / GOOGL co-cluster check (CP1 sanity carries over)
    print("\n[6] GOOG / GOOGL co-cluster check:")
    goog_mask = tickers.isin(["GOOG", "GOOGL"])
    goog_permnos = tickers[goog_mask].index.tolist()
    if len(goog_permnos) >= 2:
        goog_labels = labels.reindex(goog_permnos)
        for permno, lbl in goog_labels.items():
            tkr = tickers.get(permno, f"#{permno}")
            print(f"    {tkr:<6}  permno={permno:<6}  cluster={lbl}")
        if len(goog_labels.unique()) == 1 and goog_labels.iloc[0] != -1:
            print(f"    ✅ co-clustered")
        else:
            print(f"    ⚠ NOT co-clustered — investigate")

    pairs = clusters_to_pairs(labels)
    purity = purity_index(labels, sectors)

    print(f"\n[7] Total within-cluster pairs : {len(pairs):,}")
    print(f"\n[8] Scorecard vs paper:")
    print(f"    {'metric':<28} | {'ours':>8} | {'paper':>6} | tolerance")
    print(f"    {'-' * 28}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 12}")
    print(f"    {'# PC clusters':<28} | {summary['n_clusters']:>8} | {'109':>6} | ±10")
    print(f"    {'purity vs SIC division':<28} | {purity:>8.3f} | {'0.84':>6} | ±0.05")

    count_ok = abs(summary["n_clusters"] - 109) <= 10
    purity_ok = abs(purity - 0.84) <= 0.05
    print()
    if count_ok and purity_ok:
        print("    ✅ Both metrics within tolerance of paper.")
    else:
        miss = []
        if not count_ok:
            miss.append(f"cluster count {summary['n_clusters']} vs target 109")
        if not purity_ok:
            miss.append(f"purity {purity:.3f} vs target 0.84")
        print(f"    ⚠ Out of tolerance: {'; '.join(miss)}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
