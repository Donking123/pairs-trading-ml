"""
First real-data run — SSD + OPTICS on the Dec-2023 formation window.

Run:  python -m notebooks.01_dec2023_ssd_clustering   (from pairs-trading-ml/)

Outputs (printed to stdout):
  1. Formation-window panel summary
  2. Cluster summary (n_clusters, outliers, sizes)
  3. Per-cluster table (size, dominant SIC division, tickers)
  4. Sample within-cluster pairs as ticker tuples
  5. GOOG / GOOGL co-cluster check (CP1 sanity)
  6. Numbers vs paper (48 clusters / 0.81 purity)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# allow `from src.* import *` when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.clustering import (
    cluster_optics,
    cluster_summary,
    clusters_to_pairs,
    purity_index,
    sic_division,
)
from src.distances import ssd_distance
from src.panel import (
    formation_window_panel,
    load_crsp_daily,
    load_sp500_constituents,
    siccd_lookup,
    ticker_lookup,
)


from src.config import OPTICS_MIN_SAMPLES, OPTICS_XI, OPTICS_MIN_CLUSTER_SIZE

AS_OF = "2023-12-29"  # last trading day of 2023 (paper's eval date)


def fmt_pair(a: int, b: int, tickers: pd.Series) -> str:
    ta = tickers.get(a, f"#{a}")
    tb = tickers.get(b, f"#{b}")
    return f"({ta}, {tb})"


def main() -> None:
    print("=" * 72)
    print(f"Phase 1a — real Dec-2023 SSD + OPTICS run")
    print("=" * 72)

    # ── 1. Load + build the formation panel ──────────────────────────────
    print("\n[1] Loading CRSP daily + S&P 500 constituent panels …")
    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    print(f"    crsp_daily         : {crsp.shape[0]:>10,} rows × {crsp.shape[1]} cols")
    print(f"    sp500_constituents : {cons.shape[0]:>10,} membership intervals")

    print(f"\n[2] Building formation-window panel ending {AS_OF} …")
    panel = formation_window_panel(AS_OF, crsp=crsp, constituents=cons)
    print(f"    panel shape        : {panel.shape}  (days × stocks)")
    print(f"    window range       : {panel.index.min().date()} → {panel.index.max().date()}")

    # ── 2. SSD distance + OPTICS clustering ──────────────────────────────
    print("\n[3] Computing SSD distance matrix …")
    dmat = ssd_distance(panel)
    print(f"    distance matrix    : {dmat.shape}  ({dmat.shape[0] * (dmat.shape[0] - 1) // 2:,} unique pairs)")
    print(f"    min / median / max : {dmat.values[dmat.values > 0].min():.2f} / "
          f"{pd.Series(dmat.values.flatten()).median():.2f} / "
          f"{dmat.values.max():.2f}")

    print(f"\n[4] Running OPTICS (min_samples={OPTICS_MIN_SAMPLES}, xi={OPTICS_XI}, "
          f"min_cluster_size={OPTICS_MIN_CLUSTER_SIZE}) …")
    labels = cluster_optics(
        dmat,
        min_samples=OPTICS_MIN_SAMPLES,
        xi=OPTICS_XI,
        min_cluster_size=OPTICS_MIN_CLUSTER_SIZE,
    )
    summary = cluster_summary(labels)
    print(f"    {summary}")

    # ── 3. Look up tickers + SIC for the universe ────────────────────────
    as_of_ts = pd.Timestamp(AS_OF)
    tickers = ticker_lookup(panel.columns.tolist(), crsp=crsp, as_of=as_of_ts)
    siccds = siccd_lookup(panel.columns.tolist(), crsp=crsp, as_of=as_of_ts)
    sectors = siccds.apply(sic_division)

    # ── 4. Per-cluster table ─────────────────────────────────────────────
    print("\n[5] Per-cluster table (showing all clusters):")
    print(f"    {'cluster':>7} | {'size':>4} | {'dominant SIC division':<22} | tickers")
    print(f"    {'-' * 7}-+-{'-' * 4}-+-{'-' * 22}-+-{'-' * 40}")
    clustered = labels[labels != -1]
    for cluster_id, group in sorted(clustered.groupby(clustered), key=lambda kv: kv[0]):
        members = group.index
        member_tickers = tickers.reindex(members).fillna("?").tolist()
        member_sectors = sectors.reindex(members)
        dominant = member_sectors.value_counts().idxmax()
        ticker_str = ", ".join(sorted(map(str, member_tickers)))
        if len(ticker_str) > 60:
            ticker_str = ticker_str[:57] + "..."
        print(f"    {cluster_id:>7} | {len(members):>4} | {dominant:<22} | {ticker_str}")

    # ── 5. Sample within-cluster pairs ───────────────────────────────────
    pairs = clusters_to_pairs(labels)
    print(f"\n[6] Total within-cluster candidate pairs : {len(pairs):,}")
    print(f"    Sample (first 15):")
    for a, b in pairs[:15]:
        sa = sectors.get(a, "?")
        sb = sectors.get(b, "?")
        same = "✓ same" if sa == sb else "✗ diff"
        print(f"      {fmt_pair(a, b, tickers):<22}  {sa:<22} | {sb:<22}  [{same} sector]")

    # ── 6. GOOG / GOOGL sanity check (CP1) ───────────────────────────────
    print("\n[7] GOOG / GOOGL co-cluster check (CP1 sanity):")
    goog_mask = tickers.isin(["GOOG", "GOOGL"])
    goog_permnos = tickers[goog_mask].index.tolist()
    if len(goog_permnos) >= 2:
        goog_labels = labels.reindex(goog_permnos)
        for permno, lbl in goog_labels.items():
            tkr = tickers.get(permno, f"#{permno}")
            print(f"    {tkr:<6}  permno={permno:<6}  cluster={lbl}")
        unique = goog_labels.unique()
        if len(unique) == 1 and unique[0] != -1:
            print(f"    ✅ same cluster ({unique[0]})")
        else:
            print(f"    ⚠ GOOG/GOOGL in different OPTICS labels — check parameters")
    else:
        print(f"    ⚠ found only {len(goog_permnos)} of GOOG/GOOGL in the universe")

    # ── 7. Purity vs SIC and numbers-vs-paper ────────────────────────────
    purity = purity_index(labels, sectors)
    print("\n[8] Numbers vs paper:")
    print(f"    {'metric':<28} | {'ours':>8} | {'paper':>6} | tolerance")
    print(f"    {'-' * 28}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 12}")
    print(f"    {'# SSD clusters':<28} | {summary['n_clusters']:>8} | {'48':>6} | ±5")
    print(f"    {'purity vs SIC division':<28} | {purity:>8.3f} | {'0.81':>6} | ±0.05")

    # Cluster count must land in ±5 of paper. For purity we accept anything
    # within ±0.05 of 0.81 OR higher (higher purity = cleaner clusters, not worse).
    count_ok = abs(summary["n_clusters"] - 48) <= 5
    purity_ok = (purity >= 0.81 - 0.05)
    print()
    if count_ok and purity_ok:
        print("    ✅ Cluster count and purity within tolerance of paper.")
    else:
        print("    ⚠ One or both metrics outside paper tolerance — parameter-tune xi / min_samples.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
