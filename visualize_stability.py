"""
visualize_stability.py — ARI stability and cluster turnover diagnostics.

Generates:
  - ARI vs time (method=a/b/fused)
  - Cluster size over time heatmap

Usage:
  python visualize_stability.py --method fused --refit 21 --corr-window 126 --beta-window 252 --save-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from spx_data import load_all
from pairs_feature_matrix import rolling_cluster_labels, adjusted_rand_score, cluster_method_a, cluster_method_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["a", "b", "fused"], default="fused")
    ap.add_argument("--corr-window", type=int, default=126)
    ap.add_argument("--beta-window", type=int, default=252)
    ap.add_argument("--refit", type=int, default=21)
    ap.add_argument("--save-dir", type=str, default="figures")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    close, volume, rets, fclose, frets = load_all(refresh=False)

    labels_df = rolling_cluster_labels(
        rets, frets, method=args.method,
        corr_window=args.corr_window, beta_window=args.beta_window, refit_freq=args.refit,
    )

    # ARI between consecutive refits
    dates = labels_df.index[~labels_df.isna().all(axis=1)]
    aris = []
    prev = None
    for d in dates:
        curr = labels_df.loc[d].dropna()
        if prev is None:
            aris.append(np.nan)
        else:
            common = curr.index.intersection(prev.index)
            if len(common) > 2:
                aris.append(adjusted_rand_score(prev.loc[common], curr.loc[common]))
            else:
                aris.append(np.nan)
        prev = curr

    ari_s = pd.Series(aris, index=dates, name="ARI")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8,3))
    sns.lineplot(data=ari_s)
    plt.ylim(0,1)
    plt.title(f"ARI stability over time (method={args.method})")
    plt.tight_layout()
    plt.savefig(save_dir / f"ari_{args.method}.png", dpi=150)

    # Cluster size map
    sizes = labels_df.apply(lambda row: row.dropna().value_counts(), axis=1).fillna(0).astype(int)
    plt.figure(figsize=(8,4))
    sns.heatmap(sizes.T, cmap="Blues", cbar_kws={"shrink":0.75})
    plt.title("Cluster sizes over time")
    plt.xlabel("date")
    plt.ylabel("cluster")
    plt.tight_layout()
    plt.savefig(save_dir / f"sizes_time_{args.method}.png", dpi=150)

    print(f"Saved ARI and size plots to {save_dir}/")


if __name__ == "__main__":
    main()

