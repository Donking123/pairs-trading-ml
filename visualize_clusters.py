"""
visualize_clusters.py — Quick visuals for cluster diagnostics on SPX universe.

Creates:
  - 2D embeddings (PCA and t-SNE) colored by cluster (Method A and B)
  - Cluster size bar charts
  - Correlation heatmap reordered by cluster labels (Method A)

Usage:
  python visualize_clusters.py --method fused --save-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse

from spx_data import load_all
from pairs_feature_matrix import (
    cluster_method_a,
    cluster_method_b,
    cluster_fused,
    _lw_correlation,
    _slice_window,
    _ridge_betas_ewma,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["a", "b", "fused"], default="fused")
    ap.add_argument("--corr-window", type=int, default=126)
    ap.add_argument("--beta-window", type=int, default=252)
    ap.add_argument("--w-ret", type=float, default=0.5)
    ap.add_argument("--save-dir", type=str, default="figures")
    ap.add_argument("--n-corr-pcs", type=int, default=10, help="#PCs from corr profiles to include in fused space")
    ap.add_argument("--umap", action="store_true", help="Use UMAP for 2D embedding if available")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    close, volume, rets, fclose, frets = load_all(refresh=False)
    end_date = rets.index[-1]

    # Labels
    if args.method == "a":
        labels, k = cluster_method_a(rets, end_date=end_date, window=args.corr_window)
    elif args.method == "b":
        labels_a, k = cluster_method_a(rets, end_date=end_date, window=args.corr_window)
        labels = cluster_method_b(rets, frets, end_date=end_date, k=k)
    else:
        labels, k = cluster_fused(
            rets, frets, end_date=end_date,
            corr_window=args.corr_window, beta_window=args.beta_window, w_ret=args.w_ret,
        )

    tickers = labels.index.tolist()
    L = labels.astype(int)

    # Build a fused feature space for cleaner visuals:
    #   - corr profile PCs (from LW correlation rows)
    #   - EWMA ridge betas scaled by factor vol
    w = _slice_window(rets[tickers], end_date, args.corr_window)
    R = w.values
    C = _lw_correlation(R)
    # Corr PCs
    n_pcs = max(2, args.n_corr_pcs)
    pca_corr = PCA(n_components=min(n_pcs, C.shape[0]-1), random_state=42)
    X_corr = pca_corr.fit_transform(C)
    # Betas
    beta_df = _ridge_betas_ewma(rets, frets, window=args.beta_window, end_date=end_date)
    fw = frets.reindex(rets.index).fillna(0.0)
    factor_vol = fw.std().replace(0, np.nan).fillna(1.0)
    common_cols = beta_df.columns.intersection(factor_vol.index)
    B = (beta_df[common_cols] * factor_vol[common_cols].values).reindex(tickers).fillna(0.0).values
    # Fuse matrices (align tickers ordering)
    X_fused = np.concatenate([X_corr[:len(tickers), :n_pcs], B], axis=1)

    # Aesthetics
    sns.set_theme(style="white", context="talk")

    # PCA on fused features
    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_fused)

    # 2D embedding: UMAP if requested and available, else t-SNE
    X_2d = None
    method_2d = "tsne"
    if args.umap:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, random_state=42)
            X_2d = reducer.fit_transform(X_fused)
            method_2d = "umap"
        except Exception:
            pass
    if X_2d is None:
        perplexity = min(30, max(5, len(tickers)//10))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        X_2d = tsne.fit_transform(X_fused)
        method_2d = f"tsne_p{perplexity}"

    # Plot PCA
    df_pca = pd.DataFrame({"x": X_pca[:,0], "y": X_pca[:,1], "cluster": L.values, "ticker": tickers})
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=df_pca, x="x", y="y", hue="cluster", palette="tab10", s=18, linewidth=0.2, edgecolor="white", alpha=0.85)
    plt.title(f"PCA embedding (method={args.method}, k≈{L.nunique()})")
    plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_dir / f"pca_{args.method}.png", dpi=150)

    # Plot t-SNE
    df_tsne = pd.DataFrame({"x": X_2d[:,0], "y": X_2d[:,1], "cluster": L.values, "ticker": tickers})
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=df_tsne, x="x", y="y", hue="cluster", palette="tab10", s=18, linewidth=0.2, edgecolor="white", alpha=0.85)
    plt.title(f"2D embedding ({method_2d}, method={args.method})")
    plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_dir / f"embed2d_{args.method}.png", dpi=150)

    # Draw 1-sigma covariance ellipses per cluster on PCA plot for professionalism
    fig, ax = plt.subplots(figsize=(7,6))
    sns.scatterplot(ax=ax, data=df_pca, x="x", y="y", hue="cluster", palette="tab10", s=18, linewidth=0.2, edgecolor="white", alpha=0.85, legend=False)
    for cl in sorted(df_pca.cluster.unique()):
        pts = df_pca[df_pca.cluster == cl][["x","y"]].values
        if len(pts) < 5:
            continue
        mu = pts.mean(axis=0)
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        width, height = 2*np.sqrt(vals)  # 1-sigma ellipse
        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        e = Ellipse(xy=mu, width=width, height=height, angle=angle, edgecolor='black', facecolor='none', lw=1.0, alpha=0.8)
        ax.add_patch(e)
        ax.text(mu[0], mu[1], str(int(cl)), fontsize=10, weight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.6))
    ax.set_title(f"PCA embedding with 1σ ellipses (method={args.method})")
    plt.tight_layout()
    plt.savefig(save_dir / f"pca_ellipses_{args.method}.png", dpi=150)

    # Sector vs cluster heatmap for professionalism
    # Infer sector via best-correlated sector ETF over corr_window
    from visualize_dashboard import assign_sector_by_corr
    sector = assign_sector_by_corr(rets[tickers], frets, window=args.corr_window, end_date=end_date)
    ctab = pd.crosstab(sector.fillna('NA'), L)
    plt.figure(figsize=(8,5))
    sns.heatmap(ctab, annot=False, cmap="Greens", cbar_kws={"shrink":0.7})
    plt.title("Sector (rows) vs Cluster (cols)")
    plt.tight_layout()
    plt.savefig(save_dir / f"sector_cluster_heatmap_{args.method}.png", dpi=150)

    # Cluster size bar
    plt.figure(figsize=(6,4))
    L.value_counts().sort_index().plot(kind="bar", color="#4C78A8")
    plt.ylabel("count")
    plt.xlabel("cluster")
    plt.title("Cluster sizes")
    plt.tight_layout()
    plt.savefig(save_dir / f"sizes_{args.method}.png", dpi=150)

    # Correlation heatmap reordered by cluster labels (Method A view)
    order = np.argsort(L.values)
    C_ord = C[order][:, order]
    plt.figure(figsize=(8,6))
    sns.heatmap(C_ord, cmap="coolwarm", center=0.0, xticklabels=False, yticklabels=False, cbar_kws={"shrink":0.75})
    plt.title("LW-shrunk return correlation (ordered by cluster)")
    plt.tight_layout()
    plt.savefig(save_dir / f"corr_heatmap_{args.method}.png", dpi=150)

    print(f"Saved figures to {save_dir}/")


if __name__ == "__main__":
    main()
