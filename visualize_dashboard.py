"""
visualize_dashboard.py — Interactive Plotly dashboard for clustering diagnostics (SPX).

Outputs a single HTML with:
  - 3D PCA embedding of fused features (corr PCs + EWMA ridge betas)
  - 2D t-SNE embedding (fused features)
  - Cluster size bar chart
  - LW-shrunk return correlation heatmap ordered by cluster
  - ARI stability over time (monthly refit)

Usage:
  python visualize_dashboard.py --method fused --save "figures/cluster_dashboard.html"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from spx_data import load_all
from pairs_feature_matrix import (
    cluster_method_a,
    cluster_method_b,
    cluster_fused,
    _lw_correlation,
    _slice_window,
    _ridge_betas_ewma,
)
from sklearn.metrics import adjusted_rand_score


SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE"]


def assign_sector_by_corr(stock_rets: pd.DataFrame, factor_rets: pd.DataFrame, window: int, end_date: pd.Timestamp) -> pd.Series:
    w = _slice_window(stock_rets, end_date, window)
    fr = factor_rets[SECTOR_ETFS].reindex(w.index).fillna(0.0)
    sectors = {}
    for tkr in w.columns:
        r = w[tkr].fillna(0.0).values
        best_s, best_c = None, -1
        for s in fr.columns:
            c = np.corrcoef(r, fr[s].values)[0, 1]
            if np.isnan(c):
                c = -1
            if c > best_c:
                best_c, best_s = c, s
        sectors[tkr] = best_s
    return pd.Series(sectors)


def build_fused_features(rets: pd.DataFrame, frets: pd.DataFrame, tickers: list[str], end_date: pd.Timestamp, corr_window: int, beta_window: int, n_corr_pcs: int = 10) -> tuple[np.ndarray, np.ndarray]:
    # Corr PCs
    w = _slice_window(rets[tickers], end_date, corr_window)
    C = _lw_correlation(w.values)
    n_pcs = max(2, min(n_corr_pcs, C.shape[0] - 1))
    X_corr = PCA(n_components=n_pcs, random_state=42).fit_transform(C)
    # Betas scaled by factor vol
    beta_df = _ridge_betas_ewma(rets, frets, window=beta_window, end_date=end_date)
    factor_vol = frets.std().replace(0, np.nan).fillna(1.0)
    common = beta_df.columns.intersection(factor_vol.index)
    B = (beta_df[common] * factor_vol[common].values).reindex(tickers).fillna(0.0).values
    X = np.concatenate([X_corr[: len(tickers), :n_pcs], B], axis=1)
    return C, X


def ari_series(rets: pd.DataFrame, frets: pd.DataFrame, method: str, corr_window: int, beta_window: int, refit_freq: int = 21) -> pd.Series:
    burn_in = max(corr_window, beta_window)
    idxs = list(range(burn_in, len(rets), refit_freq))
    prev = None
    aris = []
    dates = []
    for i in idxs:
        end_date = rets.index[i]
        if method == "a":
            labels, _ = cluster_method_a(rets, end_date=end_date, window=corr_window)
        elif method == "b":
            _, k = cluster_method_a(rets, end_date=end_date, window=corr_window)
            labels = cluster_method_b(rets, frets, end_date=end_date, k=k)
        else:
            labels, _ = cluster_fused(rets, frets, end_date=end_date, corr_window=corr_window, beta_window=beta_window)
        if prev is None:
            aris.append(np.nan)
        else:
            common = labels.index.intersection(prev.index)
            val = adjusted_rand_score(prev.loc[common], labels.loc[common]) if len(common) > 5 else np.nan
            aris.append(val)
        prev = labels
        dates.append(end_date)
    return pd.Series(aris, index=dates, name="ARI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["a", "b", "fused"], default="fused")
    ap.add_argument("--corr-window", type=int, default=126)
    ap.add_argument("--beta-window", type=int, default=252)
    ap.add_argument("--n-corr-pcs", type=int, default=10)
    ap.add_argument("--save", type=str, default="figures/cluster_dashboard.html")
    args = ap.parse_args()

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)

    close, volume, rets, fclose, frets = load_all(refresh=False)
    end_date = rets.index[-1]
    # Labels
    if args.method == "a":
        labels, k = cluster_method_a(rets, end_date=end_date, window=args.corr_window)
    elif args.method == "b":
        labels_a, k = cluster_method_a(rets, end_date=end_date, window=args.corr_window)
        labels = cluster_method_b(rets, frets, end_date=end_date, k=k)
    else:
        labels, k = cluster_fused(rets, frets, end_date=end_date, corr_window=args.corr_window, beta_window=args.beta_window)

    tickers = labels.index.tolist()
    cluster = labels.astype(int)
    sector = assign_sector_by_corr(rets[tickers], frets, window=args.corr_window, end_date=end_date)

    # Features
    C, X = build_fused_features(rets, frets, tickers, end_date, args.corr_window, args.beta_window, args.n_corr_pcs)

    # 3D PCA of fused features
    X_pca3 = PCA(n_components=3, random_state=42).fit_transform(X)
    fig3d = px.scatter_3d(
        x=X_pca3[:, 0], y=X_pca3[:, 1], z=X_pca3[:, 2],
        color=cluster.astype(str), symbol=sector, hover_name=tickers,
        labels={"color": "cluster", "symbol": "sector"},
        title=f"3D PCA of fused features (k≈{cluster.nunique()}, method={args.method})",
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    # 2D t-SNE of fused features
    X_tsne2 = TSNE(n_components=2, perplexity=min(30, max(5, len(tickers)//10)), random_state=42, init="pca").fit_transform(X)
    fig2d = px.scatter(
        x=X_tsne2[:, 0], y=X_tsne2[:, 1], color=cluster.astype(str), hover_name=tickers,
        labels={"color": "cluster"}, title="t-SNE (2D) of fused features",
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    # Cluster sizes
    size_s = cluster.value_counts().sort_index()
    fig_sizes = px.bar(x=size_s.index.astype(str), y=size_s.values, labels={"x": "cluster", "y": "count"}, title="Cluster sizes")

    # Correlation heatmap ordered by cluster
    order = np.argsort(cluster.values)
    C_ord = C[order][:, order]
    fig_heat = go.Figure(data=go.Heatmap(z=C_ord, colorscale="RdBu", zmid=0))
    fig_heat.update_layout(title="LW-shrunk return correlation ordered by cluster", xaxis_showticklabels=False, yaxis_showticklabels=False)

    # ARI stability
    ari = ari_series(rets, frets, args.method, args.corr_window, args.beta_window)
    fig_ari = px.line(x=ari.index, y=ari.values, labels={"x": "date", "y": "ARI"}, title="ARI stability over time")
    fig_ari.update_yaxes(range=[0, 1])

    # Write out a simple HTML with stacked figures
    html_parts = [
        fig3d.to_html(full_html=False, include_plotlyjs="cdn"),
        fig2d.to_html(full_html=False, include_plotlyjs=False),
        fig_sizes.to_html(full_html=False, include_plotlyjs=False),
        fig_heat.to_html(full_html=False, include_plotlyjs=False),
        fig_ari.to_html(full_html=False, include_plotlyjs=False),
    ]
    html = "\n<hr/>\n".join(html_parts)
    out.write_text(html, encoding="utf-8")
    print(f"Saved dashboard to {out}")


if __name__ == "__main__":
    main()

