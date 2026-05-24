"""
03_clustering.py
────────────────
For each formation window, clusters stocks by the similarity of their
factor-beta vectors (i.e. how similar their factor exposures are).

Distance metric: 1 − correlation(beta_i, beta_j)
  → two stocks with identical factor loadings have distance 0
  → two stocks with uncorrelated factor loadings have distance 1

Algorithm: Agglomerative Hierarchical Clustering with average linkage.
  - No need to specify k upfront
  - distance_threshold controls coarseness (tune via diagnostics printed below)
  - Average linkage is less sensitive to outlier stocks than complete or ward

Output:
  data/processed/clusters/clusters_YYYYMMDD_YYYYMMDD.parquet
    → columns: permno, cluster_id
  data/processed/cluster_diagnostics.csv
    → per-window cluster count and size distribution (use to tune threshold)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from tqdm import tqdm
import warnings

from config import (
    DATA_PROC,
    CLUSTER_DISTANCE_THRESHOLD,
    MIN_CLUSTER_SIZE,
    FACTOR_NAMES,
)

BETAS_DIR   = DATA_PROC / "betas"
CLUSTER_DIR = DATA_PROC / "clusters"
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)


# ── Cluster one window ────────────────────────────────────────────────────────
def cluster_one_window(beta_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Given a (n_stocks × n_factors) beta matrix, returns a DataFrame with
    columns [permno, cluster_id].

    Stocks assigned to clusters smaller than MIN_CLUSTER_SIZE get
    cluster_id = -1 (excluded from pair formation, like DBSCAN noise).
    """
    # Drop stocks with any NaN beta (incomplete regression)
    beta_clean = beta_df.dropna()
    if len(beta_clean) < 2:
        return pd.DataFrame(columns=["permno", "cluster_id"])

    # Standardise each factor column (unit variance) so no single factor
    # dominates the distance metric just because it has larger beta magnitudes
    beta_std = (beta_clean - beta_clean.mean()) / (beta_clean.std() + 1e-9)

    # Correlation-based distance matrix
    corr = beta_std.T.corr()          # (n_stocks × n_stocks) correlation of beta vectors
    dist = (1 - corr).clip(lower=0)   # clip tiny negatives from float precision

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage="average",
            metric="precomputed",
        )
        labels = model.fit_predict(dist.values)

    result = pd.DataFrame({
        "permno":     beta_clean.index.astype(int),
        "cluster_id": labels,
    })

    # Remove clusters that are too small (professor's guidance: avoid tiny clusters)
    cluster_sizes = result["cluster_id"].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index
    result.loc[~result["cluster_id"].isin(valid_clusters), "cluster_id"] = -1

    return result


# ── Diagnostics helper ────────────────────────────────────────────────────────
def cluster_diagnostics(result: pd.DataFrame) -> dict:
    """Returns summary stats for one window's clustering."""
    valid = result[result["cluster_id"] >= 0]
    if valid.empty:
        return {"n_clusters": 0, "n_stocks": 0, "mean_size": 0,
                "min_size": 0, "max_size": 0, "n_unclustered": len(result)}
    sizes = valid["cluster_id"].value_counts()
    return {
        "n_clusters":    len(sizes),
        "n_stocks":      len(valid),
        "mean_size":     round(sizes.mean(), 1),
        "min_size":      int(sizes.min()),
        "max_size":      int(sizes.max()),
        "n_unclustered": int((result["cluster_id"] == -1).sum()),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    windows = pd.read_parquet(DATA_PROC / "window_index.parquet")
    beta_files = sorted(BETAS_DIR.glob("betas_*.parquet"))

    if not beta_files:
        raise FileNotFoundError(f"No beta files found in {BETAS_DIR}. "
                                "Run 02_rolling_betas.py first.")

    print(f"Clustering {len(beta_files)} windows  "
          f"(distance_threshold={CLUSTER_DISTANCE_THRESHOLD}, "
          f"min_cluster_size={MIN_CLUSTER_SIZE})…\n")

    all_diagnostics = []

    for bf in tqdm(beta_files, desc="  Windows"):
        beta_df = pd.read_parquet(bf)

        # Keep only the factor columns defined in config
        available = [c for c in FACTOR_NAMES if c in beta_df.columns]
        beta_df   = beta_df[available]

        result = cluster_one_window(beta_df, CLUSTER_DISTANCE_THRESHOLD)

        # Save clusters
        date_str = bf.stem.replace("betas_", "")   # YYYYMMDD_YYYYMMDD
        out      = CLUSTER_DIR / f"clusters_{date_str}.parquet"
        result.to_parquet(out, index=False)

        diag = cluster_diagnostics(result)
        diag["window"] = date_str
        all_diagnostics.append(diag)

    # Save diagnostics
    diag_df = pd.DataFrame(all_diagnostics).set_index("window")
    diag_out = DATA_PROC / "cluster_diagnostics.csv"
    diag_df.to_csv(diag_out)

    # Print summary so you can judge whether to tune the threshold
    print("\n── Cluster size diagnostics (across all windows) ──")
    print(f"  Mean clusters per window : {diag_df['n_clusters'].mean():.1f}")
    print(f"  Mean stocks per cluster  : {diag_df['mean_size'].mean():.1f}  "
          f"(target: 8–20)")
    print(f"  Overall min cluster size : {diag_df['min_size'].min()}")
    print(f"  Overall max cluster size : {diag_df['max_size'].max()}")
    print(f"  Mean unclustered stocks  : {diag_df['n_unclustered'].mean():.1f}")
    print()
    print("  If mean_size < 8  → lower CLUSTER_DISTANCE_THRESHOLD in config.py")
    print("  If mean_size > 20 → raise  CLUSTER_DISTANCE_THRESHOLD in config.py")
    print()
    print(f"  Diagnostics saved → {diag_out}")
    print("\nNext: python 04_cointegration.py")
