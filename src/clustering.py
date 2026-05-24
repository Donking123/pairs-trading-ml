"""
Clustering and cluster validation  [Pipeline Stage 2].

Takes a distance matrix from distances.py and groups stocks into clusters with
OPTICS. Stocks in the same cluster are candidate pair partners; everything else is
an outlier (filtered out). purity_index() checks the clusters against SIC industry
sectors - a sanity check that the unsupervised algorithm found economically
meaningful groups.

OPTICS (Ordering Points To Identify the Clustering Structure):
  * density-based - clusters are dense regions; sparse points become outliers
  * does NOT need the number of clusters specified up front (unlike k-means)
  * handles clusters of differing density (unlike DBSCAN)
  * the paper feeds it a precomputed distance matrix, min cluster size = 2
"""
from __future__ import annotations

import itertools

import pandas as pd
from sklearn.cluster import OPTICS


def sic_division(siccd) -> str:
    """Map a 4-digit SIC code to one of the 10 major industry divisions.

    Ranges per the paper's footnote 2 (standard SIC division structure).
    """
    try:
        d = int(siccd) // 100  # first two digits of the SIC code
    except (TypeError, ValueError):
        return "unknown"
    if d <= 9:
        return "agriculture"
    if d <= 14:
        return "mining"
    if d <= 17:
        return "construction"
    if d <= 39:
        return "manufacturing"
    if d <= 49:
        return "transport_utilities"
    if d <= 51:
        return "wholesale"
    if d <= 59:
        return "retail"
    if d <= 67:
        return "finance"
    if d <= 89:
        return "services"
    return "public_admin"


def cluster_optics(
    distance_matrix: pd.DataFrame,
    min_samples: int = 2,
    xi: float = 0.05,
    min_cluster_size: int = 2,
) -> pd.Series:
    """Cluster stocks with OPTICS on a precomputed distance matrix.

    Parameters
    ----------
    distance_matrix : DataFrame
        Square distance matrix from distances.py.
    min_samples : int
        Neighbourhood size for a point to count as 'core'. Not published by the
        paper - a hyperparameter to tune so cluster counts match (plan: CP1).
    xi : float
        Steepness threshold for the OPTICS 'xi' cluster-extraction method.
    min_cluster_size : int
        Minimum stocks per cluster. Paper: 2.

    Returns
    -------
    Series
        Cluster label per stock (index = permno). Label -1 = outlier (unclustered).
    """
    model = OPTICS(
        metric="precomputed",
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        cluster_method="xi",
    )
    labels = model.fit_predict(distance_matrix.to_numpy())
    return pd.Series(labels, index=distance_matrix.index, name="cluster")


def clusters_to_pairs(labels: pd.Series) -> list[tuple]:
    """Expand cluster labels into the list of within-cluster candidate pairs.

    This is the filtering payoff: ~80,000 possible pairs collapse to a few hundred
    economically related ones.
    """
    pairs: list[tuple] = []
    clustered = labels[labels != -1]
    for _, group in clustered.groupby(clustered):
        pairs.extend(itertools.combinations(sorted(group.index), 2))
    return pairs


def purity_index(labels: pd.Series, sectors: pd.Series) -> float:
    """Cluster purity vs industry sectors - Rotondi & Russo (2025), Eq. (2).

    For each cluster, count the stocks in its majority sector; sum across clusters
    and divide by the number of *clustered* stocks. Purity ~ 1 means clusters are
    almost entirely single-sector.

    Note on N: the paper's Eq. (2) writes 'overall stocks', but with their reported
    outlier counts that would cap purity near 0.3 - inconsistent with their reported
    0.81-0.94. Purity is therefore taken over clustered (non-outlier) stocks.
    """
    clustered = labels[labels != -1]
    if len(clustered) == 0:
        return float("nan")
    majority_total = 0
    for _, group in clustered.groupby(clustered):
        sec = sectors.reindex(group.index)
        majority_total += sec.value_counts().max()
    return majority_total / len(clustered)


def cluster_summary(labels: pd.Series) -> dict:
    """Quick descriptive stats for a clustering result."""
    clustered = labels[labels != -1]
    sizes = clustered.value_counts()
    return {
        "n_clusters": int(clustered.nunique()),
        "n_clustered_stocks": int(len(clustered)),
        "n_outliers": int((labels == -1).sum()),
        "mean_cluster_size": round(float(sizes.mean()), 2) if len(sizes) else 0.0,
        "max_cluster_size": int(sizes.max()) if len(sizes) else 0,
    }
