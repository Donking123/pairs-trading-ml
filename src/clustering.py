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

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, OPTICS, AgglomerativeClustering


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


def cluster_hdbscan(
    distance_matrix: pd.DataFrame,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
) -> pd.Series:
    """Cluster stocks with HDBSCAN on a precomputed distance matrix  [Phase 3a].

    Robustness alternative to OPTICS. HDBSCAN is also density-based but extracts a
    flat clustering from the condensed tree via cluster stability, rather than
    OPTICS's reachability-steepness (xi) rule. It auto-determines the number of
    clusters — there is no xi-equivalent to tune.

    Parameters
    ----------
    distance_matrix : DataFrame
        Square precomputed distance matrix from distances.py.
    min_cluster_size : int
        Smallest grouping to call a cluster (paper convention: 2).
    min_samples : int, optional
        Conservativeness of the density estimate. Defaults to `min_cluster_size`
        when None (sklearn's own default behaviour).

    Returns
    -------
    Series
        Cluster label per stock (index = permno). Label -1 = outlier.
    """
    model = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    # HDBSCAN needs a float64 contiguous matrix; precomputed must be non-negative.
    dm = np.ascontiguousarray(distance_matrix.to_numpy(), dtype="float64")
    labels = model.fit_predict(dm)
    return pd.Series(labels, index=distance_matrix.index, name="cluster")


def cluster_hierarchical(
    distance_matrix: pd.DataFrame,
    distance_threshold: float | None = None,
    distance_quantile: float | None = None,
    linkage: str = "average",
    min_cluster_size: int = 2,
) -> pd.Series:
    """Agglomerative (hierarchical) clustering on a precomputed distance matrix [Phase 3a].

    Robustness alternative to OPTICS. Builds a dendrogram by repeatedly merging the
    closest groups, then cuts it at a height.

    The cut height can be set two ways (exactly one required):
      * `distance_threshold` — an absolute height (used by unit tests).
      * `distance_quantile`  — a quantile of the off-diagonal pairwise distances,
        computed per call. **This is the backtest default**: the absolute distance
        scale drifts over the 21-year sample, so a fixed height gives wildly different
        cluster counts across time (it would test threshold drift, not the algorithm).
        A fixed *quantile* auto-adapts to the scale, keeping the candidate-pair count
        temporally stable.

    Note: **Ward linkage is not available here** — it requires raw Euclidean features,
    whereas SSD/PC supply only a precomputed distance matrix. Average linkage is the
    standard, correct choice for precomputed distances (complete/single also valid).

    Clusters smaller than `min_cluster_size` (singletons when =2) are relabelled as
    outliers (-1), matching the OPTICS/HDBSCAN convention.

    Returns
    -------
    Series
        Cluster label per stock (index = permno). Label -1 = outlier.
    """
    if (distance_threshold is None) == (distance_quantile is None):
        raise ValueError("pass exactly one of distance_threshold or distance_quantile")

    dm = distance_matrix.to_numpy()
    if distance_quantile is not None:
        iu = np.triu_indices_from(dm, k=1)
        distance_threshold = float(np.quantile(dm[iu], distance_quantile))

    model = AgglomerativeClustering(
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    raw = model.fit_predict(dm)
    labels = pd.Series(raw, index=distance_matrix.index, name="cluster")

    # Relabel singletons / sub-threshold groups as outliers (-1) for parity with
    # the density-based methods.
    sizes = labels.value_counts()
    small = sizes.index[sizes < min_cluster_size]
    labels = labels.where(~labels.isin(small), -1)
    return labels


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
