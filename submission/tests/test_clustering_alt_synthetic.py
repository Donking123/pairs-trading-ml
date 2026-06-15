"""
Synthetic-data sanity check for the Phase 3a alternative clusterers
(clustering.cluster_hdbscan / cluster_hierarchical).

We plant K obvious clusters of near-identical stocks plus noise, build an SSD
distance matrix, and require both new clusterers to:
  * recover ~K clusters,
  * keep planted cluster-mates together (co-membership),
  * leave noise stocks as outliers (label -1) most of the time.

Also checks the quantile-vs-absolute threshold API for cluster_hierarchical.

Run:  python -m tests.test_clustering_alt_synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering import (
    cluster_hdbscan,
    cluster_hierarchical,
    clusters_to_pairs,
    purity_index,
)
from src.distances import ssd_distance


def _planted_panel(n_clusters=4, per=5, n_noise=6, n_days=252, seed=7):
    rng = np.random.default_rng(seed)
    cols, rets, truth = [], [], {}
    for c in range(n_clusters):
        common = rng.normal(0.0005, 0.015, size=n_days)
        for j in range(per):
            sid = f"c{c}_s{j}"
            rets.append(common + rng.normal(0.0, 0.0008, size=n_days))
            cols.append(sid); truth[sid] = c
    for j in range(n_noise):
        sid = f"noise_{j}"
        rets.append(rng.normal(0.0, 0.020, size=n_days))
        cols.append(sid); truth[sid] = -1
    prices = 100.0 * (1.0 + pd.DataFrame(np.asarray(rets).T, columns=cols)).cumprod()
    return prices, pd.Series(truth, name="planted")


def _comembership_purity(labels: pd.Series, planted: pd.Series) -> float:
    """Purity of recovered clusters against the planted cluster ids."""
    return purity_index(labels, planted)


def test_hdbscan_recovers_planted_clusters() -> None:
    prices, planted = _planted_panel()
    dmat = ssd_distance(prices)
    labels = cluster_hdbscan(dmat, min_cluster_size=2)
    n_clusters = len(set(labels[labels >= 0]))
    pur = _comembership_purity(labels, planted)
    print(f"  HDBSCAN: {n_clusters} clusters, purity vs planted = {pur:.3f}")
    assert 3 <= n_clusters <= 6, f"expected ~4 clusters, got {n_clusters}"
    assert pur > 0.9, f"clusters not pure vs planted truth: {pur:.3f}"


def test_hierarchical_recovers_planted_clusters() -> None:
    prices, planted = _planted_panel()
    dmat = ssd_distance(prices)
    # absolute threshold: planted clusters are very tight, noise is far
    labels = cluster_hierarchical(dmat, distance_threshold=5.0, min_cluster_size=2)
    n_clusters = len(set(labels[labels >= 0]))
    pur = _comembership_purity(labels, planted)
    print(f"  hierarchical(abs): {n_clusters} clusters, purity vs planted = {pur:.3f}")
    assert 3 <= n_clusters <= 6, f"expected ~4 clusters, got {n_clusters}"
    assert pur > 0.9, f"clusters not pure vs planted truth: {pur:.3f}"


def test_hierarchical_quantile_mode_runs_and_groups() -> None:
    prices, planted = _planted_panel()
    dmat = ssd_distance(prices)
    labels = cluster_hierarchical(dmat, distance_quantile=0.05, min_cluster_size=2)
    pairs = clusters_to_pairs(labels)
    print(f"  hierarchical(q=0.05): {len(set(labels[labels>=0]))} clusters, {len(pairs)} pairs")
    assert len(pairs) > 0, "quantile-mode hierarchical produced no pairs"


def test_hierarchical_requires_exactly_one_threshold() -> None:
    prices, _ = _planted_panel()
    dmat = ssd_distance(prices)
    for kwargs in [{}, {"distance_threshold": 5.0, "distance_quantile": 0.05}]:
        try:
            cluster_hierarchical(dmat, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for kwargs={kwargs}")
    print("  rejects 0 and 2 thresholds correctly ✓")


def test_singletons_become_outliers() -> None:
    """A lone far-away stock must be labelled -1, not its own cluster."""
    prices, planted = _planted_panel(n_clusters=2, per=4, n_noise=1)
    dmat = ssd_distance(prices)
    labels = cluster_hierarchical(dmat, distance_threshold=5.0, min_cluster_size=2)
    # the single noise stock should be an outlier
    noise_label = labels[[c for c in labels.index if c.startswith("noise")][0]]
    print(f"  lone noise stock label = {noise_label} (expect -1)")
    assert noise_label == -1, "singleton was not relabelled as outlier"


if __name__ == "__main__":
    tests = [
        test_hdbscan_recovers_planted_clusters,
        test_hierarchical_recovers_planted_clusters,
        test_hierarchical_quantile_mode_runs_and_groups,
        test_hierarchical_requires_exactly_one_threshold,
        test_singletons_become_outliers,
    ]
    failures = 0
    for t in tests:
        print(f"\n▶ {t.__name__}")
        try:
            t(); print("  ✅ PASS")
        except AssertionError as e:
            failures += 1; print(f"  ❌ FAIL — {e}")
        except Exception as e:
            failures += 1; print(f"  💥 ERROR — {type(e).__name__}: {e}")
    print(f"\n{'─'*60}")
    print(f"{'✅ all passed' if not failures else f'❌ {failures} failed'}")
    sys.exit(1 if failures else 0)
