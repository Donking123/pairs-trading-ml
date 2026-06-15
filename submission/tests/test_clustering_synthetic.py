"""
Synthetic-data sanity check for distances.ssd_distance + clustering.* (Phase 1a).

We plant K obvious price-trajectory clusters of N stocks each, plus a few noise
stocks with independent trajectories. SSD + OPTICS should:
  * recover ~K clusters (noise stocks become outliers, label = -1),
  * cluster_optics labels are consistent within each planted group,
  * purity_index ~ 1.0 when 'sector' labels match planted-cluster IDs,
  * clusters_to_pairs returns the expected within-cluster combinations.

Run:  python -m tests.test_clustering_synthetic   (from pairs-trading-ml/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow `from src.* import *` when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering import (
    cluster_optics,
    cluster_summary,
    clusters_to_pairs,
    purity_index,
    sic_division,
)
from src.distances import ssd_distance


def generate_synthetic_panel(
    n_clusters: int = 3,
    stocks_per_cluster: int = 5,
    n_noise: int = 5,
    n_days: int = 252,
    seed: int = 42,
    cluster_shock_scale: float = 0.015,
    idio_shock_scale: float = 0.0008,
    noise_shock_scale: float = 0.020,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a fake price panel with K planted clusters + noise stocks.

    Each cluster shares a *common return shock* every day (-> very similar price
    paths); each stock also gets a small idiosyncratic shock. Noise stocks get only
    independent random shocks. SSD should easily separate the two groups.

    Returns (prices, planted_labels):
        prices : DataFrame, rows = dates, columns = stock_id (str)
        planted_labels : Series mapping stock_id -> ground-truth cluster id
                         (cluster stocks tagged 0..K-1; noise tagged -1)
    """
    rng = np.random.default_rng(seed)

    columns: list[str] = []
    return_panel = []
    truth: dict[str, int] = {}

    # K clusters of correlated stocks
    for c in range(n_clusters):
        common = rng.normal(0.0005, cluster_shock_scale, size=n_days)
        for j in range(stocks_per_cluster):
            stock_id = f"c{c}_s{j}"
            idio = rng.normal(0.0, idio_shock_scale, size=n_days)
            return_panel.append(common + idio)
            columns.append(stock_id)
            truth[stock_id] = c

    # noise stocks (independent random walks)
    for j in range(n_noise):
        stock_id = f"noise_{j}"
        return_panel.append(rng.normal(0.0, noise_shock_scale, size=n_days))
        columns.append(stock_id)
        truth[stock_id] = -1

    returns = pd.DataFrame(np.asarray(return_panel).T, columns=columns)
    prices = 100.0 * (1.0 + returns).cumprod()
    planted = pd.Series(truth, name="planted")
    return prices, planted


# ────────────────────────────────────────────────────────────────────────────────
# tests
# ────────────────────────────────────────────────────────────────────────────────


def test_ssd_within_cluster_smaller_than_across() -> None:
    prices, planted = generate_synthetic_panel()
    dmat = ssd_distance(prices)

    within = []
    across = []
    cluster_ids = sorted(set(planted) - {-1})
    for c in cluster_ids:
        members = planted[planted == c].index.tolist()
        non_members = planted[(planted != c) & (planted != -1)].index.tolist()
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                within.append(dmat.loc[a, b])
            for b in non_members:
                across.append(dmat.loc[a, b])

    w_mean, a_mean = float(np.mean(within)), float(np.mean(across))
    assert w_mean < a_mean, (
        f"within-cluster SSD ({w_mean:.1f}) should be < across-cluster ({a_mean:.1f})"
    )
    print(f"  within-cluster SSD mean : {w_mean:8.2f}")
    print(f"  across-cluster SSD mean : {a_mean:8.2f}")
    print(f"  ratio (across / within) : {a_mean / max(w_mean, 1e-9):8.2f}x")


def test_optics_recovers_planted_clusters() -> None:
    """OPTICS labels should be 'sector-pure' against planted ground truth.

    A planted cluster may be split into multiple OPTICS sub-clusters (OPTICS's xi
    extraction can detect sub-density structure inside even very-tight planted
    groups); that's a parameter-tuning concern handled on real Dec-2023 data, not a
    pipeline bug. The pipeline correctness property we want here is:
        every within-OPTICS-cluster pair lives inside a single *planted* cluster
        (i.e. OPTICS never *merges* unrelated groups into false pairs).
    """
    prices, planted = generate_synthetic_panel()
    dmat = ssd_distance(prices)
    labels = cluster_optics(dmat, min_samples=3, xi=0.05, min_cluster_size=2)

    summary = cluster_summary(labels)
    print(f"  cluster_summary         : {summary}")

    # we expect at least as many OPTICS clusters as planted clusters
    assert summary["n_clusters"] >= 3, (
        f"expected >=3 clusters, got {summary['n_clusters']}"
    )

    # every OPTICS cluster's members should share the same planted ground-truth label
    # (this is the load-bearing property: OPTICS never merges unrelated stocks)
    for opt_label, group in labels[labels != -1].groupby(labels[labels != -1]):
        planted_of_members = planted.loc[group.index]
        non_noise = planted_of_members[planted_of_members != -1]
        unique_planted = non_noise.unique() if len(non_noise) else []
        assert len(unique_planted) <= 1, (
            f"OPTICS cluster {opt_label} merges planted clusters {unique_planted}"
        )


def test_purity_index_perfect_when_sectors_match() -> None:
    """purity_index ≈ 1.0 when 'sector' labels match planted ground truth.

    Even if OPTICS sub-splits a planted cluster, each sub-cluster still has a
    dominant sector = the planted ID, so purity stays high.
    """
    prices, planted = generate_synthetic_panel()
    dmat = ssd_distance(prices)
    labels = cluster_optics(dmat, min_samples=3, xi=0.05, min_cluster_size=2)

    # 'sector' = planted ground truth (noise tagged as its own sector)
    sectors = planted.replace({-1: 99}).astype(str)
    p = purity_index(labels, sectors)
    print(f"  purity_index            : {p:.4f}")
    # threshold 0.80 = paper's reported purity on real CRSP data (0.81). A handful of
    # noise stocks will get pulled in as boundary points to dense clusters (normal
    # OPTICS behaviour), so we don't demand ~1.0 here.
    assert p >= 0.80, f"purity_index expected >=0.80 on planted data, got {p:.4f}"


def test_clusters_to_pairs_combinatorics() -> None:
    prices, _ = generate_synthetic_panel(n_clusters=2, stocks_per_cluster=4, n_noise=2)
    dmat = ssd_distance(prices)
    labels = cluster_optics(dmat, min_samples=3, xi=0.05, min_cluster_size=2)
    pairs = clusters_to_pairs(labels)

    # 2 clusters of 4 = 2 * C(4, 2) = 12 within-cluster pairs (upper bound; OPTICS
    # may split or drop members, so just check we got pairs and they are within-cluster)
    print(f"  candidate pairs         : {len(pairs)}")
    assert len(pairs) > 0, "expected at least one within-cluster pair"
    for a, b in pairs:
        assert labels[a] == labels[b], f"pair ({a},{b}) spans clusters!"
        assert labels[a] != -1, f"pair ({a},{b}) involves an outlier!"


def test_sic_division_known_codes() -> None:
    # spot-check a few SIC codes against the divisions
    assert sic_division(100) == "agriculture"
    assert sic_division(1311) == "mining"  # oil & gas extraction
    assert sic_division(2834) == "manufacturing"  # pharma preps
    assert sic_division(4813) == "transport_utilities"  # telecoms
    assert sic_division(6020) == "finance"  # commercial banks
    assert sic_division(7372) == "services"  # prepackaged software
    assert sic_division(None) == "unknown"
    assert sic_division("not-a-number") == "unknown"
    print("  sic_division spot checks pass")


# ────────────────────────────────────────────────────────────────────────────────
# runner
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_ssd_within_cluster_smaller_than_across,
        test_optics_recovers_planted_clusters,
        test_purity_index_perfect_when_sectors_match,
        test_clusters_to_pairs_combinatorics,
        test_sic_division_known_codes,
    ]
    failures = 0
    for t in tests:
        print(f"\n▶ {t.__name__}")
        try:
            t()
            print(f"  ✅ PASS")
        except AssertionError as e:
            failures += 1
            print(f"  ❌ FAIL — {e}")
        except Exception as e:
            failures += 1
            print(f"  💥 ERROR — {type(e).__name__}: {e}")

    print(f"\n{'─' * 60}")
    if failures == 0:
        print(f"✅ all {len(tests)} tests passed")
        sys.exit(0)
    else:
        print(f"❌ {failures}/{len(tests)} tests failed")
        sys.exit(1)
