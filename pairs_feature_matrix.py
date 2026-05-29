"""
pairs_feature_matrix.py — Clustering feature matrix for equity pairs discovery.

Implements EquityPairsTradingProposal.md Section 3 (Clustering for peer discovery):

Method A — returns-spectral:
  1. 126d return correlation matrix C
  2. Ledoit-Wolf shrinkage → Ĉ
  3. k from Marchenko-Pastur upper bound: λ+ = (1+√q)², q=N/T
  4. Graph Laplacian L = D - Ĉ; k smallest eigenvectors (L2-normalized rows)
  5. K-Means on eigenvector embedding (best of 10 seeds)

Method B — factor-beta:
  1. 17-20 factor ETF returns
  2. EWMA-weighted ridge betas per stock (λ_ewma=0.97, 126-252d window)
  3. Scale betas by factor vol (makes distance scale-invariant)
  4. Cosine affinity A_beta = exp(-d_cos²/σ²)
  5. Spectral clustering on A_beta (same k as Method A ± 20%)

Auxiliary feature matrix (Appendix A):
  mom_3m, mom_12m, vol_21d, beta_spy_63d, corr_to_sector_63d, ADV, Amihud

Fusion (optional):
  A_fused = w * A_ret + (1-w) * A_beta; spectral cluster on A_fused

ARI stability gate:
  When ARI(prev_labels, curr_labels) > 0.6 → freeze labels (no churn).

Usage:
    from spx_data import load_all
    from pairs_feature_matrix import build_feature_matrix, cluster_method_a, cluster_method_b

    close, volume, rets, fclose, frets = load_all()
    end_date = rets.index[-1]

    labels_a, k = cluster_method_a(rets, end_date=end_date)
    labels_b = cluster_method_b(rets, frets, end_date=end_date, k=k)
    labels_fused = cluster_fused(rets, frets, end_date=end_date)
    aux = build_aux_features(close, volume, rets, frets, end_date=end_date)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.covariance import LedoitWolf
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize, StandardScaler


# ── Constants ──────────────────────────────────────────────────────────

CORR_WINDOW = 126
BETA_WINDOW = 252
EWMA_LAM = 0.97
RIDGE_ALPHA = 10.0
KMEANS_SEEDS = 10
ARI_FREEZE_THRESHOLD = 0.6
K_MIN, K_MAX = 2, 25
AFFINITY_SIGMA_PCTILE = 50  # median pairwise cosine distance as σ


# ── Shared utilities ───────────────────────────────────────────────────

def _slice_window(
    df: pd.DataFrame,
    end_date: Optional[pd.Timestamp],
    window: int,
    min_obs: Optional[int] = None,
) -> pd.DataFrame:
    """Return trailing `window` rows ending at end_date; drop sparse columns."""
    if end_date is None:
        end_pos = len(df)
    else:
        end_pos = df.index.get_indexer([end_date], method="pad")[0] + 1
    start_pos = max(0, end_pos - window)
    w = df.iloc[start_pos:end_pos]
    if min_obs is None:
        min_obs = window // 2
    valid = w.columns[w.notna().sum() >= min_obs]
    return w[valid].fillna(0.0)


def _lw_correlation(R: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk correlation matrix from (T × N) returns array."""
    lw = LedoitWolf().fit(R)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-10, 1.0, std)
    C = cov / np.outer(std, std)
    np.fill_diagonal(C, 1.0)
    return C


def _corr_to_distance(C: np.ndarray) -> np.ndarray:
    """Map correlation to distance in [0,2]: D = 1 - C (diag=0)."""
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return D


def _kmeans_best(X: np.ndarray, k: int, seeds: int = KMEANS_SEEDS) -> np.ndarray:
    """K-Means with multiple random seeds; return best-inertia labels."""
    best_labels, best_inertia = None, np.inf
    for seed in range(seeds):
        km = KMeans(n_clusters=k, n_init=1, max_iter=500, random_state=seed)
        labels = km.fit_predict(X)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_labels = labels
    return best_labels


# ── Marchenko-Pastur k selection ───────────────────────────────────────

def mp_upper_bound(N: int, T: int) -> float:
    """MP noise ceiling for correlation matrix eigenvalues.

    λ+ = (1 + √(N/T))²
    Eigenvalues above this carry signal; below is noise.
    """
    q = N / T
    return (1.0 + np.sqrt(q)) ** 2


def select_k_mp(C: np.ndarray, T: int) -> int:
    """Count eigenvalues of C above the MP upper bound → number of clusters."""
    N = C.shape[0]
    lam_plus = mp_upper_bound(N, T)
    eigvals = np.linalg.eigvalsh(C)  # ascending
    k = int(np.sum(eigvals > lam_plus))
    return int(np.clip(k, K_MIN, K_MAX))


# ── Method A: Returns-spectral clustering ─────────────────────────────

def _spectral_embedding(C: np.ndarray, k: int) -> np.ndarray:
    """Compute k-dimensional spectral embedding from correlation matrix C.

    Steps:
      1. Shift C to non-negative affinity A = (C + 1) / 2
      2. Build unnormalized Laplacian L = D - A
      3. Find k smallest eigenvectors of L
      4. L2-normalize rows (standard spectral clustering step)

    Returns (N × k) embedding matrix.
    """
    A = (C + 1.0) / 2.0  # map [-1,1] → [0,1]; keeps relative structure
    d = A.sum(axis=1)
    L = np.diag(d) - A

    # k smallest eigenvectors (ascending order from eigh)
    eigvals, eigvecs = np.linalg.eigh(L)
    V = eigvecs[:, :k]  # (N × k)

    # L2-normalize rows so KMeans distances are meaningful
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return V / norms


def cluster_method_a(
    return_df: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    window: int = CORR_WINDOW,
    k_override: Optional[int] = None,
) -> Tuple[pd.Series, int]:
    """Method A: spectral clustering on LW-shrunk return correlation.

    Args:
        return_df:  (dates × tickers) simple daily returns
        end_date:   formation date; defaults to last available
        window:     trailing days for correlation estimation (126d default)
        k_override: fix k instead of deriving from MP bound

    Returns:
        labels:  pd.Series (ticker → cluster_int)
        k:       number of clusters used
    """
    w = _slice_window(return_df, end_date, window)
    tickers = w.columns.tolist()
    R = w.values  # T × N
    T, N = R.shape

    C = _lw_correlation(R)

    k = k_override if k_override is not None else select_k_mp(C, T)

    V = _spectral_embedding(C, k)
    labels = _kmeans_best(V, k)

    return pd.Series(labels, index=tickers, name="cluster_a"), k


# ── Method B: Factor-beta clustering ──────────────────────────────────

def _ewma_weights(T: int, lam: float = EWMA_LAM) -> np.ndarray:
    """EWMA weights: w_t = λ^(T-1-t), normalized to sum=1, then scaled to T."""
    t = np.arange(T)
    w = lam ** (T - 1 - t)
    w /= w.sum()
    return w * T  # scale so WLS residuals are on original scale


def _orthogonalize_factors(
    factor_rets: pd.DataFrame,
    n_components: int = 8,
) -> pd.DataFrame:
    """PCA on factor returns → orthogonal risk factors.

    WHY ORTHOGONALIZE FIRST (not after betas):
    Our factor set (FF5+MOM + 17 ETFs) is highly collinear: SPY ≈ FF_MktRF,
    XLK ≈ tech loading, TLT ≈ FF_HML, etc. Ridge regression on correlated
    inputs produces unstable betas even with regularization — the alpha
    merely shrinks the instability, it doesn't remove the collinearity.

    Correct order (Barra-style):
      correlated factors → PCA → orthogonal factors → ridge → betas

    This gives betas to independent risk factors (similar to Barra's
    statistical factor model). Betas are much more stable quarter-to-quarter
    because each orthogonal factor represents a distinct risk dimension with
    no overlap.

    Returns:
        DataFrame (dates × n_components) of orthogonal factor returns, columns PC1..PCn
    """
    from sklearn.decomposition import PCA

    X = factor_rets.fillna(0.0).values
    n = min(n_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n, random_state=42)
    ortho = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n)]
    return pd.DataFrame(ortho, index=factor_rets.index, columns=cols)


def _ridge_betas_ewma(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    window: int = BETA_WINDOW,
    end_date: Optional[pd.Timestamp] = None,
    ridge_alpha: float = RIDGE_ALPHA,
    ewma_lam: float = EWMA_LAM,
    ortho_factors: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate EWMA-weighted ridge betas for each stock vs factor ETFs.

    When ortho_factors=N: PCA on the factor set first to get N orthogonal
    factors, then regress stocks on those. This resolves collinearity and
    produces more stable betas (correct order: PCA factors → betas, not
    PCA betas → clustering).

    Returns:
        beta_df: (tickers × factors) beta matrix
    """
    sw = _slice_window(stock_rets, end_date, window)
    fw = factor_rets.reindex(sw.index).fillna(0.0)
    fw = fw.dropna(axis=1, how="all").fillna(0.0)

    # Orthogonalize factor returns BEFORE regression.
    # CRITICAL: fit PCA on the full available factor history up to end_date
    # (not just the estimation window), so the orthogonal basis is stable
    # across refits. Using window-only PCA makes PC1 incomparable across quarters.
    if ortho_factors is not None and ortho_factors > 0:
        # Full history up to end_date for PCA basis (T+1 safe: no future data)
        if end_date is not None:
            full_factors = factor_rets.loc[:end_date].fillna(0.0)
        else:
            full_factors = factor_rets.fillna(0.0)
        full_factors = full_factors.dropna(axis=1, how="all").fillna(0.0)
        from sklearn.decomposition import PCA
        n = min(ortho_factors, full_factors.shape[1], full_factors.shape[0] - 1)
        pca = PCA(n_components=n, random_state=42)
        pca.fit(full_factors.values)
        # Transform only the estimation window using the stable basis
        fw_vals = pca.transform(fw[full_factors.columns].fillna(0.0).values)
        fw = pd.DataFrame(fw_vals, index=fw.index,
                          columns=[f"PC{i+1}" for i in range(n)])

    T = len(sw)
    w_sqrt = np.sqrt(_ewma_weights(T, ewma_lam))  # (T,)

    X = fw.values  # T × K
    X_aug = np.c_[np.ones(T), X]  # add intercept column
    X_w = X_aug * w_sqrt[:, np.newaxis]  # WLS transform

    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    betas: dict[str, np.ndarray] = {}

    for tkr in sw.columns:
        y = sw[tkr].values
        y_w = y * w_sqrt
        ridge.fit(X_w, y_w)
        betas[tkr] = ridge.coef_[1:]  # drop intercept; (K,)

    beta_df = pd.DataFrame.from_dict(betas, orient="index", columns=fw.columns)
    return beta_df


def _cosine_affinity(beta_df: pd.DataFrame, sigma_pctile: int = AFFINITY_SIGMA_PCTILE) -> np.ndarray:
    """Build affinity matrix A_beta = exp(-d_cos² / σ²) from beta matrix.

    Cosine distance d_cos(i,j) = 1 - cos_similarity(i,j).
    σ² = variance at `sigma_pctile` percentile of pairwise distances.
    """
    B = normalize(beta_df.values, norm="l2")  # L2-normalize rows
    cos_sim = B @ B.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    d_cos = 1.0 - cos_sim  # (N × N) cosine distances in [0, 2]
    np.fill_diagonal(d_cos, 0.0)

    # Adaptive σ: median (or chosen percentile) of upper-triangle distances
    upper = d_cos[np.triu_indices_from(d_cos, k=1)]
    sigma2 = np.percentile(upper, sigma_pctile) ** 2
    if sigma2 < 1e-12:
        sigma2 = 1e-12

    A = np.exp(-d_cos**2 / sigma2)
    np.fill_diagonal(A, 1.0)
    return A


def _spectral_on_affinity(A: np.ndarray, k: int) -> np.ndarray:
    """Spectral clustering on a precomputed affinity matrix A (N × N).

    Build Laplacian from A, take k smallest eigenvectors, K-Means.
    Identical procedure to Method A but with A_beta as input.
    """
    d = A.sum(axis=1)
    L = np.diag(d) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    V = eigvecs[:, :k]
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    V = V / norms
    return _kmeans_best(V, k)


# ── Method C: Partial-correlation + density clustering (optional) ──────

def _residualize_vs_controls(
    stock_rets: pd.DataFrame,
    control_rets: pd.DataFrame,
    ridge_alpha: float = RIDGE_ALPHA,
    use_ols: bool = False,
) -> pd.DataFrame:
    """Regress each stock's returns on controls; return residual returns.

    use_ols=True: plain OLS via lstsq (remote PC-core approach).
    use_ols=False: Ridge(alpha) for regularized stability (default).
    """
    ctrl = control_rets.reindex(stock_rets.index).dropna(how="all", axis=1).fillna(0.0)
    if ctrl.empty:
        return stock_rets.fillna(0.0)
    X = np.c_[np.ones(len(ctrl)), ctrl.values]
    out = {}
    for tkr in stock_rets.columns:
        y = stock_rets[tkr].fillna(0.0).values
        try:
            if use_ols:
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                out[tkr] = y - X @ coef
            else:
                model = Ridge(alpha=ridge_alpha, fit_intercept=True)
                model.fit(X[:, 1:], y)
                out[tkr] = y - model.predict(X[:, 1:])
        except Exception:
            out[tkr] = y * 0.0
    return pd.DataFrame(out, index=stock_rets.index)


def cluster_method_c_partialcorr(
    stock_rets: pd.DataFrame,
    control_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    window: int = CORR_WINDOW,
    algo: str = "spectral",
    k_override: Optional[int] = None,
    optics_min_samples: int = 2,
    optics_xi: float = 0.05,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 2,
    use_ols: bool = False,
):
    """Cluster on residual-return correlations (partial corr proxy).

    algo:
      - 'spectral': spectral + KMeans(k) on LW corr of residuals (default)
      - 'optics'  : OPTICS(metric='precomputed') on distance D=1-C (outliers=-1)
      - 'dbscan'  : DBSCAN(metric='precomputed') on D (outliers=-1)

    Returns (labels: pd.Series, k: int). For optics/dbscan, k counts non-outlier clusters.
    """
    w = _slice_window(stock_rets, end_date, window)
    tickers = w.columns.tolist()
    res = _residualize_vs_controls(w, control_rets, use_ols=use_ols)
    C = _lw_correlation(res.values)
    if algo == "spectral":
        T, N = res.shape
        k = k_override if k_override is not None else select_k_mp(C, T)
        V = _spectral_embedding(C, k)
        labels = _kmeans_best(V, k)
        return pd.Series(labels, index=tickers, name="cluster_c"), k
    # Density-based with precomputed distance
    D = _corr_to_distance(C)
    if algo == "optics":
        optics = OPTICS(metric="precomputed", min_samples=optics_min_samples, xi=optics_xi)
        optics.fit(D)
        labels = optics.labels_.astype(int)
        k = int(len(set(labels)) - (1 if -1 in labels else 0))
        return pd.Series(labels, index=tickers, name="cluster_c_optics"), max(k, 0)
    if algo == "dbscan":
        db = DBSCAN(metric="precomputed", eps=dbscan_eps, min_samples=dbscan_min_samples)
        db.fit(D)
        labels = db.labels_.astype(int)
        k = int(len(set(labels)) - (1 if -1 in labels else 0))
        return pd.Series(labels, index=tickers, name="cluster_c_dbscan"), max(k, 0)
    # Fallback
    T, N = res.shape
    k = k_override if k_override is not None else select_k_mp(C, T)
    V = _spectral_embedding(C, k)
    labels = _kmeans_best(V, k)
    return pd.Series(labels, index=tickers, name="cluster_c"), k


# ── Diagnostics: purity vs sector proxy (optional helper) ──────────────

def purity_index(labels: pd.Series, sector_map: dict[str, str]) -> float:
    """Compute cluster purity relative to sector_map {ticker: sector}.

    Returns [0,1]; NaN if insufficient mapping. Non-invasive utility.
    """
    lbl = labels.dropna()
    if lbl.empty:
        return float("nan")
    buckets: dict[int, list[str]] = {}
    for t, c in lbl.items():
        s = sector_map.get(t)
        if s is not None:
            buckets.setdefault(int(c), []).append(s)
    N = sum(len(v) for v in buckets.values())
    if N == 0:
        return float("nan")
    top = 0
    for v in buckets.values():
        vc = pd.Series(v).value_counts()
        top += int(vc.iloc[0]) if not vc.empty else 0
    return float(top / N)


def cluster_method_b(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    window: int = BETA_WINDOW,
    k: int = 10,
    k_guard_frac: float = 0.20,
    ridge_alpha: float = RIDGE_ALPHA,
    ewma_lam: float = EWMA_LAM,
    pca_components: Optional[int] = None,
    ortho_factors: Optional[int] = None,
) -> pd.Series:
    """Method B: spectral clustering in EWMA-ridge beta-space.

    k is clipped within ±k_guard_frac of the provided k (from Method A)
    to keep the two views consistent.

    Returns:
        labels: pd.Series (ticker → cluster_int)
    """
    beta_df = _ridge_betas_ewma(
        stock_rets, factor_rets, window=window, end_date=end_date,
        ridge_alpha=ridge_alpha, ewma_lam=ewma_lam, ortho_factors=ortho_factors,
    )

    # Scale betas by factor vol (scale-invariant cosine distance).
    # When ortho_factors is set, PCs are already variance-normalized by PCA
    # so we skip vol-scaling (common_cols would be empty since columns are "PC1"... not ETF names).
    if ortho_factors is not None and ortho_factors > 0:
        beta_scaled = beta_df.fillna(0.0)  # PCs already orthonormal
    else:
        fw = factor_rets.reindex(stock_rets.index).fillna(0.0)
        factor_vol = fw.std().replace(0, np.nan).fillna(1.0)
        common_cols = beta_df.columns.intersection(factor_vol.index)
        beta_scaled = beta_df[common_cols] * factor_vol[common_cols].values

    # Optional PCA on betas: reduces redundant factor blocks (e.g., SPY vs MktRF)
    # and decorrelates the feature space before cosine affinity.
    if pca_components is not None and pca_components < beta_scaled.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components, random_state=42)
        beta_arr = pca.fit_transform(beta_scaled.fillna(0.0).values)
        beta_scaled = pd.DataFrame(beta_arr, index=beta_scaled.index)

    # Guardrail: k within ±guard of Method A k
    k_lo = max(K_MIN, int(k * (1 - k_guard_frac)))
    k_hi = min(K_MAX, int(k * (1 + k_guard_frac)) + 1)
    k_b = int(np.clip(k, k_lo, k_hi))

    A_beta = _cosine_affinity(beta_scaled)
    labels = _spectral_on_affinity(A_beta, k_b)

    return pd.Series(labels, index=beta_scaled.index, name="cluster_b")


def cluster_method_b_agglo(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    window: int = BETA_WINDOW,
    ridge_alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
    distance_threshold: float = 0.4,
) -> tuple:
    """Remote PC-core clustering: RidgeCV betas → correlation distance → agglomerative.

    Exact replication of Rotondi & Russo (2025) 03_clustering.py:
      1. StandardScaler on factor returns
      2. RidgeCV (cross-validated alpha) per stock, unscale betas back
      3. Pairwise distance = 1 - corr(beta_i, beta_j)
      4. AgglomerativeClustering(linkage='average', distance_threshold=0.4)

    Returns (labels: pd.Series, k: int).
    """
    sw = _slice_window(stock_rets, end_date, window)
    fw = factor_rets.reindex(sw.index).fillna(0.0).dropna(axis=1, how="all")
    if fw.empty or sw.empty:
        return pd.Series(dtype=int), 0

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(fw.values)
    ridge = RidgeCV(alphas=list(ridge_alphas), fit_intercept=True)

    betas: dict = {}
    for tkr in sw.columns:
        y = sw[tkr].fillna(0.0).values
        try:
            ridge.fit(X_sc, y)
            betas[tkr] = ridge.coef_ / scaler.scale_
        except Exception:
            betas[tkr] = np.zeros(fw.shape[1])

    beta_df = pd.DataFrame(betas, index=fw.columns).T  # (stocks × factors)
    B = beta_df.fillna(0.0).values

    if B.shape[0] < 3:
        return pd.Series(dtype=int), 0

    corr = np.corrcoef(B)
    D = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(D)
    k = len(set(labels))
    return pd.Series(labels, index=beta_df.index, name="cluster_b_agglo"), k


# ── Fusion ─────────────────────────────────────────────────────────────

def cluster_fused(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    corr_window: int = CORR_WINDOW,
    beta_window: int = BETA_WINDOW,
    w_ret: float = 0.5,
    k_override: Optional[int] = None,
) -> Tuple[pd.Series, int]:
    """Fuse Method A and B affinities; spectral cluster on blended graph.

    A_fused = w_ret * A_ret + (1 - w_ret) * A_beta

    Returns:
        labels: pd.Series (ticker → cluster_int)
        k:      clusters used
    """
    # --- Method A affinity ---
    w = _slice_window(stock_rets, end_date, corr_window)
    tickers_a = w.columns.tolist()
    R = w.values
    T, N = R.shape
    C = _lw_correlation(R)
    k = k_override if k_override is not None else select_k_mp(C, T)
    A_ret = (C + 1.0) / 2.0  # (N × N)

    # --- Method B affinity ---
    beta_df = _ridge_betas_ewma(stock_rets, factor_rets, window=beta_window, end_date=end_date)
    fw = factor_rets.reindex(stock_rets.index).fillna(0.0)
    factor_vol = fw.std().replace(0, np.nan).fillna(1.0)
    common_cols = beta_df.columns.intersection(factor_vol.index)
    beta_scaled = beta_df[common_cols] * factor_vol[common_cols].values

    # Align to common tickers (Method A may have dropped some)
    common_tickers = [t for t in tickers_a if t in beta_scaled.index]
    if not common_tickers:
        raise ValueError("No common tickers between Method A and B windows.")

    A_ret_s = A_ret[
        [tickers_a.index(t) for t in common_tickers], :
    ][:, [tickers_a.index(t) for t in common_tickers]]
    A_beta_s = _cosine_affinity(beta_scaled.loc[common_tickers])

    # Blend
    A_fused = w_ret * A_ret_s + (1.0 - w_ret) * A_beta_s

    # Clip k to guardrail
    k_b = int(np.clip(k, max(K_MIN, int(k * 0.8)), min(K_MAX, int(k * 1.2) + 1)))
    labels = _spectral_on_affinity(A_fused, k_b)

    return pd.Series(labels, index=common_tickers, name="cluster_fused"), k_b


# ── Cointegration affinity (A_coint) ──────────────────────────────────

def build_coint_affinity(
    close_df: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    corr_window: int = CORR_WINDOW,
    coint_window: int = 252,
    min_corr_prefilter: float = 0.70,
    adf_maxlag: int = 1,
) -> Tuple[np.ndarray, list]:
    """Build pairwise cointegration affinity matrix for spectral clustering.

    WHY THIS EXISTS:
    Method A clusters on returns correlation — finds stocks that co-move, but
    cointegration (stationarity of the spread) is tested post-hoc in discovery
    via a hard ADF p-value threshold. Consequence: cluster structure is blind to
    cointegration quality. ~22 false-positive pairs per cluster expected at p<0.05
    (C(30,2)=435 pairs × 0.05 = 21.8).

    A_coint bakes cointegration strength DIRECTLY into the cluster structure so
    pairs within the same cluster have ex-ante higher cointegration probability.
    Discovery still filters with ADF — this just ensures the upstream cluster
    structure supplies it with better raw material.

    Algorithm:
    1. Compute LW correlation on log-returns → prefilter to corr ≥ min_corr_prefilter.
       (Reduces 44,850 pairs for N=300 to ~2,000-4,000 — makes runtime tractable.)
    2. For each prefiltered pair (i,j): OLS beta → spread → ADF p-value (fixed maxlag=1
       for speed; autolag is ~5× slower and overkill for pairs pre-screened by HL<30d).
    3. A_coint[i,j] = exp(-3 * p_value): soft affinity in (0,1].
       Mapping: p=0 → 1.0 (perfect), p=0.05 → 0.86, p=0.10 → 0.74, p=1.0 → 0.05.
       Unprefiltered pairs (corr too low to bother): A_coint[i,j] = 0.0.

    Runtime: ~2,500 pairs × 0.5ms (maxlag=1, 252-obs) ≈ 1.3s per refit.
    Over 46 refits (11-year backtest): ~60s overhead. Full run ~2 min vs ~50s for Method A.

    Args:
        close_df:            (dates × tickers) adjusted close prices
        end_date:            formation end date; defaults to last available
        corr_window:         trailing days for LW correlation prefilter (126d default)
        coint_window:        trailing days used for ADF spread construction (252d default)
        min_corr_prefilter:  LW corr threshold below which ADF is skipped (0.70 default)
        adf_maxlag:          fixed lag for adfuller; 1 is fast and sufficient for HL<30d

    Returns:
        A_coint: (N × N) float64 affinity matrix
        tickers: list of N tickers corresponding to matrix rows/columns
    """
    from statsmodels.tsa.stattools import adfuller

    logp = np.log(close_df.replace(0, np.nan))

    # ── Slice windows ──────────────────────────────────────────────────────
    # corr_window: for the LW correlation prefilter (shorter, faster, more current)
    # coint_window: for the OLS spread + ADF (longer → more ADF power)
    w_corr = _slice_window(close_df.pct_change(), end_date, corr_window)
    tickers = w_corr.columns.tolist()
    N = len(tickers)

    if N < 2:
        return np.ones((N, N)), tickers

    w_logp = _slice_window(logp, end_date, coint_window)
    # Restrict to tickers present in both windows
    tickers = [t for t in tickers if t in w_logp.columns]
    N = len(tickers)
    if N < 2:
        return np.ones((N, N)), tickers

    w_corr = w_corr[tickers]
    w_logp = w_logp[tickers]

    # ── LW correlation for prefilter ───────────────────────────────────────
    R = w_corr.fillna(0.0).values
    C = _lw_correlation(R)   # (N × N) symmetric, diagonal = 1

    # ── Build A_coint ──────────────────────────────────────────────────────
    A = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(A, 1.0)

    logp_vals = w_logp.ffill().bfill().values  # (T × N)
    T = logp_vals.shape[0]
    ones_col = np.ones(T)

    for i in range(N):
        for j in range(i + 1, N):
            if C[i, j] < min_corr_prefilter:
                continue  # leave A[i,j]=0.0

            pi = logp_vals[:, i]
            pj = logp_vals[:, j]

            # Mask out NaN rows
            mask = np.isfinite(pi) & np.isfinite(pj)
            if mask.sum() < 30:
                continue

            pi_m, pj_m = pi[mask], pj[mask]

            # OLS: pi = beta * pj + alpha + spread
            X = np.c_[pj_m, ones_col[: mask.sum()]]
            coef = np.linalg.lstsq(X, pi_m, rcond=None)[0]
            beta, alpha = float(coef[0]), float(coef[1])
            if not np.isfinite(beta) or abs(beta) < 0.05 or abs(beta) > 20.0:
                continue

            spread = pi_m - (beta * pj_m + alpha)
            if spread.std() < 1e-8:
                continue

            try:
                pval = float(adfuller(spread, maxlag=adf_maxlag, regression="c")[1])
            except Exception:
                continue

            if not np.isfinite(pval):
                continue

            aff = float(np.exp(-3.0 * pval))  # p=0→1.0, p=0.05→0.86, p=1.0→0.05
            A[i, j] = A[j, i] = aff

    return A, tickers


def cluster_a_coint(
    stock_rets: pd.DataFrame,
    close_df: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    corr_window: int = CORR_WINDOW,
    coint_window: int = 252,
    w_coint: float = 0.5,
    min_corr_prefilter: float = 0.70,
    k_override: Optional[int] = None,
) -> Tuple[pd.Series, int]:
    """Spectral clustering on blended A_ret (Method A) + A_coint affinity.

    A_fused = (1 - w_coint) * A_ret  +  w_coint * A_coint

    Method A correlation affinity (A_ret) captures co-movement; A_coint captures
    cointegration strength. Blending ensures the cluster structure favors pairs
    that BOTH co-move AND cointegrate — better raw material for discovery.

    w_coint=0.5 is the default; w_coint=0 reduces to pure Method A.

    Args:
        stock_rets:  (dates × tickers) daily returns
        close_df:    (dates × tickers) adjusted close prices (for ADF spread)
        end_date:    formation date; defaults to last available
        corr_window: trailing days for LW correlation + Method A embedding (126d)
        coint_window: trailing days for ADF computation (252d; more ADF power)
        w_coint:     weight of cointegration affinity (default 0.5)
        min_corr_prefilter: min LW corr to run ADF test (default 0.70; ~2k-4k pairs for N=300)
        k_override:  fix k; else derived from Marchenko-Pastur on correlation matrix

    Returns:
        labels: pd.Series (ticker → cluster_int)
        k:      number of clusters used
    """
    # ── Method A path: A_ret + k from MP ──────────────────────────────────
    w = _slice_window(stock_rets, end_date, corr_window)
    tickers_a = w.columns.tolist()
    R = w.values
    T, N = R.shape
    C = _lw_correlation(R)
    k = k_override if k_override is not None else select_k_mp(C, T)
    A_ret = (C + 1.0) / 2.0   # shift from [-1,1] to [0,1]

    if w_coint <= 0.0:
        # Pure Method A (w_coint=0): skip the expensive A_coint computation
        V = _spectral_embedding(C, k)
        labels = _kmeans_best(V, k)
        return pd.Series(labels, index=tickers_a, name="cluster_a_coint"), k

    # ── Cointegration affinity ─────────────────────────────────────────────
    A_coint_full, tickers_coint = build_coint_affinity(
        close_df, end_date=end_date,
        corr_window=corr_window, coint_window=coint_window,
        min_corr_prefilter=min_corr_prefilter,
    )

    # ── Align tickers between the two affinity matrices ────────────────────
    common_tickers = [t for t in tickers_a if t in tickers_coint]
    if not common_tickers:
        # Fallback: pure Method A if no overlap
        V = _spectral_embedding(C, k)
        labels = _kmeans_best(V, k)
        return pd.Series(labels, index=tickers_a, name="cluster_a_coint"), k

    a_idx = [tickers_a.index(t) for t in common_tickers]
    c_idx = [tickers_coint.index(t) for t in common_tickers]

    A_ret_s   = A_ret[np.ix_(a_idx, a_idx)]
    A_coint_s = A_coint_full[np.ix_(c_idx, c_idx)]

    # ── Blend and cluster ──────────────────────────────────────────────────
    A_fused = (1.0 - w_coint) * A_ret_s + w_coint * A_coint_s

    k_b = int(np.clip(k, max(K_MIN, int(k * 0.8)), min(K_MAX, int(k * 1.2) + 1)))
    labels = _spectral_on_affinity(A_fused, k_b)

    return pd.Series(labels, index=common_tickers, name="cluster_a_coint"), k_b


# ── ARI stability gate ────────────────────────────────────────────────

def apply_ari_freeze(
    prev_labels: Optional[pd.Series],
    curr_labels: pd.Series,
    threshold: float = ARI_FREEZE_THRESHOLD,
) -> pd.Series:
    """Return curr_labels if ARI < threshold (changed enough); else prev_labels.

    On the first call (prev_labels=None), always accept curr_labels.
    Aligns on common tickers; tickers missing from prev get curr value.
    """
    if prev_labels is None:
        return curr_labels

    common = curr_labels.index.intersection(prev_labels.index)
    if len(common) < 2:
        return curr_labels

    ari = adjusted_rand_score(prev_labels.loc[common].values, curr_labels.loc[common].values)

    if ari >= threshold:
        # Cluster stable enough — freeze (propagate prev labels)
        merged = prev_labels.copy()
        new_tickers = curr_labels.index.difference(prev_labels.index)
        merged = pd.concat([merged, curr_labels.loc[new_tickers]])
        return merged
    else:
        return curr_labels


# ── Auxiliary feature matrix ──────────────────────────────────────────

def build_aux_features(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    end_date: Optional[pd.Timestamp] = None,
    sector_etf_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Build Appendix A auxiliary feature matrix for each stock.

    Features:
        mom_3m              63d cumulative return
        mom_12m             252d cumulative return
        vol_21d             21d annualized return volatility
        beta_spy_63d        OLS beta to SPY over 63d
        corr_to_sector_63d  63d correlation to assigned sector ETF
        adv_63d             63d average dollar volume (log-scaled)
        amihud_21d          21d Amihud illiquidity ratio (×10^6 for readability)

    Args:
        close_df:       (dates × tickers) adjusted close
        volume_df:      (dates × tickers) volume
        stock_rets:     (dates × tickers) simple daily returns
        factor_rets:    (dates × ETFs) factor ETF daily returns
        end_date:       formation date (default: last row)
        sector_etf_map: {ticker: sector_etf} e.g. {'AAPL': 'XLK'}; optional

    Returns:
        aux_df: (tickers × 7) feature DataFrame
    """
    if end_date is None:
        end_pos = len(stock_rets)
    else:
        end_pos = stock_rets.index.get_indexer([end_date], method="pad")[0] + 1

    def _tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[max(0, end_pos - n):end_pos]

    rets_63 = _tail(stock_rets, 63)
    rets_252 = _tail(stock_rets, 252)
    rets_21 = _tail(stock_rets, 21)
    close_63 = _tail(close_df, 63)
    vol_63 = _tail(volume_df, 63)

    tickers = stock_rets.columns.tolist()
    features: dict[str, pd.Series] = {}

    # mom_3m: 63d compound return
    features["mom_3m"] = (1 + rets_63).prod() - 1

    # mom_12m: 252d compound return (may be shorter if near start)
    features["mom_12m"] = (1 + rets_252).prod() - 1

    # vol_21d: annualized volatility
    features["vol_21d"] = rets_21.std() * np.sqrt(252)

    # beta_spy_63d: simple OLS beta to SPY
    spy_63 = factor_rets["SPY"].reindex(rets_63.index).fillna(0.0).values
    spy_var = np.var(spy_63)
    betas_spy = {}
    for tkr in tickers:
        y = rets_63[tkr].fillna(0.0).values
        if spy_var > 1e-12:
            betas_spy[tkr] = float(np.cov(y, spy_63)[0, 1] / spy_var)
        else:
            betas_spy[tkr] = np.nan
    features["beta_spy_63d"] = pd.Series(betas_spy)

    # corr_to_sector_63d
    if sector_etf_map is not None:
        corr_sector: dict[str, float] = {}
        for tkr in tickers:
            etf = sector_etf_map.get(tkr)
            if etf and etf in factor_rets.columns:
                etf_63 = factor_rets[etf].reindex(rets_63.index).fillna(0.0)
                tkr_63 = rets_63[tkr].fillna(0.0)
                c = np.corrcoef(tkr_63.values, etf_63.values)
                corr_sector[tkr] = float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0
            else:
                corr_sector[tkr] = np.nan
        features["corr_to_sector_63d"] = pd.Series(corr_sector)
    else:
        features["corr_to_sector_63d"] = pd.Series(np.nan, index=tickers)

    # adv_63d: average daily dollar volume (log)
    dollar_vol_63 = close_63 * vol_63
    adv = dollar_vol_63.mean()
    features["adv_63d"] = np.log1p(adv)

    # amihud_21d: mean(|return| / dollar_volume) × 1e6
    dv_21 = _tail(close_df, 21) * _tail(volume_df, 21)
    r_21 = rets_21.abs()
    dv_21 = dv_21.reindex(r_21.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amihud = (r_21 / dv_21.replace(0, np.nan)).mean() * 1e6
    features["amihud_21d"] = amihud

    aux_df = pd.DataFrame(features).loc[tickers]
    return aux_df


# ── Walk-forward clustering with ARI freeze ───────────────────────────

def hungarian_persistence(prev_labels: pd.Series, curr_labels: pd.Series) -> float:
    """Fraction of stocks staying in the same cluster after optimal label alignment.

    K-Means cluster numbers are arbitrary permutations each run. ARI handles
    this but gives a scalar similarity, not an interpretable retention rate.
    Hungarian alignment finds the best label mapping, then counts how many
    stocks retained their cluster membership under that mapping.

    Returns a float in [0, 1]: 1.0 = perfectly stable, 0.0 = complete churn.
    """
    from scipy.optimize import linear_sum_assignment

    common = prev_labels.index.intersection(curr_labels.index)
    if len(common) < 2:
        return np.nan

    pv = prev_labels.loc[common].values.astype(int)
    cv = curr_labels.loc[common].values.astype(int)
    pk, ck = np.unique(pv), np.unique(cv)
    pm = {v: i for i, v in enumerate(pk)}
    cm = {v: i for i, v in enumerate(ck)}

    # Contingency matrix: C[i,j] = stocks in prev-cluster i AND curr-cluster j
    C = np.zeros((len(pk), len(ck)), dtype=int)
    for p_, c_ in zip(pv, cv):
        C[pm[p_], cm[c_]] += 1

    row_ind, col_ind = linear_sum_assignment(-C)  # maximise overlap
    return float(C[row_ind, col_ind].sum()) / len(common)


def rolling_cluster_labels(
    stock_rets: pd.DataFrame,
    factor_rets: pd.DataFrame,
    method: str = "fused",
    corr_window: int = CORR_WINDOW,
    beta_window: int = BETA_WINDOW,
    refit_freq: int = 21,
    w_ret: float = 0.5,
    ari_threshold: float = ARI_FREEZE_THRESHOLD,
    ridge_alpha: float = RIDGE_ALPHA,
    pca_betas: Optional[int] = None,
    ortho_factors: Optional[int] = None,
    # ── a_coint method parameters ──────────────────────────────────────────
    close_df: Optional[pd.DataFrame] = None,
    coint_window: int = 252,
    w_coint: float = 0.5,
    min_corr_prefilter: float = 0.70,
    # ── Method C (partial-correlation) parameters ──────────────────────────
    control_rets: Optional[pd.DataFrame] = None,
    c_algo: str = "optics",
    c_optics_min_samples: int = 2,
    c_optics_xi: float = 0.05,
    c_dbscan_eps: float = 0.5,
    c_use_ols: bool = False,
    # ── Method b_agglo (remote exact replication) ───────────────────────────
    b_agglo_threshold: float = 0.4,
    b_agglo_ridge_alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> pd.DataFrame:
    """Compute cluster labels for every refit date; apply ARI freeze gate.

    Args:
        method:      'a' | 'b' | 'fused' | 'a_coint' | 'c' | 'c_optics' | 'c_dbscan'
        refit_freq:  trading days between re-estimations (default 21 ≈ monthly)
        close_df:    required when method='a_coint' (adjusted close for ADF spread)
        control_rets: required when method starts with 'c'; residualise stock returns
                      vs these before computing partial-correlation distance.
                      Pass frets[['SPY']] for market-only (Rotondi-Russo paper),
                      or full frets for multi-factor residualisation.
        c_algo:      clustering algo for method='c' — 'spectral'|'optics'|'dbscan'
                     (method='c_optics'/'c_dbscan' override this)

    Returns:
        label_df: (dates × tickers) cluster label ints; NaN before first valid window
    """
    is_c = method in ("c", "c_optics", "c_dbscan")
    is_b_agglo = method == "b_agglo"
    burn_in = max(corr_window, beta_window, coint_window if method == "a_coint" else 0)
    n_dates = len(stock_rets)
    tickers = stock_rets.columns.tolist()

    label_df = pd.DataFrame(
        np.nan, index=stock_rets.index, columns=tickers, dtype="float64"
    )
    prev_labels: Optional[pd.Series] = None
    current_labels: Optional[pd.Series] = None

    refit_indices = range(burn_in, n_dates, refit_freq)

    # Resolve algo for method C variants
    _c_algo = {"c": c_algo, "c_optics": "optics", "c_dbscan": "dbscan"}.get(method, c_algo)

    for idx in refit_indices:
        end_date = stock_rets.index[idx]
        try:
            if is_b_agglo:
                new_labels, k = cluster_method_b_agglo(
                    stock_rets, factor_rets, end_date=end_date, window=beta_window,
                    ridge_alphas=b_agglo_ridge_alphas,
                    distance_threshold=b_agglo_threshold,
                )
            elif method == "a":
                new_labels, k = cluster_method_a(stock_rets, end_date=end_date, window=corr_window)
            elif method == "b":
                _, k = cluster_method_a(stock_rets, end_date=end_date, window=corr_window)
                new_labels = cluster_method_b(stock_rets, factor_rets, end_date=end_date, k=k,
                                              window=beta_window, ridge_alpha=ridge_alpha,
                                              pca_components=pca_betas,
                                              ortho_factors=ortho_factors)
            elif method == "a_coint":
                if close_df is None:
                    raise ValueError("method='a_coint' requires close_df to be passed.")
                new_labels, k = cluster_a_coint(
                    stock_rets, close_df, end_date=end_date,
                    corr_window=corr_window, coint_window=coint_window,
                    w_coint=w_coint, min_corr_prefilter=min_corr_prefilter,
                )
            elif is_c:
                ctrl = control_rets if control_rets is not None else factor_rets
                new_labels, k = cluster_method_c_partialcorr(
                    stock_rets, ctrl, end_date=end_date, window=corr_window,
                    algo=_c_algo,
                    optics_min_samples=c_optics_min_samples,
                    optics_xi=c_optics_xi,
                    dbscan_eps=c_dbscan_eps,
                    use_ols=c_use_ols,
                )
            else:
                new_labels, k = cluster_fused(
                    stock_rets, factor_rets, end_date=end_date,
                    corr_window=corr_window, beta_window=beta_window, w_ret=w_ret,
                )
        except Exception as e:
            warnings.warn(f"Clustering failed at {end_date}: {e}; freezing previous labels.")
            new_labels = current_labels if current_labels is not None else None

        if new_labels is not None:
            current_labels = apply_ari_freeze(prev_labels, new_labels, ari_threshold)
            prev_labels = current_labels

        if current_labels is not None:
            next_idx = min(idx + refit_freq, n_dates)
            for tkr, lab in current_labels.items():
                if tkr in label_df.columns:
                    label_df.loc[label_df.index[idx:next_idx], tkr] = lab

    return label_df


if __name__ == "__main__":
    from spx_data import load_all

    print("Loading data...")
    close, volume, rets, fclose, frets = load_all()

    end_date = rets.index[-1]
    print(f"Formation date: {end_date.date()}, universe: {rets.shape[1]} stocks")

    print("\nMethod A (returns-spectral)...")
    labels_a, k = cluster_method_a(rets, end_date=end_date)
    print(f"  k={k} clusters | {labels_a.value_counts().sort_index().to_dict()}")

    print("\nMethod B (factor-beta)...")
    labels_b = cluster_method_b(rets, frets, end_date=end_date, k=k)
    print(f"  clusters | {labels_b.value_counts().sort_index().to_dict()}")

    ari = adjusted_rand_score(
        labels_a.reindex(labels_b.index).dropna().values,
        labels_b.dropna().values,
    )
    print(f"  ARI(A, B) = {ari:.3f}")

    print("\nFused clustering...")
    labels_f, k_f = cluster_fused(rets, frets, end_date=end_date)
    print(f"  k={k_f} clusters | {labels_f.value_counts().sort_index().to_dict()}")

    print("\nAuxiliary features...")
    aux = build_aux_features(close, volume, rets, frets, end_date=end_date)
    print(aux.describe().round(3))
