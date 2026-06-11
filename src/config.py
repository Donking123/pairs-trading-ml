"""Project-wide configuration — single source of truth for paths and constants.

All parameters match the benchmark design of Rotondi & Russo (2025),
"Machine Learning for Pairs Trading: a Clustering-based Approach".
"""
from __future__ import annotations

from pathlib import Path

# --- Sample period (paper: Jan 2000 - Dec 2023) ---
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"

# --- Paths (resolved relative to the repo root) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PHASES_DIR = PROJECT_ROOT / "phases"

# Per-phase paths. Each phase has its own notebooks/, results/, and README.
PHASE1_DIR = PHASES_DIR / "phase1"
PHASE2_DIR = PHASES_DIR / "phase2"
PHASE2_5_DIR = PHASES_DIR / "phase2_5"
PHASE3_DIR = PHASES_DIR / "phase3"
PHASE4_DIR = PHASES_DIR / "phase4"
PHASE5_DIR = PHASES_DIR / "phase5"
PHASE6_DIR = PHASES_DIR / "phase6"

# Backward-compat alias used by older notebooks: defaults to current phase 1 results
# while the SSD baseline parquets live there. After Phase 2 ships, prefer the
# explicit per-phase path (e.g., PHASE2_DIR / 'results' / 'pc_core_monthly.parquet').
RESULTS_DIR = PHASE1_DIR / "results"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up the filesystem from `start` (or cwd) until we find a directory that
    looks like the project root (contains src/config.py). Useful for notebooks that
    can be opened from arbitrary working directories — call this once at the top
    of each notebook to make `from src.* import ...` work without hand-tuning depth.
    """
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / "src" / "config.py").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not locate project root (expected src/config.py somewhere above cwd)")

# --- Universe filter ---
# CRSP share codes 10 & 11 = ordinary US common stock (excludes ADRs, ETFs, units).
COMMON_SHARE_CODES = (10, 11)

# --- Trading design (paper benchmark, Section 4.1) ---
FORMATION_YEARS = 3        # rolling formation window for pair selection
TRADING_MONTHS = 1         # rolling out-of-sample trading window
ZSCORE_WINDOW_MONTHS = 6   # look-back for spread mean/std in the z-score
ENTRY_THRESHOLD = 2.0      # |z| > 2 opens a position
EXIT_THRESHOLD = 0.0       # z = 0 (zero-cross) closes the position (core variant)
STOP_LOSS_SIGMA = None     # core = no stop; realism variant sets this to 3.5

# --- Phase 5: position carry-over -------------------------------------------
# By default the engine force-closes every open position at month-end (paper /
# Gatev-Goetzmann convention). 88.4% of trades exit this way at a mean loss, the
# single biggest drag: positions are cut before spreads can revert. With CARRY_OVER
# on, a position still open at month-end is HELD into the next month instead of being
# force-closed, PROVIDED the pair is still endorsed by next month's clustering. It is
# then closed on reversion / stop / delisting / dropping out of the candidate set, or
# when it hits MAX_CARRY_MONTHS.
#
# MAX_CARRY_MONTHS is tied to HALF_LIFE_BOUNDS (60 trading days ≈ 3 months): a pair
# that has not reverted within its own selection half-life has violated its premise,
# so we stop carrying it. Locked here, before seeing any Sharpe (anti-overfit).
#
# The carried position keeps the γ it was OPENED with (frozen) — sizing is equal-dollar
# so γ never touches P&L, only the z-score exit signal, and freezing keeps that signal
# consistent with the spread actually entered. CARRY_OVER=False is bit-identical to the
# pre-Phase-5 engine.
CARRY_OVER = False
MAX_CARRY_MONTHS = 3

# --- OPTICS clustering hyperparameters ---
# `xi` is metric-specific because different distance metrics live on different
# numerical scales, so a single steepness threshold has different effective
# behaviour across metrics. min_samples and min_cluster_size are shared.
#
# OPTICS_XI (SSD) — Phase 1a, tuned 2026-05-24.
#   Sweep: phases/phase1/notebooks/02_xi_tuning_sweep.py validated on Dec 2010 /
#   Dec 2015 / Dec 2023. Chosen xi=0.10 → 47 clusters on Dec 2023 (paper: 48 ±5);
#   purity 0.871.
#
# OPTICS_XI_PC (PC distance) — Phase 2, tuned 2026-05-24.
#   Sweep: phases/phase2/notebooks/02_xi_tuning_pc.py validated on the same three
#   dates. Chosen xi=0.04 → 81 clusters on Dec 2023; purity 0.937. Paper reports
#   109 PC clusters with target 0.84 purity — our undercount is approximately
#   consistent with our smaller universe (407 vs paper's likely ~500 stocks)
#   and minor PC formulation differences; locking the value firmly inside the
#   stable region rather than at the cliff edge (xi=0.02).
#
# OPTICS_XI_FACTOR (factor-beta distance) — Phase 2.5, tuned 2026-06-07.
#   Distance is Euclidean in standardized 18-factor beta-space (NOT [0,2] like
#   SSD/PC), so it needs its own xi. Validated on Dec 2015 / Dec 2023:
#   xi=0.10 → 61 / 78 clusters (vs PC's ~81 — chosen for a fair head-to-head),
#   purity 0.903 / 0.915 vs SIC division. Chosen before running any backtest.
#
# DO NOT re-tune any xi after seeing Sharpe — that would inject overfit bias.
OPTICS_MIN_SAMPLES = 2
OPTICS_XI = 0.10              # for SSD distance
OPTICS_XI_PC = 0.04           # for PC distance (Phase 2)
OPTICS_XI_FACTOR = 0.10       # for factor-beta distance (Phase 2.5)
OPTICS_MIN_CLUSTER_SIZE = 2

# --- Phase 3a: alternative clustering algorithms (robustness) ---
# HDBSCAN needs no threshold (auto cluster count via stability). Hierarchical
# (average-linkage on the precomputed distance matrix) is cut at the q-th QUANTILE
# of off-diagonal pairwise distances — NOT a fixed height. The absolute distance
# scale drifts over 2003-2023, so a fixed height would test threshold drift, not the
# algorithm; a fixed quantile auto-adapts and keeps the candidate-pair count
# temporally stable. q=0.01 validated on Dec 2003/2010/2015/2023:
#   PC     ~724-834 pairs across all four dates (OPTICS 202-315)
#   FACTOR ~529-641 pairs across all four dates (OPTICS 103-239)
# One quantile works for both metrics. Ward is unavailable for precomputed distances.
# Chosen BEFORE any backtest, same discipline as the xi values.
HIER_LINKAGE = "average"
HIER_QUANTILE = 0.01

# --- Factor-beta hyperparameters (Phase 2.5) ---
#   Ridge L2 penalty for the per-stock factor regression. alpha=1.0 stabilizes the
#   collinear 18-factor betas; chosen before seeing any Sharpe.
RIDGE_ALPHA = 1.0

# --- Cointegration filter hyperparameters (Phase 2) ---
# Paper conventions (Rotondi & Russo 2025, §3.3):
#   - Reject the unit-root null at the 5% level.
#   - Trade pairs whose spread half-life is in [5, 60] trading days.
#     Too fast = noise; too slow = won't revert in the 21-day trading window.
# DO NOT re-tune after seeing Sharpe.
COINTEGRATION_P_THRESHOLD = 0.05
HALF_LIFE_BOUNDS = (5.0, 60.0)  # in trading days
