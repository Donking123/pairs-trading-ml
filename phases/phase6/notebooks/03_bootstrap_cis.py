"""
Phase 6 — bootstrap confidence intervals for every reported Sharpe.

Review improvement (reporting only, no engine change): the writeup compares point
Sharpes (1.028 vs 0.752, in-sample 1.028 vs OOS 0.858, ...) without error bars. This
script puts a CI on each cell and on the key PAIRED differences, so conclusions are
drawn at the right confidence level.

Method: circular block bootstrap on the monthly return series (block = 6 months to
preserve short-range autocorrelation/regime clustering), 10,000 resamples. For paired
differences the SAME month indices are drawn for both cells, so common-month variation
cancels and the CI reflects the difference only.

Run (from pairs-trading-ml/):
  python phases/phase6/notebooks/03_bootstrap_cis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# walk up to project root
_p = Path(__file__).resolve()
while _p != _p.parent:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.config import PHASES_DIR

N_BOOT = 10_000
BLOCK = 6
SEED = 0
MONTHS = 12

# label -> monthly parquet (relative to phases/)
CELLS = {
    "ssd_core (phase1)":        "phase1/results/ssd_core_monthly.parquet",
    "pc_core":                  "phase2/results/pc_core_monthly.parquet",
    "pc_filtered (raw ADF)":    "phase2/results/pc_filtered_monthly.parquet",
    "factor_core":              "phase2_5/results/factor_core_monthly.parquet",
    "pc_realism (phase4)":      "phase4/results/pc_realism_monthly.parquet",
    "OOS forward_pc 2024-25":   "phase4/results/forward_pc_monthly.parquet",
    "OOS forward_factor":       "phase4/results/forward_factor_monthly.parquet",
    "a1_core_delist (P6)":      "phase6/results/a1_core_delist_monthly.parquet",
    "b1_filt_mackinnon (P6)":   "phase6/results/b1_filt_mackinnon_monthly.parquet",
    "c0_real_baseline (P6)":    "phase6/results/c0_real_baseline_monthly.parquet",
    "c1_real_cooldown (P6)":    "phase6/results/c1_real_cooldown_monthly.parquet",
    "a2_core_delay (P6)":       "phase6/results/a2_core_delay_monthly.parquet",
}

# paired comparisons (same sample period → joint resampling)
PAIRS = [
    ("pc_core", "pc_filtered (raw ADF)", "does the (weak) filter hurt?"),
    ("pc_filtered (raw ADF)", "b1_filt_mackinnon (P6)", "does the CORRECT filter hurt more?"),
    ("pc_core", "a1_core_delist (P6)", "delisting fix impact"),
    ("c0_real_baseline (P6)", "c1_real_cooldown (P6)", "stop cooldown impact"),
    ("pc_core", "factor_core", "PC vs factor-beta metric"),
    ("pc_core", "a2_core_delay (P6)", "one-day execution delay impact"),
]


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(MONTHS)) if sd > 1e-12 else float("nan")


def block_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) circular block-bootstrap index matrix."""
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n] % n


def boot_sharpes(r: np.ndarray, idx: np.ndarray) -> np.ndarray:
    samples = r[idx]                                    # (n_boot, n)
    mu = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    return np.where(sd > 1e-12, mu / sd * np.sqrt(MONTHS), np.nan)


def main() -> None:
    rng = np.random.default_rng(SEED)
    series: dict[str, pd.Series] = {}
    for label, rel in CELLS.items():
        f = PHASES_DIR / rel
        if f.exists():
            series[label] = pd.read_parquet(f)["monthly_return"]

    print(f"\nBlock bootstrap (block={BLOCK}m, {N_BOOT:,} resamples) — Sharpe 95% CIs\n")
    rows = []
    for label, s in series.items():
        r = s.to_numpy()
        idx = block_indices(len(r), N_BOOT, rng)
        bs = boot_sharpes(r, idx)
        rows.append({
            "cell": label, "n_months": len(r), "sharpe": round(sharpe(r), 3),
            "ci_lo": round(float(np.nanpercentile(bs, 2.5)), 3),
            "ci_hi": round(float(np.nanpercentile(bs, 97.5)), 3),
            "P(SR<=0)": f"{float(np.nanmean(bs <= 0)):.1%}",
        })
    print(pd.DataFrame(rows).set_index("cell").to_string())

    print("\nPaired differences (joint month resampling; Δ = first − second)\n")
    rows = []
    for a, b, why in PAIRS:
        if a not in series or b not in series:
            continue
        sa, sb = series[a].align(series[b], join="inner")
        ra, rb = sa.to_numpy(), sb.to_numpy()
        idx = block_indices(len(ra), N_BOOT, rng)
        d = boot_sharpes(ra, idx) - boot_sharpes(rb, idx)
        rows.append({
            "comparison": f"{a}  −  {b}", "why": why,
            "Δsharpe": round(sharpe(ra) - sharpe(rb), 3),
            "ci_lo": round(float(np.nanpercentile(d, 2.5)), 3),
            "ci_hi": round(float(np.nanpercentile(d, 97.5)), 3),
            "P(Δ<=0)": f"{float(np.nanmean(d <= 0)):.1%}",
        })
    print(pd.DataFrame(rows).set_index("comparison").to_string())

    if "pc_core" in series and "OOS forward_pc 2024-25" in series:
        # independent periods → compare via independent bootstrap distributions
        r_is = series["pc_core"].to_numpy()
        r_oos = series["OOS forward_pc 2024-25"].to_numpy()
        d = (boot_sharpes(r_is, block_indices(len(r_is), N_BOOT, rng))
             - boot_sharpes(r_oos, block_indices(len(r_oos), N_BOOT, rng)))
        print(
            f"\nIn-sample pc_core vs OOS 2024-25 (independent periods): "
            f"Δ={sharpe(r_is) - sharpe(r_oos):+.3f}, "
            f"95% CI [{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}], "
            f"P(OOS>=IS)={float(np.nanmean(d <= 0)):.1%}"
        )


if __name__ == "__main__":
    main()
