"""
Synthetic-data sanity check for spread.fit_hedge_ratio + spread_series +
rolling_zscore  (Phase 1b).

We plant a known γ in synthetic prices and verify:
  * fit_hedge_ratio recovers γ ≈ planted γ
  * spread_series matches the formula
  * rolling_zscore is look-ahead-safe (z_t cannot see spread_t)
  * a synthetic mean-revert + diverge pattern triggers the expected z values

Run:  python -m tests.test_spread_synthetic   (from pairs-trading-ml/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spread import fit_hedge_ratio, rolling_zscore, spread_series


# ────────────────────────────────────────────────────────────────────────────────
# tests
# ────────────────────────────────────────────────────────────────────────────────


def test_fit_hedge_ratio_recovers_planted_gamma() -> None:
    """Plant γ=2.5, α=10, and verify OLS recovers them within tolerance."""
    rng = np.random.default_rng(42)
    n = 756
    b = pd.Series(100.0 + np.cumsum(rng.normal(0.05, 1.0, size=n)))
    noise = rng.normal(0.0, 0.5, size=n)
    a = 10.0 + 2.5 * b + noise

    fit = fit_hedge_ratio(a, b)
    print(f"  planted γ=2.50, α=10.00   →   fit γ={fit.gamma:.4f}, α={fit.alpha:.4f}")
    print(f"  residual_std={fit.residual_std:.4f}   n_obs={fit.n_obs}")
    assert abs(fit.gamma - 2.5) < 0.01, f"γ off: {fit.gamma}"
    assert abs(fit.alpha - 10.0) < 0.5, f"α off: {fit.alpha}"
    assert 0.45 < fit.residual_std < 0.55, f"residual_std off: {fit.residual_std}"
    assert fit.n_obs == n


def test_fit_hedge_ratio_worked_example() -> None:
    """Reproduce the worked example from notes (WMT/COST toy data)."""
    wmt = pd.Series([100.0, 102.5, 100.5, 103.5, 103.5])
    cost = pd.Series([500.0, 510.0, 505.0, 515.0, 520.0])
    fit = fit_hedge_ratio(wmt, cost)
    print(f"  WMT/COST toy → γ={fit.gamma:.4f} (expected 0.20), α={fit.alpha:.4f} (expected 0.00)")
    assert abs(fit.gamma - 0.20) < 1e-9
    assert abs(fit.alpha - 0.00) < 1e-9


def test_spread_series_matches_formula() -> None:
    """spread_t = A_t - γ·B_t — direct numeric check."""
    a = pd.Series([105.0, 106.0, 107.0, 106.0, 103.5])
    b = pd.Series([521.0, 520.0, 519.0, 518.0, 518.0])
    s = spread_series(a, b, hedge_ratio=0.20)
    expected = pd.Series([0.8, 2.0, 3.2, 2.4, -0.1])
    print(f"  spread (γ=0.20) → {s.tolist()}")
    assert np.allclose(s.to_numpy(), expected.to_numpy(), atol=1e-9)


def test_rolling_zscore_no_lookahead() -> None:
    """z_t must not depend on spread_t — verified by changing spread_t and
    confirming z_t (recomputed at same index) does change *only* via the
    numerator, never via μ/σ. This catches a missing .shift(1)."""
    rng = np.random.default_rng(0)
    spread = pd.Series(rng.normal(0, 1, size=200))
    z = rolling_zscore(spread, window=20)

    # mutate one observation deep in the series and recompute
    spread2 = spread.copy()
    spread2.iloc[50] = 99.0  # huge outlier
    z2 = rolling_zscore(spread2, window=20)

    # If μ/σ used spread_50 itself, z2[50] would be drastically different from
    # what (spread2[50] - μ_old) / σ_old gives. Check: z2[50] differs only via
    # the numerator change.
    mu_50 = spread.iloc[30:50].mean()  # days 30..49 (last 20 before day 50)
    sigma_50 = spread.iloc[30:50].std(ddof=1)
    expected_z2_50 = (99.0 - mu_50) / sigma_50
    print(f"  z[50] before mutation : {z.iloc[50]:.4f}")
    print(f"  z[50] after  mutation : {z2.iloc[50]:.4f}")
    print(f"  expected (lookback-only): {expected_z2_50:.4f}")
    assert abs(z2.iloc[50] - expected_z2_50) < 1e-9, (
        "rolling_zscore appears to use spread_t in its own mu/sigma (look-ahead leak)"
    )

    # Also: z2[51] onwards SHOULD reflect the new spread[50] in its rolling
    # window (it's lookback for them). Confirm μ_51 includes spread2[50].
    mu_51_with = spread2.iloc[31:51].mean()
    mu_51_naive = spread.iloc[31:51].mean()
    assert mu_51_with != mu_51_naive, "μ_51 should include the mutated spread[50]"


def test_rolling_zscore_initial_window_is_nan() -> None:
    """First `window` entries of z should be NaN (no past history yet)."""
    spread = pd.Series(np.arange(50.0))
    z = rolling_zscore(spread, window=10)
    assert z.iloc[:10].isna().all(), "first 10 z-values should be NaN"
    assert not z.iloc[10:].isna().any(), "z should be defined after the warmup"
    print(f"  first 10 z values = NaN ✓, days 10..49 defined ✓")


def test_entry_exit_signal_on_synthetic_diverge_revert() -> None:
    """End-to-end: plant a clean cointegrated pair, then a divergence, and
    verify z crosses +2σ (entry) and then returns to 0 (exit)."""
    rng = np.random.default_rng(7)
    n_form = 300        # formation
    n_trade = 60        # trading
    b = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=n_form + n_trade))
    a = np.zeros_like(b)
    # cointegrated in formation: a = 2*b + noise
    a[:n_form] = 2.0 * b[:n_form] + rng.normal(0, 0.3, size=n_form)
    # in trading window, drift the spread upward (mean reversion later)
    drift = np.concatenate([
        np.linspace(0, 5.0, n_trade // 2),       # divergence
        np.linspace(5.0, 0.0, n_trade - n_trade // 2),  # reversion
    ])
    a[n_form:] = 2.0 * b[n_form:] + drift + rng.normal(0, 0.3, size=n_trade)

    a_s = pd.Series(a)
    b_s = pd.Series(b)
    fit = fit_hedge_ratio(a_s.iloc[:n_form], b_s.iloc[:n_form])
    print(f"  fitted γ={fit.gamma:.4f} (expected ≈2.0)")
    assert abs(fit.gamma - 2.0) < 0.02

    spread = spread_series(a_s, b_s, fit)
    z = rolling_zscore(spread, window=126)
    # check that some trading-window day breaches +2σ then returns to ≤0
    trading_z = z.iloc[n_form:]
    print(f"  z over trading window: min={trading_z.min():.2f}, max={trading_z.max():.2f}")
    assert trading_z.max() >= 2.0, f"divergence should hit +2σ, got max {trading_z.max():.2f}"
    # entry occurs at first day z >= 2.0; exit when z crosses ≤ 0 after entry
    entry_day = (trading_z >= 2.0).idxmax()  # first True
    after_entry = trading_z.loc[entry_day:]
    assert (after_entry <= 0).any(), "no zero-cross exit observed after entry"
    print(f"  entry at trading day {entry_day - n_form}, "
          f"exit at relative day {(after_entry <= 0).idxmax() - n_form}")


# ────────────────────────────────────────────────────────────────────────────────
# runner
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_fit_hedge_ratio_recovers_planted_gamma,
        test_fit_hedge_ratio_worked_example,
        test_spread_series_matches_formula,
        test_rolling_zscore_no_lookahead,
        test_rolling_zscore_initial_window_is_nan,
        test_entry_exit_signal_on_synthetic_diverge_revert,
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
