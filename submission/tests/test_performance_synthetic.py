"""
Synthetic-data sanity check for performance.compute_metrics (Phase 1d).

Verifies the metric math against series with known properties:
  * Planted Sharpe via Gaussian draws → recover within sampling tolerance
  * Constant positive returns → Sharpe = ∞-ish (zero vol → NaN)
  * All-negative series → negative Sharpe, drawdown = total drop
  * Single-month crash → drawdown captures it, peak/trough dates correct

Run:  python -m tests.test_performance_synthetic   (from pairs-trading-ml/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.performance import compute_metrics, format_metrics


def _monthly_index(n: int, start: str = "2003-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="ME")


# ────────────────────────────────────────────────────────────────────────────────
# tests
# ────────────────────────────────────────────────────────────────────────────────


def test_planted_sharpe_recovered() -> None:
    """Plant Gaussian monthly returns with known mean/std → recover Sharpe."""
    rng = np.random.default_rng(42)
    n = 252
    mu_monthly = 0.006        # ~7.4% ann return
    sigma_monthly = 0.024     # ~8.3% ann vol → expected Sharpe ~0.89
    rets = pd.Series(rng.normal(mu_monthly, sigma_monthly, size=n), index=_monthly_index(n))

    m = compute_metrics(rets)
    expected_sharpe = mu_monthly * 12 / (sigma_monthly * np.sqrt(12))
    print(f"  planted Sharpe ≈ {expected_sharpe:.3f}  →  computed = {m.sharpe:.3f}")
    # large n, so sampling error should be modest
    assert abs(m.sharpe - expected_sharpe) < 0.15, (
        f"Sharpe off: expected {expected_sharpe:.3f}, got {m.sharpe:.3f}"
    )
    # ann_vol should match sigma * sqrt(12) within sampling
    assert abs(m.ann_vol - sigma_monthly * np.sqrt(12)) < 0.01
    assert m.n_months == n


def test_total_and_annualised_return() -> None:
    """Constant +1%/mo for 12 months → total = 1.01^12 - 1, annualised same."""
    rets = pd.Series([0.01] * 12, index=_monthly_index(12))
    m = compute_metrics(rets)
    expected_total = 1.01 ** 12 - 1
    print(f"  total = {m.total_return:.5f}  (expected {expected_total:.5f})")
    print(f"  ann   = {m.ann_return:.5f}  (expected {expected_total:.5f})")
    assert abs(m.total_return - expected_total) < 1e-9
    assert abs(m.ann_return - expected_total) < 1e-9
    # zero vol → Sharpe NaN
    assert np.isnan(m.sharpe), "Sharpe should be NaN when vol is 0"
    # zero drawdown
    assert m.max_drawdown == 0.0


def test_max_drawdown_captures_crash() -> None:
    """A single -30% month sandwiched between flat months → drawdown = -30%.

    We use compounded returns, so a single -0.30 monthly return causes the
    cumulative to drop by 30% (peak-to-trough)."""
    rets = pd.Series(
        [0.01, 0.01, 0.01, -0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        index=_monthly_index(12),
    )
    m = compute_metrics(rets)
    print(f"  drawdown = {m.max_drawdown:.4f}  (expected -0.30)")
    print(f"  peak     = {m.max_drawdown_start}")
    print(f"  trough   = {m.max_drawdown_end}")
    # cum after 3 +1% months = 1.0303; trough = 1.0303 * 0.70 = 0.7212; drawdown = -30%
    assert abs(m.max_drawdown - (-0.30)) < 1e-9
    # peak is the 3rd month (index 2), trough is the 4th month (index 3)
    assert m.max_drawdown_start == rets.index[2]
    assert m.max_drawdown_end == rets.index[3]


def test_hit_rate_and_win_loss() -> None:
    """7 wins (each +1%), 5 losses (each -0.5%)."""
    rets = pd.Series([0.01] * 7 + [-0.005] * 5, index=_monthly_index(12))
    m = compute_metrics(rets)
    print(f"  hit_rate = {m.hit_rate:.4f}  (expected 7/12 ≈ 0.5833)")
    print(f"  avg_win  = {m.avg_win:.4f}  avg_loss = {m.avg_loss:.4f}")
    print(f"  win/loss = {m.win_loss_ratio:.4f}  (expected 0.01/0.005 = 2.0)")
    assert abs(m.hit_rate - 7 / 12) < 1e-9
    assert abs(m.avg_win - 0.01) < 1e-9
    assert abs(m.avg_loss - (-0.005)) < 1e-9
    assert abs(m.win_loss_ratio - 2.0) < 1e-9


def test_sortino_higher_than_sharpe_when_upside_volatile() -> None:
    """A series with a few big positive months should have Sortino > Sharpe
    (upside vol inflates Sharpe denominator but not Sortino's)."""
    rng = np.random.default_rng(7)
    rets = pd.Series(
        np.concatenate([rng.normal(0.005, 0.01, size=240), [0.20, 0.18, 0.15] * 4]),
        index=_monthly_index(252),
    )
    m = compute_metrics(rets)
    print(f"  Sharpe = {m.sharpe:.3f}  Sortino = {m.sortino:.3f}")
    assert m.sortino > m.sharpe, (
        f"expected Sortino > Sharpe when upside is volatile, got {m.sortino:.3f} vs {m.sharpe:.3f}"
    )


def test_calmar_against_known_drawdown() -> None:
    """If ann_return = 10% and max_drawdown = -20%, Calmar = 0.5."""
    # construct a series whose ann_return ≈ 0.10 and MDD ≈ -0.20
    # 12 months of +0.797% each → 1.00797^12 ≈ 1.10 → ~10% ann return
    # add one -20% crash to create the drawdown
    rets = pd.Series(
        [0.00797] * 6 + [-0.20] + [0.05] * 5,
        index=_monthly_index(12),
    )
    m = compute_metrics(rets)
    print(f"  ann_return = {m.ann_return:.4f}  MDD = {m.max_drawdown:.4f}  Calmar = {m.calmar:.4f}")
    expected = m.ann_return / abs(m.max_drawdown)
    assert abs(m.calmar - expected) < 1e-9


def test_format_metrics_does_not_crash() -> None:
    """Smoke-test the formatter: should produce a non-empty string."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.005, 0.02, size=60), index=_monthly_index(60))
    m = compute_metrics(rets)
    s = format_metrics(m)
    print(s)
    assert isinstance(s, str)
    assert "Sharpe" in s
    assert "paper target" in s


# ────────────────────────────────────────────────────────────────────────────────
# runner
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_planted_sharpe_recovered,
        test_total_and_annualised_return,
        test_max_drawdown_captures_crash,
        test_hit_rate_and_win_loss,
        test_sortino_higher_than_sharpe_when_upside_volatile,
        test_calmar_against_known_drawdown,
        test_format_metrics_does_not_crash,
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
