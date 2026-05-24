"""
Synthetic-data sanity check for distances.pc_distance + market_adjusted_returns
(Phase 2 first build).

The key property we test: PC sees through the market-beta confound that SSD doesn't.
We plant a 3-stock universe where:
  Stock A: beta=1.2 + idiosyncratic shock pattern X
  Stock B: beta=0.8 + idiosyncratic shock pattern X   (same idio as A -> TRUE pair)
  Stock C: beta=1.0 + idiosyncratic shock pattern Y   (independent idio)

Expectations:
  PC distance(A, B) ~ 0  (residuals share idio)
  PC distance(A, C) ~ 1  (residuals uncorrelated)
  SSD on the same prices would say A-C are similar (both follow market) -- shown
  here for contrast.

Run:  python -m tests.test_distances_pc_synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distances import market_adjusted_returns, pc_distance, ssd_distance


def _build_universe(
    n_days: int = 756,
    beta_A: float = 1.2,
    beta_B: float = 0.8,
    beta_C: float = 1.0,
    idio_sigma: float = 0.005,
    mkt_sigma: float = 0.012,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the 3-stock + market test universe used across all tests."""
    rng = np.random.default_rng(seed)
    mkt_ret = rng.normal(0.0003, mkt_sigma, size=n_days)
    idio_AB = rng.normal(0.0, idio_sigma, size=n_days)
    idio_C = rng.normal(0.0, idio_sigma, size=n_days)

    returns = pd.DataFrame({
        "A": beta_A * mkt_ret + idio_AB,
        "B": beta_B * mkt_ret + idio_AB,   # shared idio with A
        "C": beta_C * mkt_ret + idio_C,    # independent
    })
    dates = pd.date_range("2021-01-04", periods=n_days, freq="B")
    returns.index = dates
    market_returns = pd.Series(mkt_ret, index=dates, name="mkt")

    # Convert returns to prices (start at 100); shift by 1 so prices line up with
    # returns naturally - first price = 100, then 100*(1+r_1), 100*(1+r_1)*(1+r_2), ...
    prices = 100.0 * (1.0 + returns).cumprod()
    # Prepend a price-100 row so pct_change later recovers the original returns.
    initial = pd.Series(100.0, index=returns.columns, name=dates[0] - pd.Timedelta(days=1))
    prices = pd.concat([initial.to_frame().T, prices])
    return prices, market_returns


# ────────────────────────────────────────────────────────────────────────────────
# tests
# ────────────────────────────────────────────────────────────────────────────────


def test_shared_idio_pair_has_near_zero_pc_distance() -> None:
    """A and B share the same idio shocks -> PC distance(A,B) should be ~0."""
    prices, mkt = _build_universe()
    dist = pc_distance(prices, mkt)
    d_AB = float(dist.loc["A", "B"])
    print(f"  PC distance(A, B) = {d_AB:.4f}   (planted: ~0)")
    assert d_AB < 0.05, f"expected near-zero distance for shared-idio pair, got {d_AB:.4f}"


def test_market_only_pair_has_pc_distance_near_one() -> None:
    """A and C have independent idio shocks -> residuals uncorrelated -> PC distance ~1."""
    prices, mkt = _build_universe()
    dist = pc_distance(prices, mkt)
    d_AC = float(dist.loc["A", "C"])
    d_BC = float(dist.loc["B", "C"])
    print(f"  PC distance(A, C) = {d_AC:.4f}   (planted: ~1)")
    print(f"  PC distance(B, C) = {d_BC:.4f}   (planted: ~1)")
    assert 0.90 <= d_AC <= 1.10, f"expected ~1.0 for market-only pair, got {d_AC:.4f}"
    assert 0.90 <= d_BC <= 1.10, f"expected ~1.0 for market-only pair, got {d_BC:.4f}"


def test_pc_vs_ssd_contrast() -> None:
    """SSD groups A and C (both follow market); PC separates them.

    This is the headline difference between the metrics - PC's value proposition.
    """
    prices, mkt = _build_universe()
    pc = pc_distance(prices, mkt)
    ssd = ssd_distance(prices)

    pc_AB, pc_AC = float(pc.loc["A", "B"]), float(pc.loc["A", "C"])
    ssd_AB, ssd_AC = float(ssd.loc["A", "B"]), float(ssd.loc["A", "C"])
    print(f"  SSD: A-B = {ssd_AB:8.3f}    A-C = {ssd_AC:8.3f}")
    print(f"  PC : A-B = {pc_AB:8.4f}    A-C = {pc_AC:8.4f}")

    # PC must separate A-C from A-B clearly (ratio >> 10 expected)
    pc_ratio = pc_AC / max(pc_AB, 1e-6)
    print(f"  PC ratio A-C / A-B = {pc_ratio:.1f}x   (large = PC discriminates well)")
    assert pc_ratio > 5, (
        f"PC should clearly separate the shared-idio pair from the market-only pair; "
        f"ratio = {pc_ratio:.2f}"
    )


def test_diagonal_exactly_zero() -> None:
    """Distance matrix diagonal must be exactly 0 (a stock vs itself)."""
    prices, mkt = _build_universe()
    dist = pc_distance(prices, mkt)
    diag = np.diag(dist.values)
    print(f"  diagonal: {diag}")
    assert (diag == 0).all(), f"diagonal contains non-zero values: {diag}"


def test_symmetry() -> None:
    """Distance matrix must be symmetric: d(X, Y) = d(Y, X)."""
    prices, mkt = _build_universe()
    dist = pc_distance(prices, mkt)
    asymmetry = (dist - dist.T).abs().to_numpy().max()
    print(f"  max |d(X,Y) - d(Y,X)| = {asymmetry:.2e}")
    assert asymmetry < 1e-10, f"distance matrix is not symmetric: {asymmetry:.4e}"


def test_market_adjusted_returns_recovers_planted_betas() -> None:
    """Cross-check the helper: market-adjusted returns should have low correlation
    with the market return (because the market component has been stripped out)."""
    prices, mkt = _build_universe()
    resid = market_adjusted_returns(prices, mkt)
    # Re-align market to residual's index
    mkt_aligned = mkt.reindex(resid.index)
    for stock in ["A", "B", "C"]:
        corr_with_mkt = float(resid[stock].corr(mkt_aligned))
        print(f"  corr(residual_{stock}, mkt) = {corr_with_mkt:+.4f}  (expected ~0)")
        assert abs(corr_with_mkt) < 0.05, (
            f"residual for {stock} still correlated with market: {corr_with_mkt:.4f}"
        )


def test_anti_correlated_residuals_give_distance_near_two() -> None:
    """Plant a pair whose residuals are NEGATIVELY correlated -> distance ~2."""
    rng = np.random.default_rng(7)
    n = 500
    mkt = rng.normal(0.0003, 0.012, size=n)
    idio = rng.normal(0.0, 0.005, size=n)
    returns = pd.DataFrame({
        "X": 1.0 * mkt + idio,
        "Y": 1.0 * mkt - idio,   # mirror-image residual
    })
    dates = pd.date_range("2021-01-04", periods=n, freq="B")
    returns.index = dates
    initial = pd.Series(100.0, index=returns.columns)
    prices = 100.0 * (1.0 + returns).cumprod()
    prices = pd.concat([initial.to_frame().T.set_index(pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1)])), prices])
    mkt_s = pd.Series(mkt, index=dates)

    dist = pc_distance(prices, mkt_s)
    d_XY = float(dist.loc["X", "Y"])
    print(f"  PC distance(X, Y) with anti-correlated residuals = {d_XY:.4f}  (planted: ~2.0)")
    assert 1.90 <= d_XY <= 2.10, f"expected ~2.0 for anti-correlated residuals, got {d_XY:.4f}"


# ────────────────────────────────────────────────────────────────────────────────
# runner
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_shared_idio_pair_has_near_zero_pc_distance,
        test_market_only_pair_has_pc_distance_near_one,
        test_pc_vs_ssd_contrast,
        test_diagonal_exactly_zero,
        test_symmetry,
        test_market_adjusted_returns_recovers_planted_betas,
        test_anti_correlated_residuals_give_distance_near_two,
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
