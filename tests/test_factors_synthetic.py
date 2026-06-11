"""
Synthetic-data sanity check for factors.* + distances.factor_beta_distance
(Phase 2.5 — factor-beta clustering extension).

Two groups of checks:

1. FF12 classification (factors.sic_to_ff12 / FF12_DEFINITIONS)
   * the canonical SIC ranges are disjoint,
   * known SIC codes map to the expected industry.

2. Factor-beta distance (distances.ridge_betas / factor_beta_distance)
   We plant a 2-factor world where each stock's return is a known linear combo of
   F1, F2 plus small idio noise:
     A: (β_F1, β_F2) = (1.5, 0.0)
     B: (1.5, 0.0)   -> same exposure as A  -> TRUE pair, distance ~ 0
     C: (0.0, 1.5)   -> orthogonal exposure -> far from A
   Expectations:
     * ridge_betas recovers the planted betas (small ridge),
     * factor_beta_distance(A,B) << factor_beta_distance(A,C),
     * diagonal exactly 0, matrix symmetric.

Run:  python -m tests.test_factors_synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distances import factor_beta_distance, ridge_betas
from src.factors import (
    FF12_DEFINITIONS,
    build_factor_panel,
    sic_to_ff12,
)


# ────────────────────────────────────────────────────────────────────────────────
# 1. FF12 classification
# ────────────────────────────────────────────────────────────────────────────────


def test_ff12_ranges_are_disjoint() -> None:
    """No SIC code may fall into two FF12 industries (first-match must be unique)."""
    seen: dict[int, str] = {}
    for name, ranges in FF12_DEFINITIONS:
        for lo, hi in ranges:
            for code in range(lo, hi + 1):
                assert code not in seen, (
                    f"SIC {code} appears in both {seen[code]} and {name}"
                )
                seen[code] = name
    print(f"  {len(seen)} SIC codes covered, all disjoint")


def test_ff12_known_codes_map_correctly() -> None:
    """Spot-check representative SIC codes against their FF12 industry."""
    cases = {
        7372: "BusEq",   # prepackaged software
        3674: "BusEq",   # semiconductors
        2834: "Hlth",    # pharmaceutical preparations
        6020: "Money",   # commercial banks
        2911: "Enrgy",   # petroleum refining
        5411: "Shops",   # grocery stores
        4911: "Utils",   # electric services
        4813: "Telcm",   # telephone communications
        3711: "Durbl",   # motor vehicles
        2821: "Chems",   # plastics materials
        9995: "Other",   # non-classifiable
    }
    for sic, expected in cases.items():
        got = sic_to_ff12(sic)
        print(f"  SIC {sic} -> {got:6s} (expect {expected})")
        assert got == expected, f"SIC {sic}: expected {expected}, got {got}"


def test_ff12_handles_bad_input() -> None:
    """Unparseable / missing SIC codes fall into the 'Other' residual bucket."""
    for bad in [None, np.nan, "abc", ""]:
        assert sic_to_ff12(bad) == "Other", f"{bad!r} should map to 'Other'"
    print("  None / NaN / non-numeric -> 'Other' ✓")


# ────────────────────────────────────────────────────────────────────────────────
# 2. Factor-beta distance
# ────────────────────────────────────────────────────────────────────────────────


def _build_two_factor_world(
    n_days: int = 756,
    idio_sigma: float = 0.002,
    seed: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    """Plant A, B (same exposure) and C (orthogonal). Return (prices, factors, betas)."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0.0, 0.010, size=n_days)
    f2 = rng.normal(0.0, 0.010, size=n_days)
    planted = {"A": (1.5, 0.0), "B": (1.5, 0.0), "C": (0.0, 1.5)}

    returns = pd.DataFrame({
        s: b1 * f1 + b2 * f2 + rng.normal(0.0, idio_sigma, size=n_days)
        for s, (b1, b2) in planted.items()
    })
    dates = pd.date_range("2021-01-04", periods=n_days, freq="B")
    returns.index = dates

    prices = 100.0 * (1.0 + returns).cumprod()
    initial = pd.Series(100.0, index=returns.columns,
                        name=dates[0] - pd.Timedelta(days=1))
    prices = pd.concat([initial.to_frame().T, prices])

    factors = pd.DataFrame({"F1": f1, "F2": f2}, index=dates)
    return prices, factors, planted


def test_ridge_betas_recovers_planted_exposures() -> None:
    """With small ridge, recovered betas should match the planted (β_F1, β_F2)."""
    prices, factors, planted = _build_two_factor_world()
    betas = ridge_betas(prices, factors, ridge_alpha=1e-4)
    for s, (b1, b2) in planted.items():
        got1, got2 = float(betas.loc[s, "F1"]), float(betas.loc[s, "F2"])
        print(f"  {s}: recovered ({got1:+.3f}, {got2:+.3f})  planted ({b1}, {b2})")
        assert abs(got1 - b1) < 0.1 and abs(got2 - b2) < 0.1, (
            f"{s}: recovered ({got1:.3f},{got2:.3f}) vs planted ({b1},{b2})"
        )


def test_same_exposure_pair_is_close_orthogonal_is_far() -> None:
    """factor_beta_distance(A,B) << factor_beta_distance(A,C)."""
    prices, factors, _ = _build_two_factor_world()
    dist = factor_beta_distance(prices, factors, ridge_alpha=1e-4)
    d_AB, d_AC = float(dist.loc["A", "B"]), float(dist.loc["A", "C"])
    print(f"  distance(A,B) = {d_AB:.4f}  (same exposure -> small)")
    print(f"  distance(A,C) = {d_AC:.4f}  (orthogonal    -> large)")
    assert d_AB < d_AC, "same-exposure pair must be closer than orthogonal pair"
    assert d_AC / max(d_AB, 1e-9) > 5, (
        f"expected clear separation; ratio = {d_AC / max(d_AB, 1e-9):.1f}x"
    )


def test_distance_diagonal_zero_and_symmetric() -> None:
    """Diagonal exactly 0; matrix symmetric."""
    prices, factors, _ = _build_two_factor_world()
    dist = factor_beta_distance(prices, factors, ridge_alpha=1.0)
    diag = np.diag(dist.values)
    asym = (dist - dist.T).abs().to_numpy().max()
    print(f"  diagonal = {diag} | max asymmetry = {asym:.2e}")
    assert (diag == 0).all(), f"non-zero diagonal: {diag}"
    assert asym < 1e-10, f"not symmetric: {asym:.2e}"


def test_build_factor_panel_shape_and_columns() -> None:
    """build_factor_panel returns 6 style + industry factors, no NaNs.

    Plant a 9-stock universe spanning 3 FF12 industries (3 names each, clearing the
    default min_stocks_per_industry=3).
    """
    rng = np.random.default_rng(3)
    n_days = 400
    # permno -> SIC (3 banks=Money, 3 oils=Enrgy, 3 software=BusEq)
    sic_map = pd.Series({
        1: 6020, 2: 6021, 3: 6022,        # Money
        4: 2911, 5: 1311, 6: 2990,        # Enrgy
        7: 7372, 8: 7373, 9: 3674,        # BusEq
    })
    rets = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(n_days, len(sic_map))),
        columns=list(sic_map.index),
        index=pd.date_range("2010-01-04", periods=n_days, freq="B"),
    )
    prices = 100.0 * (1.0 + rets).cumprod()

    # Synthetic style factors aligned to the same dates.
    from src.factors import STYLE_FACTORS
    style = pd.DataFrame(
        rng.normal(0.0, 0.008, size=(n_days, len(STYLE_FACTORS))),
        columns=STYLE_FACTORS, index=prices.index,
    )
    panel = build_factor_panel(prices, sic_map, style_factors=style)
    print(f"  factor panel shape: {panel.shape} | cols: {list(panel.columns)}")
    assert not panel.isna().any().any(), "factor panel has NaNs"
    # 6 style + 3 industries (each has exactly 3 names)
    for col in STYLE_FACTORS + ["Money", "Enrgy", "BusEq"]:
        assert col in panel.columns, f"missing factor column {col}"
    assert panel.shape[1] == 9, f"expected 9 factor columns, got {panel.shape[1]}"


# ────────────────────────────────────────────────────────────────────────────────
# runner
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_ff12_ranges_are_disjoint,
        test_ff12_known_codes_map_correctly,
        test_ff12_handles_bad_input,
        test_ridge_betas_recovers_planted_exposures,
        test_same_exposure_pair_is_close_orthogonal_is_far,
        test_distance_diagonal_zero_and_symmetric,
        test_build_factor_panel_shape_and_columns,
    ]
    failures = 0
    for t in tests:
        print(f"\n▶ {t.__name__}")
        try:
            t()
            print("  ✅ PASS")
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
