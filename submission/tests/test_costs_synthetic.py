"""
Synthetic-data check for costs.py  [Phase 4a realism].

Tests the cost primitives directly (no backtest):
  * build_spread_panel computes (ask-bid)/mid and NaNs invalid quotes,
  * transaction_cost = 0.5·(½·spread_a) + 0.5·(½·spread_b), with fallback,
  * borrow_cost_daily math,
  * RealismConfig().frictionless is True by default.

Run:  python -m tests.test_costs_synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.costs import (
    RealismConfig,
    borrow_cost_daily,
    build_spread_panel,
    transaction_cost,
)


def test_default_config_is_frictionless() -> None:
    assert RealismConfig().frictionless is True
    assert RealismConfig(transaction_costs=True).frictionless is False
    assert RealismConfig(borrow_bps_annual=35).frictionless is False
    print("  RealismConfig() frictionless ✓")


def test_build_spread_panel_basic_and_invalid() -> None:
    crsp = pd.DataFrame({
        "permno": [1, 2, 1, 2],
        "date": pd.to_datetime(["2010-01-04", "2010-01-04", "2010-01-05", "2010-01-05"]),
        "bid": [99.0, 50.0, 100.0, -1.0],     # last is an invalid quote
        "ask": [101.0, 50.5, 100.0, 0.0],     # 1: 2/100=2%; 2 day1: 0.5/50.25
    })
    panel = build_spread_panel(crsp)
    # permno 1, day1: (101-99)/100 = 0.02
    assert abs(panel.at[pd.Timestamp("2010-01-04"), 1] - 0.02) < 1e-9
    # permno 2 day1: (50.5-50)/50.25
    assert abs(panel.at[pd.Timestamp("2010-01-04"), 2] - (0.5 / 50.25)) < 1e-9
    # permno 2 day2: invalid (ask<=0) -> NaN
    assert np.isnan(panel.at[pd.Timestamp("2010-01-05"), 2])
    print("  spread panel: valid frac + NaN on bad quote ✓")


def test_transaction_cost_value_and_fallback() -> None:
    # spread_frac panel: pair (1,2) both 0.02 full spread -> half=0.01 each
    panel = pd.DataFrame({1: [0.02], 2: [0.02]},
                         index=pd.to_datetime(["2010-01-04"]))
    d = pd.Timestamp("2010-01-04")
    cost = transaction_cost(panel, d, 1, 2, default_spread_bps=10.0)
    # 0.5*(0.01) + 0.5*(0.01) = 0.01
    assert abs(cost - 0.01) < 1e-12, cost

    # missing permno -> fallback default (10 bps full -> 5 bps half), one leg known
    panel2 = pd.DataFrame({1: [0.02]}, index=pd.to_datetime(["2010-01-04"]))
    cost2 = transaction_cost(panel2, d, 1, 999, default_spread_bps=10.0)
    # leg1 half = 0.01; leg999 fallback full=0.001 -> half=0.0005
    expected = 0.5 * 0.01 + 0.5 * 0.0005
    assert abs(cost2 - expected) < 1e-12, cost2

    # None panel -> both legs fallback
    cost3 = transaction_cost(None, d, 1, 2, default_spread_bps=10.0)
    assert abs(cost3 - 0.0005) < 1e-12, cost3   # 0.5*0.0005 + 0.5*0.0005
    print("  transaction_cost value + fallback ✓")


def test_borrow_cost_daily() -> None:
    # 35 bps annual on the 0.5 short leg, /252
    bc = borrow_cost_daily(35.0)
    assert abs(bc - (0.0035 / 252 * 0.5)) < 1e-15, bc
    assert borrow_cost_daily(0.0) == 0.0
    print("  borrow_cost_daily ✓")


if __name__ == "__main__":
    tests = [
        test_default_config_is_frictionless,
        test_build_spread_panel_basic_and_invalid,
        test_transaction_cost_value_and_fallback,
        test_borrow_cost_daily,
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
