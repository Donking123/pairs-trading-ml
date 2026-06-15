"""
Synthetic-data check for lookahead.py  [Phase 4b].

Tests the reconstruction + comparison machinery directly on hand-built trade records
(no backtest run needed):
  * reconstruct_daily_positions lays direction on [entry, exit) only,
  * identical runs → PASS (no mismatch),
  * a planted change to a PAST position → FAIL (detected),
  * a pair present in only one run → FAIL,
  * differences AT/AFTER the cut date are ignored (only the overlap matters).

Run:  python -m tests.test_lookahead_synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lookahead import (
    compare_position_panels,
    reconstruct_daily_positions,
)

CAL = pd.bdate_range("2010-01-04", "2010-03-31")  # ~62 business days


def _trade(a, b, direction, entry, exit_):
    return {"permno_a": a, "permno_b": b, "direction": direction,
            "entry_date": pd.Timestamp(entry), "exit_date": pd.Timestamp(exit_)}


def test_reconstruct_holds_on_half_open_interval() -> None:
    trades = pd.DataFrame([_trade(10, 20, 1, "2010-01-11", "2010-01-15")])
    panel = reconstruct_daily_positions(trades, CAL)
    held = panel["10_20"]
    assert held.loc["2010-01-11"] == 1, "entry day should be held"
    assert held.loc["2010-01-14"] == 1, "mid-trade day should be held"
    assert held.loc["2010-01-15"] == 0, "exit day should be flat (half-open)"
    assert held.loc["2010-01-08"] == 0, "before entry should be flat"
    print("  [entry, exit) reconstruction correct ✓")


def test_identical_runs_pass() -> None:
    trades = pd.DataFrame([
        _trade(10, 20, 1, "2010-01-11", "2010-01-29"),
        _trade(10, 30, -1, "2010-02-01", "2010-02-26"),
    ])
    full = reconstruct_daily_positions(trades, CAL)
    short = reconstruct_daily_positions(trades, CAL)  # same trades = no future leak
    res = compare_position_panels(full, short, cut_date="2010-02-15")
    print(" ", res.summary())
    assert res.passed, "identical runs must pass"


def test_planted_past_change_is_detected() -> None:
    full_trades = pd.DataFrame([_trade(10, 20, 1, "2010-01-11", "2010-01-29")])
    # the SHORT run sees a different PAST position for the same pair/date → bias
    short_trades = pd.DataFrame([_trade(10, 20, -1, "2010-01-11", "2010-01-29")])
    full = reconstruct_daily_positions(full_trades, CAL)
    short = reconstruct_daily_positions(short_trades, CAL)
    res = compare_position_panels(full, short, cut_date="2010-02-15")
    print(" ", res.summary())
    assert not res.passed, "a flipped past position must be flagged"
    assert (res.mismatches["pair"] == "10_20").all()


def test_pair_only_in_one_run_is_detected() -> None:
    full = reconstruct_daily_positions(
        pd.DataFrame([_trade(10, 20, 1, "2010-01-11", "2010-01-29")]), CAL)
    short = reconstruct_daily_positions(
        pd.DataFrame([_trade(40, 50, 1, "2010-01-11", "2010-01-29")]), CAL)
    res = compare_position_panels(full, short, cut_date="2010-02-15")
    print(" ", res.summary())
    assert not res.passed, "a pair traded in only one run must be flagged"


def test_differences_after_cut_are_ignored() -> None:
    # identical before the cut; the full run trades MORE after the cut → must still PASS
    full_trades = pd.DataFrame([
        _trade(10, 20, 1, "2010-01-11", "2010-01-29"),   # before cut (shared)
        _trade(10, 30, 1, "2010-03-01", "2010-03-15"),   # after cut (full only)
    ])
    short_trades = pd.DataFrame([_trade(10, 20, 1, "2010-01-11", "2010-01-29")])
    full = reconstruct_daily_positions(full_trades, CAL)
    short = reconstruct_daily_positions(short_trades, CAL)
    res = compare_position_panels(full, short, cut_date="2010-02-15")
    print(" ", res.summary())
    assert res.passed, "post-cut divergence must not count against the overlap"


if __name__ == "__main__":
    tests = [
        test_reconstruct_holds_on_half_open_interval,
        test_identical_runs_pass,
        test_planted_past_change_is_detected,
        test_pair_only_in_one_run_is_detected,
        test_differences_after_cut_are_ignored,
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
