"""
Synthetic-data tests for the Phase 5 position carry-over logic in
src.backtest.simulate_pair_in_month.

We build a deterministic two-window price panel:
  * formation : a small noisy spread (so the z-score has a defined mean/std)
  * trading-1 : the spread diverges upward and does NOT revert → a position opens
                (short spread) and is still open at month-end
  * trading-2 : the spread either reverts to 0 (position closes) or keeps diverging
                (position would stay open) — chosen per test

Tested behaviours:
  1. carry_over=False force-closes at month-end (pre-Phase-5 default unchanged)
  2. carry_over=True holds the open position into next month (CarryState returned,
     no force_close trade, γ frozen, months_carried=1)
  3. a carried position continues in month-2 and closes there, with the trade record
     spanning BOTH months (entry_date preserved, round-trip P&L accumulated)
  4. no P&L double-count or gap at the month boundary (the critical correctness test)
  5. MAX_CARRY_MONTHS cap force-closes a position that never reverts
  6. the carried position's month-2 signal uses the FROZEN γ (not a re-fit)

Run:  python -m tests.test_backtest_carryover_synthetic   (from pairs-trading-ml/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import CarryState, simulate_pair_in_month
from src.spread import rolling_zscore, spread_series

PERMNO_A, PERMNO_B = 101, 202
GAMMA = 2.0
ZWIN = 30
ENTRY, EXIT = 2.0, 0.0


def _two_month_panel(
    t2_spread_end: float,
    n_form: int = 40,
    n_t1: int = 15,
    n_t2: int = 15,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Build a continuous panel and return (panel, trading-1 dates, trading-2 dates).

    Spread path: formation noise ~N(0,0.5) → trading-1 ramps 0.3→3.0 (diverge, opens a
    short) → trading-2 ramps 3.0→`t2_spread_end` (revert if 0.0, persist if >3).
    B carries a mild trend so γ genuinely affects the spread shape (and thus the z).
    """
    rng = np.random.default_rng(seed)
    n = n_form + n_t1 + n_t2
    dates = pd.bdate_range("2020-01-01", periods=n)
    b = 100.0 + 0.1 * np.arange(n)
    s = np.empty(n)
    s[:n_form] = rng.normal(0.0, 0.5, size=n_form)
    s[n_form:n_form + n_t1] = np.linspace(0.3, 3.0, n_t1)
    s[n_form + n_t1:] = np.linspace(3.0, t2_spread_end, n_t2)
    a = GAMMA * b + s
    panel = pd.DataFrame({PERMNO_A: a, PERMNO_B: b}, index=dates)
    t1 = dates[n_form:n_form + n_t1]
    t2 = dates[n_form + n_t1:]
    return panel, t1, t2


def _sim(panel, trading_dates, *, carry=None, carry_over=False, max_carry_months=1):
    return simulate_pair_in_month(
        PERMNO_A, PERMNO_B, GAMMA, panel, trading_dates,
        entry_sigma=ENTRY, exit_sigma=EXIT, stop_sigma=None,
        zscore_window=ZWIN, delisting_events={},
        carry=carry, carry_over=carry_over, max_carry_months=max_carry_months,
    )


# ────────────────────────────────────────────────────────────────────────────────
# tests
# ────────────────────────────────────────────────────────────────────────────────


def test_carry_off_force_closes_at_month_end() -> None:
    """Default (carry_over=False): an open position is force-closed at month-end and
    nothing is carried — the pre-Phase-5 contract."""
    panel, t1, _ = _two_month_panel(t2_spread_end=5.0)
    trades, _pnl, _w, _days, carry_out = _sim(panel, t1, carry_over=False)

    assert carry_out is None, "carry_over=False must never carry a position"
    reasons = [t.exit_reason for t in trades]
    assert "force_close" in reasons, f"expected a force_close, got {reasons}"
    # the position opened short (spread rose above its mean)
    assert trades[-1].direction == -1
    print(f"  carry off → trades={reasons}, carry_out=None ✓")


def test_carry_on_holds_position() -> None:
    """carry_over=True: the open position is carried, not force-closed."""
    panel, t1, _ = _two_month_panel(t2_spread_end=5.0)
    trades, _pnl, _w, _days, carry_out = _sim(
        panel, t1, carry_over=True, max_carry_months=3)

    assert carry_out is not None, "an open position should have been carried"
    assert all(t.exit_reason != "force_close" for t in trades), \
        "carried month must not emit a force_close"
    assert carry_out.direction == -1
    assert carry_out.gamma_frozen == GAMMA, "γ must be frozen at the value used"
    assert carry_out.months_carried == 1, "first carry → months_carried == 1"
    print(f"  carry on → carry_out(dir={carry_out.direction}, "
          f"γ={carry_out.gamma_frozen}, m={carry_out.months_carried}) ✓")


def test_carried_position_continues_and_closes_next_month() -> None:
    """A position carried out of month-1 continues in month-2 and reverts there. The
    closing trade spans both months (entry_date from month-1)."""
    panel, t1, t2 = _two_month_panel(t2_spread_end=0.0)   # month-2 reverts to 0
    _tr1, _p1, _w1, _d1, carry1 = _sim(panel, t1, carry_over=True, max_carry_months=3)
    assert carry1 is not None
    entry_date = carry1.entry_date

    trades2, _p2, _w2, _d2, carry2 = _sim(
        panel, t2, carry=carry1, carry_over=True, max_carry_months=3)

    rev = [t for t in trades2 if t.exit_reason == "reversion"]
    assert rev, f"expected a reversion close in month-2, got {[t.exit_reason for t in trades2]}"
    assert carry2 is None, "position closed in month-2, nothing left to carry"
    assert rev[0].entry_date == entry_date, "trade must keep its original entry date"
    print(f"  carried → reverted in month-2; entry preserved "
          f"({pd.Timestamp(entry_date).date()}) ✓")


def test_no_pnl_double_count_at_boundary() -> None:
    """The carried trade's round-trip P&L must equal the naive close-to-close
    equal-dollar P&L over [entry, exit] — no day dropped or double-counted at the
    month boundary. (Frictionless, so borrow/costs are zero.)"""
    panel, t1, t2 = _two_month_panel(t2_spread_end=0.0)
    _t1, _p1, _w1, _d1, carry1 = _sim(panel, t1, carry_over=True, max_carry_months=3)
    trades2, _p2, _w2, _d2, _c2 = _sim(
        panel, t2, carry=carry1, carry_over=True, max_carry_months=3)
    rev = [t for t in trades2 if t.exit_reason == "reversion"][0]

    a, b = panel[PERMNO_A], panel[PERMNO_B]
    direction = carry1.direction
    hold = panel.index[(panel.index > carry1.entry_date) & (panel.index <= rev.exit_date)]
    naive = 0.0
    for t in hold:
        i = panel.index.get_loc(t)
        prev = panel.index[i - 1]
        ret_a = a[t] / a[prev] - 1.0
        ret_b = b[t] / b[prev] - 1.0
        naive += direction * 0.5 * (ret_a - ret_b)

    print(f"  round_trip={rev.round_trip_return:.8f}  naive={naive:.8f}  "
          f"(hold {len(hold)} days across the boundary)")
    assert abs(rev.round_trip_return - naive) < 1e-9, \
        "carried round-trip P&L disagrees with the naive close-to-close hold"


def test_max_carry_months_cap_force_closes() -> None:
    """With max_carry_months=2, a never-reverting position is force-closed at the end of
    its 2nd month instead of carrying a third time; with max_carry_months=3 it carries."""
    panel, t1, t2 = _two_month_panel(t2_spread_end=5.0)   # month-2 keeps diverging
    _t1, _p1, _w1, _d1, carry1 = _sim(panel, t1, carry_over=True, max_carry_months=2)
    assert carry1 is not None and carry1.months_carried == 1

    # cap = 2 → cannot carry again → force-close in month-2
    trades_cap, _pc, _wc, _dc, carry_cap = _sim(
        panel, t2, carry=carry1, carry_over=True, max_carry_months=2)
    assert carry_cap is None, "cap=2 must stop the carry at the 2nd month"
    assert any(t.exit_reason == "force_close" for t in trades_cap)

    # cap = 3 → the same position is allowed to carry again
    _tr3, _p3, _w3, _d3, carry3 = _sim(
        panel, t2, carry=carry1, carry_over=True, max_carry_months=3)
    assert carry3 is not None and carry3.months_carried == 2
    print("  cap=2 force-closes month-2; cap=3 carries (months_carried=2) ✓")


def test_carried_signal_uses_frozen_gamma() -> None:
    """The month-2 exit is driven by the FROZEN γ. Reconstruct the spread/z with the
    carry's gamma_frozen and verify the reversion fires on the first day z crosses ≤ 0."""
    panel, t1, t2 = _two_month_panel(t2_spread_end=0.0)
    _t1, _p1, _w1, _d1, carry1 = _sim(panel, t1, carry_over=True, max_carry_months=3)
    trades2, _p2, _w2, _d2, _c2 = _sim(
        panel, t2, carry=carry1, carry_over=True, max_carry_months=3)
    rev = [t for t in trades2 if t.exit_reason == "reversion"][0]

    # independent reconstruction with the frozen γ
    spread = spread_series(panel[PERMNO_A], panel[PERMNO_B], carry1.gamma_frozen)
    z = rolling_zscore(spread, window=ZWIN)
    z_t2 = z.loc[t2]
    # short position (-1) exits on the first month-2 day with z <= 0
    predicted_exit = z_t2[z_t2 <= 0.0].index[0]

    print(f"  frozen γ={carry1.gamma_frozen}: predicted exit "
          f"{pd.Timestamp(predicted_exit).date()}, actual {pd.Timestamp(rev.exit_date).date()}")
    assert rev.exit_date == predicted_exit, \
        "reversion did not fire where the frozen-γ z-score crosses zero"


# ────────────────────────────────────────────────────────────────────────────────
# runner
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_carry_off_force_closes_at_month_end,
        test_carry_on_holds_position,
        test_carried_position_continues_and_closes_next_month,
        test_no_pnl_double_count_at_boundary,
        test_max_carry_months_cap_force_closes,
        test_carried_signal_uses_frozen_gamma,
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
