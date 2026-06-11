"""
Synthetic-data tests for the Phase 6 corrections (phases/phase6/decisions.md).

Each correction lives behind a flag whose DEFAULT reproduces the Phase 5 engine
bit-identically (verified by the pre-existing test suite still passing, plus the
inert-flag test below). Tested here:

  D6.1  MacKinnon p-values are stricter than raw-ADF p-values on non-cointegrated
        pairs, and a genuinely cointegrated pair still passes.
  D6.2  Corrected delisting-code fallback map + compounded delisting-day return.
  D6.3  Stop cooldown blocks the next-day re-entry that the plain stop allows.
  D6.4  block_last_day_entry suppresses the degenerate last-day round trip.
  D6.5  execution_delay=1 shifts fills (and first P&L day) by one day, keeping
        the signal-day z as entry_z.
  D6.7  Flags that have nothing to act on leave the simulation bit-identical.

Run:  python tests/test_corrections_synthetic.py   (from pairs-trading-ml/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import (
    _delisting_day_return,
    _delisting_fallback_return,
    _delisting_fallback_return_fixed,
    simulate_pair_in_month,
)
from src.cointegration import engle_granger
from src.spread import rolling_zscore, spread_series

PERMNO_A, PERMNO_B = 101, 202
GAMMA = 2.0
N_FORM = 40
ZWIN = 40          # = formation length → every trading day has a valid z
ENTRY, EXIT = 2.0, 0.0


def _panel_from_spread(trading_spread: list[float]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Panel whose spread (A - γB) equals ±0.5 alternating noise over the formation
    window, then exactly `trading_spread` over the trading month. The z-score inside
    simulate_pair_in_month is fully determined by this spread path."""
    n_t = len(trading_spread)
    n = N_FORM + n_t
    dates = pd.bdate_range("2020-01-01", periods=n)
    b = 100.0 + 0.1 * np.arange(n)
    s = np.concatenate([0.5 * np.tile([1.0, -1.0], N_FORM // 2), trading_spread])
    a = s + GAMMA * b
    panel = pd.DataFrame({PERMNO_A: a, PERMNO_B: b}, index=dates)
    return panel, dates[N_FORM:]


def _zscores(panel: pd.DataFrame) -> pd.Series:
    return rolling_zscore(
        spread_series(panel[PERMNO_A], panel[PERMNO_B], GAMMA), window=ZWIN
    )


def _simulate(panel, trading, **kwargs):
    defaults = dict(
        entry_sigma=ENTRY, exit_sigma=EXIT, stop_sigma=None,
        zscore_window=ZWIN, delisting_events={},
    )
    defaults.update(kwargs)
    return simulate_pair_in_month(
        PERMNO_A, PERMNO_B, GAMMA, panel, trading, **defaults
    )


# ────────────────────────────────────────────────────────────────────────────────
# D6.2 — delisting corrections
# ────────────────────────────────────────────────────────────────────────────────


def test_delisting_fallback_fixed_map():
    # mergers: unchanged neutral
    assert _delisting_fallback_return_fixed(231) == 0.0
    # 300s are EXCHANGES (neutral) — old map wrongly penalised them -30%
    assert _delisting_fallback_return(350) == -0.30
    assert _delisting_fallback_return_fixed(350) == 0.0
    # 400s liquidations: -30% in both
    assert _delisting_fallback_return_fixed(450) == -0.30
    # 500 / 520-584 are the performance-related "dropped" class (Shumway -30%) —
    # old map gave them a token -5%. 500 is also the v2-pull sentinel.
    assert _delisting_fallback_return(500) == -0.05
    assert _delisting_fallback_return_fixed(500) == -0.30
    assert _delisting_fallback_return_fixed(552) == -0.30
    assert _delisting_fallback_return_fixed(574) == -0.30
    # moved to another exchange: neutral
    assert _delisting_fallback_return_fixed(503) == 0.0


def test_delisting_day_return_compounds():
    # default: overwrite (Phases 1-5)
    assert _delisting_day_return(0.02, -0.50, delisting_fix=False) == -0.50
    # fix: compound (1+ret)(1+dlret)-1
    fixed = _delisting_day_return(0.02, -0.50, delisting_fix=True)
    assert abs(fixed - (1.02 * 0.50 - 1.0)) < 1e-12
    # fix with no usable market return: fall back to dlret alone
    assert _delisting_day_return(float("nan"), -0.50, delisting_fix=True) == -0.50


# ────────────────────────────────────────────────────────────────────────────────
# D6.3 — stop cooldown
# ────────────────────────────────────────────────────────────────────────────────


def _stop_panel():
    """d0: entry long (z≤-2, >-3.5) · d1: stop (z≤-3.5) · d2: still beyond entry
    (-3.5<z≤-2 → naked re-entry without cooldown) · d3: back inside the band."""
    panel, trading = _panel_from_spread([-1.2, -2.0, -1.5, -0.5])
    z = _zscores(panel).loc[trading]
    assert -3.5 < z.iloc[0] <= -2.0, f"bad construction: z0={z.iloc[0]:.2f}"
    assert z.iloc[1] <= -3.5, f"bad construction: z1={z.iloc[1]:.2f}"
    assert -3.5 < z.iloc[2] <= -2.0, f"bad construction: z2={z.iloc[2]:.2f}"
    assert abs(z.iloc[3]) < 2.0, f"bad construction: z3={z.iloc[3]:.2f}"
    return panel, trading


def test_stop_without_cooldown_reenters_next_day():
    panel, trading = _stop_panel()
    trades, _, _, _, _ = _simulate(panel, trading, stop_sigma=3.5)
    # stop at d1, re-entry at d2 (force-closed at d3 month-end) → 2 trades
    assert len(trades) == 2, f"expected stop + re-entry, got {trades}"
    assert trades[0].exit_reason == "stop_loss"
    assert trades[1].entry_date == trading[2], "re-entry should be the day after the stop"


def test_stop_cooldown_blocks_reentry():
    panel, trading = _stop_panel()
    trades, _, _, _, _ = _simulate(panel, trading, stop_sigma=3.5, stop_cooldown=True)
    assert len(trades) == 1, f"cooldown should leave only the stop trade, got {trades}"
    assert trades[0].exit_reason == "stop_loss"


# ────────────────────────────────────────────────────────────────────────────────
# D6.4 — block last-day entry
# ────────────────────────────────────────────────────────────────────────────────


def test_block_last_day_entry():
    # entry signal fires ONLY on the final trading day
    panel, trading = _panel_from_spread([0.0, 0.0, 0.0, -1.2])
    z = _zscores(panel).loc[trading]
    assert z.iloc[-1] <= -2.0 and (z.iloc[:-1] > -2.0).all(), "bad construction"

    trades_default, _, _, days_default, _ = _simulate(panel, trading)
    assert len(trades_default) == 1 and trades_default[0].exit_reason == "force_close"
    assert trades_default[0].entry_date == trades_default[0].exit_date == trading[-1]
    assert days_default == 0   # the degenerate round trip run_one_month silently drops

    trades_fixed, _, _, _, _ = _simulate(panel, trading, block_last_day_entry=True)
    assert trades_fixed == [], f"last-day entry should be blocked, got {trades_fixed}"


# ────────────────────────────────────────────────────────────────────────────────
# D6.5 — execution delay
# ────────────────────────────────────────────────────────────────────────────────


def test_execution_delay_shifts_fill_by_one_day():
    # d0: entry signal · d1-d2: hold · d3: reversion signal
    panel, trading = _panel_from_spread([-1.2, -1.2, -1.2, 1.0])
    z = _zscores(panel).loc[trading]
    assert z.iloc[0] <= -2.0 and z.iloc[3] >= 0.0, "bad construction"

    trades0, pnl0, _, _, _ = _simulate(panel, trading)
    trades1, pnl1, _, _, _ = _simulate(panel, trading, execution_delay=1)

    assert len(trades0) == len(trades1) == 1
    # fill day shifts d0 → d1; entry_z stays the SIGNAL-day dislocation
    assert trades0[0].entry_date == trading[0]
    assert trades1[0].entry_date == trading[1]
    assert trades1[0].entry_z == trades0[0].entry_z
    # first P&L day shifts d1 → d2
    assert pnl0.loc[trading[1]] != 0.0
    assert pnl1.loc[trading[1]] == 0.0
    assert pnl1.loc[trading[2]] != 0.0
    # the delayed exit signal (d3) lapses at month-end → force-close, same close
    assert trades0[0].exit_reason == "reversion"
    assert trades1[0].exit_reason == "force_close"
    assert trades0[0].exit_date == trades1[0].exit_date == trading[3]


# ────────────────────────────────────────────────────────────────────────────────
# D6.1 — MacKinnon vs raw-ADF p-values
# ────────────────────────────────────────────────────────────────────────────────


def test_mackinnon_stricter_than_adf():
    n = 500
    diffs = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
        y = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
        p_adf = engle_granger(y, x, pvalue_method="adf").adf_pvalue
        p_mack = engle_granger(y, x, pvalue_method="mackinnon").adf_pvalue
        diffs.append(p_mack - p_adf)
    # raw ADF on estimated residuals is anti-conservative → MacKinnon p higher
    assert np.mean(diffs) > 0, f"expected MacKinnon > ADF on average, diffs={diffs}"
    assert sum(d >= 0 for d in diffs) >= 6, f"diffs={diffs}"

    # a genuinely cointegrated pair must still pass the correct test
    rng = np.random.default_rng(42)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.5)
    y = 2.0 * x + noise
    res = engle_granger(y, x, pvalue_method="mackinnon")
    assert res.adf_pvalue < 0.05, f"cointegrated pair should pass, p={res.adf_pvalue:.4f}"


# ────────────────────────────────────────────────────────────────────────────────
# D6.7 — flags with nothing to act on are inert (bit-identical)
# ────────────────────────────────────────────────────────────────────────────────


def test_inert_flags_are_bit_identical():
    # plain entry → reversion path: no stop, no delisting, no last-day entry
    panel, trading = _panel_from_spread([-1.2, -1.2, 1.0, 0.0])
    base_trades, base_pnl, base_wt, base_days, base_carry = _simulate(panel, trading)
    fix_trades, fix_pnl, fix_wt, fix_days, fix_carry = _simulate(
        panel, trading,
        stop_cooldown=True, block_last_day_entry=True, delisting_fix=True,
    )
    assert base_trades == fix_trades
    assert base_pnl.equals(fix_pnl)
    assert base_wt.equals(fix_wt)
    assert base_days == fix_days
    assert base_carry == fix_carry


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_delisting_fallback_fixed_map,
        test_delisting_day_return_compounds,
        test_stop_without_cooldown_reenters_next_day,
        test_stop_cooldown_blocks_reentry,
        test_block_last_day_entry,
        test_execution_delay_shifts_fill_by_one_day,
        test_mackinnon_stricter_than_adf,
        test_inert_flags_are_bit_identical,
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
