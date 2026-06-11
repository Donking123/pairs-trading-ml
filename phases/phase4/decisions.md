# Phase 4 — Decisions Log

**Status:** 🔵 IN PROGRESS — started 2026-06-07.

Phase 4 = realism + validation + writeup. The first deliverable built here is **4b**, the
**lookahead-bias test**, which *replaces* the originally-planned Alpaca paper-trade forward
test (per the QF621 prof's guidance — see below).

New folder; Phases 1/2/2.5/3 frozen. `src/` extended with a new module only.

## D4b.1 — Lookahead test replaces the Alpaca forward test

The prof advised that paper trading is **not needed** for the project. Paper trading only
adds two things over a backtest holdout: (1) proof of real-time operational workflow —
trivial for a monthly-rebalanced strategy; and (2) immunity to lookahead bias. He gave an
automated, black-box test for (2) that needs no paper trading and no code-reading:

> Run the backtest over X→Y; record daily target-position vectors. Re-run over X→Y′ (Y′<Y).
> For every overlapping date, the position vectors **must be byte-identical**. If cutting off
> the future changes a past position, the strategy is reaching into the future = lookahead
> bias. Try many cut dates (subtle leaks hide around delistings / dividends / universe
> selection / full-sample normalization).

**Decision:** implement this as `src/lookahead.py` + a Phase 4b runner, and drop Alpaca.
It is cheaper (no weeks of waiting), directly validates the look-ahead protections we built
(t+1 execution, rolling 3y formation window, `rolling_zscore` `.shift(1)`, point-in-time
universe + delisting), and is a strong, defensible artefact for the writeup.

## D4b.2 — Position granularity: per-PAIR signed position

The "instrument" we trade is the spread (a pair). We record, per trading day, each pair's
signed target position ∈ {−1, 0, +1} (+1 = long spread = long A / short B). This is:
- the truest representation of the strategy's *decisions* (open/close which pair, which way),
- more sensitive than per-asset netting (a per-pair change can't be masked by asset-level
  cancellation), and
- independent of the allocation weighting (equal vs zweight) — bias detection is about the
  *decisions*, not the dollar sizing. (Per-asset/dollar aggregation is a possible extension.)

## D4b.3 — Reconstruct positions from the saved `Trade` records

Rather than instrument the engine, reconstruct the daily position panel from the `Trade`
records each run already produces (permno_a, permno_b, direction, entry_date, exit_date).
A trade holds `direction` on trading days in **[entry_date, exit_date)** (flat at exit). The
same deterministic reconstruction is applied to both runs, so any consistent convention
detects bias correctly — what matters is whether the two runs AGREE on overlapping dates.

## D4b.4 — Why our structure should pass (and what would fail)

Each month is independent and self-contained (force-close at month-end), and the end date
enters only via which months are simulated. So truncating Y→Y′ should leave all months ≤ Y′
identical. A FAILURE would mean some computation leaked future data — e.g. universe/date
selection using max(sample date), global normalization, or delisting/dividend adjustment
applied with full-sample knowledge. Running the test on the headline metrics (PC, factor)
turns "we were careful" into evidence.

## Build status (2026-06-07)
- [x] `src/lookahead.py` — `reconstruct_daily_positions`, `compare_position_panels`,
      `LookaheadResult`.
- [x] `tests/test_lookahead_synthetic.py` — 5 tests; suite 49/49.
- [x] `notebooks/01_lookahead_test.py` — full + 3 cuts on PC/factor → report.
- [x] Real smoke (PC 2003→2004-12 vs cut 2004-06): 375 days × 491 pairs, 0 mismatches PASS.
- [x] **Full test COMPLETE (2026-06-07): 6/6 PASS** — PC + factor, cuts 2009/2013/2017, 0
      mismatches (up to 3,866 pairs × 3,776 days). No lookahead bias. 4b DONE.

## D4a — Realism variant (built 2026-06-07)

Re-run the headline strategies with real frictions to see if the edge survives.

**D4a.1 Transaction costs = actual CRSP bid/ask half-spread.** 99% of quotes are valid and
realistically time-varying (median full spread ~27 bps in 2000-04 → ~2.5 bps by 2015+).
Using real per-name spreads is far more defensible than a fixed bps and captures the wide
crisis-era costs that matter (our edge is crisis-dependent). Each $0.50 leg crosses HALF the
spread, at entry and at exit. Fallback `default_spread_bps=10` for the ~1% bad quotes.

**D4a.2 Borrow = 35 bps annual on the short leg** ($0.50 of every long-short pair), accrued
daily while in position.

**D4a.3 Stop = 3.5σ** via the existing `stop_sigma` knob (set at call time).

**D4a.4 Frictionless by default.** `RealismConfig()` is all-off; `run_backtest` with no
`realism` arg is bit-identical to the core (verified: pc 2003-Q1 = 160 trades, identical
returns). Costs are threaded run_backtest → run_one_month → simulate_pair_in_month and
subtracted from daily P&L (so the portfolio Sharpe is exact; trade-level round_trip_return
is a diagnostic and omits the entry cost only).

### 4a build status
- [x] `src/costs.py` — `RealismConfig`, `build_spread_panel`, `transaction_cost`,
      `borrow_cost_daily`. 4 unit tests; suite 53/53.
- [x] `src/backtest.py` — `realism=` knob + bid/ask spread panel threaded through; default
      frictionless bit-identical.
- [x] `notebooks/02_run_realism.py` — pc_realism + factor_realism (costs+borrow+3.5σ stop)
      vs frictionless baselines. Real-data smoke (pc 2003) confirms costs apply.
- [ ] **Run the realism backtest** (user) → fill the README realism table.

## D4a.5 — Net-of-cost optimisation sweep (built 2026-06-07)

Baseline net Sharpe ~0.57 (cost haircut ~45%). Drag = churn on marginal force-close trades ×
wide early-2000s spreads. Levers (all via existing knobs + the new `spread_cost_multiplier`),
in `notebooks/03_cost_optimization.py`, ranked by net Sharpe:
- **coint** — Engle-Granger filter (cut turnover ~3-4×)
- **entry2.5 / entry3.0** — higher conviction entries
- **nostop** — the 3.5σ stop may be net-negative (adds churn)
- **passive** — `spread_cost_multiplier=0.5` (limit orders vs marketable); 2003 H1 smoke
  flipped −0.70% → +0.74%, so execution is a big lever in the wide-spread era
- **combo** — coint + entry2.5 + passive

`RealismConfig.spread_cost_multiplier` added (1.0 marketable / 0.5 passive / 0 mid). 53 tests.
**Exploratory / in-sample** — validate any winner out-of-sample (lookahead/holdout) before
claiming it. User runs the sweep in the background.

## D4d — True out-of-sample forward test (built 2026-06-07)

**Gap identified:** all results live inside 2003-2023, and xi/ridge-α were tuned on Dec-2023,
so even 2023 is not a clean holdout. The only genuine generalisation test is data AFTER the
development sample. Today (2026-06) WRDS has CRSP through ~2025, giving ~2 years of unseen data
— far more than an Alpaca paper-trade would yield.

**Design:**
- `src/wrds_pull.py` refactored to parametrise `--start/--end/--data-dir` (default path
  unchanged). Forward pull writes to a SEPARATE dir (`data_through_2025/`) so the validated
  2003-2023 cache stays pristine and the locked results stay reproducible.
- `src/backtest.py` gained a `style_factors=` passthrough (so the factor metric can read FF
  factors from the extended dir; default None = load from `data/`, bit-identical).
- `notebooks/05_forward_test.py` runs FROZEN PC + factor over 2024-01→2025-12 and compares
  forward Sharpe to in-sample 1.028 / 1.013. NO re-tuning. ~24 monthly returns → indicative,
  noisier than the 251-month figure.

### 4d build status
- [x] wrds_pull parametrised; backtest `style_factors` passthrough (53/53 tests, bit-identical).
- [x] `05_forward_test.py` runner.
- [x] **Forward test RUN (2026-06-07).** First attempt (legacy `dsf`) only reached 2024
      (11 mo). Migrated the pull to CRSP **CIZ/v2 tables** (`--source v2`) → data through 2025,
      giving a **23-month OOS window (2024-2025)**. Frozen result vs in-sample 1.028/1.013:
      **PC 0.858 (generalises!), factor 0.117 (does not).** Year split: PC 2024 +0.16 / 2025
      +1.40; factor 2024 −0.46 / 2025 +0.47. Both weak in calm 2024, recovered 2025
      (regime-dependent). The 23-mo read SUPERSEDES the earlier gloomy 11-mo conclusion —
      methodological lesson: a single short OOS window misleads. 4d DONE.
      v2 pull added to `wrds_pull.py` (CIZ schema mapped to legacy; common-stock filter =
      EQTY/COM/NS/usinc=Y/ACOR-CORP; delisting from dlydelflg). Data verified clean (99.1% valid
      bid/ask, 2025 = 250 days / 590 names).

## D4a.5 result — cost-opt sweep DONE (2026-06-07)
Net-of-cost ranking (baseline PC 0.572 / factor 0.578): **passive execution wins** (PC 0.782,
factor 0.773, +~0.20), **dropping the 3.5σ stop is #2** (+0.11–0.14, stop is net-negative).
Cointegration filter HURTS PC (0.289) / neutral factor; higher entry thresholds hurt both. Best
defensible operating point = passive + no stop (not data-mined: execution-realism + structural).
Writeup §6.1 written.

## PHASE 4 COMPLETE — project build done
4a realism ✅ · 4b lookahead 6/6 PASS ✅ · 4d forward (PC generalises 0.858) ✅ · cost-opt ✅ ·
4c writeup drafted (all §s final). Remaining: submission-format pass + optional GitHub sync.
