# Phase 4 — Realism, Validation & Writeup

**Status:** 🔵 4a realism ✅ · 4b lookahead ✅ (6/6 PASS) · 4d forward test ✅ · 4c writeup
drafted. Started 2026-06-07.

> New folder; Phases 1/2/2.5/3 frozen.

## 4d — True out-of-sample forward test ✅ (the headline limitation)

Frozen strategies (no re-tuning) run on **2024** — data past the 2003–2023 development sample
(WRDS CRSP daily available through 2024-12-31). The cleanest generalisation test; even 2023
isn't a clean holdout since ξ/ridge-α were tuned on Dec-2023.

Run on **2024–2025** (~23 months) via the CIZ/v2 CRSP tables (current through 2025).

| Strategy | In-sample (2003–2023) | OOS 2024 | OOS 2025 | OOS full (23mo) |
|---|---:|---:|---:|---:|
| **PC** | 1.028 | +0.16 | +1.40 | **0.858** |
| Factor-beta | 1.013 | −0.46 | +0.47 | **0.117** |

**PC generalises** (0.858 ≈ in-sample 1.028; 2025 even exceeded it). **Factor-beta does not**
(0.117) — a reversal, since it was the sturdier metric under in-sample perturbations. Strong
year-to-year regime variation (both weak in calm 2024, recovered 2025). Methodological note: the
2024-only window (PC 0.16) would have wrongly signalled "edge vanished" — a single short OOS
window is unreliable. Harness: `src/wrds_pull.py --source v2` (CIZ tables) +
`notebooks/05_forward_test.py`; data in `data_through_2025/` (separate from the 2003–2023 cache).

## 4b — Lookahead-bias test ✅ (replaces the Alpaca forward test)

Per the QF621 prof: paper trading is **not needed** for the project. Its only real benefit
over a backtest holdout is immunity to lookahead bias — and that can be tested directly,
as a black box, without paper trading or reading the strategy code.

**The test.** Run the backtest over X→Y and again over X→Y′ (Y′<Y). For every overlapping
date, the daily target-position vectors must be **byte-identical**. If cutting off the
future changes a past position, the strategy is reaching into the future = lookahead bias.
Run several cut dates — subtle leaks hide around delistings / dividends / universe
selection (the prof's advice).

**Why our design should pass:** each month is self-contained (force-close at month-end) and
the end date only selects which months run, so truncating Y→Y′ leaves earlier months
untouched. A failure would expose a real future leak.

**What's built:**
- `src/lookahead.py` — `reconstruct_daily_positions` (per-pair signed {−1,0,+1} from the
  saved `Trade` records) + `compare_position_panels` (overlap diff → `LookaheadResult`).
- `tests/test_lookahead_synthetic.py` — 5 tests (clean pass; planted past-change detected;
  pair-only-in-one-run detected; post-cut divergence ignored). Suite 49/49.
- `notebooks/01_lookahead_test.py` — runs full + 3 cuts on PC & factor, writes a report.

**Smoke check (real data):** PC, full 2003→2004-12 vs cut 2003→2004-06 → 375 overlap days
× 491 pairs, **0 mismatches (PASS)**. Confirms the machinery and a clean pipeline.

### Run the full test
```bash
nohup python phases/phase4/notebooks/01_lookahead_test.py \
  > phases/phase4/results/lookahead_run.log 2>&1 &
tail -f phases/phase4/results/lookahead_run.log
```
Output: `results/lookahead_{pc,factor}.txt` + `lookahead_summary.csv`. Expected: ALL PASS.

## Results — _full run pending_

| Metric | Cut 2017 | Cut 2013 | Cut 2009 |
|---|---|---|---|
| PC | PASS ✅ | PASS ✅ | PASS ✅ |
| factor | PASS ✅ | PASS ✅ | PASS ✅ |

**6/6 PASS — no lookahead bias.** 0 mismatched cells across both metrics and all three cuts
(up to 3,866 pairs × 3,776 overlap days). Confirms the look-ahead protections (t+1 execution,
rolling formation window, `.shift(1)` z-score, point-in-time universe/delisting) hold.

## 4a — Realism variant ✅ (built; full run pending)

Re-runs the headline strategies with all frictions on, to see if the edge survives real
costs:
- **Transaction costs** = actual CRSP bid/ask half-spread at entry & exit (real per-name,
  time-varying ~27 bps in 2000-04 → ~2.5 bps by 2015+; 10 bps fallback for ~1% bad quotes).
- **Borrow** = 35 bps annual on the short leg.
- **Stop** = 3.5σ.

**What's built:** `src/costs.py` (`RealismConfig`, `build_spread_panel`, `transaction_cost`,
`borrow_cost_daily`) + `realism=` knob on `run_backtest` (default frictionless = bit-identical
to core). `notebooks/02_run_realism.py` runs pc_realism + factor_realism vs the frictionless
baselines. 4 unit tests; suite 53/53. Real-data smoke (pc 2003) confirms costs apply.

### Run the realism variant
```bash
nohup python phases/phase4/notebooks/02_run_realism.py \
  > phases/phase4/results/realism_run.log 2>&1 &
```

| Strategy | Frictionless | Realism (net of costs) | Δ | ann.ret | MDD |
|---|---:|---:|---:|---:|---:|
| PC core | 1.028 | **0.572** | −0.456 | +1.66% | −9.8% |
| Factor-beta core | 1.013 | **0.578** | −0.436 | +1.81% | −6.6% |

**Finding:** real frictions (CRSP bid/ask half-spread + 35bps borrow + 3.5σ stop) cut the
Sharpe by ~45%, but both strategies stay **net-positive (~0.57)** — a genuine risk-adjusted
edge survives realistic costs. Drag is heaviest in the wide-spread early-2000s; the stop adds
round-trips (more cost incidence). Both metrics converge to ~0.57 net, factor-beta with the
tighter drawdown (−6.6% vs −9.8%).

## Still pending in Phase 4
- **4c final writeup** — pull Phases 0–4 together.

## Files
| Path | What |
|---|---|
| **`notebooks/phase4_complete_reference.ipynb`** | **detailed walkthrough (executed, with plots): realism, cost-opt, lookahead, forward test** |
| `decisions.md` | design decisions (D4a.*, D4b.*, D4d) |
| `src/lookahead.py`, `src/costs.py` | lookahead audit + cost model engines |
| `notebooks/01_lookahead_test.py` · `02_run_realism.py` · `03_cost_optimization.py` · `05_forward_test.py` | runners |
| `notebooks/06_probe_crsp_currency.py` · `07_probe_v2_schema.py` | WRDS CIZ probes (forward-data migration) |
