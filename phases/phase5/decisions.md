# Phase 5 — Decisions Log

**Status:** 🔵 IN PROGRESS — started 2026-06-08.

Phase 5 = two improvements over the force-close-every-month engine, motivated by the Phase-4
finding that **88.4% of trades exit via force-close at a mean loss** (the single biggest drag):

1. **Position carry-over** — hold a still-open position into the next month instead of
   force-closing it at month-end.
2. **Passive execution** — make limit-order pricing (`spread_cost_multiplier=0.5`) a shared
   preset; it was the Phase-4 cost-opt winner (+~0.20 net Sharpe).

`src/` extended only; Phases 1–4 frozen. `carry_over=False` is bit-identical to the
pre-Phase-5 engine (verified — see D5.5).

## D5.1 — Freeze γ at entry for a carried position (NOT re-fit each month)

A carried position keeps the γ it was OPENED with (`CarryState.gamma_frozen`). Rationale:
equal-dollar sizing makes γ irrelevant to P&L — γ only drives the z-score *exit* signal — so
freezing keeps the exit consistent with the spread actually entered, is more realistic (the
legs were never rebalanced), and keeps the position byte-identical under the lookahead audit.
Re-fitting γ mid-trade would let the monitoring hedge ratio drift under a position you never
re-sized. (Original plan said "re-fit"; the audit corrected this — it was the core fix.)

## D5.2 — MAX_CARRY_MONTHS = 3, tied to the half-life ceiling

A position may stay open during at most 3 distinct months. This is not arbitrary: the
cointegration filter already admits half-lives up to `HALF_LIFE_BOUNDS = (5, 60)` trading
days ≈ 3 months. A pair that has not reverted within its own selection half-life has violated
its premise → stop carrying it. Bounds broken-pair bleed. Locked BEFORE seeing any Sharpe
(anti-overfit), consistent with the xi / ridge-α discipline. `MAX_CARRY_MONTHS = 1` recovers
pure force-close (a sanity property used in tests).

## D5.3 — Drop-out rule: force-close a carried pair the cluster no longer endorses

Carry only while the unsupervised clustering still groups the pair. A carried pair absent
from the new month's candidate set (or whose leg left the universe / has a non-delisting data
gap) is force-closed at the new month's first trading day. Its P&L was already booked through
the prior month-end, so the close-out adds only a diagnostic Trade record, no portfolio P&L.
(Known minor simplification: dropped-carry close-outs are not charged an exit transaction
cost — rare event, second-order, conservative.)

## D5.4 — Portfolio P&L is independent of carry bookkeeping

Portfolio returns are aggregated from each pair's per-month daily return series, which is
byte-unchanged. `CarryState.cum_pnl` feeds ONLY the diagnostic `Trade.round_trip_return`. So
a carry bookkeeping bug can move a diagnostic field but never the headline Sharpe — this is
what makes the change low-risk.

Side fix: the round-trip P&L is now accumulated per-trade instead of sliced as
`daily_pnl[-days_in_position:]`. The old slice mis-attributed P&L for intra-month re-entries
(a latent diagnostic bug); portfolio returns were and remain unaffected.

## D5.5 — Backward compatibility verified

- 53 prior tests pass unchanged; 6 new carry tests (`tests/test_backtest_carryover_synthetic.py`)
  pass → suite 59/59.
- Bit-identical: SSD core, `carry_over=False`, 2003 (11 months) vs the frozen
  `phase1/results/ssd_core_monthly.parquet` → **max |Δ monthly_return| = 0.0**.

## D5.6 — Grid design

Cells (OPTICS, OLS hedge, equal allocation), run by `01_run_carryover_grid.py`:
- **E** carry + frictionless → signal effect vs stored **F** (no-carry frictionless core).
- **B** no-carry + passive (no stop) / **D** carry + passive (no stop) → operating point.
- PC-only sensitivity: **Dstop** (carry+passive+3.5σ), **Dmkt** (carry+marketable).
F is NOT re-run (reuses phase1/2/2.5 core parquets). Forward test (`03_`) runs frozen E/D on
2024-2025. Lookahead re-audit with carry on is expected to pass unmodified (the comparator
already restricts to `< cut_date` and reconstruction is span-agnostic).

## Build status (2026-06-08)
- [x] `src/config.py` — `CARRY_OVER`, `MAX_CARRY_MONTHS=3`, `PHASE5_DIR`.
- [x] `src/costs.py` — shared `REALISM_FULL` / `REALISM_PASSIVE` presets.
- [x] `src/backtest.py` — `CarryState`, carry threaded through `simulate_pair_in_month` /
      `run_one_month` / `run_backtest`; default `carry_over=False` bit-identical.
- [x] `tests/test_backtest_carryover_synthetic.py` — 6 tests; suite 59/59.
- [x] `phases/phase5/notebooks/` — `01_run_carryover_grid`, `02_evaluate_phase5`,
      `03_forward_test_carry`.
- [~] Grid runs (user). **PC-E (carry frictionless) DONE 2026-06-08.**

## D5.7 — PC-E result: carry is a COST lever, not a signal lever

PC frictionless, carry (E) vs no-carry (F):
- Sharpe **1.019 vs 1.028** — essentially flat (marginally down). The "force-close is a drag,
  removing it lifts Sharpe" thesis is **NOT** a free signal gain.
- Mechanism confirmed working: force-close **91.5%→62.8%**, reversion **8.3%→36.7%**, mean
  hold **19.9→49.7 days**, trades **30,050→18,318 (−39%)**, max-DD **−5.7%→−3.9%**, Calmar
  0.594→0.639. (Sortino 2.531→2.091 and hit 62.2%→59.8% slightly worse — carry holds some
  losers longer.)
- **Implication:** the payoff is the 39% turnover cut, which only shows up NET of costs.
  Decisive test is therefore **D vs B** (passive). The "stop if PC-E ≤ 1.028" fail-fast rule
  was too blunt — proceed to the realism cells, where carry must earn its keep.

## D5.8 — PC D-vs-B result: carry CONFIRMED as a net-of-cost win

PC passive (no stop), carry (D) vs no-carry (B):
- Sharpe **0.900 vs 0.855 → +0.045**. Carry-over earns its keep once costs are on.
- Decomposition: frictionless −0.009 (E−F) → passive +0.045 (D−B), a **+0.054 swing = pure
  turnover savings** from the 39% fewer trades. Carry is a cost lever and the cost paid off.
- Max-DD **−6.8%→−4.2%**, Calmar **0.380→0.492** (better). Sortino 1.951→1.775 (holds losers
  longer) but aggregate DD improved, so net tail is better not worse. Hit 59.0% both.
- Stacking passive + no-stop + carry moved net PC Sharpe **0.572 (Phase-4 baseline) → 0.900**,
  recovering most of the gap to frictionless 1.028.
- **Verdict:** both L1-audit improvements (carry-over + passive execution) validated on PC.
  Next: extend to factor/ssd + the Dstop/Dmkt sensitivity, then the frozen forward test.

## D5.9 — Full PC sensitivity + factor: carry-over is METRIC-DEPENDENT

**PC sensitivity (passive):** D (carry, no-stop) **0.900** > Dstop (carry+3.5σ) **0.812**
> Dmkt (carry, marketable) **0.823** > B (no-carry) 0.855. Two design choices confirmed
under carry: the 3.5σ stop still hurts (−0.088 vs D), and marketable execution costs ~0.077
of Sharpe vs passive. Best PC operating point = **carry + passive + no-stop = 0.900**
(Phase-4 baseline was 0.572).

**Factor — carry HURTS:** frictionless E 0.929 < F 1.013 (−0.084); passive D 0.823 < B 0.877
(−0.054). The 32% turnover saving does not compensate. Root cause = the same one behind
factor's OOS collapse (0.117): factor-beta loadings DRIFT over weeks, so holding a position
longer exposes it to a decaying relationship. (Carry still improved factor's drawdown
−5.4%→−4.0% and Calmar 0.525→0.562, but Sharpe/Sortino fell.)

**Synthesis (PC + factor):** carry-over pays when the ENTRY RELATIONSHIP IS STATIONARY — it
helps PC (idiosyncratic-correlation signal, stable; +0.045 net) and hurts factor (drifting
betas; −0.054 net). Same stationarity axis that made PC generalise OOS and factor fail.

## D5.10 — SSD completes the grid: carry wins BOTH frictionless and net

SSD carry vs no-carry:
- Frictionless: E **0.686** vs F **0.589** → **+0.097** (helps even before costs).
- Passive net-of-cost: D **0.649** vs B **0.572** → **+0.077**.
- Max-DD: frictionless −14.3%→−6.8% (halved); passive −9.9%→−7.0%. force-close 88.4%→65.5%,
  hold 19.2→38.7d, trades 12,255→8,350.

SSD benefits MOST because it had the worst baseline (highest force-close 88.4%, deepest
drawdown −14.3%): the crudest selector (raw normalized price-path similarity) picks
structurally-linked, SLOW-reverting pairs that force-close was guillotining before they
completed. Carry lets them finish.

### Final grid (in-sample 2003-2023), frictionless E−F / net-of-cost D−B
| metric | E−F (frictionless) | D−B (net) | verdict |
|--------|--------------------|-----------|---------|
| SSD    | +0.097 (.589→.686) | +0.077 (.572→.649) | carry wins big, both |
| PC     | −0.009 (1.028→1.019) | +0.045 (.855→.900) | carry wins net-of-cost |
| factor | −0.084 (1.013→.929) | −0.054 (.877→.823) | carry hurts |

**Two robust findings:**
1. **Carry-over pays in proportion to (a) how persistent the entry relationship is and (b) how
   much premature-cut damage the force-close was doing.** SSD (persistent + worst baseline)
   gains most; PC (persistent + clean) gains via cost only; factor (drifting betas) loses.
2. **Carry improves drawdown for ALL THREE metrics** (SSD −14.3→−6.8, PC −5.7→−3.9, factor
   −5.9→−3.9 frictionless) — even where Sharpe falls. Month-end is an arbitrary clock;
   force-closing into a dislocation crystallises a mark-to-market loss that would have
   reverted. This is metric-independent and is the cleanest single result of Phase 5.

**In-sample config (SUPERSEDED OOS — see D5.11):** carry ON for SSD and PC, OFF for factor.

## D5.11 — Forward test (2024-2025): carry-over does NOT generalise ⚠️

Frozen carry cells, 23 unseen months:
| metric | E carry frictionless | D carry passive | in-sample E/D | Phase-4 no-carry OOS |
|--------|----------------------|-----------------|---------------|----------------------|
| pc     | **0.434**            | 0.366           | 1.019 / 0.900 | **0.858**            |
| factor | −0.016               | −0.019          | 0.929 / 0.823 | 0.117                |
| ssd    | 0.071                | 0.098           | 0.686 / 0.649 | —                    |

- **PC carry frictionless OOS (0.434) ≈ HALF the no-carry PC OOS (0.858, Phase 4, same
  data/code).** The in-sample +0.045 net gain does NOT survive — carry roughly halved PC's
  OOS Sharpe. SSD collapses 0.686→~0.08. Factor weak either way.
- **Mechanism / caveat:** 2024-2025 was a strongly trending, low-dispersion regime — adverse
  for ALL mean-reversion (PC no-carry itself fell 1.028→0.858) and worst for LONG holds. Carry
  (19.9→49.7d) was punished hardest: patient reversion is the wrong bet when dislocations keep
  widening. So the failure may be REGIME-SPECIFIC (carry helps mean-reverting regimes, hurts
  trending ones) — coherent but unproven on one 23-mo window.
- **Anomaly flagged + TRACED:** SSD D (passive, costs-on) 0.098 > E (frictionless) 0.071
  violates cost monotonicity (10/23 months D>E). Root cause = the equal-weight aggregation in
  `run_one_month` masks "open today" with `all_returns != 0`. A frictionless ENTRY day has P&L
  exactly 0 → excluded from the daily mean; with costs it's `0 − entry_cost ≠ 0` → included. So
  cost-on vs frictionless average over slightly DIFFERENT pair sets on entry/exit days, breaking
  monotonicity. **Pre-existing (affects all realism cells incl. Phase-4), NOT a carry bug.**
  Immaterial to the headline (PC obeys D<E; the 0.434-vs-0.858 gap is huge). Only visible for
  SSD because its OOS returns are ~0, so the quirk dominates. Clean fix: mask "open" by position
  (the existing `daily_weight` panel), not by `return != 0` — pre-existing scope, would shift
  Phase-4 realism slightly. Does not change any Phase-5 conclusion.

**VERDICT:** Phase-5 carry-over is an in-sample improvement that FAILS to generalise on
2024-2025. **Do NOT ship as a default.** The OOS test did its job (cf. factor in Phase 4:
sturdy in-sample, failed OOS). Still standing: (a) passive execution is a separate lever;
(b) the universal in-sample DRAWDOWN reduction (all 3 metrics) — but OOS returns are too small
to confirm it held. Open question worth one run: is carry's value regime-conditional?

### To confirm apples-to-apples (recommended next)
Run no-carry OOS baselines (carry_over=False) frictionless + passive on 2024-2025 for all 3
metrics in THIS harness, so the carry-vs-no-carry OOS delta doesn't rely on cross-referencing
the Phase-4 number. Expected: confirms PC no-carry ≈ 0.858 ≫ carry 0.434.
