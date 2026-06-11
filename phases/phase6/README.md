# Phase 6 — Implementation-Correctness Fixes (code-review corrections)

**Status: engine + tests complete · core ladder run (a1, a2, b1, c0, c1 done; rest optional) — 2026-06-10**

A line-by-line review of `src/` (2026-06-10) found six implementation flaws — none of
them lookahead bias (the 6/6 audit stands), but several touch reported numbers. Phase 6
fixes each one **behind its own engine flag** (defaults = bit-identical Phase 5 engine,
verified by the unchanged pre-existing test suite) and re-runs the affected cells **one
fix at a time** so every Sharpe change is attributable.

Full rationale per fix: [`decisions.md`](decisions.md). Synthetic tests:
`tests/test_corrections_synthetic.py` (8/8 passing; full suite 67/67).

## The corrections

| ID | Flag | What was wrong | Cells affected |
|---|---|---|---|
| D6.1 | `coint_pvalue="mackinnon"` | Engle-Granger used raw-ADF p-values (anti-conservative on estimated residuals) instead of MacKinnon's cointegration distribution | filtered |
| D6.2 | `delisting_fix=True` | Fallback code map shifted one CRSP bucket (500s "dropped" got −5% instead of Shumway's −30%; the v2/OOS sentinel is 500!); dlret overwrote instead of compounding; weekend dlstdt silently never applied; NaN-z delist day skipped the close | all (esp. OOS) |
| D6.3 | `stop_cooldown=True` | Stop-out at 3.5σ re-entered the same position next day (|z| still ≥ 2) → stop = pure churn; contaminates the "stop is net-negative" finding | realism |
| D6.4 | `block_last_day_entry=True` | Last-day entries force-closed at the same close; their costs + trade records were silently dropped (`days_open == 0` gate) | realism / carry |
| D6.5 | `execution_delay=1` | Fills at the signal close (can't observe a close and trade at it); first P&L day was t+1 but fill price was day-t close | all |
| D6.6 | `use_coint_gamma=True` | Filter validated the min-p direction's spread, engine traded A-on-B γ regardless | filtered |
| D6.7 | (always on) | NaN pair-day returns silently skipped by `.mean(skipna)` while counting as "open" → now a loud warning + explicit fill | none (guard) |

## Running the ladder

```bash
# list cells + recommended order
python phases/phase6/notebooks/01_run_corrections_ladder.py --list

# highest-information cells first (~1-2h each):
python phases/phase6/notebooks/01_run_corrections_ladder.py \
  --cells b1_filt_mackinnon,a1_core_delist,c0_real_baseline,c1_real_cooldown,a2_core_delay

# or everything (≈12-20h):
nohup python phases/phase6/notebooks/01_run_corrections_ladder.py --all \
  > phases/phase6/results/phase6_ladder_log.txt 2>&1 &

# compare (also adds an excess-of-rf Sharpe column):
python phases/phase6/notebooks/02_evaluate_corrections.py

# reporting analyses (no backtest, run on existing parquets):
python phases/phase6/notebooks/03_bootstrap_cis.py        # Sharpe 95% CIs + paired Δs
python phases/phase6/notebooks/04_regime_dispersion.py    # dispersion-regime study
```

Cells already on disk are skipped (`--force` to re-run), so the ladder can be run in
several sittings.

## Results

| Cell | Fix | Sharpe | Δ vs baseline | Verdict |
|---|---|---:|---:|---|
| phase2 pc_core (baseline) | — | 1.028 | — | reference |
| a1_core_delist | D6.2 | 1.007 | −0.021 | ✅ headline **robust** to the delisting correction |
| a2_core_delay | D6.5 | 0.882 | −0.145 | ✅ **edge survives honest fills** (see finding 5) |
| a3_core_allfix | D6.2+5+4 | | | optional |
| phase2 pc_filtered (baseline) | — | 0.752 | — | reference (raw ADF = paper convention) |
| b1_filt_mackinnon | D6.1 | 0.299 | −0.453 | ❌ the **correct** EG test guts the filter (see finding 1) |
| b2_filt_gamma | D6.6 | | | optional (pairs with raw ADF) |
| b3_filt_allfix | all | | | skipped — superseded by b1 finding |
| c0_real_baseline | — | 0.572 | — | reference; reproduces Phase 4's 0.572 exactly |
| c1_real_cooldown | D6.3 | 0.566 | −0.005 | ✅ churn removed (37.6k → 30.7k trades), Sharpe unchanged (see finding 2) |
| c2_real_lastday | D6.4 | | | optional (expected ≈ 0) |
| c3_real_allfix | all | | | optional |

## Findings so far (2026-06-10)

1. **The cointegration filter's apparent viability was a statistical artifact.**
   With MacKinnon p-values the pass rate drops 26% → 11% of candidates (Dec 2015:
   83 → 36 pairs; MacKinnon-kept is a strict subset, p-values uniformly ~3× larger),
   pairs traded fall 36 → 16/month, and Sharpe collapses 0.752 → 0.299. Trade-level
   decomposition of the Phase 2 filtered cell: pairs that also pass MacKinnon earned
   **+12.1 bps/trade** vs **+19.3 bps/trade** for raw-ADF-only pairs (hit rates equal,
   53.6%/53.5%) — stronger cointegration evidence does **not** select better trades;
   the filter's only effect is shrinking diversification. Together with the existing
   dose-response (core 1.028 → weak filter 0.752 → correct filter 0.299), conclusion:
   the edge is short-horizon dislocation reversion, not EG-style equilibrium reversion.
   The raw-ADF cell stays in the 2×2 as the paper-faithful replication arm (the paper
   very likely used the same anti-conservative test — our raw-ADF cell matches its
   target); the writeup must describe it as a ~12–15%-effective threshold, not "5%".
2. **"The stop is net-negative" survives its fairness test.** The cooldown removes the
   stop→re-enter churn (−18% trades) but Sharpe is unchanged (0.572 → 0.566, paired
   bootstrap P(Δ≤0) = 47%). Phase 4's recommendation to drop the stop stands, now
   clean of the implementation artifact.
3. **Error bars change two writeup claims** (`03_bootstrap_cis.py`, block bootstrap):
   pc_core 1.028 [0.60, 1.43]; OOS pc 0.858 [0.18, 1.63] — *consistent with* in-sample
   (P(OOS≥IS) = 33%), strengthening the generalisation claim; but OOS factor 0.117
   [−0.52, +1.12] — 23 months **cannot** distinguish failure from ~1.0, so soften
   "factor-beta does not generalise" to "no OOS evidence either way yet". The
   core-vs-filter gap is borderline-significant (P=5.8%); filtered-vs-MacKinnon P=3.4%.
4. **Dispersion regime dependence: mechanism confirmed, timing signal weak**
   (`04_regime_dispersion.py`): contemporaneous dispersion quartiles are cleanly
   monotone (Sharpe 0.73/0.71/1.03/1.56) — the strategy eats dislocations. But the
   *deployable* (prior-month) signal is non-monotone (1.02/1.32/0.54/1.51): only the
   extreme top quartile is reliable, so CONCLUSIONS' "trade only when dispersion is
   elevated" should read "**scale up** in high-dispersion regimes; a binary on/off
   rule would have skipped calm months that earned ~1.0–1.3". GFC window = 10% of
   months, 28% of arithmetic return.
5. **The edge survives honest fill timing.** With fills delayed to the close AFTER the
   signal (`execution_delay=1`), pc_core drops 1.028 → **0.882** — a real cost
   (paired bootstrap Δ = 0.145, 95% CI [0.02, 0.31], P(Δ≤0) = 1.4%), but the strategy
   stays clearly profitable (CI [0.45, 1.26], P(SR≤0) = 0.0%), with *lower* vol
   (3.07% vs 3.32%) and a shallower max drawdown (−5.1% vs −5.7%). ~86% of the
   frictionless headline is genuine signal, ~14% was the fill-at-signal-close
   convention. This mirrors Gatev et al.'s one-day-waiting result and preempts the
   most likely examiner challenge.

## After the ladder

1. Update this table + `CONCLUSIONS.md` (especially the "stop is net-negative" claim —
   re-judge it from c1, and the OOS numbers if D6.2 moves them).
2. Optional: re-run the 2024–25 forward test with `delisting_fix=True` (the v2 sentinel
   bug lives exactly there) via the Phase 4 `05_forward_test.py` pattern.
3. Out of scope, documented in `decisions.md`: reporting-convention changes,
   multiple-testing correction, capacity/liquidity modelling.
