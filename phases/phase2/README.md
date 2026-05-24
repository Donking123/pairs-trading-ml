# Phase 2 — PC Distance + Cointegration Filter

**Status:** ✅ **COMPLETE 2026-05-24** — paper replicated. PC core Sharpe **1.028** vs paper **1.01** (Δ +0.018, within ±0.15 tolerance).

This phase adds the **second distance metric** (PC, the paper's *winning* metric) and
the **Engle-Granger cointegration filter** to the existing pipeline. It targets the
paper's headline result (Sharpe 1.01) and the specific *force-close-drag-reduction*
lever identified in Phase 1's P&L attribution.

> **📖 Before starting:** read [`carryover-from-phase1.md`](carryover-from-phase1.md) —
> the consolidated walkthrough of what's settled, what's still open, and what
> discipline to apply. Six items: (1) pipeline soundness, (2) bimodal duration
> finding, (3) locked decisions, (4) regime dependency, (5) methodology, (6) open
> Phase 2 questions.

## Final scorecard (2026-05-24)

| Cell | Ours Sharpe | Paper | Δ | Verdict |
|---|---:|---:|---:|---|
| **PC core** | **1.028** | **1.01 ±0.15** | **+0.018** | **✅** |
| PC + filter | 0.752 | 0.80 ±0.15 | −0.048 | ✅ |
| SSD + filter | 0.731 | 0.75 ±0.15 | −0.019 | ✅ |
| SSD core | 0.589 | 0.88 ±0.15 | −0.291 | ❌ (same as Phase 1) |

**Phase 1 invariant ✅**: `ssd_core` matches Phase 1 to Δ −0.0005.

### Bimodal lever — moved as predicted

| Cell | force_close mean | Outliers \|rt\| > 50% |
|---|---:|---:|
| SSD core | −32 bps | 5 |
| **PC core** | **−11 bps** ⭐ | 3 |
| PC + filter | −18 bps | **0** ⭐ |

PC core cut the force-close drag by 65% per trade. PC + filter eliminated *every*
outlier trade in 21 years.

### Read the deliverables

- `notebooks/phase2_complete_reference.ipynb` — concept → worked examples → real-data
  outputs → 4-cell scorecard → CP2 verdict → Phase 3 roadmap (32 cells, all rendered)
- `notebooks/phase2_pnl_attribution.ipynb` — bimodal lever check, regime-by-regime
  Sharpe, sector/direction attribution, cross-cell pair overlap (18 cells, all rendered)

---

## Why this phase (from Phase 1 evidence)

Phase 1's P&L attribution revealed a **bimodal duration pattern**: the strategy's
+30.4 net per-trade P&L comes from the 11.4% of trades that fully revert (+471 bps
each) being partly offset by the 88.4% of trades that get force-closed (−32 bps each):

$$
\underbrace{+65.58}_{\text{reversion (11.4\%)}} + \underbrace{-34.25}_{\text{force\_close (88.4\%)}} + \underbrace{-0.92}_{\text{delisting (0.3\%)}} = +30.41
$$

**The single biggest lever to lift Sharpe is reducing the force-close drag.** That's
what Phase 2's cointegration filter is designed to do — reject pairs whose residual
spreads fail the ADF stationarity test (i.e., pairs that won't revert). PC distance
complements this by finding more *idiosyncratic* mean-reverting pairs in the first
place (it strips out market beta, isolating the cointegrating component).

If the cointegration filter halves the force-close drag (−34.3 → −17.2), net P&L
jumps from +30 to +48 and Sharpe rises from **0.59 → ~0.9**, closer to the paper's 1.0.

---

## Goals

### CP2 (Phase 2 gate) — must hit to declare phase complete

| Metric | Paper target | Tolerance |
|---|---:|---:|
| # PC clusters (Dec 2023) | 109 | ±10 |
| Purity vs SIC division (PC, Dec 2023) | 0.84 | ±0.05 |
| # PC + cointegration-filtered pairs (Dec 2023) | 78 | ±10 |
| **Annualised Sharpe — PC core** | **1.01** | **±0.15** |
| Annualised Sharpe — PC + cointegration filter | 0.80 | ±0.15 |

### Deliverable scorecard (2×2 grid)

|  | no cointegration filter | with cointegration filter |
|---|---|---|
| **SSD** | 0.589 (done, from Phase 1) | (Phase 2) |
| **PC** | (Phase 2) | (Phase 2) |

---

## Planned build order

1. **`src/distances.py::pc_distance`** — Partial correlation distance on
   market-adjusted returns.
   - For each stock, regress daily returns on SPY → residual return series.
   - Pairwise: distance = `1 − corr(residual_X, residual_Y)`.
   - Add synthetic test that recovers a planted PC structure.

2. **`src/cointegration.py`** (new module) — Engle-Granger ADF test + half-life filter.
   - `engle_granger(prices_a, prices_b)` → returns ADF p-value, hedge ratio γ_eg,
     residual half-life (Ornstein-Uhlenbeck fit), pass/fail flag.
   - Default thresholds (paper): `p < 0.05`, `5 ≤ half-life ≤ 60` trading days.
   - Add synthetic test with planted cointegrated pair + non-cointegrated control.

3. **Extend `src/backtest.py`** — accept `metric: 'ssd' | 'pc'` and
   `cointegration_filter: bool` arguments.
   - Wire the new metric into the `formation` phase (replace `ssd_distance` call).
   - Insert optional cointegration-filter step between `clusters_to_pairs` and γ-fit.
   - Backward-compatible: existing Phase 1 calls keep working with default args.

4. **Run the 2×2 backtest grid** — 4 backtests total:
   - SSD core / SSD + filter / PC core / PC + filter
   - Save outputs to `phases/phase2/results/{ssd_core, ssd_filtered, pc_core, pc_filtered}_{monthly, trades}.parquet`.

5. **Build `notebooks/phase2_complete_reference.ipynb`** — same structure as
   Phase 1's reference notebook:
   - PC distance concept + worked example + Dec-2023 real-data clusters
   - Cointegration filter concept + worked example + Dec-2023 acceptance rate
   - 4-way scorecard with CP2 verdict
   - Updated force-close-drag arithmetic showing whether the lever moved

6. **Build `notebooks/phase2_pnl_attribution.ipynb`** — same diagnostic as Phase 1:
   - Did force-close drag actually shrink?
   - Did the bimodal pattern soften or sharpen?
   - Pareto / concentration: more diversified or more concentrated?

7. **Update `phases/phase2/decisions.md` and `phases/phase2/README.md`** with results.

---

## How to read this phase (once complete)

To **understand what we did and why**: open `README.md` (this file) and `decisions.md`.

To **see the build**: open `notebooks/phase2_complete_reference.ipynb`.

To **see the attribution**: open `notebooks/phase2_pnl_attribution.ipynb`.

To **compare with Phase 1**: open both `phases/phase1/README.md` and this file side by side.

---

## Cross-phase references

- Phase 1 baseline numbers: `phases/phase1/README.md`
- Phase 1 attribution (the lever evidence): `phases/phase1/notebooks/phase1_pnl_attribution.ipynb`
- Shared concepts / glossary: `notes/concepts-walkthrough.md`
- Paper-vs-proposal decisions: `notes/strategy-reconciliation.md`
