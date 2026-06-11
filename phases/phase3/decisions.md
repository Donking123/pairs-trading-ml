# Phase 3 — Decisions Log (Robustness Cells)

**Status:** 🔵 IN PROGRESS — started 2026-06-07.

Phase 3 turns the headline point estimates into **confidence intervals** by re-running
the strategy under deliberately different methodological choices. If PC core 1.028 and
factor-beta core 1.013 survive these perturbations, the result is robust; if they swing
wildly, we learn which choice was load-bearing.

This is a NEW phase — Phases 1/2/2.5 artifacts are frozen. `src/` is EXTENDED with
backward-compatible additions only: every new knob defaults to current behavior, so the
Phase 1 invariant (ssd_core 0.589) and the 39-test suite stay intact.

## The four robustness dimensions (user-selected)

| Cell | Perturbs | Default (unchanged) | Variant |
|---|---|---|---|
| **3a Clustering algorithm** | how stocks are grouped | OPTICS | HDBSCAN, hierarchical (avg-linkage) |
| **3b Hedge ratio** | spread construction | OLS γ | RLM (Huber) robust γ |
| **3c Allocation** | position sizing | equal-weight across open pairs | \|entry-z\|-weighted |
| **3d Hyperparameter sensitivity** | the locked constants | xi / ridge-α point values | sweep → Sharpe band |

## Backtest dispatch design (D3.0)

`run_one_month` / `run_backtest` gain three new args, all backward-compatible:

```
clusterer:    "optics" (default) | "hdbscan" | "hierarchical"
hedge_method: "ols"    (default) | "rlm"
allocation:   "equal"  (default) | "zweight"
```

Existing calls (no new args) reproduce Phase 1/2/2.5 results bit-for-bit.

## D3a — Clustering algorithm

- **HDBSCAN** — `sklearn.cluster.HDBSCAN(metric="precomputed", min_cluster_size=2)`.
  Density-based like OPTICS but with a different cluster-extraction rule; auto-determines
  cluster count (no xi-equivalent to tune).
- **Hierarchical** — `AgglomerativeClustering(metric="precomputed", linkage="average",
  distance_threshold=τ)`. **Ward is NOT usable** — it requires raw Euclidean features,
  but SSD/PC give precomputed distance matrices. Average-linkage is the correct choice
  for precomputed distances. τ is tuned per metric on Dec-2023 to give a cluster count
  comparable to OPTICS (parallels how xi was tuned). Singletons → label −1.

## D3b — RLM hedge ratio

Replace the closed-form OLS γ with `statsmodels.RLM` (Huber's T norm) so a few outlier
days don't tilt the hedge ratio. Same spread/z-score machinery downstream. Robustness
question: does the spread definition matter?

## D3c — \|entry-z\|-weighted allocation

Currently the daily portfolio return is the equal-weight mean across pairs open that day.
Variant: weight each open pair by the **\|entry-z\| of its currently-open trade** (bet
more on stronger dislocations), renormalized daily. `simulate_pair_in_month` gains a
parallel per-day weight series (\|entry_z\| while in position, 0 when flat).

## D3d — Hyperparameter sensitivity

Sweep OPTICS `xi` (and factor-beta `ridge-α`) around their locked values and report the
resulting Sharpe band, so the headline reads "1.0xx ± y" rather than a single point.
Cheap Dec-2023 cluster-count sweeps frame it; a few full backtests anchor the band.

## CP3 (gate)

No paper benchmark. Success = the headline Sharpes (PC 1.028, factor 1.013) stay within a
defensible band (target: within ±0.15) across 3a–3c, and 3d quantifies the band. Report
*whatever* happens — a cell that breaks the result is itself a finding.

## Build status (2026-06-07)

- [x] 3a clustering — `cluster_hdbscan`, `cluster_hierarchical` (quantile-based) + backtest
      `clusterer=` dispatch. Tuned HIER_QUANTILE=0.01 (scale-adaptive, fixes the fixed-τ
      drift problem). 5 new tests; suite 44/44.
- [x] 3b RLM — `spread.fit_hedge_ratio_rlm` (Huber T) + backtest `hedge_method=` dispatch.
- [x] 3c allocation — `simulate_pair_in_month` now returns a per-day \|entry_z\| weight series;
      backtest `allocation="zweight"` renormalizes daily. Default "equal" bit-identical.
- [x] All three verified: 44/44 tests; default path reproduces prior trades exactly
      (pc 2003-Q1 = 160 trades unchanged); rlm/zweight produce expected deltas.
- [x] 3d sensitivity — `03_xi_alpha_sensitivity.py`. Dec 2015/2023 sweep shows the locked
      xi (PC) and ridge-α (factor) sit on a STABLE plateau (cluster count + purity flat),
      not a cliff edge → headline not a knife-edge artefact. Full Sharpe band = re-run grid
      at a few config values.
- [x] grid runner `01_run_robustness_grid.py` (8 cells) + comparison `02_compare_robustness.py`
      (prints per-metric Sharpe band; reproduces baselines pc 1.028 / factor 1.013).
- [x] `phase3_complete_reference.ipynb` — polished deliverable (concept → 3d plateau (live)
      → band scorecard, reads results dynamically). Executed; baselines + 3d baked, variants
      [pending]. Re-run `_build_phase3_reference_notebook.py` after the grid to fill the band.
- [x] **8-cell grid COMPLETE (2026-06-07).** PC band 0.485–1.046, factor band 0.615–1.060.
      Finding: robust to RLM hedge + z-weight; clustering is the load-bearing choice (HDBSCAN
      dilutes both; PC also breaks under hierarchical, factor does NOT — factor-beta is the
      sturdier metric). README + writeup tables filled. PHASE 3 COMPLETE.

## Build COMPLETE (2026-06-07) — backtests pending

All engine changes done, 44/44 tests, backward-compatible (default path bit-identical).
3d sensitivity shows hyperparameter stability. Remaining = run `01_run_robustness_grid.py`
(8 backtests, ~1-3h each) then `02_compare_robustness.py` to read the Sharpe bands.

## Robustness grid (what to run)

Baselines already exist (don't re-run): pc_core 1.028, factor_core 1.013.
New cells = {pc, factor} × {hdbscan, hierarchical, rlm, zweight} = 8 backtests.
Each ~1-3h (hdbscan/hierarchical are denser → slower). Run subsets via the CELLS list.
