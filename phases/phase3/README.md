# Phase 3 — Robustness Cells

**Status:** 🔵 Build complete + tested (44/44); 8-cell robustness grid pending (user runs
`notebooks/01_run_robustness_grid.py`). Started 2026-06-07.

Phase 3 turns the headline point estimates (**PC core 1.028**, **factor-beta core 1.013**)
into **confidence intervals** by re-running each under deliberately different
methodological choices. If the Sharpes survive, the result is robust; if a cell breaks it,
that's itself a finding (it identifies the load-bearing choice).

> New folder; Phases 1/2/2.5 frozen. `src/` extended with backward-compatible knobs only —
> the default path is bit-identical to before (verified: pc 2003-Q1 = 160 trades unchanged),
> Phase 1 invariant and all prior tests intact.

## The four robustness dimensions

| Cell | Perturbs | Default (unchanged) | Variant | Code |
|---|---|---|---|---|
| **3a** | clustering algorithm | OPTICS | HDBSCAN; hierarchical (avg-linkage, quantile cut) | `clustering.cluster_hdbscan/_hierarchical`, backtest `clusterer=` |
| **3b** | hedge ratio (spread) | OLS γ | RLM (Huber) robust γ | `spread.fit_hedge_ratio_rlm`, backtest `hedge_method=` |
| **3c** | position sizing | equal-weight | \|entry-z\|-weighted | backtest `allocation="zweight"` |
| **3d** | locked hyperparams | point values | sweep → stability/band | `03_xi_alpha_sensitivity.py` |

Design notes in `decisions.md` (esp. D3a: **Ward unavailable for precomputed distances** →
average linkage; and the **quantile cut** that makes hierarchical scale-adaptive instead of
drifting over the 21-year sample).

## 3d sensitivity (done — no backtest needed)

Locked hyperparameters sit on a **flat plateau**, not a cliff edge:
- **PC xi**: 74–84 clusters across xi 0.02→0.06 (locked 0.04), purity ~0.93.
- **Factor ridge-α**: 59–83 clusters across α 0.25→4.0 (locked 1.0), purity ~0.88–0.95.

→ the headline is not an artefact of a knife-edge hyperparameter.

## Robustness band — _grid pending_

Baselines are frozen (PC 1.028 from phase2, factor 1.013 from phase2_5); only the 8
variant cells are run here. Populate with `02_compare_robustness.py` after the grid.

### PC (headline 1.028) — COMPLETE
| Variant | Sharpe | Δ vs baseline | Read |
|---|---:|---:|---|
| baseline (OPTICS/OLS/equal) | 1.028 | — | — |
| 3b RLM hedge | 1.046 | +0.018 | ✅ robust |
| 3c z-weighted | 1.012 | −0.016 | ✅ robust |
| 3a HDBSCAN | 0.616 | −0.412 | ⚠️ sensitive |
| 3a hierarchical | 0.485 | −0.543 | ⚠️ sensitive |

**Band 0.485–1.046.** Robust to hedge ratio (RLM) and allocation (z-weight); **sensitive to
the clustering algorithm.**

### Factor-beta (headline 1.013) — COMPLETE
| Variant | Sharpe | Δ vs baseline | Read |
|---|---:|---:|---|
| baseline (OPTICS/OLS/equal) | 1.013 | — | — |
| 3b RLM hedge | 1.060 | +0.047 | ✅ robust |
| 3c z-weighted | 1.027 | +0.014 | ✅ robust |
| 3a hierarchical | 0.991 | −0.022 | ✅ robust |
| 3a HDBSCAN | 0.615 | −0.398 | ⚠️ sensitive |

**Band 0.615–1.060.** Robust to RLM hedge, z-weight, AND hierarchical clustering; only
HDBSCAN dilutes.

## Key finding

Both headlines survive the **hedge-ratio** (RLM) and **position-sizing** (z-weight)
perturbations essentially unchanged. The sensitivity is to the **clustering algorithm**, and
it traces to **selectivity**: HDBSCAN and hierarchical produce **~3× more candidate pairs**
than OPTICS (denser clusters, comparable purity), so they trade a larger, more **diluted** set.

Crucially, the two metrics differ in *how* sensitive:
- **PC** drops under both HDBSCAN (0.616) and hierarchical (0.485, MDD −11.8%).
- **Factor-beta** drops only under HDBSCAN (0.615); its hierarchical variant **holds** (0.991,
  MDD −3.9%, its tightest).

So **factor-beta is the more robust metric** — 3 of 4 perturbations leave it ~1.0, vs PC's 2
of 4. Honest conclusion: *the strategy is robust to spread construction and position sizing;
its headline depends on clustering selectivity, and the factor-beta extension is sturdier
across clustering choices than the paper's PC metric.* (Hyperparameter sweep shows xi/ridge-α
on a stable plateau, not a cliff.)

## How to run

```bash
# 8 backtests (~1-3h each; edit CELLS to run a subset). Runs in background:
nohup python phases/phase3/notebooks/01_run_robustness_grid.py \
  > phases/phase3/results/phase3_grid_log.txt 2>&1 &

python phases/phase3/notebooks/02_compare_robustness.py   # Sharpe bands
python phases/phase3/notebooks/03_xi_alpha_sensitivity.py  # already runnable (no backtest)
```

## Files
| Path | What |
|---|---|
| `decisions.md` | design decisions + build log |
| `notebooks/01_run_robustness_grid.py` | runs the 8 variant cells |
| `notebooks/02_compare_robustness.py` | per-metric Sharpe band |
| `notebooks/03_xi_alpha_sensitivity.py` | 3d hyperparameter plateau check |
| `src/clustering.py` | `cluster_hdbscan`, `cluster_hierarchical` |
| `src/spread.py::fit_hedge_ratio_rlm` | robust hedge ratio |
| `src/backtest.py` | `clusterer` / `hedge_method` / `allocation` dispatch |
