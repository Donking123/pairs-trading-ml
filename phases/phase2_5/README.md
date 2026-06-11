# Phase 2.5 — Factor-Beta Clustering Extension (QF621 contribution)

**Status:** 🔵 Build complete + validated; full backtest pending (run
`notebooks/01_run_factor_backtest.py`). Started 2026-06-07.

This phase is the QF621 group's **original contribution** beyond the Rotondi & Russo
(2025) replication. It adds a **third distance metric** to the pluggable pipeline:
cluster stocks by their **risk-factor exposure (beta) vector** instead of by price
trajectory (SSD, Phase 1) or idiosyncratic-return correlation (PC, Phase 2).

> Phase 2 artifacts (`phases/phase2/`) are **frozen** — this is a new folder. The
> shared `src/` modules were extended with backward-compatible additions only; the
> Phase 1 invariant (ssd_core Sharpe 0.589) and the full test suite still pass
> (39/39, incl. 7 new factor tests).

---

## The idea

SSD asks "do these two stocks' prices move together?" PC asks "do their
*market-stripped* returns move together?" Factor-beta asks a different question:
**"are these two stocks exposed to the same risks?"**

For each stock we ridge-regress its daily returns on 18 risk factors and keep the
**beta vector** (its loadings). Two stocks with similar betas react the same way to a
shared shock (an oil move, a rate move, a tech selloff), so when their spread
dislocates it should mean-revert. We cluster on these beta vectors (OPTICS), then
trade within-cluster pairs with the exact same engine as Phases 1–2.

## Factor set (18, all self-contained — see `decisions.md` D2.5.1)

| Group | Count | Source |
|---|---:|---|
| Style (FF5 + momentum: `mktrf, smb, hml, rmw, cma, umd`) | 6 | `data/ff_factors.parquet` |
| Industry (Fama-French 12, equal-weight returns from our universe) | 12 | built from CRSP `siccd` |

No external ETF / commodity pull — every factor traces to our own WRDS/FF data, which
keeps the survivorship-bias discipline intact and avoids ETF coverage gaps.

## Method

1. `factors.build_factor_panel` — assemble the 18-factor daily panel for the formation
   window.
2. `distances.ridge_betas` — ridge-regress (α=1.0) each stock's returns on the panel →
   β matrix (stocks × 18). Ridge, not OLS, because the factors are collinear.
3. `distances.factor_beta_distance` — z-score each β dimension across stocks, then
   Euclidean distance between β vectors.
4. OPTICS (`xi=0.10`, the factor-space value) → clusters → within-cluster pairs →
   existing backtest engine.

## Hyperparameters (locked BEFORE any backtest — no overfit)

| Param | Value | Where |
|---|---|---|
| Ridge α | 1.0 | `config.RIDGE_ALPHA` |
| OPTICS xi (factor) | 0.10 | `config.OPTICS_XI_FACTOR` |
| Distance | standardized Euclidean on β | `distances.factor_beta_distance` |

**Dec-2023 validation (pre-backtest):** betas are economically sensible — top Enrgy
loaders APA/MRO/DVN/OXY/FANG, top BusEq NVDA/AMD/LRCX/AMAT, top Money
ZION/CMA/KEY/CFG/FITB. Clusters: 78 (Dec 2023) / 61 (Dec 2015), purity vs SIC division
0.915 / 0.903.

---

## CP2.5 (gate)

No paper benchmark — this is our extension. Success = (a) economically-coherent
clusters ✅ and (b) Sharpe reported **head-to-head** vs PC core (1.028) and SSD core
(0.589). No pass/fail Sharpe target.

## Results scorecard (2026-06-07) — COMPLETE

Full 5-cell grid. Regenerate with `notebooks/02_compare_to_pc.py`.

| Cell | Sharpe | Sortino | Calmar | Ann.ret | Ann.vol | MDD | vs PC peer |
|---|---:|---:|---:|---:|---:|---:|---|
| SSD core (Phase 1 baseline) | 0.589 | 1.102 | 0.211 | 3.01% | 5.28% | −14.3% | — |
| PC core (paper headline) | 1.028 | 2.531 | 0.594 | 3.41% | 3.32% | −5.8% | — |
| PC + cointeg. filter | 0.752 | 1.396 | 0.440 | 2.59% | 3.48% | −5.9% | — |
| **Factor-beta core (EXTENSION)** | **1.013** | **2.598** | **0.605** | 3.60% | 3.55% | −6.0% | **Δ −0.014 vs PC core ✅** |
| **Factor-beta + cointeg. filter** | **0.858** | 1.954 | 0.584 | 4.45% | 5.24% | −7.6% | **+0.106 vs PC filtered ✅** |

**Headline.** Factor-beta clustering — an independent, economically-motivated metric —
**reproduces the paper's ~1.0 Sharpe** (core 1.013 vs PC core 1.028), with marginally
better Sortino/Calmar and fewer, more-selective trades (19,044 vs PC's 30,050). Its
filtered variant (0.858) **beats PC's filtered variant** (0.752) and, like PC+filter,
has **zero** |return|>50% outlier trades in 21 years. Both factor cells match the PC
force-close lever (−12 / −15 bps vs −11 / −18). Confirms the headline result is not an
artifact of one specific similarity measure.

Both metrics pair overwhelmingly within-industry (~80% of trades); factor-beta concentrates
its P&L a bit more in financials (Money/Money ≈27% of P&L vs PC's ≈23%).

---

## Files

| Path | What |
|---|---|
| `decisions.md` | Design decisions (factor set, FF12, ridge, distance, xi) + build status |
| `notebooks/01_run_factor_backtest.py` | Runs factor_core + factor_filtered → `results/` |
| `notebooks/02_compare_to_pc.py` | Head-to-head scorecard + bimodal-lever diagnostic |
| `results/` | Parquets + `phase2_5_scorecard.csv` (after the run) |
| `src/factors.py` | Factor panel + FF12 mapper (shared module) |
| `src/distances.py::factor_beta_distance` | The new metric (shared module) |
