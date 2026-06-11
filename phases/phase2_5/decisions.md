# Phase 2.5 — Decisions Log (Factor-Beta Clustering Extension)

**Status:** 🔵 IN PROGRESS — started 2026-06-07.

Phase 2.5 is the QF621 group's **original contribution** beyond the Rotondi & Russo
(2025) replication: cluster stocks by their **factor-exposure (beta) vector** rather
than by price trajectory (SSD) or idiosyncratic-return correlation (PC). Two stocks
with similar betas to the same risk factors should mean-revert against each other.

This is a NEW phase — Phase 2 artifacts (`phases/phase2/`) are frozen and untouched.
`src/` modules are EXTENDED (new functions, backward-compatible) — existing behavior
and the 32 passing tests stay intact.

---

## D2.5.1 — Factor set: SELF-CONTAINED (no external data)

**Decision:** Build the factor panel entirely from data already pulled in Phase 0.
No external ETF / commodity pull.

**Factor set (18 total):**
- **6 style factors** from `data/ff_factors.parquet`: `mktrf, smb, hml, rmw, cma, umd`
  (Fama-French 5-factor + momentum). `rf` excluded — it's the risk-free rate, not an
  exposure.
- **12 industry factors** built internally: equal-weight daily return of the universe
  stocks in each **Fama-French 12-industry** bucket, mapped from CRSP `siccd`.

**Why self-contained (vs external SPDR ETFs + commodity):**
1. **Defensibility** — every factor traces to our own WRDS CRSP pull; nothing external
   to justify in the QF621 defense. Consistent with the project's survivorship-bias
   discipline (991-stock bias-free universe).
2. **No coverage gaps** — internal industry factors span 2000–2023 cleanly. SPDR sector
   ETFs start mid-sample (XLRE 2015, XLC 2018) → NaNs inside formation windows.
3. **Pipeline coherence** — PC distance already market-adjusts vs the S&P index from the
   same data ecosystem; factor-beta clustering stays in-ecosystem.
4. **The contribution is the idea, not the tickers.**

**Deferred:** GLD/USO commodity factors can be added as a Phase 3 robustness cell if a
grader asks for orthogonal (non-equity) exposure.

## D2.5.2 — Industry granularity: Fama-French 12, not SIC divisions

**Decision:** Map `siccd` → Fama-French 12 industries (FF12), not the 10 one-letter SIC
divisions used in `clustering.py::sic_division`.

**Why:** SIC divisions are too coarse for risk factors — division "D Manufacturing"
lumps pharma, semiconductors, autos, and food together, which have very different factor
behavior. FF12 is the academic standard and separates these (e.g. Chems, BusEq, Hlth,
Manuf as distinct buckets). `sic_division` stays in use for the *purity* metric only.

## D2.5.3 — Beta estimation: Ridge regression

**Decision:** For each stock, estimate its 18-dim beta vector by **ridge regression** of
its formation-window daily excess returns on the 18 factors.

**Why ridge (not OLS):** the 12 industry factors + 6 style factors are correlated
(multicollinearity), so OLS betas are unstable. Ridge (L2) shrinks betas toward zero,
stabilizing the vectors that we then cluster on. Penalty `alpha` is a hyperparameter —
default TBD (start `alpha=1.0`, tune on Dec-2023 cluster count vs sensibility).

## D2.5.4 — Distance metric on beta vectors: standardized Euclidean (TENTATIVE)

**Decision (tentative):** z-score each beta dimension across the universe (so no single
factor dominates by scale), then Euclidean distance between standardized beta vectors.
Feeds the same OPTICS clustering as SSD/PC.

**Open:** may switch to `1 − corr(beta_i, beta_j)` (cosine-like) to match the PC/SSD
"1 − similarity" convention. Decide after seeing Dec-2023 cluster behavior.

## D2.5.5 — Self-inclusion (circularity) in industry factors

**Note:** a stock contributes to its own industry factor's equal-weight return. With
dozens of stocks per FF12 industry the contribution is negligible. **Decision:** accept
it for the core; add a leave-one-out variant only if Dec-2023 betas look degenerate.

---

## Resolved during the build (2026-06-07, all chosen BEFORE any backtest)

- [x] **Ridge `alpha` = 1.0** (D2.5.3) — `config.RIDGE_ALPHA`. Stabilizes the collinear
      18-factor betas. Validated: Dec-2023 betas are economically sensible (top Enrgy
      loaders = APA/MRO/DVN/OXY/FANG; top BusEq = NVDA/AMD/LRCX/AMAT; top Money =
      ZION/CMA/KEY/CFG/FITB).
- [x] **Distance = standardized Euclidean** (D2.5.4) — z-score each beta dimension across
      stocks, then Euclidean. (1−corr not needed; Euclidean separated the synthetic
      planted clusters cleanly, ratio >5×.)
- [x] **OPTICS `xi` = 0.10** for the factor metric — `config.OPTICS_XI_FACTOR`. Distance
      is Euclidean in 18-dim β-space (not [0,2]), so it gets its own xi. Validated on
      Dec 2015 / 2023: 61 / 78 clusters (vs PC's ~81 — chosen for a fair head-to-head),
      purity 0.903 / 0.915 vs SIC division. Chosen before any Sharpe.

## CP2.5 targets (no paper benchmark — this is our extension)

Success = (a) economically-coherent Dec-2023 clusters ✅ (done, purity ~0.90), and
(b) Sharpe reported head-to-head vs **PC core 1.028** and **SSD core 0.589**. No
pass/fail Sharpe gate — the contribution is the method + the comparison, whatever the
number turns out to be.

## Build status (2026-06-07)

- [x] `src/factors.py` — FF12 mapper (canonical Siccodes12 ranges, disjoint-verified) +
      `build_factor_panel` (6 style + 12 industry).
- [x] `src/distances.py` — `ridge_betas` + `factor_beta_distance` (Euclidean on z-betas).
- [x] `src/config.py` — `OPTICS_XI_FACTOR`, `RIDGE_ALPHA`, `PHASE2_5_DIR`.
- [x] `src/backtest.py` — `metric="factor"` branch (backward-compatible).
- [x] `tests/test_factors_synthetic.py` — 7 tests; full suite 39/39 green.
- [x] `phases/phase2_5/notebooks/01_run_factor_backtest.py` — 2-cell grid runner.
- [x] `notebooks/02_compare_to_pc.py` — head-to-head scorecard (reproduces SSD/PC baselines).
- [x] `notebooks/03_factor_attribution.py` — bimodal lever + regime + sector battery.
- [x] `notebooks/phase2_5_complete_reference.ipynb` — polished deliverable (executed, 0 errors).
- [x] **factor_core backtest DONE (2026-06-07): Sharpe 1.013** vs PC core 1.028
      (Δ −0.014) — matches. Sortino 2.598 / Calmar 0.605 (both edge PC); ann.ret 3.60%,
      vol 3.55%, MDD −6.0%; 19,044 trades (vs PC 30,050); force-close −12 bps (≈ PC −11).
- [x] **factor_filtered backtest DONE (2026-06-07): Sharpe 0.858** — BEATS PC+filter
      (0.752, Δ +0.106). Ann.ret 4.45% / vol 5.24% / MDD −7.6%; 5,643 trades; **0 outliers**
      |r|>50% in 21 years (like PC+filter); force-close −15 bps.

## PHASE 2.5 COMPLETE (2026-06-07)

Headline: factor-beta clustering — a structurally different, economically-motivated
metric — **independently reproduces the paper's ~1.0 Sharpe** (core 1.013 vs PC 1.028),
and its filtered variant (0.858) **beats** PC+filter (0.752). Strengthens the claim that
the headline result is not an artifact of the partial-correlation metric. Both PC and
factor-beta pair ~80% within-industry; factor-beta concentrates P&L a bit more in
financials (Money/Money ≈27% vs ≈23%). All deliverables executed & baked
(`phase2_5_complete_reference.ipynb`). Tests 39/39. Next: Phase 3 robustness.
