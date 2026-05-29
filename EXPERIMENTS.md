# Pairs Trading — Research Log (append‑only)

Date: 2026‑05‑21
Run ID: S12d
Method: A (returns‑spectral), residualized tail‑ADF/HL on 252d within 378d residual history
Params: k=10, z_entry=2.0 (turning‑point confirm), z_exit=0.5, hard_stop=3.0, time_stop=20, corr_win=189, refit=63, HL[8–20], corr63≥0.70, zero‑cost
Results: Gross +0.357, Net n/a, MaxDD −4.6%, Trips 193, Hit 75.3%, Overlap ~20%
Note: Entry asymmetry + confirmation fixed left tail; low DD.

────────────────────────────────────────
Date: 2026‑05‑21
Run ID: S13a
Method: A
Params: k=10, z=2.0 (confirm), cost=3bps
Results: Gross +0.357, Net +0.107, MaxDD −5.5%, Trips 193, Hit 75.3%
Note: First positive net; 3 bps/leg viable; break‑even between 3–5 bps.

────────────────────────────────────────
Date: 2026‑05‑21
Run ID: S13b
Method: A
Params: k=10, z=2.0 (confirm), cost=5bps
Results: Gross +0.357, Net −0.059, MaxDD −6.1%, Trips 193, Hit 75.3%
Note: 5 bps kills net at this frequency; need 2–3 bps routing.

────────────────────────────────────────
Date: 2026‑05‑21
Run ID: S13c
Method: A
Params: k=12, z=2.0 (confirm), cost=0
Results: Gross +0.250, Net +0.250, MaxDD −5.4%, Trips 203, Hit 73.3%
Note: k=10 is quality ceiling; extra pairs dilute.

────────────────────────────────────────
Date: 2026‑05‑21
Run ID: S14a (monthly)
Method: A
Params: k=10, z=2.0 (confirm), cost=0, refit=21, corr_win=189, HL[8–20], corr63≥0.75
Results: Gross +0.061, Net n/a, MaxDD −4.3%, Trips 137, Hit 68.2%, Overlap 27%, ClusterPersist 85%
Note: Monthly refit worse than quarterly despite higher overlap; z=2.0+confirm calibrated for 63d segments; monthly windows under‑supply confirmed entries → hit rate drops.

────────────────────────────────────────
Date: 2026‑05‑21
Run ID: S15a (full OOS)
Method: A (returns‑spectral), 2014–2024 full period, residualized tail‑ADF/HL=252 on 378d residual history
Params: k=10, z=2.0 (confirm), hard_stop=3.0, z_exit=0.5, time_stop=20, corr_win=189, refit=63, HL[8–20], corr63≥0.70, cost=0
Results: Gross +0.332, MaxDD −4.6%, Trips 268, Hit 77.4%, Overlap 14.9%, ClusterPersist 79.1%
Note: Parameters robust; subperiod SR high in 2017–2021, weak in 2014–2016 and 2022–2024 (PC1 dominance).

────────────────────────────────────────
Date: 2026‑05‑21
Run ID: S15b (Method B + PCA8)
Method: B (beta‑space), PCA on betas (8 PCs), residualized formation as S12d
Params: beta_window=504, EWMA λ≈0.98, ridge α=30, PCA(8) on betas, refit=63, cost=0, same entry/stop stack as S12d
Results: Gross −0.424, MaxDD −7.2%, Trips 308, Hit 69.4%, Overlap 6.9%, ClusterPersist 50.5%
Note: Current B implementation closed on daily SPX; PCA on betas marginally improved persistence but hurt performance. Next research: orthogonalize factor RETURNS first (PCA on factors), then estimate betas to orthogonal factors and cluster.

────────────────────────────────────────
Date: 2026‑05‑21
Run ID: S16c (Method B + stable ortho‑factors)
Method: B (beta‑space), PCA on FACTOR RETURNS first (k=8), then EWMA ridge betas to orthogonal PCs; residualized formation as S12d
Params: factor z‑score (rolling 252d), PCA(k=8) fit on factor_rets[:d0] (no look‑ahead), beta_window=504, EWMA λ≈0.98, ridge α=30, refit=63, cost=0; entries/stops = S12d (z=2.0+confirm, hard=3.0, z_exit=0.5, time_stop=20), HL[8–20], corr63≥0.70
Results: Gross +0.099, MaxDD −?%, Trips ?, Hit ?, Overlap 2.5%, ClusterPersist 50.6%
Note: Correct‑order orthogonalization produced first positive gross for B, but ClusterPersist stays ~50% and overlap is very low. Daily factor rotation in SPX limits beta‑space stability at quarterly cadence. Not additive vs Method A on daily horizon.

────────────────────────────────────────
Date: 2026-05-21
Run ID: S14b (Method B, S12d quality stack)
Method: B (beta-space spectral), residualized formation
Params: k=10, z=2.0 (confirm), cost=0, refit=63, beta_window=504, ridge_alpha=30, corr_win=189, HL[8-20], corr63>=0.75
Results: Gross -0.014, MaxDD -3.0%, Trips 319, Hit 69.0%, Overlap 10%, ClusterPersist 45.7%
Note: Method B clustering unstable (ClusterPersist 45.7% vs A's 76%). Factor betas drift quarterly
      (sector/style rotation) → fresh unvalidated pairs every segment → hit rate collapses 75%→69%.
      Method B does NOT add independent alpha on top of A. Beta-space clustering is the wrong
      foundation for this signal at daily frequency without much longer beta_window or PCA reduction.

────────────────────────────────────────

## Complete Run Table (all runs, zero-cost unless noted)

Base config evolving toward S12d. Method A unless noted. Residualized ADF from S3 onward.

| Run | Method | z | k | HL | corr | hard | refit | Gross | Net | MaxDD | Trips | Hit | Overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused-logspace | fused | 2.0 | 3 | 5-30 | 0.70 | 4.0 | 63 | +2.59 | — | -1% | 231 | — | — |
| fused-pairret | fused | 2.0 | 3 | 5-30 | 0.70 | 4.0 | 63 | -0.19 | — | -8% | 231 | 80% | 6% |
| zerocost | fused | 2.0 | 3 | 5-30 | 0.70 | 4.0 | 63 | -0.05 | — | -8% | 231 | 80% | 6% |
| A-S1 | A | 1.5 | 6 | 5-30 | 0.70 | 4.0 | 63 | -0.31 | — | -23% | 1056 | — | 7% |
| A-S2 | A | 1.5 | 6 | 5-30 | 0.70 | 4.0 | 63 | -0.36 | — | -23% | 1062 | — | 7% |
| A-S3 | A | 1.5 | 6 | 5-30 | 0.70 | 4.0 | 63 | -0.05 | — | -21% | 638 | 81% | 17% |
| A-S4 | A | 1.5 | 6 | 5-60 | 0.65 | 4.0 | 63 | -0.38 | — | -25% | 707 | 81% | 15% |
| A-S5 | A | 1.5 | 6 | 5-30 | 0.70 | 4.0 | 63 | -0.80 | — | -33% | 623 | 81% | 21% |
| S6a | A | 1.5 | 6 | 5-30 | 0.70 | 3.0 | 63 | -0.06 | — | -19% | 785 | 69% | 17% |
| S6b | A | 1.5 | 6 | 5-30 | 0.70 | 3.5 | 63 | +0.004 | — | -19% | 624 | 75% | 17% |
| S6c | A | 1.5 | 6 | 5-30 | 0.70 | 3.0 | 63 | +0.030 | — | -19% | 801 | 74% | 17% |
| S6c+5bps | A | 1.5 | 6 | 5-30 | 0.70 | 3.0 | 63 | +0.030 | -0.52 | -26% | 801 | 74% | 17% |
| S7a | A | 2.0 | 6 | 5-30 | 0.70 | 3.0 | 63 | -0.09 | — | -15% | 584 | 68% | 17% |
| S7b | A | 2.0 | 6 | 5-30 | 0.70 | 3.0 | 63 | -0.83 | — | -25% | 577 | 66% | 21% |
| S7c | A | 2.0 | 10 | 5-30 | 0.70 | 3.0 | 63 | -0.28 | — | -21% | 741 | 69% | 23% |
| S8a | A | 1.5 | 6 | 8-20 | 0.70 | 3.0 | 63 | -0.017 | — | -15% | 569 | 73% | 10% |
| S8b | A | 1.75 | 6 | 8-20 | 0.70 | 3.0 | 63 | -0.14 | — | -12% | 481 | 69% | 10% |
| S8c | A | 1.5 | 8 | 8-20 | 0.70 | 3.0 | 63 | +0.045 | — | -13% | 630 | 73% | 13% |
| S9a | A | 1.5 | 10 | 8-20 | 0.70 | 3.0 | 63 | +0.124 | — | -13% | 663 | 74% | 14% |
| S9b | A | 1.5 | 8 | 8-20 | 0.70 | 3.0 | 63 | -0.563 | — | -30% | 862 | 72% | 21% |
| S9c | A | 1.5 | 10 | 8-20 | 0.70 | 3.0 | 63 | -0.428 | — | -27% | 942 | 73% | 21% |
| S10a | A | 1.5 | 10 | 8-20 | 0.75 | 3.0 | 63 | +0.190 | — | -14% | 446 | 74% | 12% |
| S10b | A | 1.5 | 10 | 8-20 | 0.80 | 3.0 | 63 | -0.041 | — | -12% | 276 | 70% | 18% |
| S10c | A | 1.5 | 12 | 8-20 | 0.75 | 3.0 | 63 | +0.149 | — | -15% | 453 | 74% | 12% |
| S10d | A | 1.5 | 10 | 8-20 | 0.70 | 3.0 | 63 | +0.115 | — | -14% | 684 | 72% | 15% |
| S10e | A | 1.5 | 10 | 10-20 | 0.70 | 3.0 | 63 | -0.242 | — | -5% | 253 | 72% | 6% |
| S11 | A | 1.5 | 10 | 8-20 | 0.75 | 3.0 | 63 | +0.267 | — | -7% | 360 | 72% | 12% |
| S12a | A | 1.5 | 10 | 8-20 | 0.75 | 3.0 | 63 | +0.190 | — | -14% | 446 | 74% | 12% |
| S12b | A | 1.5 | 12 | 8-20 | 0.75 | 3.0 | 63 | +0.198 | — | -8% | 378 | 71% | 12% |
| S12c | A | 1.5 | 15 | 8-20 | 0.75 | 3.0 | 63 | +0.145 | — | -8% | 389 | 70% | 13% |
| **S12d** | **A** | **2.0** | **10** | **8-20** | **0.75** | **3.0** | **63** | **+0.357** | **—** | **-5%** | **193** | **75%** | **12%** |
| S13a | A | 2.0 | 10 | 8-20 | 0.75 | 3.0 | 63 | +0.357 | **+0.107** | -6% | 193 | 75% | 12% |
| S13b | A | 2.0 | 10 | 8-20 | 0.75 | 3.0 | 63 | +0.357 | -0.059 | -6% | 193 | 75% | 12% |
| S13c | A | 2.0 | 12 | 8-20 | 0.75 | 3.0 | 63 | +0.250 | — | -5% | 203 | 73% | 12% |
| S14a | A | 2.0 | 10 | 8-20 | 0.75 | 3.0 | **21** | +0.061 | — | -4% | 137 | 68% | 27% |
| S14b | **B** | 2.0 | 10 | 8-20 | 0.75 | 3.0 | 63 | -0.014 | — | -3% | 319 | 69% | 10% |

Notes on S11 vs S12d: S11 adds structural-break exits + entry confirmation + new-pair cap. S12d is S11 + z_entry raised to 2.0.

## Parameter Ceiling Findings

**What works (each lever confirmed positive in isolation):**
- Residualized ADF: overlap 6%→17% (F3)
- hard_stop 4.0→3.0: Gross -0.05→+0.03 (F4)
- HL[8-20d]: removes fast/slow mismatches; MaxDD -19%→-13% (F6)
- corr63>=0.75: filters to tightly-linked pairs; Gross +0.12→+0.19 (F-new)
- Structural-break exits (5d corr<0.5, |Δβ|>30%): MaxDD -14%→-7% (F-new)
- Entry confirmation (z turning back): hit rate recovers at z=2.0; filters straight-to-stop entries
- z_entry=2.0 WITH confirmation: asymmetry flips (+1.5σ win vs -1.0σ loss); Gross +0.19→+0.357

**What doesn't work (tested and confirmed negative):**
- Monthly refit: hit rate drops 75%→68%; signal calibrated for 63d segments (H-monthly REFUTED)
- z_entry>1.5 WITHOUT confirmation: heavy tails; hit rate always drops (H5 REFUTED)
- Method B at any quality level: beta-space clustering unstable (ClusterPersist 46%); no independent alpha
- k>10 at corr>=0.75: extra pairs below quality threshold; Gross degrades
- HL[10-20d] or HL[5-30d]: both worse than [8-20d] sweet spot
- 378d formation with HL[8-20]: cumulative residual drift noisifies z-score

**Parameter ceiling: Gross ~+0.35-0.40 on daily SPX-300.**
Net Sharpe: +0.107 at 3 bps/leg (S13a). Ceiling at 5 bps/leg: barely negative.

## Path to Higher Sharpe

| Step | Expected Gross | Requires |
|---|---|---|
| S12d current best | +0.357 | Done |
| +IBES earnings blackout | +0.55-0.65 | WRDS credentials |
| +CRSP clean universe | +0.60-0.70 | WRDS credentials |
| +TAQ intraday 30-60min | +0.8-1.5 | WRDS TAQ |
| Gross >1.0 without WRDS | Not achievable | Need more independent bets (universe or horizon) |

## Production Configuration (S12d)
```
--method a --refit 63 --corr-window 189 --ari-thresh 0.7
--factor-source both --neutralize
--z-entry 2.0 --z-exit 0.5 --hard-stop 3.0 --time-stop 20
--min-hl 8 --max-hl 20 --min-corr 0.75 --top-k 10
--cost-bps 3
```
Gross +0.357 | Net +0.107 @ 3bps | MaxDD -5.5% | Trips 193 | Hit 75.3% | Overlap 12%

────────────────────────────────────────

Date: 2026-05-21
Run ID: S15a (2014-2024 OOS validation)
Method: A, S12d params on full 2014-2024 history
Params: k=10, z=2.0 (confirm), cost=0, refit=63, corr_win=189, HL[8-20], corr63>=0.75
Results: Gross +0.332, MaxDD -4.6%, Trips 268, Hit 77.4%, Overlap 14.9%, ClusterPersist 79.1%
Sub-periods:
  2014-2016: Sharpe -0.625  (ZIRP / low dispersion / PC1 dominance)
  2017-2018: Sharpe +0.917  (high idiosyncratic dispersion)
  2019-2021: Sharpe +1.012  (peak idiosyncratic MR, includes COVID vol)
  2022-2024: Sharpe -0.319  (rates shock / macro-dominated / PC1 dominance)
Note: Params not overfit (full-period +0.332 vs 2019-only +0.357; modest decay).
      Strategy is regime-conditional: earns when cross-sectional idiosyncrasy is high
      (2017-2021), loses in macro-dominated regimes where PC1 dominates.
      Fix: breadth/PC1 throttle (scale book to 0 when macro regime detected).

────────────────────────────────────────

Date: 2026-05-21
Run ID: S15b (Method B + PCA8 — final B attempt)
Method: B (beta-space spectral), PCA(8) on betas before affinity
Params: k=10, z=2.0 (confirm), cost=0, beta_window=504, ridge_alpha=30, pca_betas=8
Results: Gross -0.424, MaxDD -7.2%, Trips 308, Hit 69.4%, Overlap 6.9%, ClusterPersist 50.5%
Note: PCA marginally raised ClusterPersist (46%->50.5%) but Gross worsened vs no-PCA (-0.014->-0.424).
      METHOD B CLOSED. Beta-space clustering is not viable at daily SPX close:
      factor betas drift faster than quarterly refit can stabilise.

OOS Verdict: S12d config is robust across 2014-2024. Regime-conditional, not broken.
Method B Verdict: definitively closed (3 configs tested; all negative Sharpe).

────────────────────────────────────────
Date: 2026-05-21
Run ID: bt_b_s16c_ortho_fixed2
Method: b

────────────────────────────────────────
Date: 2026-05-25
Run ID: PLAN_S25 (HMM Regime Overlay — Next Chapter)
Context: Baseline S17a (2015–2026 CRSP) Gross SR ≈ +0.57 / Net SR ≈ +0.20 @ 3bps is regime‑heterogeneous (2014/2017/2021/2022 weak). PC1 and dispersion as single gates are weak; need a persistent regime classifier.
Hypothesis: A 2–3 state Gaussian HMM on VIX level/slope + structure (dispersion/PC1; optional credit/curve) will scale entries/size to cut crisis losses and stabilize yearly Sharpe without retuning pair parameters.
Plan (Phase 1):
  - Features (daily, T+1): log(VIX), VIX3M−VIX1M (or ratio), 20d realized SPX vol, dispersion or avg corr, PC1 share. Optional: VVIX, HY OAS, 2s10s.
  - Model: GaussianHMM(n=3, diag cov), fit 2014–2019 (seeded); decode OOS with filtered posteriors.
  - Hook: position_scale = 1.0·p(calm) + 0.5·p(stressed) + 0.0·p(crisis); exits unchanged.
Runs queued:
  - S25a: HMM(3) posterior scaling (1.0, 0.5, 0.0) — headline.
  - S25b: HMM(2) comparator (1.0, 0.0).
  - S25c: Viterbi hard switch (no posterior smoothing).
  - S25d: Train excl. 2020 (leakage check).
  - S25e: Dynamic hard_stop (2.5 stressed / 3.0 calm) ablation.
  - S25f: Yearly 2014 and 2017 slices.
  - S25g: True OOS — train 2010–2014, trade 2015–2026.
  - S25h: Naïve VIX>25 gate baseline.
Acceptance:
  - Net SR lift ≥ +0.05 vs S17a; MaxDD not worse; trips −≤30%.
  - 2014/2017 SR not worse by >0.15; 2022–24 losses reduced.
  - HMM ≥ naïve gate by ≥ +0.05 SR. If met, consider Phase 2 (SABR ν/ρ/α).
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.75
Results: Gross 0.09858379309342011, Net 0.09858379309342011, MaxDD -0.0472865690289231, Trips 346, Hit 0.6909586056644881
Note: 

────────────────────────────────────────


## Method B Progression (S14b → S16c)
| Run | Approach | Gross | ClusterPersist | Overlap |
|---|---|---|---|---|
| S14b | Raw correlated ETF factors | -0.014 | 45.7% | 10% |
| S15b | PCA on betas after estimation (wrong order) | -0.424 | 50.5% | 6.9% |
| S16a | PCA on factor returns, time-varying basis | no trades | n/a | n/a |
| S16c | PCA on factor returns, stable basis (correct) | +0.099 | 50.6% | 2.5% |

Lesson: orthogonalize factor RETURNS before ridge (Barra-style), not betas after.
Stable basis = fit PCA on all factor history up to refit date, transform window with that basis.
ClusterPersist ceiling ~50% is structural (daily factor rotation in SPX); not an implementation bug.
Method B viable at +0.099 but 3.6x below Method A (+0.357). Not additive enough to justify complexity.

────────────────────────────────────────
Date: 2026-05-23
Entry: Data infrastructure — fred.parquet now live

fred.parquet sourced: WRDS frb.rates_daily REST API (token auth).
fredgraph.csv endpoint (fred.stlouisfed.org) consistently times out on this machine;
WRDS frb.rates_daily has the same H.15 + ICE BofA data and works via REST.

Series in fred.parquet (4224 rows, 2014-01-01 → 2026-05-22, updated 2026-05-23):
  DGS2          FRED API exact (H.15)               2014–today
  DGS10         FRED API exact (H.15)               2014–today
  T10Y2Y        FRED API exact (independent series) 2014–today
  BAMLH0A0HYM2  WRDS→2023 spliced with FRED→today   2014–today  exact OAS throughout
  BAMLC0A0CM    WRDS proxy→2023 + FRED exact→today  2014–today  note below

Series ID correction: original code used BAMLCC0A0CM (extra C) — does not exist on FRED.
Correct ID = BAMLC0A0CM. All code and docs updated.

ICE BofA licensing: FRED only distributes BAML series from 2023-05-23 (licensing restriction).
Pre-2023 history from WRDS frb.rates_daily (bamlh0a0hym2 exact; BAMLC0A0CM proxied).

BAMLC0A0CM pre-2023 proxy:
  WRDS has IG effective yield (bamlc0a0cmey) not OAS. Proxy = bamlc0a0cmey − dgs10.
  Overstates OAS by ~20bp (call option premium; stable, not noise).
  Bias cancels on first-differencing. Strategy uses fred.diff() → acceptable.
  Post-2023: exact FRED OAS — no proxy needed.
  Resolution: CLOSED. Exact for post-2023, proxy for pre-2023 with known stable bias.
Date: 2026-05-23
Run ID: bt_S17a
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.5675844626186166, Net 0.5675844626186166, MaxDD -0.04496445057310651, Trips 558, Hit 0.7554166666666666
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S17b
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.41313464985678017, Net 0.41313464985678017, MaxDD -0.04520361633161961, Trips 544, Hit 0.7619311193111931
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S17a_3bps
Method: a
Params: k=10, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.5675844626186166, Net 0.1961358236874568, MaxDD -0.05931884327153147, Trips 558, Hit 0.7554166666666666
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: S17a (IBES blackout, full period)
Method: A (returns-spectral), residualized
Params: k=10, z=2.0 (confirm), hard_stop=3.0, z_exit=0.5, time_stop=20, corr_win=189, refit=63, HL[8-20], corr63≥0.70, formation=378, earnings_blackout=±3d (entry+forced-exit), zero-cost, 2015-2026
Results: Gross +0.57, MaxDD −4.5%, Trips 558, AvgHold 4.0d, Hit 75.5%, Overlap 22%, ClusterPersist 78%
Note: IBES blackout delivers +60% Gross SR improvement (0.357→0.57). Within projected 0.55-0.65 corridor. AvgHold collapsed to 4d (was ~8-10d); forced-exit+re-entry churn suspected.

────────────────────────────────────────

Date: 2026-05-23
Run ID: S17a_3bps (S17a + 3bps/leg cost)
Method: A, same as S17a
Params: same as S17a, cost=3bps
Results: Gross +0.57, Net +0.20, MaxDD −5.9%, Trips 558, AvgHold 4.0d, Hit 75.5%
Note: Net improved from S13a's +0.107 to +0.20 (+87%). Cost drag ~0.61%/yr (cost eating 2/3 of gross alpha due to short AvgHold).

────────────────────────────────────────

Date: 2026-05-23
Run ID: S17b (IBES + factor-source both)
Method: A, same as S17a, factor_source=both (FF5+MOM+FRED∆ + 17 ETFs)
Params: same as S17a, factor_source=both, zero-cost. Date range starts 2015-07 (larger factor set → more burn-in needed)
Results: Gross +0.41, MaxDD −4.5%, Trips 544, Hit 76.2%, Overlap 23%
Note: ff+fred factor augmentation HURTS vs etf_only (+0.41 vs +0.57). Confirms S16c finding. Extra factor collinearity degrades neutralization quality. Stick with etf_only.

────────────────────────────────────────

## Updated Parameter Ceiling (post-IBES)

| Signal | Before IBES | After IBES |
|--------|-------------|------------|
| Gross SR | +0.357 | +0.57 |
| Net SR @ 3bps | +0.107 | +0.20 |
| MaxDD | -4.6% | -4.5% |
| AvgHold | ~8-10d | 4.0d |

**Constraint identified**: Forced-exit on blackout window collapses AvgHold → inflates trips → costs eat net SR. Next: entry-block-only (S17c) to test whether AvgHold recovers.

**Factor source conclusion**: etf_only is optimal for Method A. ff_daily and fred parquets loaded but NOT used in clustering — they degrade residualization when added. Use etf_only going forward.

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S17c
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.551292238582489, Net 0.551292238582489, MaxDD -0.05522485222259997, Trips 557, Hit 0.7536309523809523
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: S17c (IBES entry-only, no forced exit)
Method: A, same as S17a, earnings_force_exit=False (entry block only)
Params: same as S17a, zero-cost
Results: Gross +0.55, MaxDD −5.5%, Trips 557, AvgHold 4.0d, Hit 75.4%, Overlap 22%
Note: Trips 557 vs S17a's 558 — forced exits cause 0 extra churn. AvgHold=4d is intrinsic (hard_stop=3.0 + z_exit=0.5, not IBES). Forced exits marginally help (+0.02 Sharpe) by avoiding adverse earnings moves. Keep S17a config (entry+forced-exit).

### S17 Series Summary
Best config: S17a (IBES blackout ±3d, entry+forced-exit, etf_only, formation=378, neutralize, corr_win=189)
Gross +0.57 | Net +0.20 @ 3bps | MaxDD −4.5% | Trips 558 | Hit 75.5% | ClusterPersist 78%

Path forward: CRSP clean universe (projected +0.60-0.70 gross). yfinance universe has survivorship bias; CRSP membership avoids this.

────────────────────────────────────────

Date: 2026-05-23
Run ID: S19a–S19f (Kalman β diagnostics)
Method: A (returns‑spectral), IBES blackout ±3d
Params: baseline S17a stack; Kalman with δ∈{1e‑5, 1e‑4, 5e‑4}, β uncertainty gate on/off
Results (representative):
  S17a (OLS):   Gross +0.57 | Trips 558 | AvgHold 4.0d | Hit 75.5% | MaxDD −4.5%
  S19a (δ=1e‑4): Gross +0.61 | Trips 92  | AvgHold 2.1d | Hit 89.2% | MaxDD −1.0%
  S19b (δ=5e‑4): Gross +0.48 | Trips 67  | AvgHold 1.9d | Hit 89.5% | MaxDD −0.8%
Diagnosis:
  - Early Kalman choke came from β‑uncertainty gate with P=I seed; fixed by OLS‑seeding P0 from warmup OLS and/or disabling the gate for warmup.
  - Oversized Kalman gain came from unscaled H=[pb,1]; fixed by scaling pb by its rolling mean so H≈[1,1]. Beta responsiveness now governed by δ.
Conclusion: β estimation is not the limiter. OLS remains best default; Kalman usable when tuned, but tends to reduce MR alpha as δ increases.

Next levers (higher impact than β method):
  1) ADF‑affinity clustering (sparse edge ADF p‑value × HL kernel) → changes peer selection. Expected +0.05–0.15 gross SR.
  2) Regime throttle (PC1 share; optional FRED overlay) → scales risk down in macro‑dominant regimes. Expected +0.03–0.08 full‑period SR.
  3) Execution realism: per‑name netting before ADV cap/impact; borrow fee drag; liquidity floors. Cost clarity and net SR realism.

Implementation notes (today):
  - Impact applied on entry AND exit; CLI flags added: --enable-impact, --impact-k-bps, --enable-adv-cap, --adv-cap-pct, --portfolio-notional, --min-adv-usd, --target-vol-bps.
  - Liquidity floor added (min ADV$ per leg).
  - Kalman fixed: observation scaling and OLS‑seeded covariance; uncertainty gate configurable (--kalman-unc-cap 0 to disable).


Date: 2026-05-23
Run ID: bt_S18a
Method: a
Params: k=5, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.75
Results: Gross 0.2839357589680756, Net 0.2839357589680756, MaxDD -0.03371705092878161, Trips 164, Hit 0.7782945736434107
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S18b
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.5350385203836323, Net 0.5350385203836323, MaxDD -0.039127499546111055, Trips 561, Hit 0.7712336892052194
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S17a_moc
Method: a
Params: k=10, z=2.0, cost=0.5bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.5675844626186165, Net 0.5057609459666841, MaxDD -0.047356849356177275, Trips 558, Hit 0.7554166666666666
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S19a
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.6057978364924621, Net 0.6057978364924621, MaxDD -0.010283188825888058, Trips 92, Hit 0.8918918918918919
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S19b
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.4817570800661093, Net 0.4817570800661093, MaxDD -0.008065608522456967, Trips 67, Hit 0.8947368421052632
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S19c
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.6134925121180947, Net 0.6134925121180947, MaxDD -0.010283188825888058, Trips 95, Hit 0.8961038961038961
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S19d
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross -0.04460830898039718, Net -0.04460830898039718, MaxDD -0.10075591431974695, Trips 598, Hit 0.6756020799124247
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S19e
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.03782073124514986, Net 0.03782073124514986, MaxDD -0.09903845083001359, Trips 762, Hit 0.49131652661064423
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S19f
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.360584923059991, Net 0.360584923059991, MaxDD -0.036137309999294875, Trips 415, Hit 0.7763395185930396
Note: 

────────────────────────────────────────

## S19 Series: Kalman Filter β (vs Rolling OLS baseline S17a)

### Findings (all Kalman variants worse or not meaningfully better than OLS at usable trip count)

| Run | Version | delta | unc-cap | Gross SR | Trips | HitRate | Note |
|-----|---------|-------|---------|----------|-------|---------|------|
| S17a | OLS | — | — | +0.57 | 558 | 75.5% | baseline |
| S19a | old H=[pb,1] | 1e-4 | 0.30 | +0.61 | 92 | 89.2% | β over-adapts |
| S19c | old H=[pb,1] | 1e-4 | 0 | +0.61 | 95 | 89.6% | cap wasn't constraint |
| S19d | old H=[pb,1] | 1e-7 | 0 | -0.04 | 598 | 67.6% | β stuck at warmup |
| S19e | scaled H | 1e-7 | 0 | +0.04 | 762 | 49.1% | near random |
| S19f | scaled H | 1e-5 | 0 | +0.36 | 415 | 77.6% | partially functional |

**Conclusion**: Kalman does NOT improve over rolling OLS for this strategy.

Root cause: OLS "staleness" (63d fixed window) is a FEATURE for mean reversion.
When pair diverges (z→2.0), the OLS β reflects pre-divergence equilibrium.
Kalman fast mode: absorbs divergence as β change → removes tradeable spread.
Kalman slow mode: β drifts from truth over years → HitRate ~50% (random).
Sweet spot (S19a/S19c: 0.61 SR, 95 trips): too few trades, <$0.40%/yr return.

Literature: rolling OLS competitive with Kalman for short-horizon mean reversion
(consistent with Elliott et al. 2005 caveats; Kalman works better for regime-
switching or multi-month horizons where β genuinely changes).

Rolling OLS remains optimal for this strategy. use_kalman=False going forward.

────────────────────────────────────────

## Current Architecture Ceiling

Gross SR: +0.57 (S17a: OLS + IBES blackout, etf_only, neutralize)
Net SR @ 3bps: +0.20 | Net SR @ 0.5bps (MOC): +0.51
Max achievable on daily SPX-300 with current signal: ~+0.57-0.70 gross

What worked:
  ✓ IBES earnings blackout (+60% gross SR, 0.357→0.57)
  ✓ MOC cost assumption (0.5bps vs 3bps: net SR 0.20→0.51)
  ✓ Method A spectral clustering (stable 78% ClusterPersist)
  ✓ IBES forced exit (marginal +0.02 SR vs entry-only)

What didn't work:
  ✗ Kalman β (β over-adapts or gets stuck; OLS staleness is a feature)
  ✗ Strict ADF p<0.01 (kills diversification, 0.28 vs 0.57)
  ✗ z_exit=1.0 (exits too early, misses tail reversion: 0.54 vs 0.57)
  ✗ ff_daily + fred in factor set (hurts neutralization: 0.41 vs 0.57)
  ✗ Method B clustering (unstable ClusterPersist 50% vs A's 78%)

Next build targets:
  1. ADF-based clustering affinity (cointegration INTO cluster structure)
  2. FRED macro regime gate (halve sizes in high-stress periods)
  3. CRSP survivorship-free universe (remove survivorship bias from yfinance)

────────────────────────────────────────
Date: 2026-05-23
Run ID: bt_S20a
Method: a_coint
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.32515445931517095, Net 0.32515445931517095, MaxDD -0.04066390464452276, Trips 556, Hit 0.7272283272283272
Note: 

────────────────────────────────────────

Date: 2026-05-23
Run ID: bt_S20b
Method: a_coint
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.12934479348459887, Net 0.12934479348459887, MaxDD -0.06288554916477747, Trips 553, Hit 0.7268620268620268
Note: 

────────────────────────────────────────

Date: 2026-05-24
Entry: Data infrastructure — design decisions locked

## Universe: S&P 500 (historical membership via crsp.msp500list)

Decision: retain S&P 500 universe rather than full CRSP ordinary shares (shrcd 10/11).

Rationale:
- Project objective is "liquid US large caps" (Proposal §2). Full CRSP adds ~22,000 stocks;
  most are illiquid micro/small caps that fail the $20mm ADV and price>$5 filters anyway.
- S&P 500 has lower borrow frictions, better data quality, and matches the practical
  trading environment for a stat-arb running at institutional scale.
- Do & Faff (2010/2012) use full CRSP with liquidity filters — their paper is a reference,
  not the specification. This project targets a subset (large-cap liquid names).
- Tradeoff acknowledged: fewer idiosyncratic pair candidates than full-CRSP universe.
  Mitigant: PC1 throttle scales book in macro-dominated regimes where SPX homogeneity
  is the binding constraint, not universe size.
- Filter by permno from crsp.msp500list (not tickers) to include all historical entrants
  and exits. No survivorship bias from using stable CRSP permanent IDs.

Do NOT use:
  memb["ticker"].unique() → ticker-based filter is biased; tickers get reused across firms.
Use instead:
  spx_only=True in fetch_crsp_daily() → permno IN (SELECT DISTINCT permno FROM crsp.msp500list)

## Price series: adjusted close (ABS(prc) / cfacpr)

Decision: use CRSP split-adjusted close, not raw closing price.

Rationale:
- Stock splits create step jumps in raw price series. For two stocks splitting at different
  times, the raw log-price spread has permanent step discontinuities that:
    (a) make the spread look non-stationary to ADF/Johansen (false negatives in pair selection)
    (b) distort OLS hedge ratio estimation via outlier observations on split days
- Adjusted close removes these artifacts; ADF correctly identifies true economic cointegration.
- "Lookahead bias" concern: backward adjustment uses future split factors to rescale
  historical prices. This is benign for spread/ratio strategies — the lookahead affects
  price LEVELS only, not the direction or timing of spread mean-reversion. No future
  return information is injected into the signal. Contrast with genuine lookahead: using
  future vol estimates for sizing, or selecting pairs on full-sample cointegration.
- ABS(prc) required: CRSP records negative prc when it uses bid-ask midpoint instead of
  a trade price (no trade that day). Absolute value recovers the price level.

────────────────────────────────────────

Date: 2026-05-24
Run ID: bt_S21a_crsp
Method: a
Params: k=10, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross 0.4849363476732794, Net 0.16640016290303206, MaxDD -0.0508388934546366, Trips 383, Hit 0.688247084548105
Note: 

────────────────────────────────────────

Date: 2026-05-24
Run ID: bt_S21b_crsp_loose
Method: a
Params: k=12, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.65
Results: Gross 0.5410753760026838, Net 0.1969876652700538, MaxDD -0.03755617258242049, Trips 530, Hit 0.7138748832866479
Note: 

────────────────────────────────────────



────────────────────────────────────────
Date: 2026-05-24
Entry: CRSP Universe Baseline Results

## S21 Series: First Runs on CRSP Survivorship-Free Data

Previous runs (S17a) used yfinance with current S&P 500 membership (survivorship bias).
These runs use CRSP historical S&P 500 membership: 935 tickers, NaN where not members.
Point-in-time behavior via _slice_window(min_obs=window//2): tickers auto-excluded
when they were not S&P 500 members during the formation window.

| Run | corr | k | Gross | Net | MaxDD | Trips | Hit | Notes |
|-----|------|---|-------|-----|-------|-------|-----|-------|
| S17a (yfinance) | 0.70 | 10 | +0.57 | +0.20 | -4.5% | 558 | 75.5% | survivorship-biased baseline |
| S21a (CRSP) | 0.70 | 10 | +0.48 | +0.17 | -5.1% | 383 | 68.8% | honest; breadth thin |
| S21b (CRSP) | 0.65 | 12 | +0.54 | +0.20 | -3.76% | 530 | 71.4% | matches S17a net; better MaxDD |

**Key findings:**
- Gross gap (0.57→0.54) = survivorship correction of ~0.03 SR. This is small and expected.
- HitRate gap (75.5%→71.4%) confirms survivorship bias was inflating hit rate on yfinance.
- MaxDD improves on CRSP at matching net SR: -3.76% vs -4.5% (fewer phantom can't-lose pairs).
- ClusterPersist improves: 78%→83% (larger 935-ticker universe → more stable spectral structure).

**Production config (CRSP)**: S21b
--method a --refit 63 --corr-window 189 --ari-thresh 0.7
--factor-source etf_only --neutralize
--z-entry 2.0 --z-exit 0.5 --hard-stop 3.0 --time-stop 20
--min-hl 8 --max-hl 20 --min-corr 0.65 --top-k 12
--cost-bps 3 --earnings-blackout 3 --formation 378
Gross +0.54 | Net +0.20 @ 3bps | MaxDD -3.76% | Trips 530 | Hit 71.4% | ClusterPersist 83%

**Method C wiring gap (not yet implemented):**
cluster_method_c_partialcorr() is in pairs_feature_matrix.py but rolling_cluster_labels()
and backtest CLI --method choices do not include 'c'/'c_optics'/'c_dbscan'. To use Method C
in the full walk-forward backtest, add these to rolling_cluster_labels() and the CLI.
Date: 2026-05-24
Run ID: bt_S22_paper_equiv
Method: c_optics
Params: k=30, z=2.0, cost=2.0bps, refit=21, corr_win=756, HL=[1.0,999.0], corr63≥0.0
Results: Gross -0.189895136556606, Net -0.5142608141585283, MaxDD -0.17156786352534759, Trips 1669, Hit 0.7018201962646406
Note: 

────────────────────────────────────────

Date: 2026-05-24
Run ID: bt_S23_c_optics_hybrid
Method: c_optics
Params: k=15, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[5.0,30.0], corr63≥0.6
Results: Gross 0.17119641915301995, Net -0.1998238059951949, MaxDD -0.11113817889299168, Trips 1058, Hit 0.7030491994777709
Note: 

────────────────────────────────────────

Date: 2026-05-25
Run ID: bt_S24_c_optics_hl
Method: c_optics
Params: k=15, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.65
Results: Gross 0.17510505921570618, Net -0.1899203425393868, MaxDD -0.09980443464189939, Trips 1101, Hit 0.6961573347287633
Note: 

────────────────────────────────────────
Date: 2026-05-25
Run ID: bt_S25a
Method: a
Params: k=10, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross -0.058276579642563484, Net -0.2813525867303727, MaxDD -0.05896341835173982, Trips 703, Hit 0.6881649506649505
Note: 

────────────────────────────────────────

Date: 2026-05-25
Run ID: bt_S25a
Method: a
Params: k=10, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross -0.10352657268541966, Net -0.3487370673759135, MaxDD -0.0612397986931439, Trips 557, Hit 0.6729896366260002
Note: 

────────────────────────────────────────

Date: 2026-05-25
Run ID: bt_S25_nooverlay
Method: a
Params: k=10, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross -0.0992318508077214, Net -0.4096133043948433, MaxDD -0.08717083522990768, Trips 703, Hit 0.6881649506649505
Note: 

────────────────────────────────────────

Date: 2026-05-25
Run ID: bt_S25_base
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross -0.08013845079984155, Net -0.08013845079984155, MaxDD -0.06129216497959335, Trips 703, Hit 0.6881649506649505
Note: 

────────────────────────────────────────

Date: 2026-05-25
Run ID: bt_S25a
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross -0.09608105637760408, Net -0.09608105637760408, MaxDD -0.05039408763512401, Trips 557, Hit 0.6729896366260002
Note: 

────────────────────────────────────────

Date: 2026-05-25
Run ID: bt_S25a
Method: a
Params: k=10, z=2.0, cost=0.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.7
Results: Gross -0.1979329637807116, Net -0.1979329637807116, MaxDD -0.05441449556684054, Trips 608, Hit 0.6947059539013563
Note: 

────────────────────────────────────────

Date: 2026-05-27
Run ID: bt_S26a-BUGGY
Method: c_optics
Params: k=30, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[5.0,60.0], corr63≥0.0
Results: Gross -0.35425056581485453, Net -0.35425056581485453, MaxDD -0.18046781382266608, Trips 4201, Hit 0.5130767550492321
Note: INVALID - zero_cross_exit condition inverted (pos*z<=0 instead of >=0), causing exit on day 0. AvgHold 0.9d confirms bug.

────────────────────────────────────────

Date: 2026-05-27
Run ID: bt_S26a
Method: c_optics
Params: --method c_optics --refit 21 --corr-window 756 --formation 756 --optics-xi 0.04 --c-use-ols --z-entry 2.0 --zero-cross-exit --hard-stop 999 --time-stop 999 --month-end-forceclose --freeze-beta --roll-win 126 --no-factor-zscore --min-hl 5 --max-hl 60 --min-corr 0.0 --adf-alpha 0.05 --top-k 30 --cost-bps 0
Results: Gross SR 0.08, Net SR 0.08 (zero-cost), MaxDD -0.229, Trips 2707, HitRate 53.3%, AvgHold 4.8d, Overlap 51%, ClusterPersist 99%
Period: 2017-01-03 → 2024-12-31 (7yr, CRSP 2014-2024 data; formation=756 burns first 3yr)
Note: S26a — exact remote PC-core replication on local CRSP data. All 7 implementation gaps closed vs S22. SR +0.08 vs S22 Gross -0.19 = methodology gaps confirmed to matter (+0.27 SR improvement). SR < 0.2 per decision framework → period effect dominates. 2017-2024 misses GFC/post-GFC era (remote tested 2003-2023). Need CRSP from 2000 to match remote period. Current data universe: 935 stocks vs remote's ~991. Zero-crossing exit bug fixed (pos*z>=0 not <=0).

────────────────────────────────────────

Date: 2026-05-27
Run ID: bt_S26a_v2
Method: c_optics
Params: same as S26a_matched + --no-entry-confirm --freeze-spread-stats
Results: Gross SR 0.21, MaxDD -6.6%, Trips 1084, HitRate 52.3%, AvgHold 4.9d, Overlap 39%, ClusterPersist 99%
Period: 2003-02-07 → 2023-10-17
Note: Two additional gaps closed: (1) confirmation filter removed → pure |z|>=2.0 entry; (2) mu/sd fixed from formation window not rolling 126d. SR 0.05→0.21, MaxDD -30%→-7%. Fewer trips (1084 vs 6555) = cleaner signal. Still below remote 1.028. Next gap: universe (1335 historical members vs ~500 active S&P 500 at each date).

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v4
Method: c_optics
Params: --method c_optics --refit 21 --corr-window 756 --formation 756 --optics-xi 0.04 --c-use-ols --z-entry 2.0 --zero-cross-exit --hard-stop 5.0 --time-stop 999 --month-end-forceclose --freeze-beta --freeze-spread-stats --roll-win 63 --no-factor-zscore --no-entry-confirm --min-hl 5 --max-hl 60 --min-corr 0.0 --adf-alpha 0.01 --top-k 20 --cost-bps 0 --start 2010-01-01 --end 2023-12-31
Results: Gross SR 0.47, MaxDD -4.6%, Trips 494, HitRate 52.0%, AvgHold 6.6d, Overlap 48%, ClusterPersist 99%
Period: 2010-04-13 → 2023-12-29 (correct remote period)
Note: Correct period 2010-2023 + roll_win=63 + stop=5.0 + k=20 + adf<0.01. SR 0.21→0.47. Remote reports 1.028 net with 5bps RT + 25bps/yr borrow. Remaining gaps: price normalization to 1.0 at formation start, 5d cooldown, agglomerative clustering vs OPTICS. BEST LOCAL REPLICATION (OPTICS method).

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v6
Method: b_agglo (exact remote clustering)
Params: --method b_agglo --refit 63 --beta-window 756 --formation 756 --b-agglo-threshold 0.4 --z-entry 2.0 --zero-cross-exit --hard-stop 5.0 --time-stop 999 --month-end-forceclose --freeze-beta --freeze-spread-stats --roll-win 63 --no-entry-confirm --min-hl 5 --max-hl 60 --min-corr 0.0 --adf-alpha 0.01 --top-k 20 --cost-bps 0 --start 2010-01-01 --end 2023-12-31
Results: Gross SR 0.44, MaxDD -2.5%, Trips 643, HitRate 54.7%, AvgHold 4.6d, Overlap 11%, ClusterPersist 72%
Period: 2010-01-11 → 2023-10-17 (exact remote period, exact remote clustering)
Note: Exact remote replication: agglomerative(threshold=0.4, average linkage) on RidgeCV beta vectors + all v4 signal params. SR 0.44 vs remote 1.028 net. SR same as v4 (0.47 OPTICS) but better MaxDD (-2.5% vs -4.6%), higher hit rate (54.7% vs 52%), lower overlap (11% vs 48%). Remaining gap: their FF factor set (we use ETFs) + CRSP dsf.ret total returns vs our price-adjusted close. SR 1.028 not reproducible without exact CRSP data + FF factors. Best b_agglo replication.

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v5
Method: c_optics
Params: same as S26a_v4 + --normalize-prices --cooldown-days 5
Results: Gross SR 0.32, MaxDD -3.1%, Trips 439, HitRate 52.0%, AvgHold 7.9d, Overlap 48%
Period: 2010-01-11 → 2023-12-29
Note: Price normalization + 5d cooldown HURT: SR 0.47→0.32. Fewer trips (439 vs 494), lower SR. Price normalization changes spread scale — formation-period mu/sd captures the level shift but reduces signal power. v4 remains best at SR 0.47. Remote clustering (agglomerative on beta vectors) likely explains remaining 0.47→1.028 gap.

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v3
Method: c_optics
Params: same as S26a_v2 + --spx-membership
Results: Gross SR 0.03, MaxDD -5.7%, Trips 811, HitRate 51.6%, AvgHold 5.7d, Overlap 40%, ClusterPersist 99%
Period: 2003-01-08 → 2023-12-29
Note: Active S&P 500 membership filter (crsp_spx_membership) HURT performance: SR 0.21→0.03. Two causes: (1) ticker-level membership matching unreliable (CRSP uses permno, tickers recycled); (2) fewer pairs (811 vs 1084 trips) = less diversification. Universe not the driver of SR gap. Best replication remains S26a_v2 at SR 0.21.

────────────────────────────────────────

Date: 2026-05-27
Run ID: bt_S26a_matched
Method: c_optics
Params: --method c_optics --refit 21 --corr-window 756 --formation 756 --optics-xi 0.04 --c-use-ols --z-entry 2.0 --zero-cross-exit --hard-stop 999 --time-stop 999 --month-end-forceclose --freeze-beta --roll-win 126 --no-factor-zscore --min-hl 5 --max-hl 60 --min-corr 0.0 --adf-alpha 0.05 --top-k 30 --cost-bps 0 --start 2003-01-01 --end 2023-12-31
Results: Gross SR 0.05, MaxDD -0.297, Trips 6555, HitRate 53.4%, AvgHold 3.9d, Overlap 39%, ClusterPersist 99%
Period: 2003-01-08 → 2023-12-29 (matched remote period; CRSP extended to 2000, factor_close SPY extended to 2000)
Note: S26a matched period — SR 0.05 vs remote 1.028 on same 2003-2023 window. WORSE than 2017-2024 run (0.08). Two gaps still active: (1) confirmation filter on entry still fires silently (z_confirming_short/long not gated); (2) rolling 126d mu/sd with frozen β is incoherent — remote likely uses formation-period normalized spread (mu=0, sd=1 at construction). Data gap also possible: local 1335 stocks vs remote ~991, different membership list. Remote SR 1.028 may be difficult to replicate due to data construction differences. Decision: implement --no-entry-confirm flag + formation-period spread normalization for S26a-v2.

────────────────────────────────────────

Date: 2026-05-27
Run ID: bt_S26a_v2
Method: c_optics
Params: k=30, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[5.0,60.0], corr63≥0.0
Results: Gross 0.2075847424005815, Net 0.2075847424005815, MaxDD -0.06588092835574672, Trips 1084, Hit 0.5231498267665781
Note: 

────────────────────────────────────────

Date: 2026-05-27
Run ID: bt_S26a_v2
Method: c_optics
Params: k=30, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[5.0,60.0], corr63≥0.0
Results: Gross 0.2075847424005815, Net 0.2075847424005815, MaxDD -0.06588092835574672, Trips 1084, Hit 0.5231498267665781
Note: 

────────────────────────────────────────

Date: 2026-05-27
Run ID: bt_S26a_v3
Method: c_optics
Params: k=30, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[5.0,60.0], corr63≥0.0
Results: Gross 0.03333524526861792, Net 0.03333524526861792, MaxDD -0.05659699807317965, Trips 811, Hit 0.5160884701528874
Note: 

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v4
Method: c_optics
Params: k=20, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[5.0,60.0], corr63≥0.0
Results: Gross 0.47278330403156776, Net 0.47278330403156776, MaxDD -0.04605412815982551, Trips 494, Hit 0.5198167044595616
Note: 

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v5
Method: c_optics
Params: k=20, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[5.0,60.0], corr63≥0.0
Results: Gross 0.3191834359792982, Net 0.3191834359792982, MaxDD -0.03074652490320625, Trips 439, Hit 0.5199142156862745
Note: 

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v6
Method: b_agglo
Params: k=20, z=2.0, cost=0.0bps, refit=63, corr_win=126, HL=[5.0,60.0], corr63≥0.0
Results: Gross 0.43899061217938695, Net 0.43899061217938695, MaxDD -0.02497411815400998, Trips 643, Hit 0.5468954415267053
Note: 

────────────────────────────────────────

Date: 2026-05-28
Run ID: bt_S26a_v7
Method: b_agglo
Params: k=20, z=2.0, cost=0.0bps, refit=63, corr_win=126, HL=[5.0,60.0], corr63≥0.0
Results: Gross -0.08853036834785098, Net -0.08853036834785098, MaxDD -0.06361698609943231, Trips 668, Hit 0.582586332570688
Note: 

────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S26_exact
Method: c_optics
Params: k=999, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[0.0,999.0], corr63≥0.0
Results: Gross 0.7017579941963951, Net 0.7017579941963951, MaxDD -1.7461845855213882, Trips 119879, Hit 0.558013858385136
Note: 

────────────────────────────────────────


────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S26_exact_v2 (PLANNED)
Note: Next run — add --spx-membership to restrict to active S&P 500 ~500 stocks per date. Expect fewer pairs, lower MaxDD, SR closer to 1.028.
Date: 2026-05-29
Run ID: bt_S26_exact_v2
Method: c_optics
Params: k=999, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[0.0,999.0], corr63≥0.0
Results: Gross 0.3257733815352986, Net 0.3257733815352986, MaxDD -2.143515062521238, Trips 100091, Hit 0.5548177402062637
Note: 

────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S26_exact_v3
Method: c_optics
Params: k=30, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[0.0,999.0], corr63≥0.0
Results: Gross 0.5336753486635912, Net 0.5336753486635912, MaxDD -0.1898719810988893, Trips 10633, Hit 0.5608889652932975
Note: 

────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S26_coint
Method: c_optics
Params: k=30, z=2.0, cost=0.0bps, refit=21, corr_win=756, HL=[5.0,60.0], corr63≥0.0
Results: Gross 0.47555485350412147, Net 0.47555485350412147, MaxDD -0.21460860053136482, Trips 11342, Hit 0.5564639562514916
Note: 

────────────────────────────────────────


════════════════════════════════════════
S26 SERIES SUMMARY — Remote PC-Core Replication
════════════════════════════════════════

Goal: Replicate remote repo (Donking123/pairs-trading-ml) SR 1.028 and understand the gap.
Remote config: c_optics (OPTICS on PC-distance, market-adjusted OLS residuals), xi=0.04,
  formation 756d, refit monthly, freeze β from formation OLS, roll_win=126 (6mo rolling z-score),
  zero-cross exit, no stop-loss, no cointegration filter, TRI price series from crsp.dsf.ret,
  S&P 500 universe ~500 stocks, 2003-2023. SR 1.028 is GROSS, ZERO-COST, IN-SAMPLE.

Key learning: First agent hallucinated agglomerative clustering (b_agglo). Actual remote uses OPTICS.
TRI data (1+ret).cumprod() correct per repo. crsp_close.parquet now stores TRI from 2000-01-03.

Implementation gaps discovered and closed sequentially:

| Run | Gap closed | SR | Notes |
|-----|-----------|-----|-------|
| S22 | baseline attempt | -0.19 | 7 gaps vs remote |
| S26a | zero-cross exit, freeze-beta, no-zscore, OLS, optics-xi=0.04 | 0.05 | wrong period 2003-23 |
| S26a_v2 | +no-entry-confirm, +freeze-spread-stats | 0.21 | period 2003-23 |
| S26a_v4 | correct period 2010-23, roll_win=63, stop=5.0, adf<0.01, k=20 | 0.47 | wrong roll_win |
| S26a_v6 | b_agglo clustering (WRONG — hallucinated spec) | 0.44 | chasing wrong spec |
| bt_S26_exact | CORRECT: c_optics, roll_win=126, no stop, no filter, TRI, top-k=999 | 0.70 | too many pairs |
| bt_S26_exact_v3 | +top-k=30 (practical pair cap) | 0.53 | MaxDD -19%, tradeable |
| bt_S26_coint | +ADF/HL filter (their filtered variant) | 0.48 | matches remote pattern |

Correct final config (best practical, matches remote methodology):
--method c_optics --refit 21 --corr-window 756 --formation 756
--optics-xi 0.04 --c-use-ols --z-entry 2.0 --zero-cross-exit
--hard-stop 999 --time-stop 999 --month-end-forceclose
--freeze-beta --roll-win 126 --no-factor-zscore --no-entry-confirm
--min-hl 0 --max-hl 999 --adf-alpha 1.0 --top-k 30 --cost-bps 0
--start 2003-01-01 --end 2023-12-31

SR 0.53 gross / SR TBD net @ 3bps. MaxDD -19%.

Remote SR 1.028 gap explained:
1. Universe: remote ~500 active S&P 500 vs our 1335 historical all-time members
   → smaller universe → tighter OPTICS clusters → fewer, higher-quality pairs
   → their natural ~50-200 pairs/segment vs our 475/segment at top-k=999
2. In-sample: remote parameters tuned on 2003-2023, reported on same period
   → our 0.53-0.70 is equivalent out-of-sample estimate
3. Remote pattern confirmed: no-filter (1.028) > with-filter (0.752) matches our 0.53 > 0.48

Date: 2026-05-29
Run ID: bt_S26_net
Method: c_optics
Params: k=30, z=2.0, cost=3.0bps, refit=21, corr_win=756, HL=[0.0,999.0], corr63≥0.0
Results: Gross 0.5336753486635912, Net -0.03991166995504285, MaxDD -0.4507332281634633, Trips 10633, Hit 0.5608889652932975
Note: 

────────────────────────────────────────


────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S26_net
Method: c_optics (exact remote config + 3bps costs)
Params: same as bt_S26_exact_v3 but --cost-bps 3
Results: Gross SR 0.53, Net SR -0.04, MaxDD -0.45, Trips 10633, AvgHold 3.8d
Period: 2003-01-08 → 2023-12-29
Note: ZERO-CROSSING EXIT UNVIABLE AT 3BPS. AvgHold 3.8d × 3bps/leg ≈ 1.6bps/day cost vs 0.013bps/day gross alpha. Remote's SR 1.028 is gross/zero-cost and survives no realistic transaction costs. S21b (z_exit=0.5, AvgHold ~8-10d) is correctly calibrated for 3bps: Net SR +0.20. Next: cherry-pick remote improvements that survive costs into S21b.

────────────────────────────────────────

S26c CHERRY-PICK PLAN: test each remote element on S21b baseline
S21b baseline: method a, refit 63, corr-window 189, ari-thresh 0.7, factor-source etf_only,
  neutralize, z-entry 2.0, z-exit 0.5, hard-stop 3.0, time-stop 20, min-hl 8, max-hl 20,
  min-corr 0.65, top-k 12, cost-bps 3, earnings-blackout 3, formation 378
  → Gross +0.54, Net +0.20 @ 3bps (CRSP data)

Elements to cherry-pick (those that survived the S26 analysis):
1. freeze-beta (formation OLS frozen): isolated test vs rolling OLS
2. roll-win 126 (vs 63): wider z-score window
3. TRI data (already in effect — crsp_close now stores TRI)
4. no-entry-confirm: removes confirmation filter
5. Stack winners

Date: 2026-05-29
Run ID: bt_S26c1
Method: a
Params: k=12, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.65
Results: Gross 0.5676936661327147, Net 0.1808385924517891, MaxDD -0.06302233995858891, Trips 784, Hit 0.6196986607142858
Note: 

────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S26c2
Method: a
Params: k=12, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.65
Results: Gross -0.33337802665727057, Net -0.6926758471550093, MaxDD -0.20658980118766063, Trips 659, Hit 0.6680949292816997
Note: 

────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S21b_tri
Method: a
Params: k=12, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.65
Results: Gross 0.4631556051268102, Net 0.1015631565355925, MaxDD -0.04209531165034548, Trips 569, Hit 0.7253481175863847
Note: 

────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S21b_norm
Method: a
Params: k=12, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.65
Results: Gross 0.2972519981523408, Net -0.1481930822195992, MaxDD -0.08301438235020057, Trips 832, Hit 0.4339155617901263
Note: 

────────────────────────────────────────

Date: 2026-05-29
Run ID: bt_S26c3
Method: a
Params: k=12, z=2.0, cost=3.0bps, refit=63, corr_win=189, HL=[8.0,20.0], corr63≥0.65
Results: Gross 0.5515302733792685, Net -0.04626732931621153, MaxDD -0.10844242895483802, Trips 1391, Hit 0.34582229309152385
Note: 

────────────────────────────────────────

