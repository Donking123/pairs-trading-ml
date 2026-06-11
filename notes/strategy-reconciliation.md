# Strategy Reconciliation — Submitted Proposal vs Rotondi & Russo Replication

The QF621 **submitted proposal** (Strategy A) is a bespoke design; our **build**
replicates **Rotondi & Russo (2025)**. This document logs the point-by-point
reconciliation of every divergence and the resulting reworded proposal text.

**Decision rule:** the paper's design is the **faithful core** (known target numbers —
annualized Sharpe SSD 0.88 / PC 1.01); the proposal's genuinely-good ideas are adopted
as **layered variants**, which strengthens the comparison & realism pillars.

**Status: all 12 items + the live-deployment framing — decided.** ✅

**2026-05-24 update:** after comparing against the *updated* submitted proposal
(`Pair_Trading_Project_Proposal_Updated.pdf`), several proposal-only choices have been
promoted from "rejected / Phase-3-conditional" to **first-class robustness extensions**
on top of the paper-faithful core. See **"Proposal-driven extensions"** section at the
bottom.

---

## Final verdict

| # | Topic | Decision |
|---|---|---|
| 1 | Clustering algorithm | **OPTICS** core; HDBSCAN + **Hierarchical avg-link (1−corr distance)** as robustness cells |
| 2 | Clustering features | **Price/return distance** (SSD, PC) core; **factor-beta vectors promoted to Phase 2.5 first-class extension** (was Phase 3 conditional) |
| 3 | Cluster stability filter | **No filter** in core; Jaccard bootstrap = optional robustness extension |
| 4 | Cointegration role | **Tested filter, not a gate**; run & report both with / without |
| 5 | Half-life filter | **Adopt [5, 60] days** — inside the cointegration-filtered variant only |
| 6 | Z-score window | **6-month** core; 60-day as sensitivity |
| 7 | Hedge ratio | **OLS, frozen** per trading month (core); **RLM (Tukey biweight)** as robustness cell; Kalman = optional upgrade |
| 8 | Rescreen / cadence | **3-yr formation / 1-mo trading**, monthly roll; weekly = sensitivity |
| 9 | Exit threshold | **Zero-cross** core; 0.5σ as sensitivity |
| 10 | Stop-loss | **No stop** in faithful core; **3.5σ stop** in the realism variant — both reported as comparison cells; 3.0σ / 4.0σ as cheap sensitivity |
| 11 | Max open pairs | **No hard cap**; capacity analysed at 10 / 20 / 50 |
| 12 | Universe building | **CRSP-only**; drop Compustat/fundamentals & the correlation filter |
| + | Live deployment | Keep the live framing — validated by **backtest + Alpaca paper-trade** forward test; drop the hard 70 ms claim |

**Pattern:** the paper's design is the faithful core; the proposal's four good ideas —
cointegration comparison, half-life, stability check, stop-loss — are layered as
variants. Two proposal choices were rejected: sector-as-clustering-feature (circular)
and the raw-correlation universe filter (redundant, paper-divergent).

---

## Decisions (detail)

**Point 1 — Clustering algorithm.** OPTICS for the core (faithful → reproduces cluster
counts 48/109/78); HDBSCAN run as a robustness check. Close density-based cousins.
`clustering.py` already uses OPTICS — no rework.

**Point 2 — Clustering features.** Cluster on price/return distance matrices (SSD, PC).
Reject feeding sector/industry in as features — circular with the purity validation.
The "economic similarity" intent is delivered by factor-beta clustering (Phase 3).

**Point 3 — Cluster stability filter.** No stability filter in the core (faithful;
avoids a multi-hour compute blow-up; redundant with the cointegration step). Jaccard
bootstrap stability kept as an optional subsampled robustness check (Phase 4).

**Point 4 — Cointegration role.** A tested filter, not a mandatory gate. Run both —
clustering-only (paper benchmark) and clustering + cointegration (proposal's Strategy
A). The paper's §4.2 shows filtering halves traded pairs and cuts PC Sharpe 1.01→0.80;
reporting both is a headline finding. Phase 2 — no rework.

**Point 5 — Half-life filter.** Adopt the [5, 60] trading-day half-life band, but only
within the cointegration-filtered variant (computed free from the same ADF
regression). Sensitivity-test the bounds.

**Point 6 — Z-score window.** 6-month (~126-day) window for the core; 60-day as a
sensitivity cell. The paper showed (§4.3) the window barely matters.

**Point 7 — Hedge ratio.** OLS β estimated on the formation window and frozen for the
1-month trading period (β barely drifts over a month). Kalman dynamic-β = optional
upgrade.

**Point 8 — Rescreen / cadence.** Core = rolling 3-yr formation / 1-mo trading, rolled
monthly (defines the ~252 monthly returns + target Sharpes). Weekly rescreen tested as
a sensitivity. See Live Deployment below.

**Point 9 — Exit threshold.** Exit on zero-cross (z=0) for the core; 0.5σ early exit
as a sensitivity. Minor.

**Point 10 — Stop-loss.** No stop in the faithful core (matches the paper's Sharpe
targets); adopt the 3.5σ stop in the realism variant — proper pair-break protection,
should lower max drawdown. Report with / without.

**Point 11 — Max open pairs.** No hard cap in the core (the paper's capital-scaled
return handles unlimited pairs). Capacity analysed in Phase 4 at caps of 10/20/50.
Paper's natural open-pair counts: SSD ~22, PC ~44, PCA ~58 — a flat "20" would bind
hard on PC/PCA.

**Point 12 — Universe building.** CRSP-only — drop "Compustat / fundamental data" (we
do not use fundamentals) and drop the rolling-252-day-correlation ≥ 0.80 filter
(redundant with clustering; correlation ≠ cointegration). Keep the $5M ADV liquidity
screen. The real binding screen is continuous index membership over the 3-yr
formation window.

**Live deployment.** Keep the live-strategy framing — made honest by validating the
*same code path* via (a) the historical backtest and (b) a **live paper-trading
forward test on an Alpaca paper account** in the final weeks. Drop the specific
"70 ms latency" claim (irrelevant for a daily strategy, unverifiable). The
forward-test + execution pipeline is a Phase 4 deliverable, **after** the replication
core.

---

## Consolidated reworded text

### Executive Summary — Strategy A line
> Strategy A – Cointegration / Clustering: Identifies mean-reverting spread
> relationships among S&P 500 equity pairs using ML-driven density-based clustering
> (OPTICS on price/return distance metrics, with HDBSCAN as a robustness check) and
> cointegration testing (Engle-Granger ADF).

### Executive Summary — table rows
> Rescreen Frequency → Monthly (rolling 3-yr formation window)
> Exit Threshold → z = 0 (zero-cross); ±0.5σ as sensitivity
> Max Open Pairs → No hard cap; capacity analysed at 10 / 20 / 50

### Executive Summary — "Low latency" bullet
> Fully automated, low-latency-capable execution. *(Drop the specific "70 ms" figure.)*

### Strategy A — Cointegration / Clustering Pair (full reworked section)

**Investment Thesis**
Many S&P 500 stocks share common fundamental drivers — overlapping business lines,
shared input costs, or the same macro sensitivities. When two such stocks are
cointegrated, their price spread is stationary: it fluctuates around a long-run
equilibrium and reverts to the mean with a measurable half-life. Strategy A
systematically identifies these relationships and trades the temporary dislocations.

The strategy is equity-market-neutral by construction: each position is long the
relatively cheap leg and short the relatively expensive leg. Net dollar exposure is
approximately zero at the pair level, isolating the spread return from broad market
movements.

**Pair Selection Pipeline**
Pair candidates are sourced from the S&P 500 constituent universe and filtered through
a three-stage offline research pipeline:

- **Universe Building:** Daily prices (close and bid/ask) and volume for S&P 500
  constituents are pulled from WRDS CRSP — point-in-time and survivorship-bias-free. A
  stock must be continuously present in the index over the 3-year formation window; a
  minimum 30-day average dollar volume of $5M screens out illiquid names.
- **ML Clustering (OPTICS):** Stocks are clustered with OPTICS — a density-based
  algorithm that needs no pre-set cluster count and labels dissimilar stocks as
  outliers — on a pairwise distance matrix built from price/return data (SSD on
  normalised prices, PC on market-adjusted returns). Industry sector is deliberately
  excluded as a clustering input, so cluster purity against sector stays an
  independent quality check. HDBSCAN and bootstrap cluster-stability are run as
  robustness checks.
- **Cointegration Validation (Engle-Granger ADF):** Pairs within approved clusters are
  tested for cointegration via the Engle-Granger two-step procedure — an OLS spread
  regression followed by an Augmented Dickey-Fuller test on the residual spread. The
  filtered variant retains only pairs with ADF p < 0.05 and a mean-reversion half-life
  of 5–60 trading days.

The pipeline rolls forward monthly — each month a fresh 3-year formation window
re-selects pairs for the next one-month trading period. Rescreen frequency (weekly vs
monthly) is evaluated as a backtest sensitivity.

**Signal Generation**
For each approved pair (A, B) with hedge ratio β:

- **Spread:** spread = price_A − β · price_B
- **Z-score:** z = (spread − μ) / σ, with μ and σ estimated over a rolling 6-month
  (~126 trading-day) window; a 60-day window is reported as a sensitivity check.
- **Long spread signal** (buy A / sell B): z ≤ −2.0
- **Short spread signal** (sell A / buy B): z ≥ +2.0
- **Exit signal:** spread crosses zero (z = 0, full mean reversion); a |z| ≤ 0.5 early
  exit is reported as a sensitivity check.
- **Stop-loss:** |z| ≥ 3.5 (spread divergence)

The hedge ratio β is estimated by OLS over the formation window and held fixed for the
one-month trading period; a Kalman-filter dynamic-β variant is explored as an optional
extension.

---

## Proposal-driven extensions (added 2026-05-24)

After re-comparing against `Pair_Trading_Project_Proposal_Updated.pdf`, four choices
that the proposal made differently from the paper have been adopted as **first-class
robustness extensions** on top of the paper-faithful core. None displace the core; all
get reported as comparison cells.

### E1 — Hierarchical clustering as a third algorithm
- **Method:** `sklearn.cluster.AgglomerativeClustering`, average linkage, distance =
  `1 − corr(features)`. Cut height tuned to produce clusters of 8–20 stocks (prof's
  guidance for the factor-beta feature space; for SSD/PC the natural OPTICS counts
  remain the reference).
- **Where it runs:** all three feature spaces (SSD, PC, factor-beta) — one row in the
  comparison grid.
- **Cost:** one extra call per rescreen inside `clustering.py`; negligible.

### E2 — Factor-beta clustering promoted to Phase 2.5 (was Phase 3 conditional)
- **Method:** per-stock RLM (Tukey biweight) regression on ~21 risk factors (11 SPDR
  sector ETFs: XLF/XLK/XLE/XLI/XLV/XLP/XLB/XLC/XLU/XLRE/XLY; style: VTV, MTUM;
  commodity: USO; market: SPY). Output = β vector per stock per formation window.
- **Distance:** `1 − corr(βᵢ, βⱼ)` between β vectors.
- **Clustered with:** OPTICS (core), HDBSCAN (robustness), Hierarchical (E1).
- **Regression variant:** **Ridge** as the preferred default (handles factor
  multicollinearity), with OLS and RLM-Tukey as robustness cells. (Standard OLS is
  noisy on correlated factors; ridge stabilises β → clusters churn less.)
- **Build artefact:** `factors.py` (downloads sector/style/commodity ETFs; runs
  rolling RLM/ridge/OLS; outputs β panel).
- **Note:** purity-vs-sector becomes partly circular for this feature space (we're
  clustering on sector-ETF loadings), so purity here is a sanity check, not the gate.

### E3 — RLM-Tukey hedge ratio as a robustness cell
- **Method:** swap OLS for `statsmodels.RLM(M=TukeyBiweight)` in the spread
  regression. Selection (Engle-Granger ADF) stays OLS-only — that's the EG definition.
  Only the *trading* β changes.
- **Reported alongside:** OLS β (core).

### E4 — Cadence: monthly is canonical
- The updated proposal table said "Weekly (Sundays)" but the body said monthly. The
  paper is monthly; weekly is ~4× more compute, more transaction-cost drag, and
  serially correlated returns. **Monthly is canonical**; weekly stays as a sensitivity
  row only. Update the proposal exec-summary table to match the body.

### Resulting comparison grid

| Feature space | Algorithm: OPTICS | HDBSCAN | Hierarchical |
|---|---|---|---|
| **SSD** (paper core) | ✅ CP1 anchor | ✅ robustness | ✅ robustness |
| **PC** (paper core) | ✅ CP2 anchor | ✅ robustness | ✅ robustness |
| **Factor-beta** (extension, Phase 2.5) | ✅ extension | ✅ robustness | ✅ natural fit |

Layered on top: hedge-ratio (OLS / RLM), stop-loss (none / 3.5σ ± sensitivity),
cointegration filter (on / off — already in plan).

**Execution**
The strategy runs as a fully automated execution pipeline. Each signal is first
cleared by a pre-trade Risk Engine (net exposure, per-pair and gross position limits,
short-borrow availability), then generates simultaneous orders for both legs of the
pair, routed via broker REST API (Alpaca / Interactive Brokers). An Order Management
System records fills and the Position Engine reconciles them into mark-to-market P&L.
The identical code path runs in two modes — a historical backtest (2003–2023) for
statistical validation, and a live paper-trading forward test on an Alpaca paper
account in the project's final weeks — demonstrating the strategy behaves identically
on real-time data, with no look-ahead or simulation artefacts.
