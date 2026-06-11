# Machine Learning for Pairs Trading: A Clustering-Based Approach with a Factor-Beta Extension

**QF621 Term Project.** Draft 2026-06-07. All results final (Phases 0–4 complete, incl. the
out-of-sample forward test and the net-of-cost optimisation). Remaining work is submission
formatting to the QF621 spec.

---

## Abstract

We replicate Rotondi & Russo (2025), *Machine Learning for Pairs Trading: a Clustering-based
Approach*, on a survivorship-bias-free CRSP S&P 500 universe spanning 2000–2023, and extend it
with an original **factor-beta clustering** metric. The replication reproduces the paper's
headline: a partial-correlation (PC) distance fed to OPTICS clustering yields an annualised
Sharpe of **1.028**, inside the paper's 1.01 ± 0.15 band. Our extension — clustering stocks by
their **risk-factor exposure vectors** rather than by return co-movement — independently
attains **1.013**, demonstrating that the headline is not an artefact of one similarity measure.
A robustness battery shows both strategies are insensitive to the hedge-ratio estimator and the
position-sizing rule, but sensitive to the clustering algorithm through its selectivity; notably
the factor-beta metric is the *sturdier* of the two. A black-box lookahead-bias audit passes on
every configuration tested; a realistic cost model (actual bid/ask spreads, borrow, and a stop)
leaves both strategies net-positive at a Sharpe near 0.57; and on a true out-of-sample window
(frozen, 2024–2025) the PC strategy generalises with a 0.858 Sharpe, though the factor-beta
extension does not (0.117) and performance shows marked year-to-year regime dependence.

## 1. Introduction

Pairs trading is a relative-value strategy: take offsetting long/short positions in two
economically related securities when their price spread diverges, and profit as it mean-reverts.
The central difficulty is *selection* — an S&P 500 universe admits on the order of 10^5 candidate
pairs, the vast majority of which share no stable relationship. Rotondi & Russo (2025) address
this with unsupervised learning: cluster stocks by a similarity metric and trade only
within-cluster pairs, so the universe of candidates collapses to a few hundred economically
coherent ones.

This project makes four contributions. First, a faithful from-scratch replication of the paper's
pipeline on independently sourced WRDS data. Second, an original **factor-beta** distance metric
that clusters stocks by their estimated exposures to a panel of risk factors — a different and
economically motivated notion of "relatedness." Third, a robustness battery that converts the
headline point estimates into confidence intervals. Fourth, in place of a live paper-trading
exercise, an automated **lookahead-bias audit** that validates the strategy's temporal integrity
directly.

## 2. Data

All data are drawn from WRDS: CRSP daily stock files, historical S&P 500 constituents, the
Fama-French factor series, and the CRSP delisting file. We restrict to ordinary US common stock
(share codes 10 and 11), which yields a **survivorship-bias-free universe of 991 names** over
2000-01-03 to 2023-12-29 (6,037 trading days). The reduction from ~1,070 raw names is fully
accounted for by the removal of foreign-incorporated issues, REITs, and funds. Because the
strategy uses a rolling three-year formation window, the first three years are reserved for
formation and the out-of-sample trading window runs 2003–2023, giving **251 monthly returns**.
Prices are dividend-reinvested total-return indices; index membership is applied point-in-time.

## 3. Methodology

### 3.1 Distance metrics
We implement three pluggable similarity metrics, each returning a square distance matrix:
- **SSD** (sum of squared deviations of z-normalised prices) — the simple price-trajectory
  baseline.
- **PC** (one minus the correlation of market-adjusted, i.e. idiosyncratic, returns) — the
  paper's preferred metric, which strips out common market exposure.
- **Factor-beta** (our extension) — for each stock we ridge-regress its daily returns on 18 risk
  factors (six Fama-French style factors and twelve Fama-French 12-industry factors, all
  constructed from our own data) and cluster on the standardised vector of regression betas.
  Two stocks are "close" if they load similarly on the same risks, so a shock to a shared factor
  moves both alike and their spread mean-reverts.

### 3.2 Clustering
Distances are clustered with OPTICS (density-based, precomputed metric, minimum cluster size 2).
The steepness parameter ξ is tuned per metric on December 2015 and 2023 *before* any backtest, to
avoid choosing it on the basis of returns.

### 3.3 Cointegration filter
As a tested option (not a mandatory gate) we apply an Engle-Granger filter: the spread must reject
a unit root at the 5% level and exhibit an AR(1) half-life between 5 and 60 trading days.

### 3.4 Backtest engine
The engine rolls a three-year formation / one-month trading window. Signals use a six-month
rolling z-score of the spread with strict look-ahead protection (the rolling mean/standard
deviation are lagged one day). A position opens when |z| > 2 and closes at the zero-crossing;
any open position is force-closed at month-end. Execution is t+1 close-to-close; legs are
equal-dollar; delistings use a code-dependent fallback return. Crucially, each month is
self-contained, which both matches the paper and underpins the lookahead audit of §7.

## 4. Results: replication and extension

### 4.1 Replication
| Cell | Sharpe | Paper target | Verdict |
|---|---:|---:|---|
| SSD core | 0.589 | 0.88 ± 0.15 | below (return-only gap; risk profile matches) |
| **PC core** | **1.028** | **1.01 ± 0.15** | ✅ matches |
| PC + cointegration filter | 0.752 | 0.80 ± 0.15 | ✅ matches |
| SSD + cointegration filter | 0.731 | 0.75 ± 0.15 | ✅ matches |

The PC core reproduces the paper's headline almost exactly. The SSD core sits below the paper's
band, but the shortfall is confined to the return numerator — volatility, drawdown, and hit rate
all match — and is documented rather than tuned away.

### 4.2 Extension: factor-beta clustering
| Cell | Sharpe | Comparison |
|---|---:|---|
| **Factor-beta core** | **1.013** | ≈ PC core 1.028 (Δ −0.014) |
| Factor-beta + cointegration filter | 0.858 | > PC + filter 0.752 |

The factor-beta metric, an independently constructed and economically motivated similarity
measure, reproduces the ~1.0 headline. Its estimated betas are sensible (energy names load on the
energy factor, semiconductors on business equipment, banks on the financial factor), and its
filtered variant outperforms the PC filtered variant.

### 4.3 P&L attribution
Profit is bimodal: roughly 9–11% of trades are clean reversions averaging about +400 bps, while
roughly 90% force-close at month-end for a small loss. The PC and factor-beta metrics cut the
force-close drag to about −12 bps per trade (versus −32 bps for SSD), which is the mechanism
behind their higher Sharpe. Both metrics pair predominantly within-industry (~80% of trades).

## 5. Robustness

We re-ran the two headline strategies under deliberately different modelling choices.

| Variant | PC Sharpe | Factor Sharpe |
|---|---:|---:|
| baseline (OPTICS / OLS / equal-weight) | 1.028 | 1.013 |
| robust (RLM) hedge ratio | 1.046 | 1.060 |
| \|entry-z\|-weighted allocation | 1.012 | 1.027 |
| hierarchical clustering | 0.485 | 0.991 |
| HDBSCAN clustering | 0.616 | 0.615 |
| **band** | **0.485–1.046** | **0.615–1.060** |

Two findings stand out. First, both headlines are essentially unchanged under a robust hedge
ratio and under |entry-z|-weighted sizing — the result does not hinge on how the spread is
estimated or how capital is allocated. Second, the sensitivity is to the *clustering algorithm*,
and it traces to **selectivity**: HDBSCAN and hierarchical clustering produce roughly three times
as many candidate pairs as OPTICS (denser clusters of comparable purity), so they trade a larger,
more diluted set. Importantly, the two metrics differ in degree — PC collapses under hierarchical
clustering (0.485, with a −11.8% drawdown) whereas factor-beta holds (0.991, its tightest
drawdown), making **factor-beta the more robust metric**. A separate sweep confirms the locked ξ
and ridge-α sit on a stable plateau rather than a cliff edge.

## 6. Realism

We re-ran the headline strategies with realistic frictions: transaction costs equal to half the
*actual* CRSP bid/ask spread on each leg at entry and exit (these vary realistically from ~27 bps
in the early 2000s to ~2.5 bps by the mid-2010s), a 35 bps annual borrow fee on the short leg, and
a 3.5σ stop-loss.

| Strategy | Frictionless | Net of costs | Δ | Max drawdown |
|---|---:|---:|---:|---:|
| PC core | 1.028 | 0.572 | −0.456 | −9.8% |
| Factor-beta core | 1.013 | 0.578 | −0.436 | −6.6% |

Realistic costs cut the Sharpe by roughly 45%, but both strategies remain net-positive at about
0.57 — a genuine risk-adjusted edge survives. The drag concentrates in the wide-spread early
2000s and in the high-turnover "force-close" trades that barely cover their round-trip cost.

### 6.1 Net-of-cost optimisation
Because the drag is driven by churn on marginal trades and by wide early-period spreads, we
tested principled cost-reduction levers and ranked them by net Sharpe (baseline = marketable
execution + 3.5σ stop):

| Lever | PC net Sharpe | Factor net Sharpe |
|---|---:|---:|
| **passive execution** (½ spread) | **0.782** | **0.773** |
| drop the 3.5σ stop | 0.683 | 0.717 |
| baseline | 0.572 | 0.578 |
| higher entry (\|z\|>2.5 / >3.0) | 0.51 / 0.40 | 0.56 / 0.54 |
| cointegration filter | 0.289 | 0.583 |

The conclusion is consistent across both strategies. Two levers help: **passive (limit-order)
execution**, which recovers ~0.20 of Sharpe by crossing less of the bid/ask spread, and **dropping
the stop-loss**, worth +0.11–0.14 because its extra round-trips cost more than the tail protection
saved. The cointegration filter and higher entry thresholds *reduce* net Sharpe — they shed more
alpha than the turnover they save (the filter's gross Sharpe was already only 0.752 vs the core's
1.028). Notably these are not data-mined parameters: passive execution is an execution-realism
assumption appropriate to monthly rebalancing, and removing the stop is a structural choice — so
the improvement to ~0.78 is a defensible operating point rather than an overfit. (A direct
passive-plus-no-stop combination was not run but would be expected to improve further; the
`combo` cell underperformed passive alone only because it also bundled the harmful filter.)

## 7. Validation: lookahead-bias audit

Following course guidance that paper trading is unnecessary for a monthly-frequency strategy, we
instead test the only property paper trading would have guaranteed — freedom from lookahead bias —
directly and as a black box. We run the backtest over the full window and again over truncated
windows, and require that every overlapping day's target positions be identical; if cutting off
the future changes a past position, the strategy is using future information. Across both metrics
and three cut dates (2009, 2013, 2017) the test returns **6/6 PASS** with zero mismatches (up to
3,866 pairs over 3,776 overlapping days), confirming the look-ahead protections — t+1 execution,
the rolling formation window, the lagged z-score, and point-in-time universe and delisting
handling — all hold.

### 7.1 True out-of-sample forward test
The audit above rules out lookahead bias but not period-specificity: every result lives inside
2003–2023, and because ξ and ridge-α were tuned on December-2023 windows, 2023 is not a clean
holdout. We therefore ran the **frozen** strategies (no re-tuning) on **2024–2025** — ~23 months
of data past the development sample, sourced from CRSP's current CIZ tables — and compared the
forward Sharpe to the in-sample headlines.

| Strategy | In-sample (2003–2023) | OOS 2024 | OOS 2025 | **OOS full (2024–2025)** |
|---|---:|---:|---:|---:|
| **PC** | 1.028 | +0.16 | +1.40 | **0.858** |
| Factor-beta | 1.013 | −0.46 | +0.47 | **0.117** |

Three findings. First, **the PC strategy generalises**: its out-of-sample Sharpe is 0.858, close
to the in-sample 1.028, and in 2025 it actually exceeded it (1.40) — strong evidence that the
replication's edge is real rather than an in-sample artefact. Second, **factor-beta generalises
poorly** (0.117) — a notable reversal, since it was the *sturdier* metric under in-sample
robustness perturbations yet the weaker one on genuinely unseen data. Third, there is large
**year-to-year regime variation**: both strategies were weak in 2024 — a calm, strongly trending,
low-dispersion market, the regime least suited to mean reversion — and recovered in 2025,
consistent with the strategy's crisis/dispersion dependence (30–40% of in-sample P&L came from
the 2007–09 crisis).

This also carries a methodological lesson: the 2024-only window (PC ≈ 0.16) would have wrongly
suggested the edge had vanished; only the fuller 23-month read reveals that PC in fact holds up.
A single short out-of-sample window is unreliable — consistent with the guidance to evaluate
generalisation over a sufficient sample.

## 8. Conclusion, limitations, future work

The paper's ~1.0 Sharpe replicates cleanly in-sample (PC 1.028) and is corroborated by an
independent factor-beta metric (1.013). The strategy is free of lookahead bias, retains a
positive ~0.57 Sharpe after realistic costs, and — importantly — **the PC strategy generalises
out-of-sample**, posting a 0.858 Sharpe on frozen 2024–2025 data (close to the in-sample 1.028).
Two honest caveats temper this. First, performance is **regime-dependent**: ~30–40% of in-sample
P&L came from the 2007–09 crisis, and both strategies were weak in the calm, trending 2024 before
recovering in 2025. Second, the **factor-beta extension does not generalise** as well (0.117 OOS)
despite being the sturdier metric under in-sample perturbations — a caution against equating
in-sample robustness with out-of-sample reliability. Other limitations: the headline depends on
clustering selectivity, and the SSD baseline underperforms the paper on returns. Natural next
steps: a longer forward window as more data accrues; a regime/dispersion filter; isolating
"clustering algorithm" from "pair count"; and orthogonal commodity factors.

## References
- Rotondi & Russo (2025), *Machine Learning for Pairs Trading: a Clustering-based Approach.*
- K. R. French, Fama-French factor and 12-industry (Siccodes12) definitions.

## Appendix — reproducibility
Source modules under `src/` (distances, clustering, cointegration, factors, costs, lookahead,
backtest); per-phase artefacts, runners, and executed reference notebooks under `phases/`. A suite
of 53 synthetic unit tests guards the engine, and every result is regenerable from the phase
`notebooks/` runners.
