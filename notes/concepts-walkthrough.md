# Concepts & Walkthroughs — Pairs Trading ML

A topic-organised reference of the explanations & design rationale built up over
the project. Use this to refresh understanding before resuming a phase or
reviewing the codebase. **Last updated:** 2026-05-24 (Phase 2 complete).

> **Note (2026-05-24):** This doc was originally written during Phase 1; concepts
> like "what the real Dec-2023 run will produce" describe Phase 1 SSD output.
> Phase 2 walkthroughs (PC distance, cointegration filter) live in the Phase 2
> reference notebook: `phases/phase2/notebooks/phase2_complete_reference.ipynb`.

---

## Contents

1. [Project shape & phase plan](#1-project-shape--phase-plan)
2. [Key numerical conventions](#2-key-numerical-conventions)
3. [The rolling-window mechanic](#3-the-rolling-window-mechanic)
4. [Strategy reconciliation — proposal vs paper vs build](#4-strategy-reconciliation--proposal-vs-paper-vs-build)
5. [The synthetic test (Phase 1a) — full walkthrough](#5-the-synthetic-test-phase-1a--full-walkthrough)
6. [What the real Dec-2023 run will produce](#6-what-the-real-dec-2023-run-will-produce)
7. [Glossary](#7-glossary)

For **Phase 2 concept walkthroughs** (PC distance, cointegration filter, bimodal
lever check), open the Phase 2 reference notebook directly.

---

## 1. Project shape & phase plan

**Goal:** replicate Rotondi & Russo (2025) clustering-based pairs trading on
CRSP S&P 500 (2000–2023), then layer on a factor-beta clustering extension as
the original contribution.

| Phase | What it does | Status |
|---|---|---|
| **0** — Data spine | WRDS CRSP pull, survivorship-bias-free; 5 parquet panels cached | ✅ complete |
| **1** — SSD vertical slice | Whole pipeline (distances → clustering → spread → backtest → performance) on the **SSD** metric only | 🔵 in progress |
| **2** — PC distance + replication validation | Add the **PC** (partial correlation) metric; cointegration filter. CP2 = replicate paper's headline. | ⬜ |
| **2.5** — Factor-beta extension | Cluster on **risk-factor exposures** (sector ETFs + style + commodity ETFs) — the contribution | ⬜ |
| **3** — Robustness cells | Hierarchical algo, RLM hedge ratio, stop-loss variants | ⬜ |
| **4** — Realism, Alpaca forward test, writeup | Realistic costs, paper-trade forward test, post-mortem | ⬜ |

### Phase 1 internal layout

| Sub | File | Status |
|---|---|---|
| 1a | `distances.py` (SSD) + `clustering.py` (OPTICS + purity + pairs) | ✅ built + synthetic-tested; real Dec-2023 run next |
| 1b | `spread.py` — OLS hedge ratio γ, 6-month rolling z-score | ⬜ |
| 1c | `backtest.py` — rolling 3y/1m loop, t+1 exec, delisting, bid/ask, monthly returns | ⬜ |
| 1d | `performance.py` — Sharpe + Sortino + Calmar + max drawdown + hit rate | ⬜ |

### Why a "vertical slice" first
Build the *whole pipeline* end-to-end using only the simplest metric (SSD), so
we have a working backtest from cluster discovery → trade signal → P&L → Sharpe.
Once one slice replicates, every later metric / variant is a configuration swap.
Beats building each component deeply in turn — that way we'd find pipeline bugs
weeks later, when they're expensive to fix.

### CP1 (Phase 1 gate)

| Metric | Paper target | Tolerance |
|---|---|---|
| # SSD clusters (Dec 2023) | ~48 | ±5 |
| Purity vs SIC division | ~0.81 | ±0.05 |
| Annualised gross Sharpe (no costs) | 0.88 | ±0.15 |
| GOOG/GOOGL pair P&L | sane | qualitative |

---

## 2. Key numerical conventions

| Number | What it is | Where it comes from |
|---|---|---|
| **252** | Trading days per year | 365 − 104 weekends − ~9 NYSE holidays |
| **3 years** | Formation window length | Paper §4.1; long enough for business cycle, short enough that firms haven't changed identity |
| **756** | Trading days per formation window | 3 × 252 |
| **1 month** | Trading window after each formation | Paper §4.1; ~21 trading days |
| **252 months** | Total trading months Jan 2003 → Dec 2023 | 21 × 12; defines the Sharpe denominator |
| **~991** | Survivorship-bias-free stock universe | After CRSP share-code 10/11 + continuous-membership filters |
| **6 months / ~126 days** | Rolling z-score window for spread | Paper §3.2 |
| **2.0σ / 3.5σ** | Entry / stop-loss thresholds | Paper §3.3 (entry); realism variant (stop) |
| **0** | Exit threshold (zero-cross) | Paper §3.3 |

---

## 3. The rolling-window mechanic

The pipeline does *not* re-select pairs every day. The cadence is **monthly**.

```
Month 1 (Jan 2003):
├─ Formation window: Jan 2000 → Dec 2002 (3 years = 756 days)
│  ├─ Compute SSD distances
│  ├─ Run OPTICS → discover clusters
│  ├─ (optional) cointegration filter
│  └─ Estimate hedge ratio γ for each surviving pair
└─ Trade these pairs during Jan 2003 (~21 days)
   └─ Daily: compute spread, z-score, fire entry/exit signals

         ↓  end of month, slide forward 1 month  ↓

Month 2 (Feb 2003): repeat with new 3-yr window ending Jan 2003.
... and so on, monthly, until Dec 2023.
```

**Monthly slides happen** for: clustering, pair selection, cointegration test,
hedge ratio estimation.

**Daily computation happens** for: spread, z-score, entry/exit triggers — all
*using* the pair set + γ that were locked in at the start of the month.

**Why monthly, not daily** — daily re-selection would be ~5,300 reselections vs
~252 monthly ones, and pair membership would jitter day-to-day → constant
trading → drowned in transaction costs.

**Overlap:** consecutive monthly formation windows share ~35 of 36 months. Most
pairs persist month-to-month; selections shuffle slightly near regime shifts
(2008, 2020).

---

## 4. Strategy reconciliation — proposal vs paper vs build

Full detail in `strategy-reconciliation.md`. Quick summary of the four
divergences worked through on 2026-05-24:

| # | Topic | Core (paper-faithful) | Extension cell (proposal idea promoted to robustness) |
|---|---|---|---|
| 1 | Clustering algorithm | OPTICS (density-based, outliers) | **Hierarchical** (avg-link, 1−corr distance) added; HDBSCAN kept |
| 2 | Clustering features | SSD + PC distance (price-based) | **Factor-beta vectors** → Phase 2.5 first-class extension |
| 7 | Hedge ratio (univariate) | OLS, frozen | **RLM (Tukey biweight)** as robustness cell |
| 7b | Factor-beta regression (multivariate, Phase 2.5) | — | **Ridge** as preferred default (handles multicollinearity); OLS + RLM as robustness |
| 10 | Stop-loss | No stop (matches paper Sharpe) | 3.5σ realism variant — both reported; 3.0/4.0σ sensitivity |
| — | Cadence | Monthly | Weekly as sensitivity only — fix proposal exec table to say "Monthly" |

### Why these are extensions, not replacements
The paper's design is the **faithful core** with known target numbers (Sharpe
SSD 0.88 / PC 1.01). The proposal's good ideas are layered as **variants** that
strengthen the comparison & realism pillars without losing the validation
backbone.

### Key technical insight: Ridge vs RLM for different regressions

| Regression | Method | Why |
|---|---|---|
| **Hedge ratio β** (univariate: price_A on price_B) | OLS core + RLM-Tukey robustness | One regressor → no multicollinearity → ridge would just add bias. RLM handles outliers (crash days). |
| **Factor-beta loadings** (multivariate: stock on 15-30 ETFs) | **Ridge** as preferred default | 21 correlated factors → real multicollinearity → ridge stabilises β vectors → cluster membership churns less. |

---

## 5. The synthetic test (Phase 1a) — full walkthrough

File: `tests/test_clustering_synthetic.py`. Five tests; all pass on 2026-05-24.

### 5.1 Why we test before touching real data

Two modules (`distances.py`, `clustering.py`) need to be verified end-to-end
*on data where we already know the right answer* — otherwise we can't tell
whether a weird real-data result is a bug or a finding.

### 5.2 How the fake data is built (`generate_synthetic_panel`)

Plant 3 obvious clusters of 5 stocks each + 5 independent noise stocks.

```python
# Each cluster shares ONE common daily return shock (σ ≈ 1.5%/day)
# Each stock has tiny idiosyncratic noise on top  (σ ≈ 0.08%/day)
# Noise stocks have only independent random walks (σ ≈ 2.0%/day)
common = rng.normal(0.0005, 0.015, size=252)
idio   = rng.normal(0.0,    0.0008, size=252)
returns = common + idio
prices  = 100 * cumprod(1 + returns)
```

The function returns `(prices, planted_labels)` where `planted_labels` is the
ground-truth cluster ID per stock (`-1` for noise). Seeded RNG = reproducible.

### 5.3 The five tests

| # | Function tested | What's asserted | Result |
|---|---|---|---|
| 1 | `ssd_distance` | within-cluster SSD < across-cluster SSD | ratio 560× ✅ |
| 2 | `cluster_optics`, `cluster_summary` | OPTICS finds ≥3 clusters; **no cross-planted-group merges** | 3 clusters, 18 clustered, 2 outliers ✅ |
| 3 | `purity_index` | purity ≥ 0.80 (paper's real-data target) | 0.833 ✅ |
| 4 | `clusters_to_pairs` | every emitted pair is within-cluster, no outliers | 16 valid pairs ✅ |
| 5 | `sic_division` | known SIC codes map to right division; bad input → "unknown" | spot checks pass ✅ |

### 5.4 Subtle design choices

**Test 2 — "no merges" instead of "exact recovery."** OPTICS is *allowed* to
sub-split a planted cluster (with our small toy sample, `xi=0.05` is sensitive)
because each sub-cluster still emits only valid within-planted-group pairs. But
OPTICS is *not* allowed to merge across planted groups — that would fabricate
false pairs and is the actual real-money risk.

**Test 3 — threshold 0.80 matches paper's reported 0.81.** A few noise stocks
get pulled into clusters as boundary points (normal OPTICS behaviour), so
purity is ~0.83, not 1.0. We use the paper's real-data target as the threshold
— a stricter threshold would test idealised toy behaviour rather than the
property we actually care about.

**Test 2 — `min_samples=3` instead of paper's `2`.** On 20-stock toy data,
`min_samples=2` over-fragments. On real 991-stock data, `min_samples=2` will be
fine (the paper's setting); we'll tune `xi` and `min_samples` against real
Dec-2023 to hit the paper's 48-cluster target (that's CP1).

### 5.5 Why the synthetic data tells us the pipeline will work on real data

The chain: ground-truth IDs → generate correlated prices → feed *only prices*
to SSD → OPTICS → cluster labels → compare labels back to ground truth. The
algorithm never sees the planted labels; if it recovers the structure from
price data alone, the pipeline plumbing is sound. The 0.83 purity matching the
paper's 0.81 is a strong "in the right ballpark" signal — OPTICS's behaviour
of pulling boundary points in is a real phenomenon, not a bug, and our test
quantifies it correctly.

---

## 6. What the real Dec-2023 run will produce

When we run SSD + OPTICS on the real formation window ending 2023-12-29 we'll
print (in `notebooks/` or a runnable script):

1. **Cluster summary** — n_clusters, n_clustered_stocks, n_outliers, mean/max
   cluster size
2. **Per-cluster table** — cluster ID, size, dominant SIC division, ticker list
3. **Sample pairs** — first ~10 within-cluster pairs as ticker tuples, e.g.
   `(AAPL, MSFT)`, `(XOM, CVX)`, `(JPM, BAC)`
4. **GOOG/GOOGL check** — the dual-share-class pair *must* land in the same
   cluster (their prices move in near lockstep). If they don't, the pipeline is
   broken. **This is a CP1 sanity check.**
5. **Numbers vs paper** — 48 clusters target, 0.81 purity target

### What different outcomes mean

- **Same-sector pairs dominate** → boring confirmation; OPTICS found
  economically meaningful structure.
- **Cross-sector pairs that cointegrate** → real finding worth investigating.
- **Many surprising pairs** → parameters too loose (raise `xi` or
  `min_samples`).
- **0 clusters** → almost certainly a parameter problem, not a data problem.
- **GOOG/GOOGL apart** → there's a bug.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **SSD** | Sum of Squared Distance on z-normalised prices — paper's Eq. (1) |
| **PC** | Partial Correlation distance on market-adjusted returns — paper's second metric |
| **OPTICS** | Ordering Points To Identify Clustering Structure — density-based clustering, no preset K, outputs outliers |
| **HDBSCAN** | Hierarchical density-based clustering — robustness check on OPTICS |
| **Hierarchical (avg-link)** | Agglomerative clustering; distance = 1 − corr; cut at chosen height |
| **OLS** | Ordinary least squares — standard regression |
| **RLM (Tukey biweight)** | Robust regression that down-weights outlier days |
| **Ridge** | L2-penalised regression; stabilises coefficients under multicollinearity |
| **Engle-Granger ADF** | Two-step cointegration test: OLS spread regression → ADF test on residual |
| **Half-life** | Time for a mean-reverting spread to decay halfway to equilibrium |
| **Z-score** | (spread − rolling_mean) / rolling_std |
| **Hedge ratio γ** | OLS slope from regressing price_A on price_B; the "share" of B that hedges A |
| **Purity** | Avg fraction of cluster members in the dominant SIC division |
| **Formation window** | 3-year period used to select pairs and estimate γ |
| **Trading window** | 1-month period in which we actually trade the selected pairs |
| **SIC division** | One of 10 industry buckets (paper footnote 2) derived from SIC code |
| **PERMNO** | CRSP's permanent stock identifier; doesn't change on ticker rename |
| **CP1 / CP2** | Phase 1 / Phase 2 validation checkpoints |
| **Survivorship bias** | The bias from using only stocks that still exist today |
| **t+1 execution** | Signal at end of day t executes at open of day t+1 (realistic) |
