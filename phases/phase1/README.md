# Phase 1 — SSD Vertical Slice

**Status:** ✅ complete · **CP1 verdict:** partial pass (clustering ✅, Sharpe ❌ 0.589 vs 0.88 ±0.15)

This phase delivers the **whole pipeline end-to-end on the SSD (Sum of Squared
Distance) metric only** — a "vertical slice" that exercises every component
(distance → clustering → spread → backtest → performance) on real CRSP data and
validates against the paper's known target numbers.

---

## What this phase contains

| Folder | Purpose |
|---|---|
| `notebooks/` | The two master reference notebooks + 7 demo `.py` scripts + the notebook generators |
| `results/` | The Phase 1 SSD-baseline outputs (`ssd_core_monthly.parquet`, `ssd_core_trades.parquet`) |

To **read the deliverable**, open one of:

1. **`notebooks/phase1_complete_reference.ipynb`** — concept walkthrough + worked
   examples (SSD math, hedge ratio, z-score) + real-data examples (47 Dec-2023
   clusters, GOOG/GOOGL spread plot) + CP1 results + Phase 2 roadmap.
2. **`notebooks/phase1_pnl_attribution.ipynb`** — where the +30.4 net per-trade P&L
   actually comes from. The **bimodal duration finding** lives here.

To **re-run**:

```bash
# from project root (pairs-trading-ml/)
python phases/phase1/notebooks/05_run_full_backtest.py     # full 251-month backtest (~1-2 hours)
python phases/phase1/notebooks/06_evaluate_cp1.py          # CP1 verdict on saved results
python phases/phase1/notebooks/07_inspect_backtest.py      # warnings, anomalies, trade outliers
```

---

## Why this phase exists

The paper's claimed Sharpe (0.88 SSD / 1.01 PC) requires a working end-to-end pipeline.
Building all metrics at once and only running the backtest at the end would mean we'd
debug data, plumbing, and model questions all together — slow and error-prone. The
**vertical slice** strategy: build the *whole pipeline* using only the simplest paper
metric (SSD), get a real number, validate it against the paper, *then* extend.

**CP1 validates the engine**: if SSD-only reproduces the paper's cluster counts and
Sharpe within tolerance, the whole pipeline is sound and every subsequent metric is a
1-line swap. If it doesn't reproduce, we know to fix data/plumbing, not the model.

---

## What we built (in `src/`)

| File | Phase 1 responsibility |
|---|---|
| `src/wrds_pull.py` | Phase 0 raw data pull (already complete before Phase 1) |
| `src/panel.py` | Formation-window slicer; total-return prices; ticker/SIC lookup |
| `src/distances.py` | `ssd_distance()` — z-normalised price-trajectory distance |
| `src/clustering.py` | `cluster_optics`, `purity_index`, `clusters_to_pairs`, `sic_division` |
| `src/spread.py` | `fit_hedge_ratio` (OLS), `spread_series`, `rolling_zscore` |
| `src/backtest.py` | `run_one_month`, `run_backtest`, Trade/MonthResult, Option-B delisting |
| `src/performance.py` | `compute_metrics` (Sharpe/Sortino/Calmar/MDD/hit-rate) |
| `src/config.py` | Paths, constants, locked OPTICS hyperparams |

Synthetic tests (18 total, 18 pass) under `tests/`.

---

## Headline results

### Clustering scorecard (Dec 2023 formation window)

| Metric | Ours | Paper target | Verdict |
|---|---:|---:|---|
| # SSD clusters | **47** | 48 ±5 | ✅ |
| Purity vs SIC | **0.871** | 0.81 ±0.05 | ✅ |
| GOOG/GOOGL co-cluster | ✓ | ✓ | ✅ |

### Full-sample backtest scorecard (2003–2023, 251 monthly returns)

| Metric | Ours | Paper target | Verdict |
|---|---:|---:|---|
| **Annualised Sharpe** | **0.589** | **0.88 ±0.15** | ❌ outside tolerance |
| Annualised return | 3.01% | ~5% | low |
| Annualised vol | 5.28% | ~5.6% | ✓ matches |
| Max drawdown | -14.3% | ~-15% | ✓ matches |
| Hit rate | 57.4% | ~57% | ✓ matches |
| Total trades | 12,255 | — | — |

The risk profile (vol, drawdown, hit rate) matches the paper. The gap is entirely in
the *return* numerator. Diagnosis in §8.4 of `phase1_complete_reference.ipynb`.

---

## The most important finding — bimodal duration pattern

The full breakdown is in `phase1_pnl_attribution.ipynb`. Key identity:

| Exit reason | n | % | mean dur | mean P&L | win rate | Total |
|---|---:|---:|---:|---:|---:|---:|
| **reversion** | 1,392 | 11.4% | 14.5d | **+471 bps** | **99.2%** | **+65.58** |
| force_close | 10,832 | 88.4% | 19.9d | **−32 bps** | 48.2% | **−34.25** |
| delisting | 31 | 0.3% | 17.1d | −297 bps | 32% | −0.92 |
| **NET** | 12,255 | 100% | 19.2d | +25 bps | 54% | **+30.41** |

**The strategy's entire net P&L comes from the 11.4% of trades that fully revert
before month-end. The other 88.4% are force-closed at a −32 bps drag each.** This is
the load-bearing finding for Phase 2 design — see `decisions.md` and the Phase 2
README.

---

## Key decisions (locked in this phase)

See `decisions.md` for the full log. Highlights:

- **OPTICS xi=0.10** (tuned 2026-05-24 against Dec 2010/2015/2023, locked in `config.py`)
- **OLS hedge ratio**, frozen per trading month (RLM-Tukey kept as Phase 3 robustness)
- **Equal-dollar position sizing** ($0.50 long / $0.50 short)
- **Equal-weight allocation across currently-open pairs** (paper convention)
- **t+1 close-to-close execution** (no costs in core; bid/ask + 35bps borrow in Phase 4 realism)
- **No stop-loss in core** (3.5σ in Phase 4 realism)
- **Option-B code-dependent delisting fallback** (0% for M&A, −30% for bankruptcy, −5% for OTC)
- **Force-close at month end** (matches paper; not the cause of Sharpe gap)

---

## Tests (all passing)

```bash
python tests/test_clustering_synthetic.py     # 5/5
python tests/test_spread_synthetic.py         # 6/6
python tests/test_performance_synthetic.py    # 7/7
```

---

## What's next — Phase 2

See `../phase2/README.md`. In one sentence: **build PC distance + Engle-Granger
cointegration filter to reduce the force-close drag** — the lever that should lift
Sharpe from 0.59 toward the paper's 1.0+.
