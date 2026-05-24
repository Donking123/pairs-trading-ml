# Pairs Trading — ML Clustering (QF621 Group Project)

From-scratch replication of **Rotondi & Russo (2025), "Machine Learning for Pairs
Trading: a Clustering-based Approach"** (Bocconi), extended with a **factor-beta
clustering** metric.

## Current status (2026-05-24)

**Phase 2 complete — paper replicated.** Full 4-cell 2×2 grid (SSD/PC × filter on/off)
run end-to-end on 251 months (Jan 2003 → Dec 2023):

| Cell | Ours Sharpe | Paper Target | Verdict |
|---|---:|---:|---|
| **PC core** | **1.028** | **1.01 ±0.15** | **✅ matches** |
| PC + cointegration filter | 0.752 | 0.80 ±0.15 | ✅ matches |
| SSD + cointegration filter | 0.731 | 0.75 ±0.15 | ✅ matches |
| SSD core | 0.589 | 0.88 ±0.15 | ❌ below (Phase 1 baseline) |

## Where to start reading

| Doc | What it is |
|---|---|
| **`phases/phase2/notebooks/phase2_complete_reference.ipynb`** | Master Phase 2 walkthrough — concept, worked examples, real-data results, CP2 verdict |
| `phases/phase2/notebooks/phase2_pnl_attribution.ipynb` | P&L attribution — bimodal lever check, regime-by-regime Sharpe |
| `phases/phase1/notebooks/phase1_complete_reference.ipynb` | Phase 1 baseline walkthrough |
| `phases/phase1/notebooks/phase1_pnl_attribution.ipynb` | Phase 1 attribution (the bimodal duration finding) |
| `phases/README.md` | Phase folder layout & status |
| `notes/progress.md` | Top-level status board |
| `notes/strategy-reconciliation.md` | Paper-vs-proposal decisions log |

## Project layout

```
pairs-trading-ml/
├── src/                          # shared library (evolves over phases)
│   ├── config.py                 # paths + constants + locked hyperparams (xi, filter thresholds)
│   ├── wrds_pull.py              # Phase 0 — WRDS data pull
│   ├── panel.py                  # formation-window slicing, ticker/SIC lookup, market returns
│   ├── distances.py              # ssd_distance, pc_distance, market_adjusted_returns
│   ├── clustering.py             # OPTICS, purity_index, clusters_to_pairs, sic_division
│   ├── spread.py                 # OLS hedge ratio γ, spread, rolling z-score
│   ├── cointegration.py          # Engle-Granger ADF, half_life_ar1, filter_cointegrated_pairs
│   ├── backtest.py               # rolling 3y/1m loop with metric & filter args
│   └── performance.py            # Sharpe, Sortino, Calmar, MDD, hit rate
├── tests/                        # 32 synthetic tests across 5 modules (all passing)
├── data/                         # cached parquet panels (gitignored; regen via wrds_pull.py)
├── notes/                        # cross-phase reference docs
└── phases/
    ├── phase1/                   # ✅ complete — SSD vertical slice (0.589 Sharpe)
    │   ├── README.md
    │   ├── decisions.md
    │   ├── notebooks/            # 01..07 + 2 .ipynb reference notebooks
    │   └── results/              # ssd_core_{monthly,trades}.parquet
    └── phase2/                   # ✅ complete — PC + cointegration (1.028 / 0.752 / 0.731 Sharpe)
        ├── README.md
        ├── decisions.md          # D2.1 – D2.7 (all resolved)
        ├── carryover-from-phase1.md
        ├── notebooks/            # 01..06 + 2 .ipynb reference notebooks
        └── results/              # {ssd,pc}_{core,filtered}_{monthly,trades}.parquet
```

## The pipeline (6 stages)

```
Data → Pair Selection (distance metric → clustering → cointegration filter)
     → Trading Rule (hedge ratio → spread → z-score)
     → Backtest Engine (rolling formation/trading, MtM PnL, t+1 execution)
     → Performance (Sharpe / Sortino / Calmar / MDD)
     → Realism (Phase 4: bid/ask + 35bps borrow + Alpaca forward test)
```

## Setup

```bash
pip install -r requirements.txt
```

## Phase 0 — pull the data (one-time)

WRDS credential setup:

```bash
python -c "import wrds; wrds.Connection().create_pgpass_file()"
```

Pull + cache:

```bash
python src/wrds_pull.py
```

Caches five parquet files into `data/`: `sp500_constituents`, `crsp_daily`,
`delisting`, `sp500_index`, `ff_factors`.

## Reproducing Phase 1 / Phase 2 results

```bash
# Phase 1 — SSD baseline (already saved to phases/phase1/results/, but to re-run:)
python phases/phase1/notebooks/05_run_full_backtest.py
python phases/phase1/notebooks/06_evaluate_cp1.py
python phases/phase1/notebooks/07_inspect_backtest.py

# Phase 2 — 4-cell grid (already saved to phases/phase2/results/, but to re-run:)
python phases/phase2/notebooks/04_run_full_backtest_grid.py
python phases/phase2/notebooks/05_evaluate_cp2.py
python phases/phase2/notebooks/06_inspect_backtest.py
```

Each full backtest cell takes ~1-2 hours laptop wall-clock; the full Phase 2 grid is
about 6 hours total.

## Tests (all 32 passing)

```bash
python tests/test_clustering_synthetic.py        # 5/5
python tests/test_spread_synthetic.py            # 6/6
python tests/test_performance_synthetic.py       # 7/7
python tests/test_distances_pc_synthetic.py      # 7/7  (Phase 2)
python tests/test_cointegration_synthetic.py     # 7/7  (Phase 2)
```

## Realism principles (professor's requirement)

No survivorship bias · no look-ahead · **t+1 execution** · realistic delisting losses
(Option B code-dependent fallback) · **bid/ask** transaction costs (Phase 4) · borrow
costs (Phase 4) · liquidity/capacity limits · honest gross-vs-net reporting.

## Next phases

- **Phase 2.5** — Factor-beta clustering extension (QF621 group project's contribution
  on top of the replication; needs `src/factors.py` + sector/style/commodity ETFs).
- **Phase 3** — Robustness cells (hierarchical algo, RLM hedge ratio, |entry-z|-weighted
  allocation alternatives, softer survivorship filter).
- **Phase 4** — Realism variant (bid/ask + 35bps borrow + 3.5σ stop) + Alpaca paper-trade
  forward test + final writeup.
