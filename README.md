# Pairs Trading — ML Clustering (QF621 Group Project)

From-scratch replication of **Rotondi & Russo (2025), "Machine Learning for Pairs
Trading: a Clustering-based Approach"** (Bocconi), extended with a **factor-beta
clustering** metric, robustness testing, transaction costs, out-of-sample validation,
and an implementation-correctness review.

## Key results

| Stage | PC Sharpe | Notes |
|---|---:|---|
| In-sample, frictionless (signal-close fill) | 1.028 | paper-match headline |
| In-sample, frictionless (honest fill, +1d) | 0.882 | 86% genuine signal |
| Net of costs (marketable) | 0.572 | realistic worst case |
| Net + passive execution + no stop | ~0.78 | best defensible IS operating point |
| Lookahead audit | PASS 6/6 | temporally honest |
| **Out-of-sample 2024–25 (no carry)** | **0.858** | **PC generalises** |
| Out-of-sample factor-beta | 0.117 | CI includes ~1.0 — inconclusive |

**Honest deployable Sharpe: ~0.5–0.8, regime-dependent — strongest in turbulent, high-dispersion markets.**

## Phase overview

| Phase | What | Status |
|---|---|---|
| **0** | Data spine — WRDS pull, survivorship-bias-free universe (991 stocks, 2000–2025) | Done |
| **1** | SSD vertical slice — full pipeline baseline (Sharpe 0.589) | Done |
| **2** | PC distance + cointegration filter — 2x2 grid (Sharpe 1.028) | Done |
| **2.5** | Factor-beta clustering — 18 FF factors, Ridge regression (Sharpe 1.013) | Done |
| **3** | Robustness — 8-cell grid: {PC, Factor} x {HDBSCAN, Hierarchical, RLM, Z-weight} | Done |
| **4** | Realism (bid/ask + borrow + stop), lookahead audit (6/6 PASS), OOS 2024–25 | Done |
| **5** | Position carry-over — works IS (+0.045 net), fails OOS (0.434 vs 0.858) | Done |
| **6** | Correctness review — six fixes, bootstrap CIs, dispersion regime study | Done |

## Where to start reading

| Doc | What it is |
|---|---|
| **`phases/phase2/notebooks/phase2_complete_reference.ipynb`** | Phase 2 walkthrough — PC distance, cointegration filter, 2x2 grid |
| `phases/phase2_5/notebooks/phase2_5_complete_reference.ipynb` | Factor-beta extension walkthrough |
| `phases/phase3/notebooks/phase3_complete_reference.ipynb` | Robustness testing — 8-cell grid |
| `phases/phase4/notebooks/phase4_complete_reference.ipynb` | Realism, lookahead audit, OOS |
| `phases/phase5/notebooks/phase5_complete_reference.ipynb` | Carry-over design and OOS verdict |
| `phases/phase6/notebooks/phase6_complete_reference.ipynb` | Correctness fixes, bootstrap CIs, dispersion |
| `phases/phase1/notebooks/phase1_complete_reference.ipynb` | Phase 1 SSD baseline walkthrough |
| `report/QF621_pairs_trading_deck.pdf` | Main presentation deck (15 slides) |
| `report/QF621_supplementary_deck.pdf` | Supplementary deck — Phases 2.5–6 (19 slides) |
| `CONCLUSIONS.md` | Final conclusions and honest assessment |

## Project layout

```
pairs-trading-ml/
├── src/                          # shared library
│   ├── config.py                 # paths + constants + locked hyperparams
│   ├── wrds_pull.py              # Phase 0 — WRDS data pull
│   ├── panel.py                  # formation-window slicing, ticker/SIC lookup
│   ├── distances.py              # SSD, PC, market-adjusted returns
│   ├── clustering.py             # OPTICS, purity, clusters_to_pairs
│   ├── spread.py                 # OLS hedge ratio, spread, rolling z-score
│   ├── cointegration.py          # Engle-Granger ADF, half-life, MacKinnon p-values
│   ├── backtest.py               # rolling 3y/1m loop, carry-over, corrections flags
│   ├── performance.py            # Sharpe, Sortino, Calmar, MDD, hit rate
│   ├── factors.py                # Phase 2.5 — Ridge factor-beta distance
│   ├── costs.py                  # Phase 4 — bid/ask, borrow costs, stop-loss
│   └── lookahead.py              # Phase 4 — black-box lookahead audit
├── tests/                        # synthetic tests (11 files)
├── data/                         # cached parquet panels (gitignored; regen via wrds_pull.py)
├── notes/                        # cross-phase reference docs
├── report/                       # presentation decks + chart assets
│   ├── build_deck.py             # main deck generator (matplotlib → PDF)
│   ├── build_supplementary_deck.py  # supplementary deck generator
│   └── assets/                   # chart PNGs used in decks
└── phases/
    ├── phase1/                   # SSD vertical slice (0.589 Sharpe)
    ├── phase2/                   # PC + cointegration (1.028 / 0.752 / 0.731)
    ├── phase2_5/                 # Factor-beta extension (1.013 / 0.858)
    ├── phase3/                   # Robustness grid (8 cells)
    ├── phase4/                   # Realism + lookahead + OOS
    ├── phase5/                   # Position carry-over
    └── phase6/                   # Correctness review + bootstrap + dispersion
```

## The pipeline

```
Data → Pair Selection (distance metric → clustering → cointegration filter)
     → Trading Rule (hedge ratio → spread → z-score → entry/exit)
     → Backtest Engine (rolling 3y/1m, carry-over, corrections flags)
     → Performance (Sharpe / Sortino / Calmar / MDD / hit rate)
     → Realism (bid/ask + 35bps borrow + stop-loss + execution delay)
     → Validation (lookahead audit + OOS 2024–25 + bootstrap CIs)
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

## Reproducing results

```bash
# Phase 1 — SSD baseline
python phases/phase1/notebooks/05_run_full_backtest.py

# Phase 2 — 4-cell grid (SSD/PC x filter on/off)
python phases/phase2/notebooks/04_run_full_backtest_grid.py

# Phase 2.5 — Factor-beta
python phases/phase2_5/notebooks/01_run_factor_backtest.py

# Phase 3 — Robustness grid
python phases/phase3/notebooks/01_run_robustness_grid.py

# Phase 4 — Realism + lookahead + OOS
python phases/phase4/notebooks/02_run_realism.py
python phases/phase4/notebooks/01_lookahead_test.py
python phases/phase4/notebooks/05_forward_test.py

# Phase 5 — Carry-over grid + OOS
python phases/phase5/notebooks/01_run_carryover_grid.py
python phases/phase5/notebooks/03_forward_test_carry.py

# Phase 6 — Corrections ladder + bootstrap + dispersion
python phases/phase6/notebooks/01_run_corrections_ladder.py
python phases/phase6/notebooks/03_bootstrap_cis.py
python phases/phase6/notebooks/04_regime_dispersion.py
```

Each full backtest cell takes ~1–2 hours; the full Phase 2 grid is about 6 hours.

## Tests

```bash
python -m pytest tests/ -v
```

11 test files covering: clustering, distances, spread, performance, cointegration,
factors, costs, lookahead, carry-over, corrections, and alternative clustering.

## Realism principles

No survivorship bias · no look-ahead (6/6 audit PASS) · **t+1 execution delay** ·
realistic delisting returns (compounded) · **actual CRSP bid/ask** transaction costs ·
35 bps/yr borrow on short leg · 3.5-sigma stop-loss · honest gross-vs-net reporting ·
**OOS 2024–25 validation** · bootstrap 95% confidence intervals.

## License

MIT
