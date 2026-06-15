# Pairs Trading — ML Clustering (QF621 Group Project)

From-scratch replication of **Rotondi & Russo (2025), "Machine Learning for Pairs
Trading: a Clustering-based Approach"** (Bocconi), extended with a **factor-beta
clustering** metric, robustness testing, transaction costs, out-of-sample validation,
and an implementation-correctness review.

## Key results

| Stage | Sharpe | Notes |
|---|---:|---|
| In-sample factor-beta core (2003–2020) | **1.149** | best IS strategy |
| In-sample PC core (2003–2020) | 1.113 | paper-match headline |
| Net of costs (marketable) | ~0.57 | realistic worst case |
| Lookahead audit | PASS 6/6 | temporally honest |
| **OOS PC + filter (2021–2025)** | **0.461** | **best filtered OOS strategy** |
| OOS PC core | 0.412 | PC stays positive OOS |
| OOS SSD + filter | 0.224 | filter keeps SSD positive |
| OOS factor + filter | 0.131 | filter rescues factor OOS |
| OOS factor core | -0.103 | factor core does not generalise |

**Honest deployable Sharpe: ~0.2–0.8, regime-dependent — strongest in turbulent, high-dispersion markets.**

## In-sample scorecard (2003–2020, 215 months)

| Strategy | Ann. return | Ann. vol | **Sharpe** | Max DD |
|---|---:|---:|---:|---:|
| **Factor-beta core** | 4.16% | 3.61% | **1.149** | -3.59% |
| **PC core** | 3.86% | 3.46% | **1.113** | -5.75% |
| Factor + filter | 5.16% | 5.35% | 0.969 | -6.41% |
| SSD + filter | 4.97% | 6.37% | 0.794 | -9.74% |
| PC + filter | 2.79% | 3.57% | 0.788 | -5.88% |
| SSD core | 3.48% | 5.54% | 0.645 | -14.31% |

## Out-of-sample (2021–2025, 59 months)

| Strategy | Ann. return | Ann. vol | **Sharpe** | Max DD |
|---|---:|---:|---:|---:|
| **PC + filter** | 1.31% | 2.92% | **0.461** | -2.98% |
| PC core | 0.82% | 2.02% | 0.412 | -2.75% |
| SSD + filter | 0.82% | 4.01% | 0.224 | -5.52% |
| Factor + filter | 0.42% | 3.77% | 0.131 | -6.56% |
| Factor core | -0.29% | 2.56% | -0.103 | -6.99% |

## Where to start reading

| Doc | What it is |
|---|---|
| `notebooks/phase1_complete_reference.ipynb` | Phase 1 SSD baseline walkthrough |
| `notebooks/phase1_pnl_attribution.ipynb` | Bimodal trade finding — the key diagnostic |
| `notebooks/phase2_complete_reference.ipynb` | Phase 2 PC distance + cointegration filter |
| `notebooks/phase2_pnl_attribution.ipynb` | Phase 2 attribution analysis |
| `notebooks/phase2_5_complete_reference.ipynb` | Factor-beta extension walkthrough |
| `notebooks/phase3_complete_reference.ipynb` | Robustness testing — 8-cell grid |
| `notebooks/phase4_complete_reference.ipynb` | Realism, lookahead audit, OOS |
| `notebooks/phase5_complete_reference.ipynb` | Carry-over design and OOS verdict |
| `notebooks/phase6_complete_reference.ipynb` | Correctness fixes, bootstrap CIs, dispersion |
| `report/QF621_pairs_trading_deck.pdf` | Main presentation deck |
| `report/QF621_supplementary_deck.pdf` | Supplementary deck — Phases 2.5-6 |
| `CONCLUSIONS.md` | Final conclusions and honest assessment |

## Project layout

```
submission/
├── src/                          # shared library
│   ├── config.py                 # paths + constants + locked hyperparams
│   ├── wrds_pull.py              # WRDS data pull
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
├── tests/                        # synthetic tests (11 files, 67 tests)
├── notebooks/                    # reference notebooks (all phases)
├── results/                      # backtest output parquets
├── report/                       # presentation decks + chart assets
├── run_insample.py               # IS backtest runner (2003-2020)
├── run_oos.py                    # OOS backtest runner (2021-2025)
├── README.md
├── CONCLUSIONS.md
└── requirements.txt
```

## The pipeline

```
Data → Pair Selection (distance metric → clustering → cointegration filter)
     → Trading Rule (hedge ratio → spread → z-score → entry/exit)
     → Backtest Engine (rolling 3y/1m, carry-over, corrections flags)
     → Performance (Sharpe / Sortino / Calmar / MDD / hit rate)
     → Realism (bid/ask + 35bps borrow + stop-loss + execution delay)
     → Validation (lookahead audit + OOS 2021-2025 + bootstrap CIs)
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing results

```bash
# In-sample (2003-2020): SSD/PC x core/filtered
python submission/run_insample.py

# Out-of-sample (2021-2025): PC + factor-beta
python submission/run_oos.py
```

Each full backtest cell takes ~1-2 hours; the full 2x2 grid is about 4-8 hours.

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
**OOS 2021-2025 validation** · bootstrap 95% confidence intervals.

## License

MIT
