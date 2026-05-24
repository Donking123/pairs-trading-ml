# Teammate Setup Guide

Welcome — this guide gets you from a fresh `git clone` to a fully-working local copy of the project in ~10 minutes.

---

## What this project is

A from-scratch replication of **Rotondi & Russo (2025), "Machine Learning for Pairs Trading: a Clustering-based Approach"** on WRDS CRSP S&P 500 data, 2003–2023.

**Current state:** Phase 2 complete. We replicated the paper's headline result —
PC distance + clustering achieves **Sharpe 1.028** vs the paper's reported **1.01**
(within ±2%). Full scorecard in `phases/phase2/README.md`.

---

## 1. Clone the repo

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd pairs-trading-ml
```

## 2. Create a Python environment

Recommended: Python 3.11+.

Option A — `venv` (built-in):
```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# or  venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Option B — Conda:
```bash
conda create -n qf621 python=3.13
conda activate qf621
pip install -r requirements.txt
```

## 3. Get the data

The CRSP panel is **140 MB and licensed** — not in the repo. Two options:

### Option A — you have WRDS access (preferred)

```bash
# One-time WRDS credential setup (creates ~/.pgpass)
python -c "import wrds; wrds.Connection().create_pgpass_file()"

# Pull + cache the panel (takes ~15-30 minutes)
python src/wrds_pull.py
```

This creates five parquet files in `data/`:
- `crsp_daily.parquet` (the main panel — 140MB)
- `sp500_constituents.parquet`
- `delisting.parquet`
- `sp500_index.parquet`
- `ff_factors.parquet`

### Option B — get the data from Don directly

If you don't have WRDS, ask Don to share the `data/` folder via Google Drive / Dropbox (it's too big for GitHub).

## 4. Verify the install

Run the synthetic test suite (no data needed, just code):

```bash
python tests/test_clustering_synthetic.py        # 5/5
python tests/test_spread_synthetic.py            # 6/6
python tests/test_performance_synthetic.py       # 7/7
python tests/test_distances_pc_synthetic.py      # 7/7
python tests/test_cointegration_synthetic.py     # 7/7
```

All 32 should pass. If they don't, your install is broken — likely a missing dependency.

## 5. Reproduce the headline results

The result parquets are **already in the repo** under `phases/{phase1,phase2}/results/` (they're only ~3 MB total). To see the scorecard immediately without re-running:

```bash
python phases/phase1/notebooks/06_evaluate_cp1.py    # Phase 1: SSD baseline
python phases/phase2/notebooks/05_evaluate_cp2.py    # Phase 2: 2x2 grid
```

To **re-run** the full backtests (requires the CRSP data; takes ~6 hours each):

```bash
python phases/phase1/notebooks/05_run_full_backtest.py
python phases/phase2/notebooks/04_run_full_backtest_grid.py
```

## 6. Read the deliverables

Open the master reference notebooks (best with JupyterLab):

```bash
jupyter lab phases/phase1/notebooks/phase1_complete_reference.ipynb \
            phases/phase1/notebooks/phase1_pnl_attribution.ipynb \
            phases/phase2/notebooks/phase2_complete_reference.ipynb \
            phases/phase2/notebooks/phase2_pnl_attribution.ipynb
```

Each notebook has all plots already baked in (executed previously), so you can scroll through immediately. Re-run them with "Kernel → Restart & Run All" if you want fresh output.

## 7. Project layout

```
pairs-trading-ml/
├── README.md                     ← project overview + status
├── TEAMMATE_SETUP.md             ← this file
├── LICENSE                       ← MIT
├── requirements.txt              ← Python deps
├── .gitignore                    ← excludes data/ + caches
│
├── src/                          ← shared library
│   ├── config.py                 ← paths, constants, locked hyperparams
│   ├── wrds_pull.py              ← Phase 0 data pull
│   ├── panel.py                  ← formation-window slicing, lookups
│   ├── distances.py              ← ssd_distance, pc_distance
│   ├── clustering.py             ← OPTICS, purity_index, pairs
│   ├── spread.py                 ← hedge ratio γ, spread, z-score
│   ├── cointegration.py          ← Engle-Granger ADF, half-life filter
│   ├── backtest.py               ← rolling 3y/1m loop with metric+filter args
│   └── performance.py            ← Sharpe, Sortino, Calmar, MDD
│
├── tests/                        ← 32 synthetic tests (all passing)
│
├── data/                         ← raw CRSP cache (gitignored)
│
├── notes/                        ← cross-phase docs (glossary, conventions)
│   ├── progress.md               ← top-level status board
│   ├── strategy-reconciliation.md← paper vs proposal decisions
│   ├── concepts-walkthrough.md
│   └── phase-0-data-spine.md
│
└── phases/
    ├── README.md                 ← phase folder layout
    ├── phase1/                   ← ✅ SSD vertical slice (Sharpe 0.589)
    │   ├── README.md
    │   ├── decisions.md          ← D1.1 – D1.11
    │   ├── notebooks/            ← 01..07 + 2 reference .ipynb
    │   └── results/              ← ssd_core_{monthly,trades}.parquet
    │
    └── phase2/                   ← ✅ PC + filter (Sharpe 1.028 = paper match)
        ├── README.md
        ├── decisions.md          ← D2.1 – D2.7
        ├── carryover-from-phase1.md
        ├── notebooks/            ← 01..06 + 2 reference .ipynb
        └── results/              ← 4 parquet pairs (2x2 grid)
```

## 8. Headline results

| Cell | Ours Sharpe | Paper Target | Verdict |
|---|---:|---:|---|
| **PC core** | **1.028** | **1.01 ±0.15** | **✅ matches paper** |
| PC + cointegration filter | 0.752 | 0.80 ±0.15 | ✅ |
| SSD + cointegration filter | 0.731 | 0.75 ±0.15 | ✅ |
| SSD core | 0.589 | 0.88 ±0.15 | ❌ below (Phase 1 baseline) |

## 9. Next planned phases (open for you to contribute)

| Phase | What | File to start with |
|---|---|---|
| **2.5** | Factor-beta clustering extension (the QF621 group's contribution) | New `src/factors.py` + new `pc_factor_distance()` in `distances.py` |
| 3 | Robustness cells: hierarchical algo, RLM hedge ratio, equal-weight allocation alternatives | `src/clustering.py` (add hierarchical), `src/spread.py` (add RLM) |
| 4 | Realism variant (bid/ask costs + 35bps borrow + 3.5σ stop) + Alpaca paper-trade forward test + final writeup | `src/backtest.py` (add cost args), new `src/forward_test.py` |

Pick a phase or sub-task that interests you. Talk to Don before starting so we don't duplicate work.

## 10. Coding conventions (informal but applied throughout)

- **Synthetic-tests-first** — every new function in `src/` gets a unit test in `tests/` before real-data use. See existing tests for the pattern.
- **Locked hyperparameters live in `config.py`** with a "DO NOT re-tune after seeing Sharpe" comment. Don't change them mid-experiment.
- **Single-variable changes** — when comparing a new variant to the baseline, change *one thing at a time* so attribution is clean.
- **Decision logs** — significant choices go into `phases/phaseN/decisions.md` at the time of choosing, with rationale.
- **Phase folder structure** — every phase has `README.md`, `decisions.md`, `notebooks/`, `results/`.

## 11. Questions?

- **Where do I start reading?** → `phases/phase2/notebooks/phase2_complete_reference.ipynb` is the master deliverable. Read top-to-bottom.
- **What did each design decision mean?** → `phases/phase{1,2}/decisions.md`
- **Why is the SSD Sharpe below the paper but PC matches?** → `phases/phase1/notebooks/phase1_pnl_attribution.ipynb` §8 (the bimodal lever finding).
- **How do I add a new metric?** → Add a function to `src/distances.py` with the same signature as `ssd_distance` (returns a square distance matrix indexed by permno). Then it plugs into the existing pipeline.

If anything is unclear, just ping Don.

— Generated 2026-05-24
