# Cointegration-Based Pairs Trading with Factor-Driven Clustering

## Project structure

```
Quantitative Trading Strategies/
├── README.md
├── requirements.txt
├── config.py                  ← edit this first (parameters, paths, factors)
│
├── 01_fetch_data.py           ← pull stock + ETF returns from WRDS/CRSP
├── 02_rolling_betas.py        ← rolling robust regression → factor beta matrix
├── 03_clustering.py           ← hierarchical clustering on beta vectors
├── 04_cointegration.py        ← Engle-Granger test within clusters
├── 05_backtest.py             ← signal generation, P&L, costs
├── 06_results.py              ← plots and performance tables
│
└── data/
    ├── raw/                   ← parquet files from WRDS (created by 01)
    ├── processed/
    │   ├── betas/             ← rolling beta matrices per window (created by 02)
    │   ├── clusters/          ← cluster labels per window (created by 03)
    │   └── pairs/             ← cointegrating pairs per window (created by 04)
    └── results/               ← equity curve, heatmaps, performance CSV (created by 06)
```

## Setup (Mac)

```bash
# 1. Open Terminal and navigate to the project folder
cd "/Users/deepakgarrepalli/Desktop/MQF/Term 3/Quantitative Trading Strategies"

# 2. Create a virtual environment (keeps packages isolated)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## WRDS credentials

Your WRDS username (`deepakg2025`) is already set in `config.py`.
On **first run** of `01_fetch_data.py`, `wrds` will prompt:

```
Enter your WRDS password:
```

Type your password. It will offer to save it to `~/.pgpass` for future runs —
say yes so subsequent runs are fully automated.

## Run in order

```bash
python 01_fetch_data.py        # ~5-10 min — downloads ~200 MB
python 02_rolling_betas.py     # ~15-25 min — robust regression per window
python 03_clustering.py        # ~1-2  min  — read diagnostics to tune threshold
python 04_cointegration.py     # ~20-40 min — Engle-Granger across clusters
python 05_backtest.py          # ~3-5  min  — P&L simulation
python 06_results.py           # <1    min  — all charts and tables
```

## Tuning the cluster threshold (important)

After running `03_clustering.py`, it prints cluster size diagnostics:

```
Mean clusters per window : 18.4
Mean stocks per cluster  : 11.3   ← target 8–20
```

- If mean size **< 8** → lower `CLUSTER_DISTANCE_THRESHOLD` in `config.py`
- If mean size **> 20** → raise `CLUSTER_DISTANCE_THRESHOLD` in `config.py`
- Re-run 03 → 04 → 05 → 06 until you're in the 8–20 range

## Extending the factor set

The professor suggested 20–50 factors. In `config.py`, uncomment the optional
factors under `FACTOR_ETFS` (TLT, GLD, IWM, etc.) and re-run from step 01.

## Key design choices

| Choice | Decision | Rationale |
|--------|----------|-----------|
| Regression | `sklearn.linear_model.RidgeCV` | Ridge handles multicollinearity among correlated factor ETFs (XLF, XLK, SPY etc.) — OLS betas are unstable when factors are correlated; Ridge's L2 penalty shrinks and stabilises them |
| λ selection | Cross-validated from [0.01, 0.1, 1, 10, 100] | Automatic per-stock per-window tuning |
| Clustering | Agglomerative, average linkage | No k needed; distance threshold is interpretable |
| Distance metric | 1 − corr(beta vectors) | Stocks with same factor profile cluster together |
| Pair restriction | Within-cluster only | Replaces static SIC — catches cross-sector themes (AI, ESG) |
| Cointegration | Engle-Granger ADF, p < 0.05 | Standard in academic literature; see Gatev et al. 2006 |
| Entry/exit | ±2 SD entry, ±0.5 SD exit, ±4 SD stop | Conservative; avoids whipsawing around zero |
| Costs | 10 bps round-trip + 35 bps/yr borrow | Defensible for S&P 500 names (per lecture slides) |
