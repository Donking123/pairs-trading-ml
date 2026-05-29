Pairs Trading — Quickstart

Setup
- Python 3.12+
- From this folder:
  - Create venv: `python -m venv .venv` and activate it.
  - Install deps with uv (fast): `pip install uv && uv sync`
    or with pip: `python -m pip install -U pip wheel && pip install .`


WRDS connection 
- If want to just connect to `wrds` without loading all the data
  - run `python -c "import wrds; c=wrds.Connection(); print('connected'); c.close()"` in terminal

Prime data cache
- `python -c "from spx_data import load_all; load_all(refresh=True)"`
  (downloads SPX universe and factor ETFs via yfinance; parquet cached under ./data)

Diagnostics
- `python visualize_clusters.py --method a --save-dir figures`
- `python visualize_stability.py --method a --save-dir figures`
- `python visualize_dashboard.py --method a --save figures/cluster_dashboard.html`

Backtest (quarterly, Method A, breadth + earlier entry, zero-cost)
- `python backtest_pairs.py --method a --start 2019-01-01 --end 2024-12-31 --top-k 8 --refit 63 --z-entry 1.5 --zero-cost --out-dir data/bt_a_q_k8_z1p5`

Data sources
- Default: SPX yfinance (prices/volume) cached to parquet (./data/*parquet)
- Future (WRDS/Bloomberg): historical index membership, CRSP returns, FF factors, FRED rates/credit for richer beta‑space stability (not validated yet)

Notes
- All scripts assume you run from this folder (local imports).
- Use CLI flags in backtest_pairs.py and pairs_discovery.py to iterate quickly on thresholds.

