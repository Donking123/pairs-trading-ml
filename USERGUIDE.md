# Asian ADR Pairs Strategy — User Guide

This guide gives the **exact order** to run the scripts that reproduce the results
verified in the health check (in-sample backtest, walk-forward out-of-sample
validation, and the data-quality / robustness review suite).

The strategy implements the Hong & Susmel (2013) Asian ADR pairs trade:
short the ADR when its dollar spread over the FX-converted local share is wide,
buy the local leg next session, and unwind on convergence (ROCE / RUCE).

---

## 0. Prerequisites (one time)

```bash
# From the repository root.
# (Optional but recommended) create and use a virtual environment.
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Requirements split into two groups (see `requirements.txt`):

- **Core analysis** — `pandas`, `numpy`, `statsmodels`, `arch`, `pyarrow`.
  These are all you need to run screening, the backtest, the walk-forward, and
  the review suite **on the cached Parquet data that already ships in the repo**.
- **Data ingestion only** — `wrds`, `python-dotenv`. Needed *only* if you want to
  re-pull raw data from WRDS (Step 1). Skip if you use the cached Parquet.

> All commands below are run from the repository root and assume the cached
> Parquet files already exist under `datastream/data/parquet/`. If they do, you
> can **skip straight to Step 2**.

---

## 1. (Optional) Fetch raw data from WRDS

Only required if you need to rebuild the Parquet cache from scratch. Requires
WRDS credentials (`WRDS_USERNAME` / `WRDS_PASSWORD` env vars or a `~/.pgpass`
entry). Run the three fetchers **in this order** — the global and FX fetchers
read the ADR reference table produced by the first.

```bash
# 1a. ADR OHLCV + ADR→underlying reference mapping
python3 datastream/fetch_datastream_adr_data.py --out datastream/data/parquet/adr

# 1b. Asian-underlying OHLCV (reads adr_reference.parquet from 1a)
python3 datastream/fetch_datastream_global_data.py \
    --start 2011-01-01 --end 2026-04-30 \
    --adr-reference datastream/data/parquet/adr/adr_reference.parquet \
    --out datastream/data/parquet/global

# 1c. FX daily rates (USD per 1 unit of local currency)
python3 datastream/fetch_fx_history.py --out datastream/data/parquet/fx
```

Output Parquet files:

| File | Contents |
|---|---|
| `datastream/data/parquet/adr/adr_prices.parquet`     | ADR OHLCV + `adj_factor` |
| `datastream/data/parquet/adr/adr_reference.parquet`  | ADR → underlying mapping + `adr_ratio` |
| `datastream/data/parquet/global/global_prices.parquet` | Asian underlying OHLCV |
| `datastream/data/parquet/fx/fx_rates.parquet`        | Daily FX mid (USD per local unit) |

---

## 2. (Recommended) Health check — verify data readiness

Confirms every Parquet file exists, is fresh, covers all tickers, and that the
Python dependencies import. A green run here means Steps 3–6 will work.

```bash
python3 datastream/healthcheck.py
```

Exit codes: `0` healthy · `1` degraded (warnings only) · `2` failed.

---

## 3. Pair screening — build the registry

Selects tradable pairs (cointegration via ADF + Phillips–Perron, liquidity
filters, Roll effective-spread estimate) and writes the pair registry that every
downstream script consumes.

```bash
python3 datastream/run_asian_adr_screening.py \
    --adr-prices    datastream/data/parquet/adr/adr_prices.parquet \
    --adr-reference datastream/data/parquet/adr/adr_reference.parquet \
    --global-prices datastream/data/parquet/global/global_prices.parquet \
    --fx-rates      datastream/data/parquet/fx/fx_rates.parquet \
    --out           config/pairs/asian_adr_pairs.json \
    --as-of         2026-04-30
```

**Output:** `config/pairs/asian_adr_pairs.json` — the approved pair registry
(ships pre-built with ~224 pairs; re-running overwrites it).

> The repo already includes a built registry, so you may skip this step if you
> only want to reproduce the backtest on the shipped pairs.

---

## 4. In-sample backtest

Trades **all approved pairs over their full history**. This reproduces the
headline ROCE / RUCE distribution (paper Table 7-B format). It is *optimistic*
(pairs are selected on the same history they are traded on) — use Step 5 for the
unbiased number.

```bash
python3 datastream/run_backtest.py \
    --pairs   config/pairs/asian_adr_pairs.json \
    --out-dir datastream/data/backtest/run_$(date +%Y%m%d_%H%M%S) \
    --k0 2.0 --kc 0.0 --T 60 --H 90
```

(All `--adr-prices` / `--global-prices` / `--fx-rates` paths default to the
cached locations, so they can be omitted.)

**Outputs** in `--out-dir`: `summary.json`, `distribution.json`,
`trades.csv`, `trades.parquet`.

**Reference result (full history, defaults):** 224 pairs traded, ~9,500 trades,
median ROCE ≈ 2.4% / RUCE ≈ 4.6% per trade.

---

## 5. Walk-forward (out-of-sample) — the headline validation

Removes the in-sample look-ahead: pairs are re-selected using **only** the train
window, then traded **only** in the test window. This is the number to trust.

```bash
python3 datastream/run_walkforward.py \
    --train-start 2015-01-01 \
    --split       2021-12-31 \
    --test-end    2024-12-31 \
    --out-dir     datastream/data/walkforward_output/walkforward_$(date +%Y%m%d_%H%M%S)
```

- `--split` is the train/test boundary: selection uses dates `<= split`, trading
  uses dates `> split`.
- Add `--folds N` for an expanding walk-forward (train grows, test rolls forward).
- Runtime is several minutes (it re-runs the full screening pipeline on the train
  window).

**Outputs** in `--out-dir`: `summary.json`, `distribution.json`, `folds.json`,
**`trades_oos.csv`** ← this file feeds the entire review suite in Step 6.

**Reference result (single fold above):** ~148 pairs selected on train, ~1,940
OOS trades, median RUCE-net ≈ 4.9% — consistent with in-sample, confirming the
edge survives proper train/test separation.

---

## 6. Review / robustness suite

All review scripts consume the **`trades_oos.csv`** produced in Step 5 (point
`--trades` at it). They are independent of one another and can be run in any
order. Set a shell variable for convenience:

```bash
WF=datastream/data/walkforward_output/walkforward_XXXXXXXX_XXXXXX   # <- your Step 5 out-dir
```

### 6a. Portfolio-level performance
Daily equity curve + Sharpe / Sortino / max-drawdown / Calmar / hit-rate.

```bash
python3 review/portfolio.py --trades $WF/trades_oos.csv --out-dir $WF
```

### 6b. Data-quality / stale-price check
Flags pairs whose apparent mean-reversion is driven by stale (zero-return)
prices rather than genuine convergence. **(Slow — scans full price history; can
take ~15 min.)**

```bash
python3 review/run_data_quality.py --trades $WF/trades_oos.csv --out-dir $WF
```

**Reference result:** 224 pairs analysed, ~12 (5.4%) flagged, stale trade
exposure ≈ 3.7% open / 4.7% close — i.e. the edge is *not* a stale-price artifact.

### 6c. Significance test (is the edge real?)
Random-entry null distribution + bootstrap CI on median RUCE-net.

```bash
python3 review/run_benchmark.py --trades $WF/trades_oos.csv --out-dir $WF
```

### 6d. Cost sensitivity
Sweeps borrow rate × slippage; reports the breakeven cost level.

```bash
python3 review/run_cost_sensitivity.py --trades $WF/trades_oos.csv --out-dir $WF
```

### 6e. Market-neutrality check
Regresses daily portfolio returns on a local-equity basket and an FX basket to
detect hidden long-EM / long-FX beta.

```bash
python3 review/run_market_neutral.py --trades $WF/trades_oos.csv --out-dir $WF
```

### 6f. Regime dependence
Splits OOS trades by volatility regime (low / medium / high) to test whether the
edge is just a volatility bet.

```bash
python3 review/run_regime.py --trades $WF/trades_oos.csv --out-dir $WF
```

### 6g. Parameter-grid robustness (heaviest — runs many walk-forwards)
Sweeps `(k0, kc, T, H)` out-of-sample. **Long runtime.** Add `--in-sample` for a
faster (optimistic) sweep on the fixed registry.

```bash
python3 review/run_param_grid.py \
    --train-start 2015-01-01 --split 2021-12-31 --test-end 2024-12-31 \
    --out-dir review/output/grid_$(date +%Y%m%d_%H%M%S)
```

---

## Quick reference — minimal path to the verified results

```bash
# 1. setup
pip install -r requirements.txt

# 2. confirm data is ready
python3 datastream/healthcheck.py

# 3. in-sample backtest
python3 datastream/run_backtest.py --out-dir datastream/data/backtest/run_demo

# 4. out-of-sample walk-forward (the headline number)
python3 datastream/run_walkforward.py \
    --train-start 2015-01-01 --split 2021-12-31 --test-end 2024-12-31 \
    --out-dir datastream/data/walkforward_output/wf_demo

# 5. headline robustness checks on the OOS trades
python3 review/portfolio.py        --trades datastream/data/walkforward_output/wf_demo/trades_oos.csv --out-dir datastream/data/walkforward_output/wf_demo
python3 review/run_benchmark.py    --trades datastream/data/walkforward_output/wf_demo/trades_oos.csv --out-dir datastream/data/walkforward_output/wf_demo
```

## Pipeline at a glance

```
WRDS fetch (opt.)          screening                backtests                review (on trades_oos.csv)
─────────────────          ─────────                ─────────                ──────────────────────────
fetch_datastream_adr   ┐                        ┌─ run_backtest (in-sample)   portfolio
fetch_datastream_global├─►  run_asian_adr_  ─►  │                             run_data_quality
fetch_fx_history       ┘    screening            └─ run_walkforward (OOS) ──► run_benchmark
                            │  (writes               │  (writes                run_cost_sensitivity
   healthcheck (verify) ◄───┘   asian_adr_           └─  trades_oos.csv)       run_market_neutral
                                pairs.json)                                    run_regime
                                                                              run_param_grid
```
