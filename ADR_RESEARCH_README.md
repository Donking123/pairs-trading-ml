# ADR Pairs Strategy — Research Results

Hong & Susmel (2013) Asian ADR pairs trading, evaluated on Datastream data 2011–2026.

**Run date:** 2026-06-11 | **Engine:** `datastream/run_research_permutations.py` + `review/portfolio.py`

---

## Headline Numbers

| Metric | Value |
|--------|-------|
| Universe | 224 Asian ADR pairs |
| Data | Datastream, 2011-05-04 to 2026-04-30 |
| Research runs | 1,080 (30 windows × 36 param combos) |
| Negative-median runs | **0 of 1,080** |
| Median RUCE-net per trade | **4.67%** |
| Median ROCE-net per trade | **2.37%** |
| Per-trade win rate | **77.9%** |
| Median trade duration | **5 days** |
| Portfolio Sharpe (daily MTM) | **2.03** (RUCE-net) / **0.87** (ROCE-net) |
| Transaction cost drag | **0.2% of gross RUCE** (negligible) |

---

## Portfolio-Level Performance (Daily MTM)

Daily mark-to-market P&L: each trade's unrealised return computed day-by-day from live prices, aggregated across concurrent open positions. Capital base = peak observed concurrent positions (73 slots).

| Metric | ROCE-net | RUCE-net |
|--------|----------|----------|
| Sharpe | 0.87 | **2.03** |
| Sortino | 0.80 | 1.98 |
| Annualised return | 6.7% | 24.4% |
| Annualised vol | 5.3% | 7.6% |
| Max drawdown | -10.0% | -11.6% |
| Calmar | 0.67 | 2.11 |
| Hit rate (active days) | 53.4% | 59.4% |
| Avg concurrent positions | 8.5 | — |

> **Note:** These are in-sample results — pairs selected on the same 2011–2026 history they are traded on. OOS Sharpe from a walk-forward will be lower. Use RUCE-net for Reg-T margin accounts; ROCE-net for fully-funded (unlevered) accounts.

---

## Alpha Source: ADR Overnight Premium Decay

The ADR leg drives virtually all of the strategy's alpha. The local leg is structurally necessary (it defines the spread) but contributes negligible alpha.

### Leg attribution (daily MTM, in-sample)

| Leg | Ann. return | Sharpe |
|-----|------------|--------|
| **ADR short leg alone** | **16.8%** | **2.81** |
| Local long leg alone | 5.9% | 0.77 |

The ADR leg median return per trade is **+2.36%** (70.2% of trades profitable) vs **+0.21%** for the local leg (52.6% — barely above random).

### Mechanism: cross-market overnight drift

The strategy is structurally an overnight trade: entry at US close (Day D), first possible exit at Asia open (Day D+1). Every trade crosses at least one overnight window. The alpha pattern is consistent with the ADR overnight premium decay phenomenon:

1. **During US hours** — retail and momentum flows push the ADR to a z-score premium (>2.5σ) vs local fair value. Entry fires.
2. **Overnight** — the premium mean-reverts as the US–Asia price gap closes. Most of the ADR return is earned in the first 1–2 days.
3. **Duration invariance** — 2-day trades and 11-30 day trades earn nearly identical RUCE (~4.2–4.4%), confirming the alpha is front-loaded into the overnight gap, not accumulated linearly.

### The role of Reg-T leverage

ROCE Sharpe (0.87) is borderline — the unlevered strategy barely justifies itself. The Reg-T 2× leverage on the ADR leg transforms it into RUCE Sharpe 2.03. The leverage uplift accounts for 57% of RUCE Sharpe. The investment case rests on access to margin for the ADR short leg.

### Cost drag

| Metric | Gross | Net | Drag |
|--------|-------|-----|------|
| ROCE per trade | 2.41% | 2.39% | 0.016% |
| RUCE per trade | 4.64% | 4.61% | 0.016% |

Roll spread + borrow = 0.2% of gross RUCE. Effectively free. The strategy is not eroded by transaction costs at the tested scale.

---

## Parameter Optimisation — 1,080 Runs

### Experimental design

| Setting | Value |
|---------|-------|
| Train window | 5 years (fixed) |
| Test windows | 1y, 2y, 4y |
| Rolling step | 1 year |
| Windows | 30 total |
| Parameter combos | 36 (STANDARD grid) |
| Total runs | 1,080 |

**Grid:** T ∈ {30,60,90} × k0 ∈ {1.65,2.00,2.50} × kc ∈ {0.0,0.5} × H ∈ {30,90}

### Recommended parameters

```
T  = 90   (estimation window, days)
k0 = 2.50 (entry z-score — higher is more selective, better quality)
kc = 0.0  (exit at full mean-reversion)
H  = 90   (max holding — rarely binds, H has negligible effect)
```

### Parameter sensitivity

| Param | Finding |
|-------|---------|
| T | T=60/T=90 beat T=30 by ~70 bps. T=60 vs T=90 within noise. |
| k0 | Monotonic: 2.50 > 2.00 > 1.65 across all windows (+84 bps per step). Higher k0 = more selective entry = better trade quality. |
| kc | kc=0.0 beats kc=0.5 by ~30 bps. Exiting earlier at kc=0.5 loses remaining convergence alpha. |
| H | Negligible (~6 bps). Most trades close via convergence before H binds. |

### RUCE-net per trade — all 1,080 runs

| Percentile | Value |
|-----------|-------|
| Min | 2.11% |
| p10 | 3.43% |
| p25 | 4.03% |
| **Median** | **4.67%** |
| p75 | 5.45% |
| p90 | 6.44% |
| Max | 8.58% |

### By test window

| Test | Trades/run | RUCE-net (median) | Win rate |
|------|-----------|-------------------|----------|
| 1y | 365 | 4.43% | 77.6% |
| 2y | 718 | 4.65% | 77.9% |
| **4y** | **1,235** | **4.90%** | **78.1%** |

Longer test windows yield higher medians — the alpha is not concentrated in a single regime.

### Period-by-period (by test start year, all 36 combos)

| Year | Median RUCE-net | Range |
|------|----------------|-------|
| 2017 | 4.14% | [2.79%, 5.43%] |
| 2018 | 4.43% | [3.43%, 5.84%] |
| 2019 | 3.93% | [2.62%, 4.99%] |
| 2020 | 4.27% | [3.04%, 5.58%] |
| 2021 | 4.67% | [3.45%, 6.09%] |
| 2022 | 5.01% | [4.03%, 6.67%] |
| 2023 | 5.09% | [3.77%, 6.70%] |
| **2024** | **6.54%** | [4.93%, 8.46%] |
| 2025 | 5.90% | [4.66%, 8.58%] |
| 2026* | 3.39% | [2.11%, 5.08%] |

*Partial year (data through April 2026).

Every single year is profitable across all 36 parameter combinations.

### Top 10 permutations (composite score)

| Rank | Test window | T | k0 | kc | H | Pairs | Trades | ROCE-net | RUCE-net | Win% |
|------|------------|---|----|----|---|-------|--------|----------|----------|------|
| 1 | 2025 (1y) | 90 | 2.50 | 0.0 | 90 | 161 | 287 | 4.20% | 8.58% | 87.1% |
| 2 | 2025 (1y) | 90 | 2.50 | 0.0 | 30 | 162 | 291 | 4.08% | 7.91% | 85.6% |
| 3 | 2024–25 (2y) | 90 | 2.50 | 0.0 | 90 | 179 | 599 | 4.02% | 8.46% | 85.6% |
| 4 | 2025 (1y) | 90 | 2.00 | 0.0 | 90 | 173 | 457 | 3.78% | 7.55% | 86.2% |
| 5 | 2025 (1y) | 60 | 2.50 | 0.0 | 90 | 157 | 298 | 3.79% | 7.59% | 86.2% |
| 6 | 2025 (1y) | 60 | 2.50 | 0.0 | 30 | 158 | 301 | 3.65% | 7.57% | 84.4% |
| 7 | 2025 (1y) | 90 | 2.00 | 0.0 | 30 | 174 | 470 | 3.64% | 7.07% | 85.1% |
| 8 | 2024–25 (2y) | 90 | 2.00 | 0.0 | 90 | 185 | 980 | 3.72% | 7.59% | 85.7% |
| 9 | 2025 (1y) | 90 | 1.65 | 0.0 | 90 | 181 | 672 | 3.47% | 6.98% | 84.7% |
| 10 | 2025 (1y) | 90 | 2.50 | 0.5 | 90 | 162 | 217 | 3.87% | 7.17% | 87.1% |

---

## Output Files

| File | Description |
|------|-------------|
| [data/backtest/research_runs.csv](data/backtest/research_runs.csv) | All 1,080 runs — full stats per row |
| [data/backtest/best_permutations.csv](data/backtest/best_permutations.csv) | Ranked permutations |
| [data/backtest/table_7_style_summary.csv](data/backtest/table_7_style_summary.csv) | Paper Table 7-B style aggregation |
| [data/backtest/portfolio_daily_mtm_equity.csv](data/backtest/portfolio_daily_mtm_equity.csv) | Daily equity curve (MTM) |
| [data/backtest/all_trades_by_run.csv](data/backtest/all_trades_by_run.csv) | All individual trades with run_id |

---

## How to Reproduce

```bash
# Research permutations (23 min)
python datastream/run_research_permutations.py \
    --adr-prices    data/parquet/adr_prices.parquet \
    --global-prices data/parquet/global_prices.parquet \
    --fx-rates      data/parquet/fx_rates.parquet \
    --pairs-file    config/pairs/asian_adr_pairs.json \
    --out-dir       data/backtest

# Portfolio MTM Sharpe
python review/portfolio.py \
    --trades  data/backtest/run_insample/trades.csv \
    --weight  amortised \
    --metric  ruce_net

# Full parameter grid (192 combos, ~2h)
python datastream/run_research_permutations.py ... --full-grid
```

---

## Strategy Reference

Hong, H., & Susmel, R. (2013). *Pairs-trading in the Asian ADR market*. Working paper, University of Houston.

- Spread: `spread_t = P_ADR,t − (P_local,t × FX_t) / adr_ratio`
- ADR ratio is a structural constant — never OLS-estimated
- Entry: Day D US close (short ADR when z > k0); Exit: Day D+1 Asia open (long local)
- RUCE = ROCE under Reg-T margin (2× leverage on ADR leg)
- Cost proxy: Roll (1984) effective spread, halved; borrow prorated over hold period
