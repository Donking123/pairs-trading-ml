# Asian ADR Pair Selection (datastream/)

Pair selection is implemented in `datastream/run_asian_adr_screening.py` — a single self-contained script. There is no `src/asian_adr/research/` package. For weekly automated re-screening, `datastream/rescreen.py` orchestrates the full pipeline (incremental WRDS fetch → screening → changelog → registry backup).

**Inputs**: WRDS Datastream Parquet cache (`adr_prices`, `adr_reference`, `global_prices`, `fx_rates`)
**Outputs**: `config/pairs/asian_adr_pairs.json`, `config/pairs/screening_diagnostics.json`

## Scripts

| Script | Role |
|--------|------|
| `datastream/run_asian_adr_screening.py` | Full one-shot pair selection pipeline |
| `datastream/rescreen.py` | Incremental re-screening orchestrator (Phase 7) |

## Eligible Asian Exchanges

```python
ASIAN_EXCHANGES: set[str] = {
    "TKS", "TSE",   # Tokyo Stock Exchange (Japan)
    "HKG",          # Hong Kong Exchange (HKEX)
    "KRX",          # Korea Exchange
    "BOM", "BSE",   # Bombay Stock Exchange (India)
    "NSE",          # National Stock Exchange (India)
    "ASX",          # Australian Securities Exchange
    "SES",          # Singapore Exchange (SGX)
    "TAI",          # Taiwan Stock Exchange (TWSE)
    "IDX",          # Indonesia Stock Exchange
    "PHS",          # Philippine Stock Exchange (PSE)
    "KLS",          # Bursa Malaysia
}
```

## Pair Eligibility Criteria

| Criterion | Threshold |
|-----------|-----------|
| Cointegration (ADF + PP on dollar spread) | Reject unit root at 5% |
| Min continuous trading | 2 years (~504 trading days) |
| Min non-zero return days (both legs) | ≥ 50% of trading days |
| Zero-return-day % (ADR, entry ceiling) | ≤ 50% |
| ADR ratio resolvable | Required; ratio > 0 |

## Screening Steps (run_pipeline)

```
1. Filter adr_reference to ASIAN_EXCHANGES
2. For each candidate pair:
   a. Resolve adr_ratio — estimate from median(P_local × FX / P_ADR),
      snap to nearest standard ratio; reject if unresolvable
   b. Reconstruct dollar spread: P_ADR − (P_local × FX) / ratio
   c. Trim to as_of date (no look-ahead)
   d. Apply liquidity filter: non_zero_return_pct ≥ 0.50 on both legs
   e. ADF + Phillips-Perron cointegration test (p < 0.05 on both)
   f. Compute Roll (1984) effective spread: 2 × √|autocov(returns)|
3. Write approved pairs to config/pairs/asian_adr_pairs.json
4. Write per-candidate diagnostics to config/pairs/screening_diagnostics.json
```

## Roll (1984) Effective Spread

```python
def roll_effective_spread(prices: pd.Series) -> float:
    rets = prices.pct_change().dropna().values
    autocov = float(np.cov(rets[:-1], rets[1:], ddof=1)[0, 1])
    return float(2.0 * np.sqrt(abs(autocov)))
```

Used as a post-hoc round-trip cost estimate per pair — not as an entry filter.

## Design Rules

- `datastream/` scripts never import from `src/asian_adr/`
- Cointegration is always tested on the **dollar spread**, never log-prices
- β is always the registered `adr_ratio`; OLS estimation is prohibited (ADR-002)
- Pair selection is always run as-of a specific date; no future data permitted
