# Historical Data Ingestion (datastream/)

Historical data ingestion and pair selection are handled entirely by standalone scripts in `datastream/`. There is no `src/asian_adr/data/` package — the scripts serve as both the initial data pipeline and the ongoing re-screening pipeline.

## Scripts

| Script | Responsibility | Output |
|--------|---------------|--------|
| `fetch_datastream_adr_data.py` | WRDS query for U.S. ADR OHLCV + reference mapping | `data/parquet/adr/` |
| `fetch_datastream_global_data.py` | WRDS query for Asian underlying OHLCV | `data/parquet/global/` |
| `fetch_fx_history.py` | WRDS Datastream SPOT FX rates (inverted to USD/CCY) | `data/parquet/fx/` |
| `run_asian_adr_screening.py` | Full pair selection pipeline (cointegration, liquidity, Roll) | `config/pairs/asian_adr_pairs.json` |
| `rescreen.py` | Incremental re-screening orchestrator: gap-fetch + re-screen + changelog | `config/pairs/` |

**Inputs**: WRDS credentials, date range
**Outputs**: `adr_prices.parquet`, `adr_reference.parquet`, `global_prices.parquet`, `fx_rates.parquet`, `asian_adr_pairs.json`

## Parquet Cache Schemas

| File | Columns | Notes |
|------|---------|-------|
| `adr/adr_prices.parquet` | `infocode`, `marketdate`, `close`, `high`, `low`, `open`, `volume`, `adj_factor`, `ticker`, `isin` | 4M+ rows |
| `adr/adr_reference.parquet` | `adr_ticker`, `adr_isin`, `adr_infocode`, `underlying_ticker`, `underlying_exchange`, `underlying_currency`, `adr_ratio` (NULL) | `adr_ratio` always NULL from WRDS; estimated from price median in screening |
| `global/global_prices.parquet` | `infocode`, `marketdate`, `close`, `high`, `low`, `open`, `volume`, `adj_factor`, `ticker`, `exchange`, `currency` | Filtered to Asian exchange mnemonics |
| `fx/fx_rates.parquet` | `date`, `base_currency`, `quote_currency`, `currency_pair`, `mid`, `provider` | `mid` = USD per 1 unit of base; inverted from Datastream CCY/USD |

## Key WRDS Queries

```python
# U.S. ADR prices — fetch_datastream_adr_data.py
"""SELECT n.infocode, d.marketdate, d.close, d.high, d.low, d.open, d.volume,
          d.cumadjfactor AS adj_factor, n.dscode AS ticker, n.isin
   FROM tr_ds_equities.wrds_ds2dsf AS d
   JOIN tr_ds_equities.wrds_ds_names AS n ON d.infocode = n.infocode
   WHERE d.marketdate BETWEEN %(start_date)s AND %(end_date)s
     AND n.region = 'US' AND n.typecode = 'ADR'"""

# FX rates — fetch_fx_history.py (CCY/USD → inverted to USD/CCY on output)
"""SELECT c.fromcurrcode AS iso_currency, r.exratedate AS marketdate, r.midrate AS rate
   FROM trdstrm.ds2fxrate AS r
   JOIN trdstrm.ds2fxcode AS c ON r.exrateintcode = c.exrateintcode
   WHERE c.ratetypecode = 'SPOT' AND c.tocurrcode = 'USD'
     AND r.exratedate BETWEEN %(start_date)s AND %(end_date)s"""
```

## Failure Handling

- WRDS timeout → exponential backoff reconnect (1s, 2s, 4s, max 60s)
- ADR ratio unavailable from reference → estimated from `median(P_local × FX / P_ADR)`, snapped to nearest standard ratio (0.01, 0.1, 0.5, 1, 2, 5, 10 …); pairs with unresolvable ratio are rejected
- Missing adjusted price → fallback to raw price with `adj_factor = 1.0`
