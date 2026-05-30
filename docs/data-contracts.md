# Data Contracts

All events inherit from `BaseEvent` and are immutable frozen Pydantic models. No component ever sees raw exchange or WRDS data — the Feed Handler and WRDS Fetcher normalise everything before it enters the bus.

## Base Event

```python
class BaseEvent(BaseModel):
    model_config = {"frozen": True}

    event_id:             UUID     = Field(default_factory=uuid4)
    event_type:           str
    timestamp_exchange:   datetime
    timestamp_received:   datetime
    timestamp_processed:  datetime | None = None
```

## Market Events

```python
class BarEvent(BaseEvent):
    event_type:   Literal["bar"] = "bar"
    ticker:       str
    open:         Decimal
    high:         Decimal
    low:          Decimal
    close:        Decimal
    volume:       int
    bar_interval: str            # "1d"
    is_adjusted:  bool
    exchange:     str            # "NYSE", "TSE", "HKEX", "KRX", etc.
    currency:     str            # "USD", "JPY", "HKD", "KRW", etc.
```

## FX Rate Event

```python
class FXRateEvent(BaseEvent):
    event_type:     Literal["fx_rate"] = "fx_rate"
    currency_pair:  str       # e.g., "JPY_USD"
    base_currency:  str       # "JPY"
    quote_currency: str       # "USD"
    mid:            Decimal   # USD per 1 unit of base currency
    provider:       str       # "datastream" | "oanda"
    is_stale:       bool = False
```

## Hong & Susmel Signal Event

```python
class HSSignal(str, Enum):
    SHORT_ADR   = "short_adr"    # SELL ADR short; buy local next Asia open
    EXIT        = "exit"          # Spread converged; close both legs
    FORCE_CLOSE = "force_close"   # Holding period expired

class HongSusmelSignalEvent(BaseEvent):
    event_type:         Literal["hs_signal"] = "hs_signal"
    pair_id:            str
    adr_ticker:         str
    underlying_ticker:  str
    signal:             HSSignal
    spread:             Decimal     # dollar spread at signal bar
    mu:                 Decimal     # rolling mean
    sigma:              Decimal     # rolling std (ddof=1)
    kappa_open:         Decimal     # µ + k0·σ
    kappa_close:        Decimal     # µ + kc·σ
    z_score:            Decimal     # (spread − µ) / σ
    days_held:          int         # 0 for SHORT_ADR signals
```

## Risk Decision

```python
class RiskDecisionStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

class RiskDecision(BaseEvent):
    event_type:        Literal["risk_decision"] = "risk_decision"
    signal_id:         UUID
    pair_id:           str
    status:            RiskDecisionStatus
    approved_notional: Decimal | None
    rejected_reason:   str | None
    risk_rule_results: list[dict]
```

## Order Request

```python
class OrderRequest(BaseEvent):
    event_type:       Literal["order_request"] = "order_request"
    pair_id:          str
    risk_decision_id: UUID
    ticker:           str
    side:             Literal["buy", "sell"]
    quantity:         Decimal
    order_type:       OrderType
    limit_price:      Decimal | None
    venue:            Literal["us_equity", "foreign_equity"]
    currency:         str
    is_short_sale:    bool = False
    time_in_force:    str = "DAY"
```

## Fill Event

```python
class FillEvent(BaseEvent):
    event_type:          Literal["fill"] = "fill"
    fill_id:             str
    client_order_id:     UUID
    broker_order_id:     str
    pair_id:             str | None
    ticker:              str
    side:                Literal["buy", "sell"]
    fill_price:          Decimal         # in native currency
    fill_price_usd:      Decimal         # USD-converted
    fill_quantity:       Decimal
    remaining_quantity:  Decimal
    commission:          Decimal
    commission_currency: str
    sec_fee:             Decimal
    stamp_duty:          Decimal
    short_borrow_fee:    Decimal
    fx_rate_used:        Decimal | None
    is_short_sale:       bool
    venue:               Literal["us_equity", "foreign_equity"]
    exchange:            str
    metadata:            dict = {}
```

## Position Update

```python
class PositionUpdateEvent(BaseEvent):
    event_type:               Literal["position_update"] = "position_update"
    pair_id:                  str
    ticker:                   str
    venue:                    Literal["us_equity", "foreign_equity"]
    net_quantity:             Decimal
    average_entry_price:      Decimal
    average_entry_price_usd:  Decimal
    unrealized_pnl_usd:       Decimal
    realized_pnl_usd:         Decimal
    mark_price:               Decimal
    mark_price_usd:           Decimal
    notional_value_usd:       Decimal
    fx_rate:                  Decimal | None
    is_short:                 bool
    trigger:                  Literal["fill", "mark_to_market", "fx_update", "reconciliation"]
```

## Asian ADR Approved Pair (Registry)

```python
class AsianADRApprovedPair(BaseModel):
    model_config = {"frozen": True}

    pair_id:              str
    adr_ticker:           str
    underlying_ticker:    str
    underlying_exchange:  str           # "TSE", "HKEX", "KRX", "ASX", etc.
    underlying_currency:  str           # "JPY", "HKD", "KRW", "AUD", etc.
    adr_ratio:            Decimal       # local shares per 1 ADR; fixed structural constant
    estimation_days:      int = 60      # T: rolling window for µ/σ
    holding_days:         int = 90      # H: max holding period before force-close
    k0:                   Decimal = Decimal("2.0")
    kc:                   Decimal = Decimal("0.0")
    zero_return_pct_adr:  Decimal       # liquidity metric (Bekaert et al. 2007)
    roll_spread_local:    Decimal       # Roll (1984) effective spread, local leg
    roll_spread_adr:      Decimal       # Roll (1984) effective spread, ADR leg
    fx_hedge_required:    bool = False
    withholding_tax_rate: Decimal = Decimal("0.0")
    approved_date:        date
    expiry_date:          date
    is_active:            bool = True
```

## ROCE / RUCE Result

```python
class LiquidityBucket(str, Enum):
    HIGH        = "high"
    HIGH_MEDIUM = "high_medium"
    MEDIUM_LOW  = "medium_low"
    LOW         = "low"

class RoceRuceResult(BaseModel):
    model_config = {"frozen": True}

    pair_id:          str
    trade_open_date:  date
    trade_close_date: date
    duration_days:    int
    local_return:     Decimal
    adr_return:       Decimal
    roce:             Decimal
    ruce:             Decimal
    roll_cost_pct:    Decimal     # estimated round-trip cost (Roll 1984)
    roce_net:         Decimal     # roce − roll_cost_pct
    ruce_net:         Decimal     # ruce − roll_cost_pct
    liquidity_bucket: LiquidityBucket
    was_force_closed: bool
    was_aborted:      bool        # True if local leg never established (overnight reversal)
```

## Topic Map

| Topic | Producers | Consumers |
|-------|-----------|-----------|
| `market-data` | Market Data Handler | H&S Engine, Position Engine, Sequencer |
| `fx-rates` | FX Rate Feed | H&S Engine, Position Engine |
| `signals` | H&S Engine | Risk Engine |
| `risk-decisions` | Risk Engine | Asian Execution Sequencer |
| `orders` | Asian Execution Sequencer | Broker Gateway, Monitoring |
| `fills` | Broker Gateway | Sequencer, Position Engine, ROCE/RUCE Calculator |
| `positions` | Position Engine | Risk Engine, Dashboard |
| `pair-registry` | Research Engine | H&S Engine, Risk Engine |
| `alerts` | Risk, Feed, Sequencer | Monitoring, Dashboard |

## Parquet Cache Schemas

| File | Columns | Notes |
|------|---------|-------|
| `adr/adr_prices.parquet` | `infocode`, `marketdate`, `close/high/low/open`, `volume`, `adj_factor`, `ticker`, `isin` | 4M+ rows |
| `adr/adr_reference.parquet` | `adr_ticker`, `adr_isin`, `adr_infocode`, `underlying_ticker`, `underlying_exchange`, `underlying_currency`, `adr_ratio` | `adr_ratio` always NULL from WRDS; must be sourced separately |
| `global/global_prices.parquet` | `infocode`, `marketdate`, `close/high/low/open`, `volume`, `adj_factor`, `ticker`, `exchange`, `currency` | Filtered to Asian exchanges |
| `fx/fx_rates.parquet` | `date`, `base_currency`, `quote_currency`, `currency_pair`, `mid`, `provider` | `mid` = USD per 1 unit of base; inverted from Datastream CCY/USD |
