# ROCE / RUCE Calculator

**Package**: `asian_adr.backtest`

## Responsibilities

- Subscribe to closed round-trip fill pairs (ADR fill + local fill)
- Compute ROCE and RUCE per trade as defined in §2.4 of the strategy specification
- Tag each trade with a `LiquidityBucket` for post-hoc attribution
- Accumulate per-pair and aggregate distribution statistics matching paper Table 7-B format
- Produce tearsheet output integrated into the backtest report

**Inputs**: `FillEvent` (topic: `fills`) — matched ADR short, ADR cover, local buy, local sell fills
**Outputs**: `RoceRuceResult` per closed round-trip; aggregate distribution statistics

## Return Formulae

```
local_return  =  (P_local_close − P_local_open)  /  P_local_open
adr_return    =  (P_ADR_short   − P_ADR_cover)   /  P_ADR_short

ROCE  =  local_return + adr_return
RUCE  =  local_return + 2 × adr_return          # Reg-T 50% margin → 2× ADR component
```

## Implementation

```python
class RoceRuceCalculator:
    def on_round_trip_closed(
        self,
        adr_short_fill:  FillEvent,
        adr_cover_fill:  FillEvent,
        local_buy_fill:  FillEvent,
        local_sell_fill: FillEvent,
    ) -> RoceRuceResult:
        p_adr_open  = adr_short_fill.fill_price
        p_adr_close = adr_cover_fill.fill_price
        p_loc_open  = local_buy_fill.fill_price_usd
        p_loc_close = local_sell_fill.fill_price_usd

        local_return = (p_loc_close - p_loc_open) / p_loc_open
        adr_return   = (p_adr_open  - p_adr_close) / p_adr_open

        roce = local_return + adr_return
        ruce = local_return + Decimal("2") * adr_return

        duration_days = (
            local_sell_fill.timestamp_exchange.date()
            - local_buy_fill.timestamp_exchange.date()
        ).days

        return RoceRuceResult(
            pair_id=adr_short_fill.pair_id,
            roce=roce,
            ruce=ruce,
            local_return=local_return,
            adr_return=adr_return,
            duration_days=duration_days,
            liquidity_bucket=self._assign_bucket(adr_short_fill.pair_id),
            was_force_closed=adr_cover_fill.metadata.get("reason") == "force_close",
            was_aborted=False,
        )
```

## Liquidity Bucket Assignment

| Bucket | Zero-return-day % (ADR) | Paper median ROCE |
|--------|------------------------|-------------------|
| High | < 6.21% | ~2.0% |
| High-Medium | 6.21–14.57% | ~2.8% |
| Medium-Low | 14.57–29.75% | ~3.0% |
| Low | > 29.75% | ~3.7% |

Bucket monotonicity (Low > Medium-Low > High-Medium > High) is a required validation check — failure indicates a spread computation or fill-price bug.

## Distribution Statistics (Table 7-B Format)

Per-pair and aggregate statistics computed per backtest run:

| Statistic | Metrics reported |
|-----------|-----------------|
| ROCE per trade | mean, std, max, p90, p75, median, p25, p10, min |
| RUCE per trade | mean, std, max, p90, p75, median, p25, p10, min |
| Duration (days) | mean, std, max, p90, p75, median, p25, p10, min |
| Trades per firm per year | mean, std, max, p90, p75, median, p25, p10, min |
| Roll cost (round trip) | mean, std, max, p90, p75, median, p25, p10, min |

## Paper Benchmarks (k0=2, kc=0, T=60, H=90)

| Metric | Paper value |
|--------|-------------|
| Median ROCE | ~2.8% |
| Median RUCE | ~5.3% |
| Median net RUCE | ~2.7% |
| Median duration | 3 days |
| IQ range of duration | 1–6 days |
| Trades per firm per year (median) | 11.6 |
| ADR leg contribution | ~90% of ROCE |
