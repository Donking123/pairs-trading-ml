# Strategy Specification

**Reference**: Hong & Susmel (2013), *Pairs-Trading in the Asian ADR Market*

## Dollar Spread

```
spread_t  =  P_ADR,t  −  (P_local,t × FX_t) / ratio
```

| Variable | Definition |
|----------|------------|
| `P_ADR,t` | U.S. ADR closing price in USD |
| `P_local,t` | Asian underlying closing price in local currency |
| `FX_t` | FX spot rate (USD per 1 unit of local currency) |
| `ratio` | ADR conversion ratio (local shares per 1 ADR); fixed structural constant, never OLS-estimated |

## Rolling Estimation

```
µ_t  =  mean(spread[t-T : t])
σ_t  =  std(spread[t-T : t],  ddof=1)     # WelfordRollingStats
```

## Entry / Exit Thresholds

```
κ_open  =  µ_t + k0 × σ_t      # entry: emit SHORT_ADR when spread_t > κ_open
κ_close =  µ_t + kc × σ_t      # exit:  emit EXIT when spread_t < κ_close
```

Z-score equivalence: `z_t = (spread_t − µ_t) / σ_t`; entry at `z > k0`, exit at `z < kc`.

`kc = 0` (default) closes when the spread returns to its rolling mean — law-of-one-price convergence.

## Parameter Grid

| Parameter | Values tested (paper) | Platform default |
|-----------|----------------------|-----------------|
| k0 (entry multiplier) | 1.65, 2.0, 2.33 | **2.0** |
| kc (exit multiplier) | 0.0, 0.5, 1.0 | **0.0** |
| T (estimation window, days) | 30, 60, 90, 120 | **60** |
| H (max holding period, days) | 30, 90, 120 | **90** |

## Execution Model

Asian markets (TSE, HKEX, KRX, etc.) and NYSE have **zero trading-hour overlap**. Establishing both legs always crosses the overnight gap.

### Open sequence

```
Day D,  U.S. close (16:00 ET)       →  spread_D > κ_open  →  SELL ADR (short)
Day D+1, Asia open (≈09:00 local)   →  recheck spread vs κ_close:
    if spread still > κ_close       →  BUY local  (pair fully open)
    if spread reversed overnight    →  BUY ADR cover (abort — no local leg ever placed)
```

### Close sequence

```
Any day K: spread_K < κ_close  OR  days_held ≥ H (force-close)
    →  SELL local (Asia close bar)
    →  BUY ADR cover (same or next U.S. bar)
```

**Why no FX hedge**: the overnight gap makes simultaneous FX spot submission impossible. The paper accepts residual FX exposure. `fx_hedge_required = False` on all registry pairs.

## Return Metrics

```
local_return  =  (P_local_close − P_local_open)  /  P_local_open
adr_return    =  (P_ADR_short   − P_ADR_cover)   /  P_ADR_short

ROCE  =  local_return + adr_return
RUCE  =  local_return + 2 × adr_return          # Reg-T 50% margin → 2× ADR component
```

RUCE is the economically realistic measure (investor posts 50% margin on short ADR leg). ROCE is the conservative bound (full notional as denominator on both legs).

### Paper benchmarks (k0=2, kc=0, T=60, H=90)

| Metric | Paper value |
|--------|-------------|
| Median ROCE | ~2.8% |
| Median RUCE | ~5.3% |
| Median duration | 3 days |
| IQ range of duration | 1–6 days |
| Trades per firm per year (median) | 11.6 |
| ADR leg contribution to returns | ~90% |
| Median net RUCE (after Roll costs) | ~2.7% |

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
| Min continuous trading | 2 years |
| Min non-zero return days (both legs) | ≥ 50% of trading days |
| Zero-return-day % (ADR, entry ceiling) | ≤ 50% |
| ADR ratio available | Required; ratio > 0 |

## Liquidity Bucketing

| Bucket | Zero-return-day criterion | Typical median ROCE |
|--------|--------------------------|---------------------|
| High | < 6.21% | ~2.0% |
| High-Medium | 6.21–14.57% | ~2.8% |
| Medium-Low | 14.57–29.75% | ~3.0% |
| Low | > 29.75% | ~3.7% |

Higher illiquidity → higher median profit (limits-to-arbitrage premium). The `ZeroReturnDayFilter` risk rule blocks entries when the ADR zero-return-day percentage exceeds a configurable ceiling.

## Transaction Cost Benchmark

Roll (1984) effective spread: `2 × √|autocov(returns)|`

Used as a post-hoc round-trip cost estimate per pair — not as an entry signal. The paper's median total Roll cost is ~2.67%, making net RUCE (~2.7%) modestly positive.
