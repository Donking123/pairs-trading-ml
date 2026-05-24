# Phase 0 — Data Spine

**Pipeline stage:** 1 (Data) · **Status:** ✅ Complete · **Checkpoint:** passed

> Goal: assemble one clean, honest, survivorship-bias-free price panel that every
> later phase is built on. "Garbage in, garbage out" — this phase is the foundation.

---

## 1. What we built

| File | Purpose |
|---|---|
| `src/config.py` | Single source of truth — sample period, paths, trading-design constants |
| `src/wrds_pull.py` | Pulls 5 datasets from WRDS, caches them as parquet in `data/` |
| `.gitignore`, `requirements.txt`, `README.md` | Repo scaffold |

Run: `python src/wrds_pull.py` (after one-time `wrds.Connection().create_pgpass_file()`).

---

## 2. The 5 datasets pulled

| Dataset | WRDS source | What it is | Why we need it |
|---|---|---|---|
| `sp500_constituents` | `crsp.dsp500list` | Point-in-time S&P 500 membership (permno + entry/exit dates) | Defines the trading universe; **survivorship-bias fix** |
| `crsp_daily` | `crsp.dsf` + `crsp.dsenames` | Daily price, **bid/ask**, return, volume, `cfacpr`, share/exchange/SIC codes | The core dataset the whole strategy runs on |
| `delisting` | `crsp.dsedelist` | Delisting date, reason code, **delisting return** | Realistic forced-exit losses |
| `sp500_index` | `crsp.dsp500` | Daily S&P 500 level + return | PC metric's market control + buy-and-hold benchmark |
| `ff_factors` | `ff.fivefactors_daily` + `ff.factors_daily` | FF5 factors + Momentum + risk-free rate | Performance attribution (Phase 2) + factor-beta extension inputs (Phase 3) |

---

## 3. Realism baked in at Phase 0

Three of the project's realism principles are enforced *by how the data was pulled* —
not by a later overlay:

- **No survivorship bias** — `crsp.dsp500list` is point-in-time: it includes every
  stock that was *ever* in the index, including those that later went bankrupt or were
  dropped. Using "today's S&P 500" would secretly keep only the winners.
- **Costs in the data** — `crsp.dsf` carries the closing **bid and ask**. We buy at the
  ask, sell at the bid; that gap *is* the transaction cost. No separate commission model.
- **Realistic delisting** — `crsp.dsedelist` gives the delisting return, so a pair
  holding a delisted leg books the real (often large, negative) loss.

---

## 4. Checkpoint results

| Metric | Got | Paper | Verdict |
|---|---|---|---|
| Months | 288 | 288 | ✅ exact |
| Trading days | 6,037 | 6,039 | ✅ 99.97% |
| Date range | 2000-01-03 → 2023-12-29 | 2000–2023 | ✅ |
| Stocks (final universe) | 991 | ~1,098 | ✅ explained below |
| `crsp_daily` rows | 4,059,948 | — | — |
| Data quality | prc/ret 99.8%, bid/ask 98% non-null | — | ✅ |

---

## 5. Universe investigation — why 991 stocks

Worked through carefully because the count looked short of the paper's "1,098 stocks".

```
crsp.dsp500list          1,098 rows  =  1,070 unique stocks
                                          └─ 28 stocks left the index and rejoined
                                             → each membership spell is its own row
        │
        ├─ 79 removed by the 10/11 common-share filter
        ▼
   991 ordinary US common stocks   ←  the final clean trading universe
```

The 79 dropped stocks, verified against `crsp.dsenames`:

| CRSP `shrcd` | Meaning | Count |
|---|---|---|
| 12 | Ordinary common, **incorporated outside the US** | 40 |
| 18 | **REIT** | 29 |
| 48 | Closed-end / other fund | 8 |
| 31 | Certificate | 1 |
| 12→72 | Foreign-incorporated, later ADR-type | 1 |

**Verification result:** `0` dropped stocks had share code 10/11 in-window, and `0`
had missing name records — so all 79 are fully and correctly explained. The company
names confirm it (Schlumberger Ltd, Linde PLC, NXP Semiconductors NV, Unilever NV …).

**Conclusion:** no bug. Excluding REITs and foreign-incorporated stocks is the paper's
**deliberate design** (they have different return/payout dynamics). The ~2.5% gap
between our 1,070 unique stocks and the paper's stated 1,098 is most likely the paper
counting membership *spells* rather than unique stocks, or minor CRSP revisions.

---

## 6. Key concepts (career reference)

- **Survivorship bias** — backtesting only on stocks that exist *today* deletes every
  failure, inflating returns. Fix: point-in-time universe membership.
- **CRSP share codes (`shrcd`)** — 2-digit: first digit = security type, second =
  sub-class. `10/11` = ordinary US common stock; `12` = foreign-incorporated;
  `18` = REIT; `48` = fund. Filtering to 10/11 keeps clean domestic common stock.
- **Adjustment factor (`cfacpr`)** — raw prices jump on splits/dividends. `price /
  cfacpr` gives a continuous, comparable series.
- **Bid/ask** — the bid–ask gap is the realistic transaction cost; marking PnL at
  bid/ask builds the cost into the price data itself.
- **Membership spells** — a stock dropped from and re-added to the index gets multiple
  rows in `crsp.dsp500list`; rows ≠ unique stocks.

## 7. Tooling notes

- `python -m py_compile <file>` — compiles to bytecode to check **syntax only**,
  without running the code. Fast safety gate before execution.
- Shell `&&` — run the next command only if the previous one succeeded.
- `python3 - <<'EOF' … EOF` — a heredoc; runs an inline script from stdin without a
  temp file.
- **WRDS connection** — `~/.pgpass` stores the *password*; `wrds.Connection()` still
  needs the *username* (pass `wrds_username=` for non-interactive runs).

---

## 8. Status & next

Phase 0 complete and locked down. Universe = **991 survivorship-bias-free ordinary US
common stocks**, 2000–2023, with bid/ask and delisting data ready.

**Next:** Phase 1 — SSD vertical slice (build the whole pipeline for one metric).
