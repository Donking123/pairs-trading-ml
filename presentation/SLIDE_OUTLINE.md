# Asian ADR Pairs Strategy — Slide-by-Slide Outline (QF621 Project)

A build-ready outline for the PowerPoint deck. Each slide lists the **title**,
the **bullet content** to type, and the **exact chart/number source** to drop in.
Framing is methodology-first (academic). Every figure traces to a repo file — see
the "src" note under each data slide.

Numbers verified from:
- `review/output/liquidity_sweep_shares.csv` (liquidity floors)
- `review/output/equity_by_floor/floor_*/portfolio_stats.json` (per-floor stats)
- `ADR_RESEARCH_README.md` / `datastream/data/walkforward_output/walkforward_*/summary.json` (headline OOS)

---

## Slide 1 — Title

**Asian ADR Pairs Trading**
*Cointegration-screened, out-of-sample validated*

- Team members · QF621 Quantitative Trading Strategies
- Subtitle: "Datastream universe 2011–2026 · 224 ADR pairs · walk-forward OOS"

---

## Slide 2 — The trade in one sentence

- An ADR and its local Asian share are the **same company, two listings** → their
  USD prices should track. A wide gap is a temporary mispricing.
- **Short the ADR** when its USD price is at a wide premium to the FX-converted
  local share; **buy the local leg** next session; **unwind on convergence**.
- This is mean-reversion of a **structural** spread, not a statistical hedge ratio.

*(Optional: simple 2-box diagram ADR ↔ Local share, with the spread between.)*

---

## Slide 3 — Data extraction

- **Source:** WRDS **Datastream**, daily OHLCV, **2011-05-04 → 2026-04-30**.
- **Three fetchers, run in order** (global + FX read the ADR reference produced first):
  1. `fetch_datastream_adr_data.py` → ADR OHLCV + `adj_factor`, and the ADR→underlying
     reference table with **`adr_ratio`**
  2. `fetch_datastream_global_data.py` → Asian underlying OHLCV
  3. `fetch_fx_history.py` → daily FX mid (USD per 1 local-currency unit)
- **Four cached Parquet tables** feed everything downstream (no re-pull needed).
- **`adr_ratio` is taken verbatim from Datastream — never OLS-estimated** (spec ADR-002).
- **Corporate actions:** prices adjusted multiplicatively `adj = raw × adj_factor`;
  trades whose window straddles a factor change are flagged and dropped (defensive backstop).
- **Coverage:** Japan, Hong Kong, Korea, India, Australia, Singapore, Taiwan,
  Indonesia, Philippines, Malaysia.

**Chart:** `presentation/charts/03_universe_breakdown.png`
**src:** `USERGUIDE.md` §1; `run_walkforward.py` (`ASIAN_EXCHANGES`); `run_backtest._adj_prices`

---

## Slide 4 — How the spread is calculated

- **Spread formula (verbatim from the code):**

  > **spread_t = P_ADR,t − (P_local,t × FX_t) / adr_ratio**

- Term by term:
  - `P_ADR,t` — ADR price, already in USD
  - `P_local,t × FX_t` — local share converted to USD
  - `÷ adr_ratio` — puts both legs on a **per-ADR-share** basis
  - result = residual **USD mispricing** (the tradable signal)
- `adr_ratio` is a **structural constant** (e.g. 1 ADR = N local shares) → no
  estimation noise, no look-ahead, unlike a classic OLS hedge ratio.

**Chart:** `presentation/charts/02_spread_example.png` (worked single-pair example)
**src:** `run_backtest.py:260`; `run_walkforward.py` screening-engine header

---

## Slide 5 — Pair screening (building the tradable universe)

Each candidate ADR/underlying pair must pass, on its own spread series:

1. **Cointegration** — Augmented Dickey–Fuller **and** Phillips–Perron both reject
   a unit root at 5% (`cointegration_alpha = 0.05`).
2. **Liquidity filters** — ≥ 50% non-zero-return days on **both** legs; ≥ **504
   trading days (~2y)** of joint history; ADR zero-return days ≤ 50%.
3. **Roll (1984) effective spread** estimated per leg → baked-in transaction-cost proxy.

- **Output:** `config/pairs/asian_adr_pairs.json` — ~**224** approved pairs
  (~**190** selected on the train window in the OOS run).

**src:** `run_walkforward.py` `run_pipeline` (spec §3.4)

---

## Slide 6 — Entry & exit signals

**Signal layer (per pair):**
- Trailing mean/std of the spread over **T = 90 days**, computed "as of yesterday"
  (shifted by 1 day → no look-ahead).
- Entry band: **κ_open = μ_t + k0·σ_t**, with **k0 = 2.5**
- Exit band: **κ_close = μ_t + kc·σ_t**, with **kc = 0.0** (full mean-reversion)

**Execution sequencer (time-zone-aware, next-bar fills):**
- **Entry — Day D US close:** if `spread > κ_open` → **SHORT ADR** at the close.
- **Day D+1 Asia open:** if spread still `> κ_close` → **BUY local** (position OPEN);
  otherwise **overnight abort** (cover ADR → 1-day trade).
- **Exit — any day K:** `spread < κ_close` **OR** `days_held ≥ H (90)` → cover ADR
  at US close, sell local at next Asia open.
- **Realism guards:** holiday-gap filter (skip if next joint bar > 4 calendar days);
  the D+1 proceed/abort decision uses only info available at the Asia open.

**Return definitions:**
- `ROCE = local_ret + adr_ret`
- **`RUCE = local_ret + 2·adr_ret`** (Reg-T 50% margin → 2× on the ADR leg)
- `*_net` subtracts ADR borrow; Roll spread already inside the effective fill prices.

**Charts:** `presentation/charts/01_timeline_schematic.png` (entry/exit timeline),
`08_ruce_hist.png` (per-trade return distribution)
**src:** `run_backtest.py` `backtest_pair` (lines 324–624)

---

## Slide 7 — Out-of-sample discipline (why the numbers are trustworthy)

- An in-sample backtest is **optimistic** — pairs are chosen on the same history
  they are then traded on (look-ahead in selection).
- **Walk-forward removes this:** select pairs using **TRAIN [2010–2020] only**,
  then trade **TEST (2021–2025] only**. Rolling stats warm up on pre-split history,
  but **no trade opens or closes before the split**.
- Supports expanding folds (`--folds N`): train window grows, test rolls forward.
- **Optimal parameters** from a 1,080-run grid: **T=90, k0=2.50, kc=0.0, H=90**
  — higher k0 ⇒ more selective ⇒ better trade quality; kc=0 captures full
  convergence; H rarely binds.

**Chart:** `presentation/charts/04_walkforward_schematic.png`
**src:** `run_walkforward.py`; `ADR_RESEARCH_README.md`

---

## Slide 8 — Headline out-of-sample results

- Single-fold walk-forward (T=90, k0=2.5): ~148–190 pairs selected,
  ~982–2,041 OOS trades.
- Median **RUCE-net ≈ 5.3%** / **ROCE-net ≈ 2.6%** per trade; median duration
  ~5–6 days; per-trade win rate ~**78%**.
- Portfolio (full universe, equal-weight, daily MTM): **ROCE-net Sharpe ≈ 2.85**
  (lead metric); **RUCE Sharpe ≈ 2.0** as the Reg-T-levered variant.

**Charts:** `presentation/charts/05_equity_curve.png`, `06_drawdown_underwater.png`,
`07_rolling_sharpe.png`, `00_kpi_strip.png`
**src:** latest `datastream/data/walkforward_output/walkforward_*/summary.json`;
`ADR_RESEARCH_README.md`

---

## Slide 9 — Where the alpha comes from (brief)

- The trade is **structurally an overnight trade** — entry at US close, first exit
  at Asia open, so every trade crosses ≥1 overnight window.
- **ADR overnight premium decay:** US-hours flows push the ADR to a >2.5σ premium;
  the premium mean-reverts overnight as the US–Asia gap closes.
- The **ADR short leg carries essentially all the alpha** (Sharpe ~2.8 alone);
  the local long leg defines the spread but adds little.
- Returns are **duration-invariant** → alpha is front-loaded into the overnight gap.

**Chart:** `presentation/charts/10_leg_attribution.png`
**src:** `ADR_RESEARCH_README.md` ("Alpha Source")

---

## Slide 10 — Output under varying liquidity (the tradeability screen)

**Setup:** The **floor** = minimum **median ADR share-volume over the train window
(2010–2020)** — a tradeability filter. Higher floor ⇒ fewer, more liquid, more
realistically tradable pairs. (Single-fold WF; ROCE-net; equal-weight; daily MTM.)

**Results (OOS):**

| Share floor | Pairs | Trades | Sharpe | Ann. return | Max DD | Total return |
|---|---|---|---|---|---|---|
| **0 (none)** | 190 | 2,041 | **2.85** | 25.6% | −5.2% | +211.6% |
| **50k** | 25 | 271 | **1.03** | 2.3% | −1.4% | +12.1% |
| **250k** | 10 | 102 | **0.27** | 0.3% | −1.0% | +1.7% |
| **1M** | 3 | 32 | **0.39** | 0.3% | −1.0% | +1.4% |

*(Newey–West–adjusted Sharpe, same order: 3.53 / 1.40 / 0.48 / 0.76.)*

**Story — tradeability screen:**
- The full universe earns the headline Sharpe but leans on **thin names**; a
  liquidity floor gives the **honest, tradeable** subset.
- **Even the most-liquid 1M-floor subset (3 pairs, 32 trades) stays profitable**
  (Sharpe 0.39, +1.4%, shallow −0.95% drawdown) — the edge is **not** purely a
  thin-name artifact, though **capacity shrinks sharply** with the floor.
- **Recommended operating floor ≈ 50k shares/day:** 25 pairs / 271 trades keep a
  respectable **Sharpe ~1.0** with a tiny **−1.4% drawdown** — the best
  liquidity-vs-edge balance. Above ~100k the sample becomes too small to rely on.

**Charts:** `review/output/liquidity_sweep_shares.png` (Sharpe vs floor, all 9
thresholds incl. 0); `review/output/equity_by_floor/equity_curves_by_floor.png`
(growth-of-$1 overlay for the floors)
**src:** `review/output/liquidity_sweep_shares.csv`;
`review/output/equity_by_floor/floor_*/portfolio_stats.json`

---

## Slide 11 — Robustness checks

From the OOS review suite (all run on `trades_oos.csv`):
- **Stale-price check** — ~5% of pairs flagged; the edge is **not** a stale-price artifact.
- **Significance** — random-entry null + bootstrap CI confirm the median edge is real.
- **Cost sensitivity** — Roll + borrow cost ≈ **0.2% of gross RUCE** (effectively free).
- **Market-neutrality** — no hidden long-EM / long-FX beta.
- **Regime independence** — edge present across low/med/high volatility regimes.

**Charts:** `presentation/charts/09_cost_sensitivity.png`,
`12_integrity_before_after.png`
**src:** `USERGUIDE.md` §6

---

## Slide 12 — Conclusion

- A **structural spread** + a **strict cointegration screen** + **genuine OOS
  validation** → a real but **capacity-limited** edge.
- **Recommended operating point:** ~**50k share floor**, **T=90 / k0=2.5 / kc=0 /
  H=90**; add Reg-T leverage on the ADR leg for the levered (RUCE) variant.
- **Limitations:** liquidity proxy is shares (not dollars); EOD/Asia-open fills
  (no intraday); OOS window is ~2021–2025.

---

### Appendix — reproduction commands

```bash
# Pair registry ships pre-built at config/pairs/asian_adr_pairs.json
# (screening engine lives in run_walkforward.py; the walk-forward re-screens
#  pairs per train window, so no separate build step is needed).

# Out-of-sample walk-forward (headline numbers)
python datastream/run_walkforward.py \
    --train-start 2010-01-01 --split 2020-12-31 --test-end 2025-12-31

# Liquidity sweep (table on slide 10)
python review/run_sweep_liquidity.py

# Per-floor equity curves (0/50k/250k/1M overlay)
python datastream/run_walkforward.py --floors 0 50000 250000 1000000
python review/run_walkforward_portfolio.py --by-floor review/output/equity_by_floor
python review/plot_equity_curves_by_floor.py

# Pitch charts
cd presentation && python make_charts.py
```
