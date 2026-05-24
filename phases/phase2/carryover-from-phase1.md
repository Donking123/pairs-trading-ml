# Phase 1 → Phase 2 Carryover

**Created:** 2026-05-24 (right before starting Phase 2 build).
**Purpose:** consolidate everything from Phase 1 that informs Phase 2 — what's settled,
what's still open, and what discipline to apply. Read this **before** writing any
Phase 2 code; refer back when in doubt.

---

## TL;DR

Six things carry over from Phase 1 to Phase 2:

1. **The pipeline architecture is sound** — risk-side metrics match the paper; the
   Sharpe gap is a *return-numerator* issue, not a plumbing bug.
2. **The bimodal duration finding is the Phase 2 motivation** — 11.4% reversion (+471 bps)
   minus 88.4% force-close (−32 bps) = net +30. Phase 2 attacks the −32.
3. **16 decisions are locked** — OPTICS params, position sizing, allocation, etc. Don't
   re-litigate. Only the metric and the filter change in Phase 2.
4. **Crisis-regime dependency is structural** — Phase 2 will soften it modestly but
   not eliminate it. Report Sharpe by regime.
5. **Methodology discipline carries over** — synthetic-tests-first, single-variable
   changes, lock-hyperparameters-once, attribution-alongside-backtest.
6. **5 Phase 2 decisions are open** (D2.1–D2.5 in `decisions.md`). Resolve each when
   building the relevant function.

---

## Item 1 — Pipeline architecture is sound

### Evidence
- Risk-side metrics match the paper within tolerance:

  | Risk metric | Ours | Paper | Match? |
  |---|---:|---:|---|
  | Annualised vol | 5.28% | ~5.6% | ✅ |
  | Max drawdown | −14.3% | ~−15% | ✅ |
  | Hit rate | 57.4% | ~57% | ✅ |

- All 18 synthetic tests pass.
- Direction (long-spread vs short-spread) is balanced — no hidden sign-flip bug.
- GOOG/GOOGL co-cluster (CP1 sanity check) ✅.

### Implication for Phase 2
We're testing economic hypotheses on top of a working pipeline. If PC + filter
lands at Sharpe 1.0, that confirms the pipeline; if it doesn't, the gap is
architectural (allocation, survivorship) — investigated in Phase 3.

### One-sentence carryover
> *"The plumbing works. Phase 2 changes inputs to the same pipeline."*

---

## Item 2 — The bimodal duration finding (Phase 2 motivation)

### The Phase 1 identity (the load-bearing fact)

Across 12,255 trades 2003-2023:

| Exit reason | n | % | mean dur | **mean P&L** | win rate | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| **reversion** (z crossed 0) | 1,392 | 11.4% | 14.5d | **+471 bps** | **99.2%** | **+65.58** |
| **force_close** (month ended) | 10,832 | 88.4% | 19.9d | **−32 bps** | 48.2% | **−34.25** |
| delisting | 31 | 0.3% | 17.1d | −297 bps | 32% | −0.92 |
| **NET** | 12,255 | 100% | 19.2d | +25 bps | 54% | **+30.41** |

$$
\underbrace{+65.58}_{\text{reversion (11.4\%)}} + \underbrace{-34.25}_{\text{force\_close (88.4\%)}} + \underbrace{-0.92}_{\text{delisting}} = +30.41
$$

### Economic interpretation
Force-closes split into two populations:
1. **Will revert eventually, just not in 21 days** — half-life longer than window.
   Roughly neutral P&L contribution.
2. **Won't revert ever** — cointegration broke during the trading window (one stock
   went bankrupt mid-month, sector relationship shifted, etc.). These are the net
   losers (MBI/FRE, FRC/EL style trades).

The −32 bps mean is the average. Population (2) drags it below zero.

### Phase 2 lever
**Engle-Granger ADF cointegration filter rejects population (2) directly** — pairs
whose residual spread fails the stationarity test are filtered out before they
ever reach the backtest.

### Lever arithmetic

| Scenario | reversion | force_close | delisting | **Net** | Approx Sharpe |
|---|---:|---:|---:|---:|---:|
| Phase 1 actual | +65.58 | −34.25 | −0.92 | **+30.41** | 0.589 |
| Halve drag (hypothetical) | +65.58 | **−17.13** | −0.92 | **+47.53** | ~0.9 |
| Eliminate drag (hypothetical) | +65.58 | **0** | −0.92 | **+64.66** | ~1.2 |

### Caveat
The filter isn't perfect — it removes some population-(1) pairs too. Paper reports
PC-filtered Sharpe = 0.80 (slightly *lower* than unfiltered PC's 1.01) because the
filter trims case-1 as well as case-2. But filtered has **lower drawdown** and
**more stable Sharpe across regimes** — that's its other value.

### Two complementary Phase 2 levers

| Component | What it changes |
|---|---|
| **PC distance** | Replaces SSD as candidate generator → finds more idiosyncratic reverters → grows the +65.58 |
| **Cointegration filter** | Trims case-2 pairs → shrinks the −34.25 |

### One-sentence carryover
> *"Phase 2's job is to shift the +65.58 / −34.25 / +30.41 identity. PC grows the
> reversion total; the filter shrinks the force-close drag."*

---

## Item 3 — Locked decisions that carry over unchanged

These are the project's "constitution." Change any and Phase 1 ↔ Phase 2 stops
being comparable.

### The locked list

| Decision | Value | Why locked |
|---|---|---|
| **OPTICS hyperparameters** | `xi=0.10`, `min_samples=2`, `min_cluster_size=2` | Tuned via cross-date validation; locked in `src/config.py`. Same OPTICS on PC. |
| Formation window | 3 years (756 days) | Paper §4.1 |
| Trading window | 1 month (~21 days) | Paper §4.1 |
| Z-score lookback | 126 days, strict past only | Paper §3.2 |
| Entry threshold | \|z\| ≥ 2.0 | Paper §3.3 |
| Exit threshold | z = 0 (zero-cross) | Paper §3.3 |
| Stop-loss | None in core (3.5σ → realism Phase 4) | Paper-faithful Sharpe target |
| Hedge ratio | OLS, frozen per trading month | Required for EG two-step |
| Position sizing | Equal-dollar ($0.50/$0.50) | Paper / Gatev-Goetzmann |
| Allocation across pairs | Equal-weight across currently-open | Paper convention |
| Execution timing | t+1 close-to-close | Realistic, no open-price data needed |
| Delisting fallback | Option B (code-dependent) | Empirically minimal impact, right default |
| Force-close at month end | Always | Paper convention; not the cause of Sharpe gap |
| Universe filter | Share codes 10/11, continuous index membership, $5M ADV | Phase 0 baked-in |
| Risk-free rate | 0% (gross Sharpe) | Paper-faithful comparison |
| Synthetic-tests-first | Mandatory for new modules | 18/18 currently pass |

### Deferred but suspect (NOT locked — known limitations)

These are *known* potential Sharpe-gap contributors. We're not touching them in
Phase 2 — queued for Phase 3 sensitivity:

| Suspect | Why we suspect it | When to investigate |
|---|---|---|
| Equal-weight allocation | Dilutes strong (z=3.5) signals to same weight as marginal (z=2.1) | Phase 3: try \|entry-z\|-weighted |
| Strict continuous-membership universe | approximately 407 stocks/month vs paper's approximately 500 | Phase 3: soften filter |
| t+1 close-to-close (vs open[t+1]) | Modest execution lag | Phase 4: low priority |

### Phase 2 NEW hyperparameters (will need tuning with same discipline)

| Parameter | Default | Tuning discipline |
|---|---|---|
| ADF p-value threshold | 0.05 | Cross-date validate; sensitivity-test {0.01, 0.10}. Lock once chosen. |
| Half-life bounds | [5, 60] days | Paper default; sensitivity-test {[3,90], [10,40]}. Lock once chosen. |
| Market-adjustment factor | SPY beta (D2.1) | Try FF 3-factor as robustness if PC cluster count is off. |

### One-sentence carryover
> *"We change only the metric and add the filter. Everything else stays bit-identical
> so Phase 1's Sharpe (0.59) and Phase 2's Sharpes are directly comparable."*

---

## Item 4 — Crisis-regime dependency persists

### The Phase 1 evidence

| Regime | n trades | % trades | P&L | **% P&L** | P&L per trade |
|---|---:|---:|---:|---:|---:|
| 2003–06 pre-crisis | 1,987 | 16% | +4.15 | 13.6% | +21 bps |
| **2007–09 GFC** | 1,748 | 14% | +11.94 | **39.3%** | **+68 bps** |
| 2010–19 expansion (calm) | 5,578 | **46%** | +5.50 | **18.1%** | +10 bps |
| 2020–21 COVID | 1,389 | 11% | +6.58 | **21.7%** | **+47 bps** |
| 2022–23 inflation/SVB | 1,553 | 13% | +2.23 | 7.3% | +14 bps |

**Crisis windows (GFC + COVID) = 25% of trades but 61% of total P&L.**

### Why it's structural
Pairs trading earns from mean reversion. Mean reversion needs spread divergence.
Spreads diverge more during crises (forced selling, correlation breaks, idiosyncratic
news cycles). In calm bull markets, correlations rise, spreads tighten, z-scores
rarely breach ±2σ. **The strategy is doing what its physics says.**

### What Phase 2 *can* do (modestly)
- Cointegration filter removes pairs that quietly drift in calm periods (population-2
  losers in expansion years). Adds small positive P&L to 2010-19.
- PC distance finds pairs that revert on *idiosyncratic* moves (exist even in calm
  regimes). Adds modest P&L to all regimes.
- Combined: maybe 2010-19 → 25% of P&L, crisis-window concentration drops to ~50%.

### What Phase 2 *cannot* do
- Eliminate crisis dependency. Spreads don't diverge in genuinely placid markets.

### Implications for writeup & forward test
1. **Report Sharpe by regime, not just headline.** "Sharpe 1.0 over 2003–2023" can
   hide that it's "Sharpe 1.8 during crises, 0.4 in calm."
2. **Forward test (Phase 4 Alpaca paper trade) will probably underperform.** Running
   in 2026 (calm period). If live Sharpe is 0.3–0.6, that's *consistent*, not failure.
3. **Honest claim:** "This is a dislocation-harvester — it shines when markets are
   stressed." Defensible and economically interesting.
4. **Filter is partly about Sharpe stability across regimes**, not just headline number.

### Verification check in Phase 2
Replicate the regime table for PC + filter. Compare row-by-row:

| Regime | Phase 1 SSD | Phase 2 PC+filter | Δ |
|---|---|---|---|
| 2003–06 | TBD | TBD | TBD |
| 2007–09 GFC | TBD | TBD | TBD |
| 2010–19 expansion | TBD | TBD | TBD |
| 2020–21 COVID | TBD | TBD | TBD |
| 2022–23 inflation | TBD | TBD | TBD |

Equal lift across rows = broad improvement. Only crises lift = strategy is more
"crisis harvester" than we hoped.

### One-sentence carryover
> *"Phase 2 lifts the headline Sharpe but won't eliminate regime dependency. Report
> by regime; set forward-test expectations."*

---

## Item 5 — Methodology discipline carryovers

### Group A — Building discipline

**A1. Synthetic-tests-first.** Before any real-data run, write a unit test with
planted ground truth. For Phase 2:

| New function | What to plant | What to check |
|---|---|---|
| `pc_distance(returns, market_returns)` | Two stocks with planted residual ρ | Distance ≈ 1 − ρ |
| `engle_granger(prices_a, prices_b)` | Cointegrated pair vs random-walk pair | ADF p < 0.05 on first, > 0.05 on second |
| `half_life_ar1(spread)` | Planted AR(1) with known reversion speed | Recovered half-life within ±10% |

**A2. Single-variable change.** Run 2×2 grid (SSD/PC × filter on/off). Each cell
isolates one change. Never combine changes — attribution becomes impossible.

### Group B — Tuning discipline

**B1. Cross-date validation.** For every Phase 2 hyperparameter, validate on Dec 2010,
Dec 2015, Dec 2023. Don't tune to one date.

**B2. Lock once chosen — no Sharpe-driven re-tuning.** Once a hyperparameter passes
cross-date validation, lock it in `src/config.py` with a "DO NOT re-tune" comment.
Same as `xi=0.10` in Phase 1.

### Group C — Reporting discipline

**C1. Same phase-folder structure as Phase 1.**

```
phases/phase2/
├── README.md
├── decisions.md
├── notebooks/
│   ├── phase2_complete_reference.ipynb     ← concept + worked example + real data + CP2
│   ├── phase2_pnl_attribution.ipynb        ← did the lever work?
│   └── 0X_*.py demo scripts
└── results/
    ├── ssd_core_monthly.parquet            ← Phase 1's archived, copied here
    ├── ssd_filtered_monthly.parquet        ← Phase 2 new
    ├── pc_core_monthly.parquet             ← Phase 2 new
    └── pc_filtered_monthly.parquet         ← Phase 2 new
```

**C2. Match-then-critique.** Replicate paper PC numbers within tolerance. Document
gaps honestly. Don't tweak design to "beat" the paper.

**C3. Attribution alongside backtest, not after.** Build `phase2_pnl_attribution.ipynb`
*as part of the deliverable*. Check:
- Did force-close drag actually shrink?
- Did bimodal pattern soften (more reversions, fewer force-closes)?
- Did regime-by-regime Sharpe become more balanced?

### Group D — Process discipline

**D1. Document decisions immediately.** When resolving D2.1–D2.5, write the
resolution in `decisions.md` *at the time*, not after.

**D2. Use existing infrastructure.** Extend `backtest.py` with new args; don't
rewrite. The `find_project_root()` helper means notebooks work from any depth.

### One-sentence carryover
> *"Build synthetic tests before real-data runs, change one thing at a time, lock
> hyperparameters once chosen, run attribution alongside backtest, never tweak to
> 'beat' the paper."*

---

## Item 6 — Open Phase 2 decisions

Full detail in `decisions.md`. Recap with my recommended defaults:

| ID | Decision | Recommended default | Resolve when |
|---|---|---|---|
| **D2.1** | Market adjustment: SPY-only or FF 3-factor? | SPY-only (matches paper §3) | Before building `pc_distance` |
| **D2.2** | Filter as gate or comparison cell? | Comparison cell (per reconciliation #4) | Already decided ✅ |
| **D2.3** | Half-life bounds | [5, 60] days (paper default) | Before building filter loop |
| **D2.4** | Half-life estimation method | AR(1) discrete-time (simple, equivalent to OU) | Before building `half_life_ar1` |
| **D2.5** | EG test direction (A on B or B on A?) | Lower p-value of both directions | Before writing filter loop |

### Bonus open question
**D2.6 — How to handle a pair appearing in multiple result sets?** For attribution,
join on `pair_key` and compare per-pair P&L across the four backtests. (Addressed
in attribution notebook design, not a config decision.)

### One-sentence carryover
> *"Resolve each open decision in `decisions.md` at the time of build, not retroactively."*

---

## Phase 2 starting checklist

- [ ] Re-read this carryover doc (you are here)
- [ ] Decide D2.1, D2.3, D2.4, D2.5 (5 minutes — defaults are recommended)
- [ ] Build `src/distances.py::pc_distance` + unit test
- [ ] Build `src/cointegration.py::engle_granger` + `half_life_ar1` + unit tests
- [ ] Extend `src/backtest.py` with `metric` and `cointegration_filter` args
- [ ] Run 4 backtests (SSD/PC × filter on/off); save parquets to `phases/phase2/results/`
- [ ] Build `phases/phase2/notebooks/phase2_complete_reference.ipynb`
- [ ] Build `phases/phase2/notebooks/phase2_pnl_attribution.ipynb`
- [ ] Update `phases/phase2/README.md` and `decisions.md` with results
- [ ] Check CP2 verdict (paper PC Sharpe 1.01 ±0.15)
- [ ] Bimodal-pattern check: did the force-close drag shrink?

When the checklist is complete, Phase 2 is done.
