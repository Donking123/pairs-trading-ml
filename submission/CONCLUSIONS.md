# Project Conclusions — QF621 Pairs Trading (ML Clustering)

*Final synthesis. Full evidence in `notebooks/` (per-phase reference notebooks) and
the presentation decks in `report/`.*

---

## Headline scoreboard

| Stage | PC | Factor-beta |
|---|---:|---:|
| In-sample, frictionless (2003–2020) | **1.113** | **1.149** |
| In-sample, filtered (2003–2020) | **0.788** | **0.969** |
| Net of realistic costs | ~0.57 | ~0.58 |
| Lookahead-bias audit | PASS (6/6) | PASS (6/6) |
| **Out-of-sample (2021–2025, frozen)** | **0.412 (core) / 0.461 (filtered)** | **-0.103 (core) / 0.131 (filtered)** |

---

## 1. The paper replicates, and the result is real
PC core reproduces the paper's headline on the 2003–2020 in-sample window (1.113 vs
paper's 1.01 on 2003–2023). The **factor-beta extension**, a structurally different
similarity metric (shared *risk exposures* rather than return co-movement),
reaches **1.149** — actually *beating* PC core and providing strong evidence the ~1.1
Sharpe is genuine and not an artefact of one specific metric.

## 2. What drives the edge
- **A dislocation harvester:** ~9–11% of trades cleanly revert (~+400 bps each); ~90%
  force-close at month-end for tiny losses. Profit = winners outweighing the churn.
- **Regime-dependent:** 30–40% of in-sample P&L came from the 2007–09 crisis. It feeds on
  volatility and cross-sectional dispersion.
- **The metric's job** is to shrink the force-close drag — PC/factor cut it to ~-12
  bps/trade vs SSD's -32, which is why they beat SSD (0.645).

## 3. What's robust vs fragile
- **Robust** to the hedge-ratio estimator (OLS vs robust RLM) and position sizing (equal
  vs z-weighted) — Sharpe approximately unchanged.
- **Fragile** to the **clustering algorithm**, via **selectivity**: HDBSCAN/hierarchical
  form looser clusters leading to ~3x more, diluted pairs and lower Sharpe. OPTICS's
  conservatism is load-bearing.
- Factor-beta is the **more robust** metric in-sample (survives 3 of 4 perturbations near
  ~1.0 vs PC's 2 of 4).

## 4. Backtest vs deployable reality
- Realistic frictions (actual CRSP bid/ask + 35 bps borrow + 3.5-sigma stop) **roughly
  halve** the Sharpe to ~0.57 — the frictionless 1.1 overstates capturable performance.
- Sensible execution recovers most of it: **passive (limit-order) execution is the #1
  lever (+~0.20)** and **dropping the stop is #2 (+0.11–0.14, the stop is
  net-negative)** leading to ~0.78.
- The cointegration filter and higher entry thresholds **hurt** net of costs (shed more
  alpha than the turnover they save).

## 5. Integrity & generalisation (the most important part)
- **No lookahead bias** — 6/6 black-box audit pass; the look-ahead protections hold.
- **Out-of-sample (2021–2025, 59 months):** PC + filter is the **best OOS strategy
  (0.461)** — the filter actually helps PC in OOS by removing pairs that drift apart.
  All three filtered strategies stay positive (SSD 0.224, factor 0.131). Factor-beta
  core turns negative (**-0.103**), but the cointegration filter rescues it to
  **0.131**. The filter's OOS benefit is universal: it removes drifting pairs that
  hurt during the trading window.
- The decay is consistent with the strategy's regime dependence: 2021–2023 was a calm,
  low-dispersion period (bad for dislocation-based strategies).
- The OOS result is honest: the strategy works, but not in all market regimes.

## 6. Implementation-correctness review
A line-by-line code review found six implementation flaws (no lookahead bias — the 6/6
audit stands). Key findings:
1. **The cointegration filter was never working** — raw-ADF p-values are
   anti-conservative on estimated residuals. With correct MacKinnon p-values, the filter
   collapses to Sharpe 0.299. The edge is short-horizon dislocation reversion, not
   long-run equilibrium reversion.
2. **The stop is net-negative** — survives a fairness test after fixing a stop-churn bug.
3. **Delisting corrections** are robust — headline barely changes (1.028 to 1.007 on the
   full window).
4. **Honest fill timing** — execution delay cuts ~14% of the headline; ~86% is genuine
   signal.

---

## Bottom line

The clustering-based pairs strategy is **real, replicable, metric-corroborated, bias-free,
and it stays profitable out-of-sample — but with significant decay.** It is a conditional,
regime-dependent edge, not all-weather alpha. Stripped of frictionless/in-sample flattery,
the honest picture is a **~0.4–0.8 Sharpe that depends on market dispersion** — strongest
in turbulent periods, weak in calm trending ones.

**Three lessons worth keeping:**
1. **Honest validation changes the story.** The 1.1 in-sample figure is true but
   misleading; costs, regime, and out-of-sample testing cut it to a realistic 0.4–0.8.
2. **In-sample robustness does not equal out-of-sample reliability** — factor-beta was the
   sturdiest in-sample yet the weakest OOS.
3. **Regime matters more than method.** The strategy's OOS decay tracks the market regime
   (low dispersion, momentum-driven), not a flaw in the methodology.

**If deploying:** PC metric + OPTICS clustering + passive execution + no stop + a
**regime/dispersion filter** (scale up when dispersion is extreme; stay invested
otherwise). Realistic expectation: a modest, largely uncorrelated, dislocation-driven
return stream — valuable as a diversifier, not a standalone money machine.
