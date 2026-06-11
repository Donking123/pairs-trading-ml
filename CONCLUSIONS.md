# Project Conclusions — QF621 Pairs Trading (ML Clustering)

*Synthesis as of 2026-06-07. Full evidence in `phases/*/` (per-phase READMEs, decisions, and
executed reference notebooks) and the writeup `phases/phase4/QF621_writeup.md`.*

---

## Headline scoreboard

| Stage | PC | Factor-beta |
|---|---:|---:|
| In-sample, frictionless (2003–2023) | **1.028** | **1.013** |
| Robustness band (across perturbations) | 0.485–1.046 | 0.615–1.060 |
| Net of realistic costs | 0.572 | 0.578 |
| Net + passive execution (no stop) | ~0.78 | ~0.77 |
| Lookahead-bias audit | PASS (6/6) | PASS (6/6) |
| **Out-of-sample (2024–2025, frozen)** | **0.858** | **0.117** |

---

## 1. The paper replicates, and the result is real
PC core reproduces the paper's headline (1.028 vs 1.01). The **factor-beta extension**, a
structurally different similarity metric (shared *risk exposures* rather than return
co-movement), independently reaches **1.013** — strong evidence the ~1.0 Sharpe is genuine and
not an artefact of one specific metric.

## 2. What drives the edge
- **A dislocation harvester:** ~9–11% of trades cleanly revert (~+400 bps each); ~90% force-close
  at month-end for tiny losses. Profit = winners outweighing the churn.
- **Regime-dependent:** 30–40% of in-sample P&L came from the 2007–09 crisis. It feeds on
  volatility and cross-sectional dispersion.
- **The metric's job** is to shrink the force-close drag — PC/factor cut it to ~−12 bps/trade vs
  SSD's −32, which is why they beat SSD (0.589).

## 3. What's robust vs fragile
- **Robust** to the hedge-ratio estimator (OLS vs robust RLM) and position sizing (equal vs
  z-weighted) — Sharpe ≈ unchanged.
- **Fragile** to the **clustering algorithm**, via **selectivity**: HDBSCAN/hierarchical form
  looser clusters → ~3× more, diluted pairs → lower Sharpe. OPTICS's conservatism is load-bearing.
- Factor-beta is the **more robust** metric in-sample (survives 3 of 4 perturbations near ~1.0 vs
  PC's 2 of 4).

## 4. Backtest vs deployable reality
- Realistic frictions (actual CRSP bid/ask + 35 bps borrow + 3.5σ stop) **roughly halve** the
  Sharpe to ~0.57 — the frictionless 1.0 overstates capturable performance.
- Sensible execution recovers most of it: **passive (limit-order) execution is the #1 lever
  (+~0.20)** and **dropping the stop is #2 (+0.11–0.14, the stop is net-negative)** → ~0.78.
- The cointegration filter and higher entry thresholds **hurt** net of costs (shed more alpha than
  the turnover they save).

## 5. Integrity & generalisation (the most important part)
- **No lookahead bias** — 6/6 black-box audit pass; the look-ahead protections hold.
- **Out-of-sample:** on frozen 2024–2025 data, **PC genuinely generalises (0.858 ≈ in-sample
  1.028)**, but **factor-beta does not (0.117)** — despite being the sturdier metric in-sample.
  Both were weak in calm, trending 2024 and recovered in 2025 (regime dependence again).

---

## Bottom line

The clustering-based pairs strategy is **real, replicable, metric-corroborated, bias-free, and it
generalises out-of-sample for PC — but it is a conditional, regime-dependent edge, not all-weather
alpha.** Stripped of frictionless/in-sample flattery, the honest deployable picture is a **~0.5–0.8
Sharpe that depends on market dispersion** — strongest in turbulent periods, weak in calm trending
ones.

**Three lessons worth keeping:**
1. **Honest validation changes the story.** The 1.0 in-sample figure is true but misleading; costs,
   regime, and out-of-sample testing cut it to a realistic 0.5–0.8.
2. **In-sample robustness ≠ out-of-sample reliability** — factor-beta was the sturdiest in-sample
   yet the weakest OOS.
3. **Short windows lie.** 2024 alone screamed "failure" (PC 0.16); the fuller 2024–25 window showed
   PC actually holds (0.858). Read generalisation over a sufficient sample.

**If deploying:** PC metric + OPTICS clustering + passive execution + no stop + a **regime/dispersion
filter** (trade only when dispersion/volatility is elevated). Realistic expectation: a modest,
largely uncorrelated, dislocation-driven return stream — valuable as a diversifier, not a
standalone money machine.

---

## Phase 6 update (2026-06-10) — implementation-correctness review

A line-by-line code review found six implementation flaws (no lookahead bias — the 6/6
audit stands). Each was fixed behind an engine flag and re-run one at a time
(`phases/phase6/`). What changes in the story above:

1. **The cointegration filter was never working — its viability was a statistical
   artifact.** The Engle-Granger step used raw-ADF p-values, which are anti-conservative
   on estimated residuals (effective threshold ~12–15%, not 5%). With the correct
   MacKinnon distribution the filter keeps 11% instead of 26% of candidates and its cell
   collapses to **0.299**. Trade-level decomposition shows why: pairs with strong EG
   evidence earned **less** per trade (+12.1 bps) than weak-evidence pairs (+19.3 bps),
   equal hit rates — the test selects nothing; it only shrinks diversification. The
   dose-response (no filter 1.028 → weak filter 0.752 → correct filter 0.299) is the
   clearest evidence yet that the edge is short-horizon dislocation reversion, not
   long-run equilibrium reversion. (Our raw-ADF cell matching the paper's target
   suggests the paper used the same miscalibrated test.) §4's "the filter hurts net of
   costs" upgrades to: **the filter hurts, period.**
2. **"The stop is net-negative" survives a fairness test.** The stop had a churn bug
   (stop-out → same-position re-entry next day). With a cooldown fix the churn is gone
   (−18% trades) but Sharpe is unchanged (0.572 → 0.566) — dropping the stop remains
   correct, now on clean evidence.
3. **The headline is robust to the delisting corrections** (code-map fix + compounding
   + date snapping): pc_core 1.028 → 1.007. The OOS 2024–25 numbers should be re-run
   with `delisting_fix=True` (the missing-dlret sentinel bug sits in the v2 pull).
4. **Error bars (block bootstrap) revise two §5 claims.** OOS PC 0.858 has 95% CI
   [0.18, 1.63] — *consistent with* in-sample 1.028 (P(OOS≥IS)=33%): the generalisation
   claim strengthens. But OOS factor 0.117 has CI [−0.52, +1.12]: 23 months cannot
   distinguish "fails" from "≈1.0", so "factor-beta does not generalise" must soften to
   "no OOS evidence either way yet" (lesson 3 cuts both ways).
5. **The deployment advice's dispersion filter is half right.** Contemporaneous
   dispersion quartiles are cleanly monotone (Sharpe 0.73→1.56): mechanism confirmed.
   But the deployable prior-month signal is non-monotone (1.02/1.32/0.54/1.51) — only
   the extreme top quartile is reliable. "Trade only when dispersion is elevated"
   should read "**scale up** when dispersion is extreme; stay invested otherwise."
6. **The edge survives honest fill timing.** Fills delayed to the close AFTER the
   signal (`execution_delay=1`) cut pc_core 1.028 → **0.882** — a real, statistically
   significant haircut (paired bootstrap Δ = 0.145, 95% CI [0.02, 0.31]) but the
   strategy stays clearly profitable (CI [0.45, 1.26], P(SR≤0) = 0.0%), with lower vol
   and a shallower max drawdown. ~86% of the frictionless headline is genuine signal;
   ~14% was the fill-at-signal-close convention. Mirrors Gatev et al.'s
   one-day-waiting finding.

*(Also: with the risk-free rate subtracted — the engine reports rf=0 Sharpe by paper
convention — pc_core's excess-return Sharpe is 0.615, and the realism baseline's is
0.127. Worth stating alongside the headline figures.)*
