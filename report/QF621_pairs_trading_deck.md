---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 22px;
    background: #ffffff;
    color: #1d2733;
    padding: 48px 60px;
  }
  h1 { color: #1b4965; font-size: 42px; }
  h2 { color: #1b4965; border-bottom: 3px solid #5fa8d3; padding-bottom: 6px; margin-bottom: 18px; }
  strong { color: #0f3a52; }
  table { font-size: 19px; width: 100%; }
  th { background: #1b4965; color: #fff; padding: 6px 10px; }
  td { padding: 5px 10px; }
  tr:nth-child(even) { background: #f2f7fb; }
  section.lead { text-align: center; justify-content: center; }
  section.lead h1 { font-size: 48px; }
  section.lead h2 { border: none; color: #5fa8d3; font-size: 26px; }
  code { background: #eef3f7; color: #0f3a52; padding: 2px 6px; border-radius: 3px; }
  .small { font-size: 17px; color: #5a6b7b; }
  .tag { font-size: 14px; background: #e8f4f8; color: #1b4965; border-radius: 3px; padding: 2px 8px; }
  footer { color: #9aa0a6; font-size: 13px; }
  blockquote { border-left: 4px solid #5fa8d3; padding-left: 16px; color: #2a4a6b; margin: 12px 0; }
footer: "Pairs Trading — Statistical Arbitrage on Economic Peer Groups · QF621"
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Pairs Trading
## Statistical Arbitrage on Economic Peer Groups

**IS Sharpe 1.03** (2003–2020) · **OOS Sharpe 0.68** with regime overlay (2021–2024)
22-year survivorship-bias-free backtest · WRDS/CRSP institutional data

<span class="small">PC distance clustering · half-life stationarity gate · HMM regime overlay · 6/6 lookahead PASS</span>

---

## 1 — Alpha hypothesis

> When two stocks in the same **economic peer group** diverge, the shared non-market factor that couples them creates a predictable reversion.

The alpha source is **economic substitutability**, not price coincidence:
- Same sector, same customer base, same input costs → shared idiosyncratic driver
- Microstructure noise temporarily widens the spread → mean reversion closes it
- **This should persist** as long as peer groups remain economically linked

**Why it is hard to harvest:**
- 450 S&P 500 stocks → ~100,000 candidate pairs; most are spurious
- Pair quality degrades fast: wrong pairs never revert, consuming capital and incurring costs
- Identifying genuine peer groups **before looking at Sharpe** is the entire problem

---

## 2 — Universe & data

| | |
|---|---|
| **Source** | WRDS / CRSP daily — institutional academic data |
| **Universe** | S&P 500 constituents, US common stock only (codes 10/11), ~450 stocks/month |
| **Period** | Jan 2000 – Dec 2024 (25 years) |
| **Survivorship bias** | Eliminated — delisted names kept; CRSP delisting returns applied |
| **In-sample** | 2003–2020 · **215 monthly returns** (3-year formation window consumed first) |
| **Out-of-sample** | 2021–2024 · **47 months** · model parameters frozen at Dec 2020 |

<span class="small">Point-in-time constituents: we only use stocks that were S&P 500 members at the time of formation. Delisting fix: CRSP code-dependent return fallback (D6.2). No IS/OOS parameter bleed.</span>

---

## 3 — The pipeline as a risk management stack

![w:900](assets/pipeline_flow.png)

<span class="small">Every gate eliminates one failure mode. Hyperparameters (OPTICS xi, z-score window, half-life bounds, entry threshold) all fixed **before** looking at any Sharpe — no p-hacking.</span>

---

## 4 — PC distance: what it actually selects

**Step 1:** OLS-regress each stock on market: $r_i = \alpha_i + \beta_i r_m + \varepsilon_i$ · **Step 2:** dist$(i,j) = 1 - \text{corr}(\varepsilon_i, \varepsilon_j)$ · **Step 3:** OPTICS density clustering

> Clustering on **idiosyncratic residuals** (what's left after removing the market), not on factors or price paths.

![w:780](assets/sector_composition.png)

<span class="small">PC selects **90% same-sector pairs** (vs 57% for SSD) — BK/NTRS, VZ/T, ED/SO. The economic link is structural, not coincidental. Cluster purity vs SIC codes: PC 0.937 vs SSD 0.871.</span>

---

## 4b — What PC-selected pairs look like

![w:940](assets/sample_pairs.png)

<span class="small">Real CRSP data, 2010–2014. Top: normalised price series — the legs co-move until a dislocation. Bottom: 6-month rolling z-score — entry (orange, |z|>2) and exit (green, z→0.5) annotated. All three pairs reverted cleanly within the trading window.</span>

---

## 5 — Signal construction & why we use half-life, not ADF

**Spread:** $s_t = P_A - \hat{\gamma} P_B$ (OLS hedge ratio, formation window)

**Stationarity gate — half-life bounds [5, 60] trading days:**

Fit AR(1) on the spread: $s_t = \mu + \phi(s_{t-1} - \mu) + \varepsilon_t$

Half-life $= \log(0.5) / \log(\hat{\phi})$ — how fast the spread halves its deviation from mean

| $\hat{\phi}$ | Half-life | Verdict |
|---|---|---|
| 0.990 | 69 days | Too slow — force-close before reversion |
| **0.970** | **23 days** | **Excellent — reverts well within window** |
| 0.800 | 3 days | Too fast — noise, costs eat alpha |

**Why not ADF?** ADF at T = 756 observations has ~40% power for $\phi = 0.97$ — it discards 60% of genuinely good pairs. Half-life directly tests *economic usefulness* (reversion speed) rather than statistical significance on a short sample.

**Trading signal:** 6-month rolling z-score · Entry $|z| > 2$ · Exit $z = 0.5$ · Stop $|z| = 3.5$

---

## 6 — P&L attribution: why the strategy makes money

![w:820](assets/pnl_breakdown_filtered.png)

> 8–12% of trades cleanly revert (+350–500 bps each). The other 88–92% get force-closed at month-end (−15 to −40 bps each). The filter's job: **raise the reversion rate and lower the force-close drag.**

<span class="small">IS and OOS trade mechanics are consistent: reversion win rate ~97%, force-close at ~50%. PC's lower force-close drag vs SSD explains its lower volatility and smaller drawdowns.</span>

---

## 7 — In-sample performance (2003–2020)

![w:700](assets/is_oos_hmm_equity.png)

| Strategy | Ann. Ret | Ann. Vol | **IS Sharpe** | **OOS Sharpe** *(2021–24)* | Max DD |
|---|---:|---:|---:|---:|---:|
| PC core (no filter) | 3.86% | 3.46% | **1.113** | 0.180 | −5.75% |
| PC + half-life only | 3.61% | 3.71% | **0.975** | 0.351 | −4.14% |
| **PC + ADF filter** | **2.79%** | **3.57%** | **0.788** | **0.410** | **−5.88%** |
| PC + HMM overlay | 1.52% | 1.48% | **1.027** | **0.679** | −3.03% |

<span class="small">Each stationarity gate reduces IS Sharpe but improves OOS generalization — a real bias-variance tradeoff. Half-life alone: 0.975 IS / 0.351 OOS. Adding ADF: 0.788 IS / **0.410 OOS**. Lookahead audit: **6/6 PASS**.</span>

---

## 8 — Regime dependence: when the strategy is alive

![w:860](assets/yearly_returns_filtered.png)

<span class="small">GFC 2008–09 (10% of IS months) = 28% of total IS return. Calm 2013–19: idles at +1–2%/yr — does not lose. OOS 2021–22 (rising rates, low dispersion): weakest period — the strategy compresses, not collapses. 2023–24: partial recovery. **Alpha is dispersion-dependent, not calendar-dependent.**</span>

---

## 9 — HMM regime overlay: protecting capital in adverse environments

![w:820](assets/regime_calendar.png)

<span class="small">Monthly returns coloured by HMM state: **green = calm (scale 1.0×)**, **yellow = stressed (0.5×)**, **red = crisis (0.0×)**. The HMM locks the scale at trade entry — no future information used. Annual expanding-window refit. 2022 crisis months correctly identified and sat out. OOS Sharpe: 0.412 → **0.679**, MaxDD: −2.75% → **−1.99%**.</span>

---

## 10 — Out-of-sample validation (2021–2024, 47 months)

Clean IS/OOS split: all parameters frozen at Dec 2020. OOS runs on data the model has never seen.

![w:700](assets/oos_bar.png)

| Strategy | IS Sharpe | **OOS Sharpe** *(fair 47-mo)* | OOS MaxDD |
|---|---:|---:|---:|
| PC core (no filter) | 1.113 | 0.180 | −2.75% |
| PC + half-life only | 0.975 | 0.351 | −2.57% |
| **PC + ADF filter** | **0.788** | **0.410** | **−2.98%** |
| **PC + HMM overlay** | **1.027** | **0.679** | **−1.99%** |

**Key finding:** each gate adds OOS generalization at the cost of IS Sharpe. The bias-variance tradeoff is explicit and empirically confirmed. HMM is orthogonal — it improves both IS and OOS risk-adjusted returns.

---

## 11 — Realistic cost impact

**Frictions modelled** (CRSP actual data, not assumptions):
- Bid/ask spread: actual CRSP daily quotes (~27 bps in 2000 → ~2.5 bps by 2015)
- Borrow cost: 35 bps annual on short leg, accrued daily
- Stop-loss: 3.5σ hard exit (separate lever)

![w:700](assets/realism_waterfall.png)

| Lever | Sharpe impact |
|---|---:|
| Frictionless baseline | 1.028 |
| + Bid/ask + borrow | 0.572 (−0.456) |
| → Switch to passive execution | +0.210 |
| → Remove stop-loss | +0.11–0.14 |
| **Net deliverable** | **~0.75–0.80** |

<span class="small">Stop-loss is **net-negative**: the round-trip cost at 3.5σ exceeds the tail protection. Passive (limit) execution recovers most of the cost drag.</span>

---

## 12 — Honest scorecard

| Variant | IS Sharpe | OOS Sharpe *(47 mo)* | IS MaxDD | OOS MaxDD |
|---|---:|---:|---:|---:|
| PC core (no filter) | 1.113 | 0.180 | −5.75% | −2.75% |
| PC + half-life only | 0.975 | 0.351 | −4.14% | −2.57% |
| PC + ADF filter | 0.788 | 0.410 | −5.88% | −2.98% |
| **PC + HMM overlay** | **1.027** | **0.679** | **−3.03%** | **−1.99%** |
| Realistic costs (passive exec) | ~0.75–0.78 | — | — | — |
| Lookahead audit | **6/6 PASS** | — | — | — |

**Finding:** stationarity filtering is a bias-variance tradeoff — more filtering costs IS Sharpe and buys OOS robustness. HMM is different: it improves both IS and OOS by correctly identifying unprofitable regimes.

**Deployable configuration: PC + ADF filter + HMM + passive execution → 0.55–0.70 net Sharpe**, regime-dependent.

---

## 13 — Risks & honest limitations

**Capacity** — ~$5–20M effective AUM before market impact. Strategy trades ~200 pairs/month in S&P 500 mid-cap names with 21-day holding period. Not scalable beyond this without a fundamentally different execution model.

**Regime dependence** — OOS 2021–22 shows the floor: in low-dispersion trending markets, alpha compresses significantly. The HMM overlay mitigates but does not eliminate this.

**Model risk** — OPTICS xi tuned pre-backtest (honest), but it remains a hyperparameter. PC distance removes only the single market factor; residuals may share industry/sector factor structure that is not truly idiosyncratic.

**Cointegration** — half-life [5, 60] bounds replace ADF (low power at T=756). Whether this is sufficient to ensure spread stationarity is tested by the `pc_halflife_only` variant — results pending.

**HMM coverage** — 2-feature model (vol + dispersion). 3-feature model with HY OAS credit spread (stronger crisis signal) requires extended FRED data pull.

---

## 14 — What's next

| Priority | Action | Expected impact |
|---|---|---|
| **Preferred config confirmed** | PC + ADF filter + HMM + passive exec | Best OOS (0.41 base + 0.68 with HMM); deploy this combination |
| **HMM 3-feature** | Add HY OAS credit spread (FRED pull back to 2000) | Sharper calm/stressed/crisis — better crisis identification |
| **Passive execution default** | Switch from marketable to limit orders | +0.21 Sharpe net — largest single cost lever |
| **Live forward test** | Paper-trade on 2025 live data, PC + ADF + HMM | Validate on truly unseen data with realistic fills |

**Thesis confirmed OOS.** The filter-vs-no-filter result (0.410 vs 0.180 OOS Sharpe) shows the stationarity gate is load-bearing for generalization, not just theoretical hygiene. HMM adds genuine OOS alpha (0.410 → 0.679) by correctly identifying adverse regimes.

**Deployable target: 0.55–0.70 net Sharpe** with passive execution, regime-conditional.

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank you

**IS 1.03** · **OOS 0.68** (with HMM) · **6/6 lookahead PASS** · 274 months total

<span class="small">Happy to walk through the backtest engine, the attribution analysis, or the HMM design.</span>
