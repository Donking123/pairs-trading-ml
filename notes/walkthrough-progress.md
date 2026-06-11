# Phase-by-Phase Walkthrough Progress

**Last updated:** 2026-06-10
**Status:** Completed through Phase 4. Phase 5 still to cover.

---

## Covered

### Phase 0 — Data Spine
- 5 parquet datasets from WRDS (constituents, CRSP daily, delisting, S&P 500 index, FF factors)
- Universe: 991 survivorship-bias-free US common stocks, 2000–2023
- Realism baked into the data: survivorship bias, bid/ask costs, delisting returns
- config.py locks all constants before any backtest

### Phase 1 — SSD Vertical Slice
- Full pipeline walkthrough: panel → SSD distance → OPTICS clustering → spread/z-score → backtest → performance
- **Formation panel:** 3-year lookback, survivorship + continuously-priced filters, total-return prices
- **SSD distance:** z-normalize each stock, squared Euclidean distance between standardized price vectors
- **OPTICS clustering:** density-based, xi=0.10, 47 clusters / 0.871 purity — matched paper
- **Purity calculation:** for each cluster, count stocks in the majority SIC sector, sum across clusters, divide by total clustered stocks
- **Spread:** OLS hedge ratio (γ = Cov(A,B)/Var(B)), frozen for the trading month
- **Z-score:** rolling 126-day (6 months × 21 days) lookback with .shift(1) lookahead protection
- **Z-score window:** rolls forward day by day through the trading month (not frozen at formation end)
- **Entry/exit:** |z| ≥ 2.0 to enter, z crosses 0 to exit (reversion), force-close at month-end
- **No stop-loss** in core variant
- **Position sizing:** equal-dollar $0.50 long + $0.50 short, daily return = position × 0.5 × (ret_a - ret_b)
- **P&L accumulation:** per-pair daily P&L → equal-weight mean across open pairs → compound daily into monthly → Sharpe from monthly series
- **Results:** Sharpe 0.589 (paper 0.88) — clustering matched, return numerator short
- **Bimodal finding:** 11.4% reversion trades (+471 bps) generate all profit; 88.4% force-close (-32 bps) drag
- **5 suspected causes of gap:** equal-weight dilution, z-score window rolling vs frozen, position sizing, t+1 execution model — all discussed

### Phase 2 — PC Distance + Cointegration Filter
- **PC (Partial Correlation) distance:** market-adjust each stock's returns via OLS, correlate the residuals, distance = 1 - corr(residuals). Finds idiosyncratic co-movement, not just shared market beta.
- **Cointegration filter:** Engle-Granger ADF test (p < 0.05) + AR(1) half-life in [5, 60] days. Tests both A→B and B→A directions, takes lower p-value (literature convention, small acceptance bias documented).
- **Half-life:** AR(1) discrete-time, equivalent to Ornstein-Uhlenbeck continuous-time under daily sampling. half_life = -ln(2) / ln(ρ).
- **PC filter pass rate:** lower than SSD because PC clusters on residual return correlation (weaker condition than cointegration) — many PC pairs are correlated in residuals but price spreads still drift.
- **Can use PC without filter:** PC core (1.028) outperforms PC + filter (0.752). Filter adds safety (zero outlier trades) but removes profitable pairs.
- **2×2 grid results:** PC core 1.028 (paper 1.01 ✅), PC+filter 0.752 (0.80 ✅), SSD+filter 0.731 (0.75 ✅), SSD core 0.589 (unchanged ✅)
- **Why PC worked:** force-close drag cut by 65% per trade, exactly as Phase 1 attribution predicted

### Phase 2.5 — Factor-Beta Clustering (Original Extension)
- **Idea:** cluster stocks by risk-factor exposure — stocks loading similarly on the same factors
- **18-factor panel:** 6 FF style factors (mktrf, smb, hml, rmw, cma, umd) + 12 FF industry factors (equal-weight return per industry from the universe)
- **Ridge regression:** regress each stock's returns on 18 factors → 18-dimensional beta vector per stock. Ridge (α=1.0) stabilizes collinear betas. Solved in closed form: B = (F'F + αI)⁻¹ F'R
- **Distance:** standardize each beta dimension (z-score across stocks), then Euclidean distance between standardized beta vectors
- **Results:** factor core 1.013, factor+filter 0.858. Comparable to PC in-sample, filter helps factor more than PC.

### Phase 3 — Robustness Testing
- **8-cell grid:** {PC, factor} × {HDBSCAN, hierarchical, RLM hedge, z-weight}
- **HDBSCAN:** density-based, auto cluster count via stability — more inclusive, ~3× more pairs
- **Hierarchical:** agglomerative average-linkage, cut at 1st percentile of pairwise distances (quantile, not fixed height, to handle drifting distance scales)
- **RLM hedge ratio:** Huber's T robust regression, downweights outlier days
- **Z-weight:** weight each open pair by |entry_z|, bet more on deeper dislocations
- **Key finding:** clustering algorithm is load-bearing (HDBSCAN/hierarchical dilute performance). Factor-beta is sturdier — holds 0.991 under hierarchical while PC breaks to 0.485. RLM and z-weight are secondary (±0.02-0.05).
- **Robustness bands:** PC 0.485–1.046, Factor 0.615–1.060

### Phase 4 — Realism, Lookahead, Out-of-Sample
- **4a Transaction costs:** actual CRSP bid/ask quotes (time-varying: 27 bps→2.5 bps), 35 bps/yr borrow on short leg, 3.5σ stop-loss. Two presets: REALISM_FULL (marketable), REALISM_PASSIVE (limit orders).
- **Results:** full realism Sharpe ~0.572, passive+no-stop ~0.78. Stop-loss hurts — removing it improves Sharpe. Passive execution saves ~0.20 Sharpe vs marketable.
- **4b Lookahead test:** black-box test — run full (2003–2023) vs truncated (2003–cut date), compare daily position vectors on overlap. If identical = no lookahead. 6/6 PASS.
- **4c Out-of-sample (2024–2025):** PC 0.858 (generalizes), factor 0.117 (does not). Honest deployable Sharpe ~0.5–0.8.

---

## Still to cover

### Phase 5 — Position Carry-Over
- Built carry-over to address the 88.4% force-close drag
- In-sample: PC +0.045, SSD +0.077 net Sharpe, drawdown improved
- OOS: carry roughly halves PC Sharpe (0.434 vs 0.858) — does not generalize
- Verdict: do not ship as default, regime-dependent
- Open threads: OOS apples-to-apples confirmation, regime-conditionality testing, aggregation mask fix, lookahead re-audit with carry, L1-audit items, professor question

---

## Topics discussed in detail
- How purity vs SIC is calculated (majority-sector count per cluster)
- How z-score is computed (rolling 126-day, shifted)
- Rolling vs frozen z-score window
- Entry/exit criteria and daily P&L accumulation into monthly returns
- Equal-weight dilution problem and z-weight alternative
- 5 suspected causes of Phase 1 Sharpe gap
- Engle-Granger direction choice (both A→B and B→A, take lower p-value)
- AR(1) half-life vs Ornstein-Uhlenbeck equivalence
- PC without filter vs with filter trade-off
- PC lower cointegration pass rate explanation
- Ridge betas calculation and factor loading plots
- What lookahead bias is and how the black-box test catches it
- Stop-loss: none in core, 3.5σ in realism (hurts more than helps)
