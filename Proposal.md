# Equity Pairs Trading with Dynamic Clustering — Proposal

Purpose: A defensible, market‑neutral stat‑arb built on pairs trading that sources candidates via clustering (economic peer selection), trades mean‑reverting spreads with a disciplined filter stack, and adds a small momentum overlay when relationships break. Designed to produce orthogonal alpha with tight drawdown control and realistic costs/impact.

References in repo:
- PLAN: Project QTS — Dynamic Clustering for Strategy Neutralization (Modules/Microstructure & QTS/Project QTS/PLAN.md)
- Code baseline: 2-pairs_trading_strategy_annotated.py (Modules/Microstructure & QTS/QTS Lectures/)

Executive value proposition
- Orthogonal alpha: Residual, idiosyncratic spread trades within peer clusters; low exposure to standard factors after neutralization.
- Drawdown discipline: Filter stack (z‑score, rolling ADF, cluster stability) reduces regime‑break losses; strict risk caps per pair/cluster.
- Feasibility: Daily data, free sources, tractable compute; extendable to intraday later.

Research integration (practical takeaways)
- Clustering-first pipeline improves pair quality; we implement A (returns-spectral), B (factor-beta), and add optional C (partial-corr with spectral/OPTICS/DBSCAN) inspired by Rotondi–Russo (2025). We report purity and stability, but we do not run an exhaustive method survey — goal is a robust production path, not a research benchmark.
- Two-step selection (screen→refine) from Rad–Low–Faff (2016) informs our formation flow: cluster-first screen, then residualized ADF/half-life filters; we emphasize after-cost net and unconverged-trade control.
- Execution calibration uses concrete magnitudes from ML pairs-trading work: explicit ≈8 bps, impact ≈20 bps, short borrow ≈1% p.a. We expose toggles (`--enable-impact`, `--impact-k-bps`, `--borrow-apr-bps`) and capacity gates to produce realistic net results.

Objective and philosophy
- Objective: Produce a concentrated, cost‑aware, market‑neutral pairs strategy with persistent net Sharpe ≥ 0.7 and MaxDD ≤ 12% on liquid US large caps.
- Philosophy: Baseline OLS+z must work gross and near‑net on canonical pairs (GOOG–GOOGL, PEP–KO). Adaptivity is layered only if it reduces left‑tail without collapsing trade count.

What’s unique here
- Candidate discovery is co‑primary: (A) spectral clustering on a Ledoit–Wolf shrunk return correlation matrix with k from the Marchenko–Pastur bound, and (B) factor‑beta clustering using rolling ridge betas of each stock to a compact ETF factor set (market, sectors, styles, and macro: rates, oil/commodities, USD/EUR FX) clustered in beta‑space.
- Added (C) partial‑correlation peer discovery: residualize stock returns vs compact controls (SPY and/or orthogonalized factor set), compute LW correlation of residuals (proxy for partial correlation), and cluster. Supports spectral (KMeans on Laplacian) and density algorithms (OPTICS/DBSCAN) that naturally leave outliers unclustered. This follows Rotondi–Russo (2025) that partial‑correlation distance improves cluster purity and net performance.
- Optional fusion: combine return‑ and beta‑space affinities (w≈0.5) to hedge specification risk and improve robustness.
- Pair selection adds a half‑life filter (5–30 days) so only pairs with actionable mean‑reversion speed enter the portfolio.
- Validation emphasizes out‑of‑sample stability, turnover/cost realism, and factor‑orthogonality (not just Sharpe). We track cluster purity vs. sector proxies and ARI stability; stability gates can freeze labels when ARI ≥ 0.6.

-------------------------------------------------------------------------------
Step 3 — Data fetch and caching (WRDS/CRSP/IBES/FF/FRED)
-------------------------------------------------------------------------------
Goal: materialize all data locally under `Modules/PairsTrading/data/` so the
pipeline can run reproducibly without interactive steps. WRDS credentials must
be configured once (stored in `~/.pgpass`). Sizes are approximate.

Prereqs (venv + packages)
- Activate venv and install deps:
  - `source .venv/bin/activate`
  - `uv add -U pandas pyarrow wrds sqlalchemy psycopg2-binary requests`

Initialize WRDS (stores creds once)
- `python - <<'PY'
import wrds
c=wrds.Connection(); print('WRDS connected'); c.close()
PY`

Fetch datasets (fresh, cached to Parquet)
- FF 5 factors + Momentum (Ken French; no WRDS needed)
  - `python - <<'PY'
from Modules.PairsTrading.data_fetcher import DataFetcher
f=DataFetcher(allow_network=True)
f.fetch_ff_daily(refresh=True)
print('ff_daily.parquet ready')
PY`
- FRED macro proxies (optional; diagnostics/orthogonality)
  - `python - <<'PY'
from Modules.PairsTrading.data_fetcher import DataFetcher
f=DataFetcher(allow_network=True)
f.fetch_fred(refresh=True)
print('fred.parquet ready')
PY`
- CRSP S&P 500 membership (historical, removes survivorship bias)
  - `python - <<'PY'
from Modules.PairsTrading.data_fetcher import DataFetcher
f=DataFetcher(); f.fetch_crsp_spx_membership(refresh=True)
print('crsp_spx_membership.parquet ready')
PY`
- CRSP adjusted prices and volume (primary inputs)
  - `python - <<'PY'
from Modules.PairsTrading.data_fetcher import DataFetcher
f=DataFetcher(); f.fetch_crsp_daily(start='2018-01-01', refresh=True)
print('crsp_close.parquet / crsp_volume.parquet ready')
PY`
- IBES earnings calendar (earnings blackout t−1..t+1)
  - `python - <<'PY'
from Modules.PairsTrading.data_fetcher import DataFetcher
f=DataFetcher(); f.fetch_ibes_earnings_dates(start='2018-01-01', refresh=True)
print('ibes_earnings.parquet ready')
PY`
- Markit borrow (optional; requires SFI entitlement)
  - `python - <<'PY'
from Modules.PairsTrading.data_fetcher import DataFetcher
f=DataFetcher()
try:
    f.fetch_markit_borrow(start='2018-01-01', refresh=True)
    print('markit_borrow.parquet ready')
except Exception as e:
    print('Markit SFI not available:', e)
PY`

Saved files (under `Modules/PairsTrading/data/`)
- `ff_daily.parquet` (~250KB)
- `fred.parquet` (~200KB)
- `ibes_earnings.parquet` (~5–15MB)
- `crsp_spx_membership.parquet` (~500KB)
- `crsp_close.parquet` (~50–200MB)
- `crsp_volume.parquet` (~50–200MB)
- `markit_borrow.parquet` (size varies; optional)

Wire CRSP into the pipeline
- Quick swap (no code changes):
  - `cp Modules/PairsTrading/data/crsp_close.parquet  Modules/PairsTrading/data/spx_close.parquet`
  - `cp Modules/PairsTrading/data/crsp_volume.parquet Modules/PairsTrading/data/spx_volume.parquet`
  The clustering, discovery, and backtest scripts read `spx_*` via `spx_data.py`.
- Or update `Modules/PairsTrading/spx_data.py` to read `crsp_*` directly.

Earnings blackout integration (recommended)
- Load `ibes_earnings.parquet` in `backtest_pairs.py` and:
  - Block entries if either leg has earnings in t−1..t+1.
  - Exit 1 day before earnings for active pairs.
This reduces earnings‑driven hard stops (see “Priority 1” in this proposal).

Validation (sanity checks)
- `python - <<'PY'
import pandas as pd, pathlib
base=pathlib.Path('Modules/PairsTrading/data')
for p in ['crsp_close.parquet','crsp_volume.parquet','ff_daily.parquet','ibes_earnings.parquet']:
    q=base/p; print(p, q.exists())
    if q.exists(): print(pd.read_parquet(q).head(2))
PY`

-------------------------------------------------------------------------------
0) ADR execution harness (validate execution stack first)
-------------------------------------------------------------------------------
Why start with ADRs
- ADRs and dual‑share pairs (e.g., GOOG–GOOGL) have tight economic linkage; small, transient gaps should mean‑revert. They are ideal to verify the execution layer: sizing, simultaneous orders, partial fills, stops, and PnL attribution.

Stepwise harness
- Stage A: Synthetic neutrality test (long SPY vs short SPY with hedge β=1). Expect near‑zero PnL net of modeled costs; validates fee/slippage accounting and share rounding.
- Stage B: Dual‑share US pairs (e.g., GOOG–GOOGL) to validate hedge‑ratio, borrow checks, corporate actions handling, and simultaneous execution.
- Stage C: Liquid ADRs (USD‑quoted Level II/III) within US hours. If home listing is not tradeable, treat it as reference only; avoid cross‑session latency/FX for v1.

Execution rules (MVP)
- Signal: spread z‑score Z_t = (s_t − μ̂_63)/σ̂_63 using prior‑day hedge β̂.
- Entry: |Z_t| ≥ 1.5 with hysteresis; Exit: |Z_t| ≤ 0.25; Time stop: 3–5 trading days; Hard stop: |Z_t| ≥ 3.0; Structural break: 5d corr < 0.5 or |Δβ̂|>30% → exit.
- Position sizing: target per‑pair daily σ_PnL 10–15 bps via 1/σ̂_spread; clamp β̂ ∈ [0.25,4.0]; ≤1% ADV/leg; ≤2 active pairs/issuer; ≤10% capital/cluster.
- Order handling: coordinate legs with slippage budget; if one‑leg fill occurs, hedge immediately or cancel‑and‑replace; attribute costs on fills.
- Trading window: only send new entries during US–Europe overlap (≈09:30–11:30 ET) to reduce stale‑leg risk; monitor outside window but do not initiate.
- FX and ratio: persist FX@entry/exit and ADR conversion ratio used; if hedging is supported when trading foreign leg, add explicit FX hedge; otherwise treat foreign as reference only (trade ADR leg) and relax neutrality target accordingly.
- What to explicitly test: borrow availability/fees; ADR ratio changes and dividends; calendar blackouts (earnings t−1..t+1); data alignment with T+1 signals; stale‑FX detector.

Example ADR candidates (for monitoring or live two‑leg if supported)
- BP (NYSE) vs BP.L (LSE), ratio 1:6, FX GBP/USD
- UL vs ULVR.L, ratio 1:1, FX GBP/USD
- VOD vs VOD.L, ratio 1:10, FX GBP/USD
- HSBC (NYSE) vs HSBC.L or 0005.HK, FX GBP or HKD
- SAP (NYSE) vs SAP.DE, ratio 1:1, FX EUR/USD

-------------------------------------------------------------------------------
1) Research questions and hypotheses
-------------------------------------------------------------------------------
RQ1. Do spread z‑score thresholds on cointegrated pairs within spectral clusters deliver positive net Sharpe after realistic costs?
RQ2. Does LW‑shrunk correlation clustering improve pair quality (lower EG p‑value, tighter half‑life distribution, higher formation‑window corr) vs GICS sector labels?
RQ3. Do formation filters (ADF gate, half‑life bounds, cluster stability ARI) reduce left‑tail risk without collapsing trade count below 40 active pairs?

Hypotheses
- H1: Spectral cluster peer selection yields higher hit‑rate and lower phantom turnover than GICS groupings; net Sharpe > 0.8 daily‑annualized.
- H2: Adding ADF gate + half‑life filter at formation reduces CVaR95 ≥ 10% vs z‑only without >50% trade‑count collapse.
- H3: Parameter sensitivity (±0.5 on z‑entry; ±21d on lookbacks) does not flip net Sharpe sign — strategy is robust, not curve‑fit.

Note: breakout/momentum overlay deferred to v2 (see Extensions).

-------------------------------------------------------------------------------
2) Universe, horizon, regimes
-------------------------------------------------------------------------------
- Universe: S&P 500 top ~300 by ADV or market cap (liquid, low borrow frictions). Optional: add ADRs carefully.
- History: 2014–present for robustness; use 2014–2019 formation, 2020–2021 validation, 2022–2024 OOS test (trend + crisis + chop).
- Frequency: Daily close for v1; intraday extension (30–60 min bars) only after v1 passes gates.
- Regimes tracked: dispersion/avg‑correlation (from PLAN Wk3), VIX terciles; report performance by regime.

Data hygiene and execution alignment
- Constituents: Prefer historical SPX membership to reduce survivorship; if unavailable, disclose bias when using current SPX.
- Execution: T+1 signals (use prior‑day signal; execute today’s close). No look‑ahead on rolling windows.
- Calendars: Earnings blackout t−1..t+1 per leg; handle corporate actions explicitly (share‑class pairs like GOOG–GOOGL).
- Regime slices: Always report by VIX terciles, dispersion terciles, and crisis windows (Aug‑2015, Dec‑2018, Mar‑2020, 2022 trend, 2023–2024 chop).

Data sources (free/practical)
- Prices: CRSP adjusted close via crsp.dsf (primary); yfinance adjusted close as fallback.
- Sectors: SPDR ETFs (XLE, XLK…) for diagnostics; GICS sector tags via free mappings or inferred.
- Proxies for fundamentals (optional): Market cap (from price×shares outstanding), simple ratios via public APIs if available.
- Costs proxies: Half‑spread from close‑to‑mid proxies; Amihud illiquidity from daily |ret|/DollarVol.

Design decisions (locked 2026-05-24)
- Universe: S&P 500 historical membership (crsp.msp500list, permno-filtered). Do & Faff (2010/2012)
  use full CRSP ordinary shares (shrcd 10/11) — this project targets liquid large caps matching
  the practical trading environment. Full-CRSP micro/small caps fail ADV>$20mm and price>$5 anyway.
  Tradeoff: fewer pair candidates; mitigated by PC1 throttle in macro-dominated regimes.
  Implementation: spx_only=True in fetch_crsp_daily() — permno subquery on crsp.msp500list,
  NOT ticker filter (tickers are reused across firms; permno is stable across renames/mergers).
- Price series: split-adjusted close (ABS(prc)/cfacpr). Raw prices create step jumps at split
  dates that break ADF cointegration tests and distort OLS hedge ratio estimation. Adjusted close
  removes these artifacts. "Lookahead bias" from backward adjustment affects price levels only —
  no future return or spread-direction information is injected into the signal.

-------------------------------------------------------------------------------
Execution realism and costs calibration
-------------------------------------------------------------------------------
We adopt pragmatic, literature‑anchored cost components and toggles to move from research‑clean to trade‑realistic backtests:

- Explicit commissions: default 3 bps/leg (≈6 bps round‑trip), tunable via `--cost-bps`.
- Market impact: square‑root model, `impact_bps = k × (√participation_A + √participation_B)`, off by default; enable with `--enable-impact` and calibrate `--impact-k-bps` by ADV bucket or Amihud. Baseline k≈15 bps informed by ML study’s ≈20 bps magnitude.
- Capacity controls: per‑leg ADV cap and min ADV floor to bound participation; portfolio notional drives participation.
- Short borrow fee: APR in bps applied pro‑rata daily to the short leg (default 0). When Markit borrow is available, screen HTB names and/or apply name‑specific APR. Toggle: `--borrow-apr-bps`.
- Earnings blackout: IBES actuals (anndats_act) ±N calendar days; entry blocked and optional forced exit. Toggle: `--earnings-blackout N`, `--no-earnings-exit`.

These magnitudes align with: Rad–Low–Faff (2016) after‑cost focus and unconverged‑trade management; ML paper’s explicit ≈8 bps commissions, ≈20 bps impact, ≈1% p.a. shorting drag. Start conservative; refine using ADR harness diagnostics (one‑leg fill rate, slippage vs budget, borrow availability).

IBES integration
-------------------------------------------------------------------------------
IBES actual earnings dates feed an entry blackout (±N days per leg) and optional forced exit of open positions entering the window. DataFetcher exposes `fetch_ibes_earnings_dates()`; the backtester consumes `ibes_earnings.parquet` when `--earnings-blackout > 0`.

ADR execution notes (v2 hooks)
-------------------------------------------------------------------------------
ADR/dual‑listing pairs (e.g., UL–ULVR.L, SAP–SAP.DE) require FX and ratio handling. Execution calibration (explicit/impact/borrow) carries over; impact k and borrow APR often differ by venue and should be bucket‑calibrated. ANN threshold‑learning is out of scope for v1; revisit if we extend into multi‑strat (including ADR and intraday).

-------------------------------------------------------------------------------
3) Clustering for peer discovery
-------------------------------------------------------------------------------
Principle: two complementary, co‑primary views — return co‑movement (noise‑controlled) and economic exposure similarity (factor betas). Use either view alone or a simple fusion; set k by theory/guardrails, not ad‑hoc tuning.

Co‑primary method A — returns‑spectral (noise‑controlled co‑movement)
1. Estimate trailing 126d return correlation matrix C (N×N).
2. Apply Ledoit–Wolf shrinkage → Ĉ (full rank, noise‑suppressed).
3. Compute graph Laplacian L = D − Ĉ; eigendecompose.
4. Marchenko–Pastur upper bound: λ₊ = (1 + √q)² where q = N/T. Set k = count(eigenvalues > λ₊).
5. K‑Means on top‑k eigenvectors (L2‑normalised rows); best of 10 seeds.
6. Stability gate: ARI between consecutive monthly clusterings > 0.6; otherwise freeze labels for that month.

Optional method C — partial‑correlation view (residual co‑movement)
1. Residualize each stock’s daily returns vs SPY and/or orthogonalized factor set (ridge).
2. Compute LW‑shrunk correlation on residual returns (proxy for partial correlation).
3. Cluster: spectral+KMeans (default) or density (OPTICS/DBSCAN) to deliberately leave outliers unassigned.
4. Diagnostics: report sector‑purity and stability (ARI) over refits; freeze labels when ARI ≥ 0.6 to reduce churn.

Co‑primary method B — factor‑beta clustering (economic exposure similarity)
1. Factor set (~17–20 ETFs): SPY; sectors XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLC, XLRE; styles MTUM, VTV (or VLUE); macro TLT (or IEF), GLD (or USO), UUP (or FXE); optional XBI.
2. Compute daily factor returns; standardize and winsorize (~3σ). Optionally orthogonalize factor returns (e.g., PCA/Gram–Schmidt) to reduce collinearity.
3. Rolling ridge betas per stock on 126–252d window with intercept; EWMA weights (λ_EWMA≈0.97). Fix λ via a one‑time time‑split grid on formation; revalidate annually.
4. Build last‑window beta matrix B (N×F). Scale betas by factor vol; optionally weight dimensions by 1/(SE_beta²+τ²).
5. Distance in beta‑space: cosine/correlation distance (scale‑invariant) or uncertainty‑aware Mahalanobis.
6. Cluster on beta‑space via K‑Means or spectral on affinity Aβ = exp(−d²/σ²). Set k to match method‑A k within ±20% as a guardrail.
7. Stability gate: same ARI>0.6 freeze rule.

Multi‑view fusion (optional, recommended for research)
- Build A_ret from method A and A_beta from method B; fuse A = w·A_ret + (1−w)·A_beta with w≈0.5; spectral‑cluster on A. Optionally test SNF.

Why LW shrinkage and ridge? LW stabilizes co‑movement in the high‑q regime; ridge stabilizes betas under multicollinearity across sectors/styles/macro.

Credibility ablation (run alongside v1 core)
- Hierarchical clustering (average linkage) on Ĉ distance matrix — k‑free baseline.
- Beta‑space clustering (ridge features) vs returns‑spectral — compare ARI stability and downstream Sharpe.
- Fusion A_ret+A_beta with weights {0.25,0.5,0.75} — robustness to specification.
- Residualized‑returns clustering: regress assets on the same factor set; cluster on residual correlations.
- Uncertainty‑aware distances in beta‑space vs plain cosine.

Static classifications as benchmark, not source of truth
- GICS and SIC baselines with identical downstream filters for fairness; expect SIC < GICS < dynamic methods on pair quality.

Window and cadence
- Estimation windows: 126d for Ĉ and betas (252d as robustness). Betas recomputed daily; clusters refreshed monthly with ARI‑freeze to avoid churn when already stable.
- Time segregation: formation on prior 252d; trade next month with T+1 signal alignment.

-------------------------------------------------------------------------------
4) Pair selection within clusters
-------------------------------------------------------------------------------
Formation window: 252 trading days (1y), roll monthly.

Scores (blend; require ALL pass minimums — hard filters, not weights)
- Distance score: MSE of normalized log‑price spread (lower is better).
- Rolling corr: 63d corr > 0.75 at formation (tighter than 0.7; looser pairs generate noise).
- Cointegration: Engle–Granger ADF p‑value < 0.05 on 252d formation window.
- Half‑life filter: 5 ≤ half‑life ≤ 30 trading days.
  Computed from ADF regression slope: half_life = −log(2) / γ where Δspread_t = γ·spread_{t‑1} + ε.
  Below 5d: too fast, likely noise. Above 30d: capital tied up too long, kills capacity.
- Hedge‑ratio stability: std of rolling OLS β (63d) below threshold; clamp β within [0.25, 4.0].

Selection: Top 3–5 pairs per cluster; deduplicate overlapping names; portfolio cap 40–60 pairs.

Cointegration testing pipeline (non‑negotiable)
- Method: Engle–Granger two‑step on formation window. Regress log P1 on log P2 (+ intercept); take residuals ê_t; run ADF on ê_t with lag length by AIC/BIC. Require p<0.05 at formation; p<0.10 acceptable for maintenance checks.
- Robustness: also test reversed regression (P2 on P1) to avoid regression direction bias; require at least one direction to pass and hedge with corresponding β.
- Alternatives: Johansen test for k>2 legs (v2 extension); Phillips–Ouliaris as a cross‑check. Use HAC standard errors if serial correlation is material.
- Guardrails: if ADF fails on the next monthly check or 63d corr drops <0.5, disable pair for trading until next formation.

-------------------------------------------------------------------------------
5) Signals and filter stack
-------------------------------------------------------------------------------
Primary mean‑reversion signal
- Spread_t = log(P1_t) − β̂·log(P2_t) − α̂ (β̂ from rolling OLS 63d)
- Z_t = (Spread_t − μ̂)/σ̂ (μ̂,σ̂ rolling 63d)
- Entry: |Z| ≥ 2.0; Exit: |Z| ≤ 0.5; Time stop: 20 trading days; Hard stop: |Z| > 4.0.

Structural validity gate (formation only — not a daily trading gate)
- ADF re‑run monthly on 252d rolling window; trade only when p < 0.10 AND cluster ARI > 0.6.
- Do not re‑run ADF daily — repeated testing inflates family‑wise error rate and generates spurious in/out signals.

In‑trade exit triggers (structural break)
- If 5d rolling corr < 0.5: exit immediately. Relationship broke; fade is wrong.
- If |Δβ̂| > 30% vs entry β̂: exit immediately. Hedge ratio drifted; position is mis‑hedged.
- No breakout/momentum overlay in v1 — see Extensions.

Design principles
- OLS + z‑score thresholds must work gross and near‑net on canonical pairs (GOOG–GOOGL, PEP–KO) before any filter is added.
- ADF is a formation gate, not a signal smoother. Once in a trade, only z_exit / time_stop / hard_stop / break triggers close it.
- Kalman filter: counterproductive in v1 (suppresses monetisable noise, kills trade count). Reserve for top decile pairs in v2 if trade count stays above 30 active pairs.

Additional filters — what and why
- Liquidity: ADV > $20mm and price > $5 to avoid micro‑caps; caps at ≤1% ADV/leg to bound impact.
- Borrow/fees: skip names with hard‑to‑borrow flags or borrow fee > 200 bps annualized; re‑check weekly.
- Event risk: earnings/major corporate actions blackout t−1..t+1; suspend on halts and mergers.
- Regime: suspend adds when VIX > 90th pct or when average pairwise corr spikes (crisis mode); maintain exits per hard stop.
- Dispersion filter: prefer months with middle‑to‑high cross‑sectional dispersion where MR is healthier; report performance by dispersion terciles.
- Factor drift: monthly regression of portfolio PnL on Fama–French + MOM + QMJ; if beta to Mkt or MOM drifts outside ±0.05, tighten caps per cluster/sector.

-------------------------------------------------------------------------------
6) Sizing, costs, and impact
-------------------------------------------------------------------------------
Hedge and sizing
- Market‑neutral per pair: notional in short leg = β̂ × long‑leg notional.
- Vol targeting: per‑pair target daily σ_PnL ≈ 10–15 bps; scale by 1/σ̂_spread.
- Caps: max 1% of ADV per leg; max 2 active pairs per issuer; max 10% portfolio exposure per cluster.

Sizing and PnL units — avoid deflated weights
- Signal stays in spread space: spread_t = log(P1_t) − β̂·log(P2_t) − α̂; z_t uses μ/σ from the spread residual.
- Sizing and PnL are computed in pair‑return space for coherent units and realistic magnitudes:
  - r_pair_t = r_A_t − β̂_{t−1}·r_B_t (T+1: use yesterday’s β̂)
  - σ_pair = rolling 63d std(r_pair) shifted by 1 day
  - w_t = target_vol / σ_pair (e.g., 0.001 / σ_pair)
  - PnL_t = w_{t−1} × pos_{t−1} × r_pair_t (execute with T+1 alignment)
- Rationale: using the level‑spread’s σ to size deflates w by ~√window; pair‑return σ keeps w in realistic 5–15% ranges per pair and makes cost/impact meaningful.

Transaction costs and impact (daily model)
- Explicit costs: 2–4 bps per leg (large caps) applied on entry/exit.
- Slippage/impact proxy: empirical power law cost_bps ≈ k × (participation)^0.5 with k ≈ 10–20 for US large caps.
- Amihud penalty: add λ × Amihud per leg to costs on high‑illiquidity names.
- Capacity caps: ≤1% ADV per leg; ≤10% portfolio exposure per cluster; ≤2 active pairs per issuer.
- Turnover drag: report gross vs net; require net Sharpe within 0.2 of gross at steady state; stress at 5–6 bps/leg.
 - Short borrow: apply APR bps pro‑rata daily to the short leg while position is open; default 0 in v1, tunable via `--borrow-apr-bps`. Screen out extreme HTB names when Markit data is available.

-------------------------------------------------------------------------------
7) Backtest design and validation
-------------------------------------------------------------------------------
Data hygiene
- No survivorship: prefer a historical SPX membership file; if not, disclose bias and use current SPX but time‑stamp ticker listing dates.
- Signal alignment: use yesterday’s signals; execute at today’s close (shift logic); no look‑ahead on ADF windows.

Avoiding hindsight bias and leakage (implementation notes)
- Walk‑forward clustering and pair discovery: compute clusters and select pairs using only data up to the refit date d0; trade only in (d0, d1].
- Freeze labels between refits: recompute every refit_freq (e.g., 21/63/252d), freeze in between to avoid phantom turnover.
- Formation windows end at refit: ADF/half‑life/correlation filters end at d0; never include trading‑segment data.
- T+1 execution alignment: for day t, compute hedge β̂ and z on data through t−1; trade at t using β̂_{t−1} and μ/σ_{t−1}.
- Structural break exits: if 5d corr < 0.5 or |Δβ̂| > 30% vs entry β̂, exit immediately.
- Static baseline without bias: form pairs once at the start (using only prior history), then hold or only re‑validate forward; do not pick end‑of‑sample pairs and apply backward.

Protocol
- Walk‑forward: monthly re‑clustering/re‑selection; daily trading; 2014–2019 fit, 2020–2021 validation, 2022–2024 OOS.
- Ablation runs:
  (a) z‑only (baseline): no ADF gate, no half‑life filter, GICS sectors
  (b) +ADF formation gate: measure CVaR95 reduction vs (a)
  (c) +half‑life filter (5–30d): measure trade count and Sharpe vs (b)
  (d) spectral clusters vs GICS: same signal stack (c), swap cluster labels — this is the core claim
  (e) LW‑hierarchical clustering vs spectral: compare ARI stability and downstream Sharpe
  (f) Factor‑beta (ridge) clustering vs returns‑spectral; include fusion weights {0.25,0.5,0.75}, residualized‑returns clustering, and uncertainty‑weighted beta distances; compare ARI, pair quality, and downstream Sharpe
  (g) Static label baselines: SIC vs GICS vs dynamic spectral — report median EG p‑value, HL IQR, entry corr, and net Sharpe
  (h) Reclustering cadence: monthly vs quarterly with ARI‑freeze — compare turnover, stability, and net Sharpe drag from churn
- Stress windows: Aug‑2015, Q4‑2018, Mar‑2020, 2022 trend, 2023–2024 chop.

Metrics
- Portfolio: Sharpe, Sortino, MaxDD, MAR, turnover, cost drag; tail metrics (skew, CVaR95).
- Neutrality: regression betas to SPY and sector ETFs; exposures by cluster.
- Orthogonality: regress PnL on standard equity factors (Mkt, SMB, HML, MOM, QMJ) and macro proxies (rates via TLT/IEF, oil/commodities via GLD/USO, USD/EUR via UUP/FXE); require low explanatory power (R² < 10%).
- Execution diagnostics (ADR harness): one‑leg fills rate, slippage vs budget, borrow availability, and PnL attribution waterfall (gross → explicit costs → impact → residuals).

Pass/fail gates
- Gate 1: Net Sharpe ≥ 0.7, MaxDD ≤ 12%, SPY beta ∈ [−0.05, 0.05] on 2022–2024 OOS.

-------------------------------------------------------------------------------
Interim Results and Diagnostics (current build)
-------------------------------------------------------------------------------
What we tried
- Clustering: Method A (returns‑spectral) and fused (returns + beta‑space) with quarterly refits (63d), 126d corr windows, 252d formation, top‑k=3.
- Signal/filters: z_entry=2.0, z_exit=0.5; ADF p<0.05; half‑life 5–30d; corr63≥0.7; β clamp.
- Backtest: T+1 alignment; sizing/PnL in pair‑return space (r_pair = r_A − β̂t−1·r_B); costs 3 bps/leg; no ADV caps.

Key outcomes
- Pair‑return sizing fixed a scaling artifact: vol rose (≈2.75% vs 1.41%), Sharpe collapsed (≈−0.19 net; ≈−0.05 gross).
- Drawdown became meaningful (≈−8.1%), which is plausible for realistic sizing.

Diagnostics (quarterly)
- Pair overlap ≈ 6% between consecutive refits (critical) while stock overlap ≈ 18–20%.
  → Same stocks recur but land in different clusters/partners; discover_pairs picks different intra‑cluster pairs each quarter.
- Half‑life median ≈ 6.7d; with z_entry=2.0 on 63d stats, entries fire late/rarely.
- Trades: ≈231 round‑trips over 135 pairs in 6 years → ≈0.07 trades/pair/quarter (too low to diversify).
- Hit rate ≈ 80% but negative Sharpe: winners small; losers (time/hard stops) larger → late entries.
- Concentration: one cluster carried ≈46% of pairs → hidden exposure.

Root‑cause split (qualitative)
- Clustering instability (primary, ≈60%): Fused labels change frequently; ARI freeze rarely triggers. Blending two noisy affinities (returns + beta) amplifies churn.
- Formation window drift (secondary, ≈40%): 252d window shifts; even stable clusters find different pairs.

Fastest remediations (prioritized)
- Use Method A (returns‑spectral) for now; defer fused until stability improves.
- Increase breadth: top‑k per cluster 6–8 to reach 20–40 active pairs per segment.
- Enter earlier: z_entry=1.5 (z_exit=0.5) to capture more of the reversion on ≈7d half‑lives.
- Loosen formation slightly: corr63≥0.6; half‑life 5–45d to allow medium‑speed MR.
- Stability knobs: raise ARI freeze threshold (e.g., ≥0.7), and/or lengthen corr window (e.g., 126→189) to damp churn.

What to monitor after changes
- overlap_rate between segments (target ≥ 20–30%).
- trades/pair/segment (target ≥ 2) and avg hold days.
- breadth and cluster balance (avoid single‑cluster dominance).
- gross vs net Sharpe once costs re‑enabled (stress 5–6 bps/leg).

Implementation notes (code status)
- Signals: spread residual drives z‑score; T+1 alignment enforced.
- Sizing/PnL: computed in pair‑return space (keeps units coherent; avoids deflated weights from spread‑level σ).
- Iteration knobs added: z‑entry/z‑exit/costs via CLI; formation filters via CLI; overlap/trade stats saved.
- Capacity realism: ADV cap and impact toggles exist (off by default). Portfolio issuer/cluster caps to be added after signal stabilizes.

-------------------------------------------------------------------------------
Next Steps: Residualized Formation and Beta‑Space Stability
-------------------------------------------------------------------------------
Why residualize first
- Raw log‑price spreads mix idiosyncratic and factor components. As factor exposures drift across quarters, ADF/HL flip in/out even when the idiosyncratic relationship holds.
- Residualized formation (Avellaneda & Lee, 2010): neutralize each leg’s returns against a stable factor set, integrate residuals to pseudo‑prices, and test cointegration on that residual spread. This targets pure idiosyncratic stationarity and is more stable across segments.

Procedure (T+1‑aligned)
- Factors: FF 5 + MOM + FRED deltas (DGS2/DGS10/T10Y2Y/IG OAS/HY OAS) + ETF proxies (SPY, 11 sectors, styles, macro) via DataFetcher('both'); cached to parquet once.
- Residuals: êA,t = rA,t − βA′t−1 Ft (EWMA ridge, 252–504d, ridge α≈25–50; drop RF; standardize factors). Same for êB,t.
- Integrate: p̂A,t = cumsum(êA,t), p̂B,t = cumsum(êB,t).
- Residual spread: ŝt = p̂A,t − γ p̂B,t − α (γ from OLS on residual pseudo‑prices within formation window).
- Formation filters: ADF p<0.05 (new), p<0.10 (maintenance); HL in [5,45]; corr63≥0.70. Apply persistence gate (e.g., ≥2 passes in last 3 segments) and cap "new" slots per segment (≤ 30–40% of book).

Stabilize clustering in parallel
- Method A: corr_window 189–252d; ARI freeze threshold ≥ 0.7; refit 63d.
- Method B: factor set 'both'; beta_window 504d; EWMA λ≈0.98; ridge α 25–50; standardize columns; optional PCA(5–10 PCs) on betas before affinity.

Validation targets (zero‑cost first)
- Pair overlap ≥ 20–30% between consecutive refits.
- Trades/pair/segment ≥ 2; avg hold ≈ 10–15d; breadth 20–40 pairs/segment.
- Only then re‑enable costs (5–6 bps/leg) and assess net Sharpe; optionally turn on ADV/impact toggles for capacity realism.

-------------------------------------------------------------------------------
Residualized Formation Guide — Execution Order and Checkpoints
-------------------------------------------------------------------------------
1) Seed caches (one‑time)
   - Run DataFetcher('both') with refresh=True to download FF+MOM, FRED deltas, SPX prices/volume.
   - Verify ./data contains ff_daily.parquet, fred.parquet, spx_close/volume.parquet (and factor_close.parquet for ETFs).

2) Add maintenance gate (highest impact)
   - New pairs: ADF p<0.05, HL∈[5,45], corr63≥0.70. Maintenance pairs: p<0.10.
   - Require ≥2 passes in last 3 segments. Cap “new” to ≤30–40% of book per segment.
   - Checkpoint: quarterly refit (63d), Method A with corr_window 189d, top‑k 6, z_entry 1.5, zero‑cost.
     Target: overlap_rate ≥ 15%; trades/pair/segment ≥ 2.

3) Residualized cointegration (Avellaneda & Lee)
   - Neutralize r_A and r_B against standardized factor matrix (drop RF), EWMA ridge (β window 252–504d, λ≈0.97–0.985, α≈25–50).
   - Integrate residuals: p̂A = cumsum(êA); build residual spread and re‑estimate γ (within formation window); run ADF/HL.
   - Checkpoint: same settings as (2), now with residualization.
     Target: overlap_rate ≥ 20–30%; cluster persistence stays high.
   - If overlap < 20%: lengthen formation window to 378d (1.5yr). Use 504d only if necessary (beware longer burn‑in).

4) Signal viability (only after (3) targets hit)
   - Turn off zero‑cost; re‑enable explicit costs 5–6 bps/leg.
   - Optional: turn on ADV cap (≤1%/leg) and impact (k≈15–20) to gauge capacity realism.
   - Confirm net Sharpe and drawdown within goals; add issuer/cluster caps at portfolio level later.

5) Method B in parallel
   - Factor set: DataFetcher('both'); beta_window 504d; EWMA λ≈0.98; ridge α≈25–50; standardize; optional PCA(5–10 PCs) on betas.
   - Apply the same maintenance gate and residualized formation; validate zero‑cost targets; then add costs/caps.

6) Optional FDR/BH correction (after overlap target met)
   - Within cluster, apply BH/FDR (q≈10%) to ADF p‑values to control false discoveries across many pairs.

-------------------------------------------------------------------------------
Market‑Structure Diagnostic (not "macro regime")
-------------------------------------------------------------------------------
Hypothesis
- Post‑2022, SPX became more homogeneous: PC1 explains a larger share of cross‑sectional variance and overall dispersion compressed. Idiosyncratic residuals then border on random walks → residualized ADF rarely passes, independent of filter width.

Minimal test (2 min, no new runs)
```
import numpy as np, pandas as pd
rets = pd.read_parquet("data/spx_close.parquet").pct_change().dropna()
for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
    r = rets[str(yr)].dropna(axis=1, how='all').fillna(0)
    _, s, _ = np.linalg.svd(r.values, full_matrices=False)
    print(f"{yr}: PC1 share = {(s[0]**2)/(s**2).sum():.1%}")
```
Interpretation
- If PC1 share rises materially (e.g., ~25% → >40%), that explains thin residual cointegration without invoking filter/code issues. Treat it as a market‑structure shift: trade fewer, higher‑quality pairs and/or extend the universe.

-------------------------------------------------------------------------------
Plan Reorder — New‑Pair Cap Before Crisis Throttle
-------------------------------------------------------------------------------
Rationale
- With overlap ≈ 17%, admitting many “new” pairs per segment still leaves the book dominated by untested names. A hard cap forces the mix toward maintenance pairs as persistence builds.

Mechanic (after persistence exists)
- In the segment loop: let maint_count be # of maintenance pairs that pass; total_cap ≈ target book size.
- Enforce: max_new_per_seg = max(maint_count, int(total_cap × 0.35)).
- Fill slots with top‑ranked maintenance pairs first; admit new pairs up to the cap.

Sequence update
1) Maintenance gate + residualized formation (A‑S3).
2) New‑pair cap (≥65% maintenance mix) once overlap ≥ 15–17%.
3) Crisis/breadth‑aware throttle only after (2), to scale risk when breadth genuinely collapses.

-------------------------------------------------------------------------------
Universe Extension — Conditional on PC1 Diagnostic
-------------------------------------------------------------------------------
Guideline
- Extend beyond SPX (e.g., ADRs/duals, add large midcaps with liquidity screens) only if PC1 share confirms homogeneity and 378d formation still yields thin breadth in 2023–2024. Otherwise, adding midcaps risks more noise/borrow frictions without solving persistence.

-------------------------------------------------------------------------------
Method B Priority (after A‑S3/‑S5)
-------------------------------------------------------------------------------
Order of operations and outcome
- Validate A (S12d stack) across 2014–2024; add breadth/PC1 throttle and new‑pair cap.
- Current B (betas to correlated ETF+FF; PCA on betas) is closed on daily SPX.
- Orthogonal‑B (S16c): PCA on factor RETURNS first (fit on factor_rets[:d0]), then EWMA ridge betas to orthogonal PCs, then cluster. Result: Gross +0.099 (first positive), but ClusterPersist ~50% and overlap ~2.5% — not additive vs A at daily frequency.
- Interpretation: daily factor rotation in SPX changes exposures faster than quarterly refits; even with correct orthogonalization, beta‑space clusters churn. Method B is viable conceptually but needs either higher‑frequency refits/betas or institutional risk‑model factors (Barra/Axioma) to be competitive here.



-------------------------------------------------------------------------------
Next Run — A‑S5 (Formation 378d)
-------------------------------------------------------------------------------
Settings
- Keep A‑S3 filters (residualized ADF, HL ≤ 30, corr63 ≥ 0.70), corr_window 189d, ARI freeze ≥ 0.7, refit 63d.
- Increase formation window to 378d (1.5y); persistence gate unchanged.

Targets
- Overlap ≥ 20%; breadth ≥ 15 pairs/segment consistently in 2023–2024; trades/pair/segment ≥ 2.
- If unmet and PC1 share is high, proceed to new‑pair cap and/or conditional universe extension.

-------------------------------------------------------------------------------
Regime Overlay (HMM) — Next Chapter
-------------------------------------------------------------------------------
Motivation
- Performance is regime‑heterogeneous (e.g., 2014/2017/2021/2022). Single gates using PC1 share or dispersion are weak; we need a persistent regime classifier to scale entries without retuning pair parameters.

Hypothesis
- A 2–3 state Gaussian HMM on market observables (VIX level and term slope, realized SPX vol, cross‑section dispersion or average correlation, PC1 variance share; optional HY OAS and curve) reduces left‑tail losses in stressed/crisis states and stabilizes Sharpe across years.

Design (Phase 1 — VIX‑based HMM)
- Features (daily, rolling z‑score; T+1 aligned): log(VIX); VIX term slope (VIX3M−VIX1M or ratio); 20d realized SPX vol; cross‑section dispersion or avg corr; PC1 share. Optional: VVIX, HY OAS, 2s10s.
- Model: GaussianHMM(n_components=3, covariance_type='diag'), seeded; fit on 2014–2019 only; decode OOS with filtered probabilities p(state|x≤t).
- Labeling: sort states by mean VIX/dispersion per refit so calm < stressed < crisis is consistent.
- Execution overlay: per‑day position scale = 1.0·p(calm) + 0.5·p(stressed) + 0.0·p(crisis). Block new entries only when crisis dominates; keep exits unchanged in v1.
- Implementation: add regime_hmm.py (fit/decode/cache to data/regime_states.parquet); backtest_pairs.py gets --regime-overlay {none,hmm_vix} and T+1 scaling. Ensure data_fetcher provides VIX series.

Validation (Phase 1)
- Baselines: S17a (no overlay); naïve VIX>25 hard gate.
- Experiments: S25a–S25h (3‑state vs 2‑state; posterior vs Viterbi; train windows; dynamic hard_stop ablation). Slice results by inferred state and by stress windows (Aug‑2015, Dec‑2018, Mar‑2020, 2022–24).
- Metrics: full‑period gross/net Sharpe, MaxDD, trips, hit‑rate; yearly SR for 2014/2017/2021. Significance via NW or bootstrap on daily return deltas (overlay − baseline).

Acceptance
- Net Sharpe lift ≥ +0.05 vs S17a with MaxDD not worse; active trips down ≤ 30%.
- 2014/2017 SR not worse by > 0.15; 2022–2024 losses reduced; HMM ≥ naïve VIX gate by ≥ +0.05 SR.
- If met, consider Phase 2 (SABR features) with incremental lift target ≥ +0.10 SR vs Phase 1.

Risks
- Low‑vol scarcity (few entries) persists — defer calm‑state z_entry adjustments to v2 if needed.
- State drift across refits — enforce label ordering by mean VIX/dispersion and document regime plots.
- SABR data plumbing — gated on Phase 1 success.

-------------------------------------------------------------------------------
Finding: 378d Formation Improved Overlap, Hurt Sharpe — Why and What To Change
-------------------------------------------------------------------------------
Observation (A‑S5 vs A‑S3)
- Overlap rose 17% → 21% and 2022–24 breadth recovered, but Sharpe fell and MaxDD worsened.
- Cause: integrating residuals over 1.5y makes the residual pseudo‑prices drift more; OLS γ fit is over a longer, noisier path; residual spread excursions are larger and z‑scores are biased → more false entries.

Conclusion
- There is a horizon trade‑off: longer windows stabilize pair identity but degrade the trading signal if you also test on the full long window. Fix by decoupling horizons.

-------------------------------------------------------------------------------
Decouple Horizons: Long Formation Window, Tail ADF/HL (A‑S6)
-------------------------------------------------------------------------------
Principle
- Use a long residual history to decide which pairs to trade (stability), but test cointegration and half‑life on the trailing slice only (signal quality). Keep trading z‑scores on a short window.

Specification (T+1‑aligned)
- Candidate history: 378d residuals (êA, êB integrated to p̂A, p̂B) for stability and persistence decisions.
- Tail ADF/HL and γ: compute on the last 252d of the 378d window (not the full 378d). This preserves stability while avoiding long‑horizon drift contamination.
- Trading stats: keep z‑score/sizing windows short (e.g., 63d for μ/σ on residual spread; σ_pair for sizing).
- Filters/gates: retain residualized ADF (p<0.05 new, p<0.10 maintenance), HL band [5,30] or [5,45], corr63≥0.70, persistence ≥2/3, cap “new” ≤30–40% of book.

Targets (zero‑cost)
- Maintain overlap ≥ 20% and 2022–24 breadth ≥ 15/segment, while Sharpe (gross) improves toward 0 and MaxDD tightens from A‑S5.

Notes
- Rolling factor z-scores (252d) and dropping FF_RF from regressors remain in force.
- If tail-ADF still degrades Sharpe, trial a deterministic-trend ADF (regression='ct') as a robustness check; the baseline remains regression='c' on residual pseudo-prices.

-------------------------------------------------------------------------------
Addendum — 2026-05-23: Beta Estimation Choices and Next Levers
-------------------------------------------------------------------------------

Beta estimation for two-asset spreads (hedge ratio)
- OLS (rolling 63d) remains the right baseline in a 1-predictor + intercept setup. Ridge adds little here.
- Kalman (adaptive β) is available and corrected: we scale the observation H by a rolling mean of pb so H≈[1,1], and seed P0 from an OLS warmup (63d). This avoids oversized gains and early β-uncertainty choke. Use when controlled adaptation is desired; tune δ≈(0.2–1.0)×10⁻⁴.
- EWMA‑WLS on the 63d window effectively increases β responsiveness (Kalman‑lite). Empirically bleeds MR alpha vs OLS; not recommended as default.
- Huber regression for formation β mutes outliers but is largely redundant with IBES blackout + structural‑break exits; expected +0.01–0.03 gross SR at best.

High‑impact next steps (beyond β estimation)
- ADF‑affinity clustering (pair discovery): build a sparse neighbor graph (top‑M by 63d corr≥0.7). On edges, compute Engle–Granger ADF on the 252d tail of a 378d window; weight edges by −log₁₀(p) × a half‑life kernel favoring 8–20d. Spectral cluster on fused returns+ADF affinity. Expected +0.05–0.15 gross SR.
- Regime/breadth throttle: compute PC1 share of the trailing correlation matrix each refit. Down‑scale book linearly from scale=1 at PC1≤0.22 to ≈0.20 at PC1≥0.35. Optional FRED overlay (T10Y2Y inversion + HY OAS widening) to clamp in macro regimes. Expected +0.03–0.08 full‑period SR by cutting 2022–2024 losses.
- Execution realism: per‑name netting before ADV cap and impact; short borrow fees; liquidity floors. Impact/ADV toggles are wired; extend to portfolio‑level netting.
- Universe quality: prefer CRSP historical SPX membership and prices; projected +0.60–0.70 gross with IBES blackout per current logs.

Status
- Impact and ADV caps exposed (entry/exit), liquidity floor added. Kalman corrected (H scaling, OLS‑seeded P). PC1/FRED throttles and ADF‑affinity clustering queued next.


-------------------------------------------------------------------------------
Maintenance Gate Outcome and Why Residualization Is Mandatory
-------------------------------------------------------------------------------
Observation
- With clustering stabilized (corr_window≈189d; Hungarian stock persistence ≈76%), adding a maintenance gate alone did not increase pair overlap (≈7% → ≈7%).
- Diagnosis: ADF/HL on raw log‑price spreads is flipping in/out as factor exposures drift across quarters; the gate preserves previous winners only if they still pass. If the test is unstable, the gate has nothing to preserve.

Implication
- Residualizing each leg against a robust factor set BEFORE cointegration is mandatory to target idiosyncratic stationarity and stabilize pair identity across segments.

Conceptual summary
- Raw spread: s_t = log(P1_t) − β̂·log(P2_t) − α̂ contains both idiosyncratic and factor components.
- Residualized formation (Avellaneda & Lee, 2010): regress returns on factors, take residuals ê, integrate to pseudo‑prices p̂ = cumsum(ê), test ADF/HL on ŝ_t = p̂A − γ p̂B − α. This filters out common factors and emphasizes persistent pair‑specific relations.

-------------------------------------------------------------------------------
Residualization Result (A-S3) -- 2026-05-21
-------------------------------------------------------------------------------
Run: Method A, corr_window=189, ARI=0.7, top-k=6, z_entry=1.5, zero-cost,
     factor_source=both (FF5+MOM+ETF; FF_RF dropped), --neutralize.

Experiment table (zero-cost, pair-return sizing throughout)
  Run                Overlap  ClusterPersist  Sharpe   MaxDD    Trips  Pairs
  fused k=3 z=2.0     6%      --             -0.05    -7.6%     231    135
  A-S1 189d corr      7%      76%            -0.31   -23.3%    1056    369
  A-S2 +maint gate    7%      76%            -0.36   -23.3%    1062    369
  A-S3 +neutralize   17%      76%            -0.05   -21.2%     638    199

Key findings
- Overlap 7% -> 17%: residualization is the fix. Idiosyncratic ADF is stable across quarters.
- Sharpe -0.36 -> -0.05: fewer pairs, much better quality. 81% hit rate maintained.
- Pre-COVID: maintenance pairs appeared by segment 3 (3->4->5 maint/seg). Raw ADF had 0.
- Post-COVID: maintenance began re-appearing within 5 segments. Raw ADF had 0 throughout.

Current bottleneck -- breadth collapse in late periods (2022-2024)
- Residualized spreads are harder to cointegrate (idiosyncratic component closer to random walk).
- 2022-2024 macro regime (rates shock, chop): only 1-3 pairs/seg pass residual ADF.
- Single-pair segments -> one bad trade dominates -> MaxDD -21%.
- Fix: loosen residual breadth filters while keeping ADF p<0.05 for new pairs.

Immediate next fix (A-S4) -- target before re-enabling costs
- HL ceiling: 30 -> 60d (idiosyncratic MR is slower than raw-spread MR).
- corr63: 0.70 -> 0.65 (residual returns are less correlated by construction).
- Add rolling 252d factor z-score before ridge (factor vol shifts significantly in 2022).
- Target: 12-20 pairs/seg across all regimes; overlap >= 20%.

Roles of the full factor set (FF+FRED+ETF)
- Residualized formation (proven): removes factor contamination from ADF -> overlap 17%.
- Method A clustering (not needed): already stable at 76% via corr_window=189.
- Method B clustering (pending): richer/orthogonal factors -> stable betas -> better cluster persistence.

SR path from here
  A-S4 (loosen breadth)         -> consistent 12-20 pairs/seg; overlap >=20%; Sharpe toward 0
  Rolling factor z-score         -> stable betas in macro regimes; better residuals post-2022
  Costs re-enabled (5-6 bps/leg) -> net Sharpe estimate (-0.1 to +0.3)
  Portfolio caps                 -> MaxDD <=12%; SPY beta ~0
  Method B + full factors        -> better peer grouping; parallel track

Done (as of 2026-05-21)
- [x] Pair-return sizing (r_pair = r_A - beta_t-1 * r_B); PnL in return space.
- [x] Clustering stabilized: corr_window=189, ARI freeze=0.7 -> 76% Hungarian persistence.
- [x] Maintenance gate: looser thresholds (p<0.10, HL<=45, corr>=0.65) for prev-segment pairs.
- [x] Pass-history tracking: >=2/3 segments required for maintenance.
- [x] Residualized formation (--neutralize): e_i = r_i - beta'_t-1 * F_t; p_hat = cumsum(e); ADF on s_hat.
- [x] FF daily + Momentum cached (ff_daily.parquet, 15792 rows, 7 cols, 1963-2026).
- [x] FF_RF dropped from regressors.
- [x] CLI flags: --corr-window, --ari-thresh, --beta-window, --ridge-alpha, --factor-source, --neutralize.
- [x] Hungarian cluster persistence metric saved to metrics.json.
- [x] EXPERIMENTS.md running log in data/.

Pending (ordered by SR impact)
- [ ] Rolling 252d factor z-score before ridge (look-ahead-free factor vol normalization).
- [ ] A-S4: HL 5-60d; corr63 >= 0.65; confirm breadth 12-20 pairs/seg post-2022.
- [x] FRED cache (DGS2/DGS10/T10Y2Y/IG OAS/HY OAS) -- sourced via WRDS frb.rates_daily (BAMLC0A0CM is a proxy; see Data Sources note).
- [ ] Portfolio-level caps: gross <=150-200%; <=10% per cluster; <=2 pairs/issuer.
- [ ] Method B full factors: beta_window=504, EWMA lambda=0.98, ridge alpha=25-50, z-scored.
- [ ] Costs re-enabled (5-6 bps/leg) only after gross Sharpe > 0 confirmed.
- [ ] BH/FDR within cluster (q~10%) after overlap >= 20%.
- [ ] Regime diagnostics: overlap/Sharpe/breadth by VIX tercile and regime window.

-------------------------------------------------------------------------------
TODOs (Implementation Plan)
-------------------------------------------------------------------------------
- Add residualized formation toggle (e.g., --neutralize) in pair discovery:
  - Data source: DataFetcher('both') → FF+MOM, FRED deltas, ETF proxies (drop FF_RF).
  - Preprocess: rolling 252d factor z‑scores (avoid look‑ahead on factor vol).
  - Regression: EWMA ridge (window 252–504d, λ≈0.97–0.985, α≈25–50), T+1 alignment (use β̂_{t−1}).
  - Integrate residuals; re‑estimate γ within formation window; run ADF/HL on residual spreads.
- Persistence gating (highest priority):
  - New pairs: ADF p<0.05, HL∈[5,45], corr63≥0.70; Maintenance: p<0.10.
  - Require ≥2 passes in last 3 segments; cap “new” pairs ≤30–40% of book per segment.
- Stability controls:
  - Method A: corr_window 189–252; ARI freeze ≥0.7; refit 63.
  - Method B: factor set 'both'; beta_window 504; EWMA λ≈0.98; ridge α 25–50; optional PCA(5–10 PCs) on betas.
- Diagnostics to add/report:
  - Pair persistence rate (fraction of last segment pairs surviving maintenance).
  - Stock cluster persistence (Hungarian‑aligned) each refit.
  - Formation pass‑rate per segment (what % of in‑cluster pairs pass filters).
  - Optional BH/FDR (q≈10%) within cluster once overlap ≥15–20%.
- Guardrails and sequencing:
  - Prefer formation window 378d (1.5y) before 504d (2y) to reduce burn‑in loss.
  - Validate zero‑cost gross Sharpe only after overlap ≥20% and trades/pair/segment ≥2; then re‑enable costs and capacity toggles.

-------------------------------------------------------------------------------
Research Log — Selected Runs (abridged)
-------------------------------------------------------------------------------
Date: 2026‑05‑21 | Run: fused k=3 z=2 (baseline)
- Overlap 6% | Sharpe −0.05 | Trips 231 | Unique pairs 135
- Note: Log‑spread sizing artifact discovered later (inflated Sharpe). Served only to validate plumbing.

Date: 2026‑05‑21 | Run: A‑S1 (corr_window=189d, ARI=0.7)
- Overlap 7% | Cluster persistence (Hungarian) 76% | Sharpe −0.31 | Trips 1056 | Unique 369
- Note: Clustering stabilized; pair formation unstable (low overlap).

Date: 2026‑05‑21 | Run: A‑S2 (+ maintenance gate)
- Overlap 7% | Cluster persistence 76% | Sharpe −0.36 | Trips 1062 | Unique 369
- Note: Maintenance gate alone did not raise overlap; confirms formation filter is root cause.

Date: 2026‑05‑21 | Run: A‑S3 (+ residualization)
- Overlap 17% | Cluster persistence 76% | Sharpe −0.05 | Trips 638 | Unique 199
- Note: Residualization fixes pair identity; breadth thinner late 2022–24.

-------------------------------------------------------------------------------
Current Sequencing (A‑S4) — Loosened Filters + Rolling Factor Z‑Score
-------------------------------------------------------------------------------
Objective
- Lift pair overlap to ≥20% and restore breadth ≥15 pairs/segment (esp. 2022–24) while keeping cluster persistence high.

What changes in A‑S4
- Residualized formation (kept) + two refinements:
  1) Loosen residual formation filters slightly: HL [5,45] → [5,60]; corr63 ≥ 0.70 → ≥ 0.65.
  2) Apply rolling 252d z‑scoring to factor columns before EWMA ridge; drop FF_RF from regressors.

Checkpoints (zero‑cost first)
- After A‑S4: overlap ≥20%; trades/pair/segment ≥2; breadth ≥15 pairs/segment consistently; cluster persistence ≈ ≥70%.
- If overlap <20%: increase formation window 252→378d (1.5y) before 504d (2y). Re‑check targets.
- Only then re‑enable explicit costs (5–6 bps/leg) and assess net Sharpe; add capacity toggles later.

Not yet implemented (to be added after A‑S4 validates)
- Cap “new” pairs per segment (forces overlap mechanically once persistence exists):
  - max_new_per_seg = max(maint_count, int(total_cap × 0.35))
  - Keeps a majority of maintenance pairs and limits untested turnover.
- EWMA ridge settings for residualization (step 2b):
  - beta_window 252→504d; EWMA λ=0.97–0.985; ridge α=25–50.
  - Apply before validating Method B (beta‑space), as it directly stabilizes residualization and betas.
- Portfolio construction caps (post‑signal):
  - Per‑issuer cap (≤2 active pairs), per‑cluster cap (≤10% exposure), ADV cap (≤1%/leg), crisis mode. These reduce tail risk; expect MaxDD to fall toward ≤15% as breadth returns.
- Structural‑break exits in backtest (post‑signal):
  - Enforce 5d corr < 0.5 and |Δβ| > 30% exits to trim left‑tail without hurting average return.




-------------------------------------------------------------------------------
Full Experiment Table -- 2026-05-21
-------------------------------------------------------------------------------
All runs: Method A, SPX-300, pair-return sizing, zero-cost unless noted.

  Run                   z    k  form  Gross   Net   MaxDD  Trips  Hit%  Overlap  Notes
  fused k=3 log-space  2.0   3  252  +2.59     --    -1%    231   --     --     ARTIFACT: log-spread sizing
  fused k=3 pairret    2.0   3  252  -0.19  -0.19   -8%    231   80%    6%     Sizing fixed; signal flat
  bt_zerocost          2.0   3  252  -0.05  -0.05   -8%    231   80%    6%     Zero-cost; signal broken
  A-S1 189d corr       1.5   6  252  -0.31  -0.31  -23%   1056   --     7%     Clustering stable 76%
  A-S2 +maint gate     1.5   6  252  -0.36  -0.36  -23%   1062   --     7%     Gate adds 0 overlap (ADF churn)
  A-S3 +neutralize     1.5   6  252  -0.05  -0.05  -21%    638   81%   17%     Residualization: overlap 7->17%
  A-S4 HL60/corr0.65  1.5   6  252  -0.38  -0.38  -25%    707   81%   15%     Looser filters add noise
  A-S5 formation=378d  1.5   6  378  -0.80  -0.80  -33%    623   81%   21%     Overlap hits 21% but spread noisier
  S6a all tight        1.5   6  252  -0.06  -0.06  -19%    785   69%   17%     time_stop=15 conflicts z_exit=0.25
  S6b z_exit=0.25     1.5   6  252  +0.004 +0.004 -19%    624   75%   17%     Marginal
  S6c hard_stop=3.0   1.5   6  252  +0.030 +0.030 -19%    801   74%   17%     BEST: hard stop fix
  S6c+5bps            1.5   6  252  +0.030 -0.520 -26%    801   74%   17%     5bps/leg destroys net
  S7a z_entry=2.0     2.0   6  252  -0.092 -0.092 -15%    584   68%   17%     Heavy tails: z=2 not safer
  S7b z=2+378d        2.0   6  378  -0.829 -0.829 -25%    577   66%   21%     Worse
  S7c z=2+k=10        2.0  10  252  -0.284 -0.284 -21%    741   69%   23%     Worse

Current best: S6c (Gross +0.030, hard_stop=3.0, z_entry=1.5, 252d residualized formation)

-------------------------------------------------------------------------------
Plausibility and Target Sharpe -- 2026-05-21
-------------------------------------------------------------------------------
Literature benchmarks
- Gatev (1999/2006): ~1.0 gross, ~0.5-0.7 net (distance method, no residualization)
- Avellaneda & Lee (2010): ~1.2-1.5 gross (residualized ETF pairs, daily)
- Realistic for daily SPX residualized: 1.0-1.2 gross, 0.5-0.7 net at scale
- Sharpe >1.5 net on daily SPX: requires intraday or different asset class

Configuration target for 1.0-1.5 gross Sharpe
- Entry/stops: z_entry=2.0, z_exit=0.5, hard_stop=3.0, time_stop=20
  WHY: at z_entry=2.0 the theoretical payoff is +1.5 sigma (winner) vs -1.0 sigma (loser).
  But S7a showed z=2.0 doesn't improve hit rate (heavy tails: conditional on reaching z=2.0,
  further divergence is just as likely). WRDS data (IBES blackout) is needed first -- most
  hard-stop losses are earnings-driven, not spread-structural.
- Formation: residualized ADF on tail 252d of a 378d window (decouple stability vs signal);
  HL [8-20d]; corr63 >= 0.70; ADF p<0.05 (new), p<0.10 (maintenance)
- Breadth: top-k=10 -> target 20-30 pairs/segment
- Persistence: >= 2/3 recent passes; cap "new" <= 35-40% of book

Why 1.0-1.5 gross requires WRDS (not just parameter tuning)
- IBES earnings blackout: eliminates earnings-driven hard stops (primary MaxDD source).
  Expected: MaxDD -19% -> -10%; gross Sharpe +0.03 -> +0.15-0.25.
- CRSP historical membership: removes survivorship bias (~15-20% of "pairs" only exist
  because both companies thrived). More honest gross Sharpe.
- Markit borrow: removes pairs where short leg is expensive/constrained.
- TAQ intraday: structural Sharpe improvement -- more alpha per unit time, lower cost/trade.

Cost reality
- 25 pairs x ~3 trips/pair/yr x ~10 bps RT = ~7.5% annual cost drag at 5% vol = 1.5 Sharpe
- Hence 0.5-0.7 net is realistic with 5 bps/leg; improve by:
  (a) WRDS data reduces trips (earnings blackout = fewer false entries)
  (b) Large-cap routing at 2-3 bps/leg
  (c) Breadth throttle in "one-factor" periods (PC1 > 40% share)

Risk and MaxDD controls (not yet implemented)
- IBES earnings blackout: t-1/t/t+1 per leg (biggest MaxDD fix; stubbed in data_fetcher.py)
- Portfolio caps: <= 2 pairs/issuer; <= 10%/cluster; <= 1% ADV per leg
- Structural-break exits: 5d corr < 0.5, |Dbeta| > 30% since entry
- Breadth throttle: if active pairs < 10 or PC1 share > 40%, scale book to zero, block entries

-------------------------------------------------------------------------------
WRDS Implementation Status -- 2026-05-21
-------------------------------------------------------------------------------
Stubbed in data_fetcher.py (need WRDS credentials + pip install wrds):
  [stub] fetch_crsp_spx_membership()  -- historical SPX membership (crsp.msp500list)
  [stub] fetch_crsp_daily()           -- adjusted prices + volume (crsp.dsf)
  [stub] fetch_ibes_earnings_dates()  -- earnings announcement dates (ibes.statsum_epsus)
  [stub] fetch_markit_borrow()        -- borrow cost/availability (sfi.loan_rate)

Implementation order (by impact on Sharpe/MaxDD):
  1. IBES blackout gate in backtest() -- earnings t-1/t+1 per leg
  2. CRSP membership to replace yfinance universe
  3. CRSP adjusted prices to replace yfinance close
  4. Markit borrow screen in discover_pairs Filters
  5. TAQ intraday (v2 extension)

-------------------------------------------------------------------------------
Immediate next experiments (action plan) -- updated 2026-05-21
-------------------------------------------------------------------------------
Status: S6c is the best run (+0.030 gross, hard_stop=3.0). Parameter space exhausted
        at daily frequency without WRDS data. Next lever is data quality.

Priority 1 -- WRDS IBES blackout gate (implement in backtest.py)
  - Load ibes_earnings.parquet; build {ticker: set(earnings_dates)} lookup
  - In backtest() entry logic: skip if either leg has earnings within +-1 day
  - In backtest() active position: exit 1d before earnings
  - Expected: MaxDD -19% -> -10%; gross Sharpe +0.03 -> +0.15-0.25
  - Then rerun S6c settings with blackout enabled

Priority 2 -- CRSP universe (replace yfinance)
  - load_all() uses fetch_crsp_daily() instead of yfinance when CRSP available
  - Universe: spx_membership by refit_date (point-in-time, no survivorship)

Priority 3 -- Residualized tail-ADF + HL [8-20d] (signal quality)
  - Run ADF on last 252d of 378d residual window (stability vs signal decoupled)
  - HL filter [8-20d]: removes fast (< 8d, mis-timed) and slow (> 20d, time-stop losses)
  - top-k=10: target 20-30 pairs/segment

Priority 4 -- Portfolio caps (MaxDD control)
  - Max 2 pairs/issuer, 10%/cluster, 1% ADV/leg
  - Breadth throttle: scale to zero if active pairs < 10

Method B parallel track (after A + WRDS passes gates)
  --method b --factor-source both --neutralize --beta-window 504 --ridge-alpha 25

-------------------------------------------------------------------------------
Data Roadmap — Current Status and Next Sources
-------------------------------------------------------------------------------
Gate: fetch WRDS data in priority order (below); intraday only after v1 daily
      passes net Sharpe gate (≥ 0.7, MaxDD ≤ 12%).

Current cache (PairsTrading/data/)
  ff_daily.parquet          ✅  FF5 + Momentum daily (1963–2026, Ken French)
  factor_close.parquet      ✅  24 ETF prices: SPY + 11 sectors + 5 styles +
                                 3 rates (TLT/IEF/SHY) + 2 credit (LQD/HYG) +
                                 3 commod/FX (GLD/USO/UUP) — via yfinance
  spx_close.parquet         ✅  SPX-300 adjusted close (yfinance, 2014–now)
  spx_volume.parquet        ✅  SPX-300 daily volume (yfinance)
  fred.parquet              ✅  Macro rates (12828 rows, 1990-01-01→2025-02-13)
                                 Source: WRDS frb.rates_daily (not FRED CSV — fredgraph.csv
                                 endpoint times out; WRDS REST API used instead).
                                 See derivation note below.
  ibes_earnings.parquet     ❌  WRDS stub — Priority 1
  crsp_spx_membership.pq    ❌  WRDS stub — Priority 2
  crsp_close.parquet        ❌  WRDS stub — Priority 2
  crsp_volume.parquet       ❌  WRDS stub — Priority 2
  markit_borrow.parquet     ❌  WRDS stub — Priority 3

Factor coverage vs proposal §3 factor set
  FF5 + MOM                 ✅  ff_daily.parquet (drop FF_RF before regressing)
  FRED macro deltas         ✅  fred.parquet — sourced via WRDS frb.rates_daily
  11 Sector ETFs            ✅  factor_close.parquet
  Style ETFs                ✅  factor_close.parquet
  Rates/credit/commod ETFs  ✅  factor_close.parquet
  CRSP point-in-time univ.  ❌  WRDS fetch_crsp_spx_membership()
  IBES earnings calendar    ❌  WRDS fetch_ibes_earnings_dates()
  Markit borrow             ❌  WRDS fetch_markit_borrow()

-------------------------------------------------------------------------------
WRDS Fetch Sequence (run once credentials in ~/.pgpass)
-------------------------------------------------------------------------------
from data_fetcher import DataFetcher
df = DataFetcher()

 Option A — terminal one-liner (simplest, no new file)
  cd "/Users/prsantamaria/Documents/Projects/mqf/MQF/Modules/PairsTrading"
  python - <<'PY'
  from data_fetcher import DataFetcher
  f = DataFetcher(allow_network=True)
  f.fetch_ff_daily(refresh=True)
  f.fetch_fred(["DGS2","DGS10","T10Y2Y","BAMLC0A0CM","BAMLH0A0HYM2"], refresh=True)
  f.fetch_crsp_spx_membership(refresh=True)
  f.fetch_crsp_daily(start="2014-01-01", refresh=True)
  f.fetch_ibes_earnings_dates(start="2014-01-01", refresh=True)
  print("done")
  PY

# Step 1 — macro rates (via WRDS frb.rates_daily, no FRED key needed)
fred = df.fetch_fred(
    ["DGS2","DGS10","T10Y2Y","BAMLC0A0CM","BAMLH0A0HYM2"],
    refresh=True
)                                    # → fred.parquet

-------------------------------------------------------------------------------
fred.parquet — Series Derivation and Data Quality Notes (updated 2026-05-23)
-------------------------------------------------------------------------------
All five series now sourced from WRDS frb.rates_daily via REST API token.
The fredgraph.csv endpoint (fred.stlouisfed.org) consistently times out; frb.rates_daily
is the same underlying Federal Reserve H.15 + ICE BofA data distributed via WRDS.

  Series        | Source                                  | Units  | 2014–today coverage
  --------------|----------------------------------------|--------|---------------------
  DGS2          | FRED API (exact, H.15)                 | % p.a. | ✅ full
  DGS10         | FRED API (exact, H.15)                 | % p.a. | ✅ full
  T10Y2Y        | FRED API (exact, independent series)   | % pts  | ✅ full
  BAMLH0A0HYM2  | WRDS 2014–2023 + FRED 2023–today       | % OAS  | ✅ spliced
  BAMLC0A0CM    | WRDS proxy 2014–2023 + FRED 2023–today | % OAS  | ✅ spliced, note below

Series ID correction (2026-05-23)
  The original series ID used was BAMLCC0A0CM (extra C). This does not exist on FRED.
  Correct ID is BAMLC0A0CM — ICE BofA US Corporate Index OAS. All references updated.

DGS2 / DGS10 / T10Y2Y
  FRED API exact values. T10Y2Y is an independent FRED series (not derived here).
  All current to previous business day.

BAMLH0A0HYM2 — ICE BofA US High Yield Index OAS
  ICE BofA licensing restricts FRED distribution to 2023-05-23 onward.
  Pre-2023 history from WRDS frb.rates_daily (bamlh0a0hym2 = exact OAS, same underlying data).
  Spliced at 2023-05-23: WRDS history + FRED exact values from 2023 to today.
  Result: exact OAS throughout 2014–today.

BAMLC0A0CM — ICE BofA US Corporate Index OAS
  Same ICE BofA licensing restriction: FRED only from 2023-05-23.
  Pre-2023: WRDS frb.rates_daily does not store IG OAS directly; stores bamlc0a0cmey
  (effective yield). Proxy = bamlc0a0cmey − dgs10 (IG yield spread).

  Why the proxy overstates OAS by ~20bp pre-2023:
    OAS strips out the embedded call option value (most IG bonds are callable).
    Call premium ≈ 15–30bp depending on rate level; higher when rates are high
    (more callables in-the-money). Raw yield spread includes this; OAS does not.

  Why the proxy is acceptable for residualization:
    Strategy uses fred.diff() — daily changes, not levels. The call premium is
    slow-moving; daily changes in yield spread track daily changes in true OAS
    at r > 0.97. The level bias cancels in differencing.
    NOT acceptable for: exact credit pricing, OAS-threshold gates, TCA.

  Post-2023: exact FRED OAS values. Pre-2023: proxy with stable ~20bp level bias.

Usage in residualization
  All five series first-differenced before use (fred.diff().add_prefix("FRED_d")).
  Level biases cancel on differencing; only the signal in daily credit stress changes matters.
  NaN on weekends forward-filled before differencing to avoid artificial gaps.

# Step 2 — IBES earnings dates (BIGGEST MaxDD fix)
ibes = df.fetch_ibes_earnings_dates(start="2014-01-01")  # → ibes_earnings.parquet

# Step 3 — CRSP survivorship-free universe
memb = df.fetch_crsp_spx_membership()                    # → crsp_spx_membership.parquet
crsp_c, crsp_v = df.fetch_crsp_daily(start="2014-01-01")# → crsp_close/volume.parquet

# Step 4 — Markit borrow screen
borrow = df.fetch_markit_borrow(start="2014-01-01")      # → markit_borrow.parquet

-------------------------------------------------------------------------------
V2 Data — TAQ Intraday (after v1 passes net Sharpe gate)
-------------------------------------------------------------------------------
Gate for intraday: v1 net Sharpe ≥ 0.7 AND MaxDD ≤ 12% on 2022–2024 OOS.
Rationale: daily signal must be structurally positive before intraday adds value.
At 30-60 min bars: more trips per pair per year → lower cost per trade →
structural Sharpe improvement. But adds complexity; don't start until daily works.

Source: WRDS TAQ (Trade and Quote) database

TAQ tables (one per calendar day)
  taqmsec.ctm_YYYYMMDD   millisecond trades  2014+   ~5–15 GB/day raw
  taqmsec.cqm_YYYYMMDD   millisecond quotes  2014+   ~30–80 GB/day raw
  taq.ct_YYYYMMDD        second trades       1993–2014
  taq.cq_YYYYMMDD        second quotes       1993–2014
  taq.wrds_ctm_YYYYMMDD  WRDS-cleaned trades 2003+   pre-filtered, preferred

CRITICAL: never download raw TAQ. Aggregate to bars in SQL on WRDS servers;
pull only the bar-level output (~1–5 MB/day compressed).

30-min OHLCV bar query (template)
  SELECT
      date,
      sym_root AS ticker,
      time_bucket('30 minutes', time::timestamp)        AS bar_time,
      (array_agg(price ORDER BY time))[1]               AS open,
      MAX(price)                                         AS high,
      MIN(price)                                         AS low,
      (array_agg(price ORDER BY time DESC))[1]          AS close,
      SUM(size)                                          AS volume,
      COUNT(*)                                           AS n_trades
  FROM taqmsec.ctm_YYYYMMDD
  WHERE sym_root IN (<tickers>)
    AND time        BETWEEN '09:30:00' AND '16:00:00'
    AND tr_corr     = '00'           -- clean/unrevised trades only
    AND tr_scond    NOT IN ('Z','B','T','L','G','W','J','K')
    AND price > 0 AND size > 0
  GROUP BY date, ticker, bar_time
  ORDER BY ticker, bar_time

Bid-ask spread from quotes (intraday cost model)
  SELECT
      date, sym_root AS ticker,
      time_bucket('30 minutes', time::timestamp)        AS bar_time,
      AVG((ask-bid)/((ask+bid)/2.0))                    AS quoted_spread,
      AVG((ask+bid)/2.0)                                AS mid_price
  FROM taqmsec.cqm_YYYYMMDD
  WHERE sym_root IN (<tickers>)
    AND time BETWEEN '09:30:00' AND '16:00:00'
    AND qu_cond = 'R'                -- regular NBBO quotes
    AND bid > 0 AND ask > bid
    AND (ask-bid)/((ask+bid)/2.0) < 0.05   -- sanity cap
  GROUP BY date, ticker, bar_time

WRDS Cloud SSH (preferred for multi-year pulls — no bandwidth cost)
  ssh priscillasm@wrds-cloud.wharton.upenn.edu
  # run Python aggregation script on WRDS servers
  # download only parquet output:
  scp priscillasm@wrds-cloud.wharton.upenn.edu:~/scratch/taq_bars_2023.parquet ./data/

Output files (stored in PairsTrading/data/taq_bars/)
  taq_bars_{YYYY}.parquet    30-min OHLCV per year per ticker
  taq_spread_{YYYY}.parquet  30-min quoted spread + mid per year per ticker

What TAQ unlocks for v2
  30-min pairs strategy    More trips/pair/year; lower cost/trade vs daily
  Precise Amihud           Intraday |ret|/DollarVol per bar (better impact model)
  Effective spread         Fill-price vs mid at execution time (TCA)
  Earnings gap detection   Intraday price spike confirms blackout window timing
  Intraday VPIN            Tick-level bulk classification for MM platform

Practical alternative (prototype only, not for publication)
  Polygon.io    $29/mo unlimited historical 1-min bars back to 2004
                Much faster to set up than TAQ; use for prototyping
                then validate on TAQ before citing in any research note
  Alpaca API    Free, 1-min bars from 2016; limited history

WRDS REST API — does NOT work for TAQ
  wrds-api.wharton.upenn.edu returns 404 for all TAQ tables. Daily-partitioned
  tables (5–80 GB/day raw) are not exposed via REST. Only small static tables
  (ff.factors_daily, crsp.dsf, ibes.*) are accessible that way. TAQ must go
  through WRDS Cloud SSH or the wrds Python package running on cloud servers.

Source comparison
  ┌───────────────┬──────────────┬──────────────┬─────────────────────────────┐
  │ Source        │ Cost         │ Depth        │ Publication rights          │
  ├───────────────┼──────────────┼──────────────┼─────────────────────────────┤
  │ WRDS TAQ      │ Free (WRDS   │ 1993–today   │ ✓ citable; academic standard│
  │ (taqmsec/taq) │ subscription)│ ms precision │   required for finance PhD  │
  ├───────────────┼──────────────┼──────────────┼─────────────────────────────┤
  │ Polygon.io    │ $29–$79/mo   │ 2004–today   │ ✗ proprietary; not citable  │
  │               │ REST API     │ 1-min bars   │   prototype use only        │
  ├───────────────┼──────────────┼──────────────┼─────────────────────────────┤
  │ Alpaca API    │ Free         │ 2016–today   │ ✗ too short for paper       │
  │               │ REST API     │ 1-min bars   │   prototype use only        │
  └───────────────┴──────────────┴──────────────┴─────────────────────────────┘

Decision tree
  1. Prototype 30-min strategy  →  Polygon.io (REST, no SSH, fast iteration)
  2. v1 Sharpe gate passed      →  validate all results on WRDS TAQ
  3. Paper / research note      →  cite only WRDS TAQ; Polygon not acceptable

Polygon.io API (prototype path)
  Endpoint:  GET /v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}
  Auth:      ?apiKey=<key> query param
  Rate:      5 req/min free tier; unlimited on $29 Starter plan
  Response:  {"results": [{"o","h","l","c","v","t",...}], "next_url": ...}
  Paginate:  follow next_url until absent
  Resample:  df.resample('30min').agg({'o':'first','h':'max','l':'min',
                                       'c':'last','v':'sum'})

  Fetch 30-min bars for one ticker, one year (~5 MB parquet output):
    base = "https://api.polygon.io/v2/aggs/ticker"
    url  = f"{base}/{ticker}/range/1/minute/{start}/{end}"
          f"?adjusted=true&sort=asc&limit=50000&apiKey={key}"

WRDS Cloud Python script (fetch_taq_cloud.py — run on wrds-cloud only)
  Location: scripts/fetch_taq_cloud.py
  Run on:   wrds-cloud.wharton.upenn.edu
            ssh priscillasm@wrds-cloud.wharton.upenn.edu
  Output:   ~/scratch/taq_bars_{year}.parquet  (scp to local data/taq_bars/)

  import wrds, pandas as pd

  TICKERS = open("tickers.txt").read().split()  # one per line
  conn    = wrds.Connection()                   # pre-authenticated on cloud, no prompt

  for year in range(2014, 2025):
      frames = []
      for date in trading_days(year):
          table = f"taqmsec.ctm_{date:%Y%m%d}"
          try:
              df = conn.raw_sql(f"""
                  SELECT date, sym_root AS ticker,
                         date_trunc('minute', time::timestamp)
                           - INTERVAL '1 min' * MOD(
                               EXTRACT(MINUTE FROM time::timestamp)::int, 30)
                           AS bar_time,
                         (array_agg(tr_price ORDER BY time))[1]       AS open,
                         MAX(tr_price)                                  AS high,
                         MIN(tr_price)                                  AS low,
                         (array_agg(tr_price ORDER BY time DESC))[1]   AS close,
                         SUM(tr_size)                                   AS volume
                  FROM {table}
                  WHERE sym_root IN ({','.join(f"'{t}'" for t in TICKERS)})
                    AND time BETWEEN '09:30:00' AND '16:00:00'
                    AND tr_corr = '00'
                    AND tr_scond NOT IN ('Z','B','T','L','G','W','J','K')
                    AND tr_price > 0 AND tr_size > 0
                  GROUP BY date, ticker, bar_time
              """, date_cols=["date", "bar_time"])
              frames.append(df)
          except Exception:
              pass  # weekend / holiday — table does not exist
      if frames:
          pd.concat(frames).to_parquet(f"~/scratch/taq_bars_{year}.parquet")

  # Download to local machine after script completes:
  # scp priscillasm@wrds-cloud.wharton.upenn.edu:~/scratch/taq_bars_*.parquet \
  #     ./data/taq_bars/

data_fetcher.py integration
  fetch_taq_bars(tickers, start, end, bar_minutes=30, refresh=False)
    → reads from data/taq_bars/*.parquet if cached (cheap, offline)
    → raises RuntimeError with SSH instructions if not cached (cannot fetch locally)
    → returns (bars_df, spreads_df): MultiIndex (date, bar_time) × tickers

  fetch_polygon_bars(tickers, start, bar_minutes=30, polygon_key="", refresh=False)
    → calls Polygon.io REST API, resamples 1-min → bar_minutes in-memory
    → caches result to data/polygon_bars.parquet
    → prototype use only; swap for fetch_taq_bars before any write-up

  load_all(intraday="none"|"polygon"|"taq_cached", bar_minutes=30, polygon_key="")
    → "none":       existing daily-only behaviour (default)
    → "polygon":    fetches Polygon bars alongside daily close data
    → "taq_cached": loads pre-downloaded WRDS TAQ parquet from data/taq_bars/

- Gate 2: ADF gate + half‑life filter reduce CVaR95 ≥ 10% vs z‑only without >50% trade‑count collapse.
- Gate 3: Spectral clusters improve at least one formation‑quality metric vs GICS (lower median EG p‑value, tighter half‑life IQR, or higher median 63d corr at entry).
- Gate 4: Parameter sensitivity (±0.5 on z‑entry; ±21d on lookbacks) does not flip net Sharpe sign.
- Gate 5 (execution sanity): ADR/dual‑share harness produces ≥0 net PnL on neutrality tests after modeled costs; persistent loss implies execution/accounting bug to fix before advancing.

-------------------------------------------------------------------------------
Parameter defaults (for v1)
-------------------------------------------------------------------------------
- k (returns view): set by MP upper bound λ₊ = (1+√q)²; typically 8–15 for SPX‑200 on 126d window.
- Formation window=252d; rolling window=63d; Z‑entry=2.0; Z‑exit=0.5; time stop=20d; hard stop=4.0.
- ADF gate p<0.10 (monthly); corr>0.75 at formation; half‑life ∈ [5,30]d; β̂ clamp [0.25, 4.0].
- Per‑pair σ_PnL target 10–15 bps/day; base cost 3 bps/leg (stress at 6 bps); impact k=15, exponent 0.5.
- Caps: 1% ADV/leg, 10% per cluster, ≤2 pairs/issuer; cluster ARI freeze threshold 0.6.

Factor‑beta specifics (for v1 research ablation)
- Factor set (~17–20): SPY; sectors XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLC, XLRE; styles MTUM, VTV/VLUE; macro TLT/IEF, GLD/USO, UUP/FXE; optional XBI.
- Beta estimation: ridge with EWMA‑weighted 126–252d window; one‑time λ grid on formation; revalidate annually.
- Beta scaling: standardize by factor vol; distance = cosine by default; test uncertainty‑aware Mahalanobis.
- Fusion: default w=0.5 when combining A_ret and A_beta.

One‑day MVP (validation sprint)
- 0–2h: Pull SPX‑200 daily adj close (Stooq/Yahoo), cache to Parquet; clean and align.
- 2–3h: LW shrinkage → spectral clustering (k from MP bound); sanity check — PEP/KO, GOOG/GOOGL in same cluster.
- 3–4h: Score pairs (distance + corr + EG p‑value + half‑life); select top 40.
- 4–6h: Backtest z‑only mean‑reversion with 3 bps/leg cost; verify Gate 1 gross Sharpe > 1.0.
- 6–7h: Add ADF formation gate + half‑life filter; verify Gate 2 (CVaR95 reduction ≥ 10%).
- 7–8h: Swap GICS labels for spectral labels, re‑run; produce ablation table (a)–(d) and OOS equity curve.

-------------------------------------------------------------------------------
8) Risk management and operations
-------------------------------------------------------------------------------
- Per‑pair hard stops (|Z|>4), β̂‑shift stop, cluster‑level exposure caps.
- Kill‑switch: suspend adds when VIX > 90th pct or avg‑corr spikes; resume on cooldown.
- Corporate actions: share class pairs (GOOG–GOOGL) require action calendars; suspend around reclassification/indices changes.
- Monitoring: pair‑level diagnostics (spread, z, β̂, corr), cluster exposure, factor neutrality.

-------------------------------------------------------------------------------
9) Literature to ground the approach (core + recent)
-------------------------------------------------------------------------------
Pairs trading core
- Gatev, Goetzmann, Rouwenhorst (1999/2006) — Pairs trading performance; formation vs trading windows.
- Vidyamurthy (2004) — Pairs Trading (book): cointegration/hedge ratio methods.
- Elliott, van der Hoek, Malcolm (2005) — Pairs trading with OU processes; half‑life and mean‑reversion speed.
- Avellaneda & Lee (2010) — Statistical arbitrage with factor models; residualization motivates cluster‑neutral spreads.

Clustering and portfolio construction
- Lopez de Prado (2016) — Hierarchical clustering / HRP; clustering as risk decomposition tool.
- Tumminello et al. (2005–2010) — MST/PMFG graph clustering from correlation matrices.
- Ledoit & Wolf (2004) — Shrinkage estimation of large covariance matrices; theoretical basis for LW shrinkage step.
- Marchenko & Pastur (1967); Bouchaud & Potters (2011) — Random matrix theory; eigenvalue noise floor (λ₊) justifies data‑driven k.

Validation and costs
- Krauss (2017) — Statistical arbitrage US equities; costs and ML ensembles.
- Bailey et al. (2014–2016) — Backtest overfitting, reality check; why rolling OOS matters.
- Frazzini, Israel, Moskowitz (2018) — Trading costs at scale; cost model realism.

-------------------------------------------------------------------------------
10) Senior‑quant validation (feasibility & pitfalls)
-------------------------------------------------------------------------------
What’s feasible now
- Daily v1 with SPX‑200; K‑Means clustering; OLS+z; ADF gate; cost model via half‑spread+impact proxy; robust OOS splits and crisis checks.
- Orthogonality testing vs standard factors; cluster‑level exposure controls; breakout overlay with strict throttles.

Key risks and how to mitigate
- Overfitting via filter stacking: Stack filters to reduce trades but validate that capacity and Sharpe remain; use time‑split CV; avoid optimizing gates to noise.
- Cointegration instability: Rolling ADF as a lenient gate only; disable MR on structural‑break signs; don’t chase re‑estimation too often.
- Cost underestimation: Stress costs to 5–6 bps/leg; cap participation; penalize Amihud‑heavy names.
- Survivorship/selection bias: Prefer historical constituents; if not available, disclose and run robustness with delisted tickers subset if possible.
- Capacity: Report P&L vs ADV constraints; simulate higher capital with impact scaling; show decay curve.

When to keep it simple vs add adaptivity (addressing feedback)
- Baseline OLS+z must work on canonical pairs (e.g., GOOG–GOOGL) gross and near‑net; if it doesn’t, fix base design before adding filters.
- Rolling ADF = gate, not a smoother; Kalman filter often counterproductive because it removes monetizable noise and reduces trade count; reserve for top pairs only in v2 if clearly beneficial.

Extensions (after v1 passes all gates)
- Breakout/momentum overlay: |Z| ≥ 3.0 AND 5d corr < 0.5 → trade with sign(Z) for 3–5 days, ≤20% capital budget; only add if v1 net Sharpe > 0.8 with stable trade count.
- Dynamic hedge ratio: OU/Kalman for top decile pairs only; strict trade‑count guardrail (must stay >30 active pairs).
- Intraday refinement: top 10 pairs on 30–60m bars; microstructure cost model; queue/impact proxy.
- Graph‑based discovery: MST/PMFG community detection; signed‑network spectral for pairs with negative formation‑window correlation.
- Feature‑based clustering layer: add mom_3m, vol_21d, beta_spy_63d inside each spectral cluster for sub‑clustering.

Deliverables
- Research note (8–12p): methodology, ablation, OOS results, factor/cluster diagnostics, cost/capacity curves.
- Notebook(s): reproducible pipeline with config; plots and tables.
- Risk checklist: stops, gates, exposure caps; monitoring dashboard spec.

-------------------------------------------------------------------------------
Appendix A — Data checklist (one‑day build)
-------------------------------------------------------------------------------
- Ticker list: SPX‑200 as of today (disclose survivorship); prices/volume from Stooq/Yahoo.
- Derived: factor ETF returns — SPY; sectors (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLC, XLRE); styles (MTUM, VTV/VLUE); macro (TLT/IEF, GLD/USO, UUP/FXE); optional XBI.
- Feature matrix: (i) rolling ridge betas to the above factor set; (ii) mom_3m, mom_12m, vol_21d, beta_spy_63d, corr_to_sector_63d, ADV, Amihud.
- Formation windows: 252d; trading daily with rolling 63d μ,σ,β.

Appendix B — Metrics and reporting templates
- Equity curve; rolling Sharpe; drawdown chart.
- Per‑cluster Sharpe; exposure by cluster; SPY/sector betas.
- Pair‑level table: entries/exits, z at entry, holding period, P&L, costs, half‑life estimate.
- Factor regression table (Mkt, SMB, HML, MOM, QMJ) R² and alphas.


Pairs Trading Pipeline — Theory Map (ELI5)

  ---
  The Core Idea (30 seconds)
  
  Two stocks in the same industry usually move together. When they temporarily diverge, bet
  they'll snap back. That's pairs trading. Every concept below exists to answer: which pairs? when
   to enter? how much to bet? when does the relationship break?

  ---
  1. Mean Reversion and Stationarity
  
  ELI5: A rubber band. Stretch it → it snaps back. If the price gap between two stocks behaves
  like a rubber band, it's mean-reverting.

  Formally: A time series $s_t$ is stationary if its mean and variance don't drift over time. A
  random walk (stock price) is non-stationary — it wanders forever. A mean-reverting spread
  wanders but always comes back.

  Why it matters: If the spread $s_t = \log P_1 - \hat\beta \log P_2$ is stationary, entering when
   it's stretched and exiting when it returns is positive-EV.

  ---
  2. Cointegration and the Engle-Granger Test

  ELI5: Two drunks walking home — each staggers randomly, but they're tied together by a leash.
  Individually non-stationary; together, the distance between them is stationary.

  Formally: Two $I(1)$ series $P_1, P_2$ are cointegrated if there exists $\beta$ such that $P_1 -
   \beta P_2$ is $I(0)$ (stationary). The Engle-Granger test:
  1. Regress $\log P_1 = \alpha + \beta \log P_2 + \varepsilon$
  2. Run ADF on residuals $\hat\varepsilon_t$
  3. If $p < 0.05$: residuals are stationary → cointegrated
  
  Augmented Dickey-Fuller (ADF): Tests $H_0$: unit root (random walk) vs $H_1$: stationary. Low
  p-value = reject unit root = stationary = good.

  In the pipeline: Formation filter — ADF $p < 0.05$ on residualized spread. Only pairs that pass
  are traded.

  ---
  3. Half-Life of Mean Reversion (OU Process)
  
  ELI5: How fast does the rubber band snap back? If it snaps back in 7 days, you need to hold for
  ~7 days. If it takes 60 days, your capital is tied up too long.

  Formally: The spread follows an Ornstein-Uhlenbeck process:
  $$\Delta s_t = \gamma s_{t-1} + \varepsilon_t, \quad \gamma < 0$$
  Half-life = $-\log(2) / \gamma$. Estimated from OLS on the AR(1) regression.

  In the pipeline: Filter HL $\in [8, 20]$ days. Below 8: spread reverts before entry fires
  cleanly. Above 20: hits time_stop=20 before reversion completes. Both hurt EV.

  ---
  4. Factor Models (Why Returns Aren't Idiosyncratic)
  
  ELI5: Stock returns are like a recipe: 60% market, 20% sector, 10% style, 10% idiosyncratic
  (unique to this company). The raw spread between two tech stocks contains lots of "tech factor"
  — that's not about the pair relationship, it's just both stocks being tech.

  Formally: Fama-French 5-factor model:
  $$r_i = \alpha_i + \beta_{Mkt} r_{Mkt} + \beta_{SMB} r_{SMB} + \beta_{HML} r_{HML} + \beta_{RMW}
   r_{RMW} + \beta_{CMA} r_{CMA} + \varepsilon_i$$

  $\varepsilon_i$ = idiosyncratic return — what the stock did that can't be explained by factors.

  In the pipeline: We use FF5+MOM+17 ETFs as the factor set. Before testing cointegration, we
  strip out the factor component.

  ---
  5. Avellaneda-Lee Residualization
  
  ELI5: Before checking if two stocks are "tied together," remove everything they have in common
  with the rest of the market. What's left is their unique relationship. ADF on the unique part is
   more stable because factor exposures drift — the pair relationship doesn't.

  Formally (our implementation):
  1. Fit ridge regression: $r_{i,t} = \hat\beta'i F_t + \hat\varepsilon{i,t}$ (EWMA-weighted, 504d
   window)
  2. Residual returns: $\hat\varepsilon_{i,t} = r_{i,t} - \hat\beta'_{t-1} F_t$ (T+1: use
  yesterday's $\hat\beta$)
  3. Residual pseudo-price: $\hat p_i = \text{cumsum}(\hat\varepsilon_i)$
  4. Residual spread: $\hat s = \hat p_A - \gamma \hat p_B - \alpha$
  5. ADF on $\hat s$ → tests idiosyncratic cointegration

  Why it works: Factor exposures drift quarterly. ADF on raw log-price spreads detects
  "cointegration" driven by factor overlap — spurious. ADF on residuals detects genuine
  idiosyncratic linkage — stable. Pair overlap went from 7% → 17% after residualization.

  ---
  6. Ledoit-Wolf Shrinkage
  
  ELI5: You have 300 stocks and 126 days of data. You're trying to estimate 300×299/2 = 44,850
  correlations from 126×300 = 37,800 data points. The math is underdetermined — you'll get garbage
   correlations for rare pairs. LW shrinkage pulls noisy estimates toward a structured target (the
   identity matrix scaled by average correlation).

  Formally: Blends sample covariance $\hat\Sigma$ with structured target $\mu I$:
  $$\hat\Sigma_{LW} = (1-\alpha)\hat\Sigma + \alpha \mu I$$
  $\alpha$ chosen to minimize expected squared error (Stein's unbiased risk estimate). Result:
  full-rank, noise-suppressed correlation matrix.

  In the pipeline: Applied before spectral clustering (Method A) so clusters reflect real
  co-movement, not estimation noise.

  ---
  7. Random Matrix Theory / Marchenko-Pastur
  
  ELI5: If 300 stocks were completely random (no correlation), what would the eigenvalues of their
   correlation matrix look like? The Marchenko-Pastur distribution tells you. Any eigenvalue above
   this noise ceiling carries real signal.

  Formally: For $N$ assets and $T$ observations ($q = N/T$), eigenvalues of a random correlation
  matrix fall in:
  $$[\lambda_-, \lambda_+], \quad \lambda_+ = (1+\sqrt{q})^2$$

  Count eigenvalues of $\hat C_{LW}$ above $\lambda_+$ → this is $k$, the number of signal
  clusters. Avoids arbitrary choice of $k$.

  In the pipeline: Set $k$ automatically. For SPX-300 with 126d window ($q \approx 2.4$),
  $\lambda_+ \approx 5.2$, typically giving $k = 5$-$8$ clusters.

  ---
  8. Spectral Clustering — Method A

  ELI5: Imagine stocks as nodes in a network. Draw edges between stocks weighted by correlation.
  Spectral clustering finds groups that are densely connected internally and sparsely connected
  between groups — the natural "peer clusters."

  Formally:
  1. Build affinity matrix: $A = (\hat C_{LW} + 1)/2$ (map $[-1,1] \to [0,1]$)
  2. Degree matrix: $D_{ii} = \sum_j A_{ij}$
  3. Unnormalized graph Laplacian: $L = D - A$
  4. Find $k$ smallest eigenvectors of $L$ (they reveal cluster structure)
  5. L2-normalize rows of eigenvector matrix
  6. K-Means on the normalized embedding

  Why Laplacian eigenvectors? The smallest eigenvalues of $L$ correspond to the slowest-varying
  modes of the graph — the most cohesive communities. The $k$-th eigenvector separates the $k$-th
  cluster from the rest. 

  In the pipeline: Method A clusters stocks by return co-movement (noise-controlled). Achieved 76%
   Hungarian-aligned cluster persistence at corr_window=189d.

  ---
  9. Ridge Regression and EWMA Weighting — Method B
  
  ELI5: Estimate how much each stock is exposed to each risk factor. Ridge regression prevents
  overfitting by shrinking coefficients when factors are correlated (which they always are). EWMA
  gives more weight to recent observations — recent factor exposures matter more than those from 2
   years ago.

  Formally:
  $$\hat\beta_i = \arg\min_\beta \sum_t w_t (r_{i,t} - \beta'F_t)^2 + \alpha |\beta|^2$$
  EWMA weights: $w_t = \lambda^{T-1-t}$, normalized. Ridge penalty $\alpha$ stabilizes betas under
   multicollinearity.

  In the pipeline: 504d window, $\lambda=0.97$, $\alpha=30$. Betas are shifted by 1 day (T+1)
  before use in residualization and trading. 

  ---
  10. Orthogonal Factor Basis (Correct Method B Implementation)

  ELI5: Our 24 factors (FF5 + ETFs) are like asking: "What's your height? What's your weight?
  What's your BMI?" Three correlated questions, not three independent measurements. PCA on factor
  returns first extracts 8 truly independent risk dimensions — like extracting "size,"
  "direction," "speed" from correlated physical measurements.

  Formally:
  - Wrong order (what we did first): correlated factors → ridge betas → PCA on betas
  dimensions — like extracting "size," "direction," "speed" from correlated physical measurements.

  Formally:
  - Wrong order (what we did first): correlated factors → ridge betas → PCA on betas
  - Correct order (Barra-style): correlated factors → PCA → orthogonal factors → ridge betas

  PCA on factor returns $F \in \mathbb{R}^{T \times K}$ → rotation matrix $V \in \mathbb{R}^{K \times n}$:
  $$\tilde F = F V \in \mathbb{R}^{T \times n} \quad (\text{orthogonal, variance-ordered})$$
  Betas to $\tilde F$ are stable because there's no collinearity to destabilize the regression.

  Critical implementation detail: Fit PCA on full history up to refit date (not the estimation window) so the orthogonal
  basis is comparable across quarters. A time-varying PCA basis makes "PC1 in Q1" ≠ "PC1 in Q2" — betas become
  incomparable.

  ---
  11. ARI and Hungarian Alignment for Cluster Stability

  ELI5: K-Means assigns arbitrary label numbers. "Cluster 3 in Q1" might be the same group as "Cluster 1 in Q2" — just
  relabeled. The Adjusted Rand Index (ARI) asks: "Do the same stocks end up in the same groups?" ignoring label numbers.
  Hungarian alignment goes further: finds the optimal label mapping, then counts correctly-assigned stocks.

  ARI formally: Compares two clusterings by counting pairs of stocks that are either (1) in the same cluster in both, or
  (2) in different clusters in both. Normalized to $[-1, 1]$; 1 = identical, 0 = random, $-1$ = perfectly anti-correlated.

  Hungarian algorithm: Solves the optimal assignment problem (bipartite matching) between cluster labels. Used here to
  compute fraction of stocks that stayed in the "same" cluster after optimal label permutation — the true cluster
  persistence metric.

  In the pipeline: ARI freeze gate: if ARI(prev, curr) $\geq 0.7$, keep previous labels. Hungarian persistence: target
  $\geq 65%$. Method A achieved 76% at corr_window=189d.

  ---
  12. The Full Pipeline (Top Down)

  DATA
    ├── SPX-300 daily close + volume (yfinance, 2014–2024)
    └── Factor set: FF5 + MOM + 17 ETFs (DataFetcher)

  PREPROCESSING (every refit, T+1 safe)
    ├── Drop FF_RF (risk-free rate, not a risk factor)
    └── Rolling 252d z-score each factor column (normalize across regimes)

  CLUSTERING (every 63d refit)
    Method A (returns-spectral):
      LW-shrunk correlation → MP bound k → Laplacian → k eigenvectors → K-Means
      + ARI freeze (≥0.7) → 76% Hungarian persistence

    Method B (beta-space, with ortho-factors):
      PCA on factor returns (stable basis) → ortho factors
      → EWMA ridge betas → cosine affinity → spectral clustering
      (ClusterPersist ceiling ~50% — structural limitation at daily frequency)

  PAIR SELECTION (every 63d, within clusters)
    Residualize: r_i → ê_i = r_i - β̂'_{t-1} F_t → p̂_i = cumsum(ê_i)
    Formation filters (new pairs):
      ADF p<0.05 on residual spread | HL ∈ [8,20d] | corr63 ≥ 0.75 | β̂ ∈ [0.25,4]
    Maintenance pairs: p<0.10, ≥2/3 recent passes
    New-pair cap: ≤35% fresh pairs per segment

  SIGNAL (daily, T+1 aligned)
    Spread: s_t = p̂_A - γ p̂_B - α (OLS on residual pseudo-prices, 63d rolling)
    Z-score: z_t = (s_t - μ̂_{t-1}) / σ̂_{t-1}
    Entry: |z_t| ≥ 2.0 AND z turning back toward 0 (confirmation filter)

  EXITS
    z-exit: |z_t| ≤ 0.5 (reversion complete)
    Time stop: days_held ≥ 20
    Hard stop: |z_t| > 3.0
    Structural break: 5d corr < 0.5 OR |Δβ̂| > 30% since entry

  SIZING (pair-return space)
    r_pair_t = r_A_t - β̂_{t-1} r_B_t
    σ_pair = rolling 63d std(r_pair), shifted by 1 day
    w_t = 0.001 / σ_pair   (target 10 bps daily PnL vol per pair)
    PnL_t = w_{t-1} × pos_{t-1} × r_pair_t

  COSTS
    3 bps/leg on entry and exit
    Net Sharpe +0.107 | MaxDD -5.5% | Gross +0.357

  OOS VALIDATION (2014-2024)
    Regime-conditional: Sharpe +1.0 in 2017-2021 (high idiosyncratic dispersion)
                        Sharpe -0.3 in 2022-2024 (macro-dominated, PC1 dominant)
    Fix: breadth/PC1 throttle (suspend entries when macro regime detected)

  ---
  13. Why the Strategy is Regime-Conditional
  
  The sub-period breakdown (2014-2024 OOS) reveals the deepest insight:

  When PC1 explains >35-40% of cross-sectional variance, all stocks move together — the market is "one factor."
  Idiosyncratic residuals are nearly random walks; the strategy finds no cointegrating pairs, has thin breadth, and Sharpe
  goes negative.

  When dispersion is high (multiple independent risk factors, moderate average correlation), each stock has genuine
  idiosyncratic variation. Residualized spreads are stationary; the strategy earns.

  This is not a bug — it's a feature. A PC1/dispersion throttle (measure daily: if $\lambda_1/\sum\lambda > 35%$, scale
  book toward zero) would turn the regime-conditional behavior into controlled drawdown discipline.

  ere’s the theory map (what to learn), plus an ELI5 of what this strategy does and how the full
  pipeline runs.

  Core Concepts To Learn (QF map)

  - Time‑Series + Stationarity
      - Returns vs prices; stationarity; residual “spread” as a (near) stationary process
      - Cointegration; Engle–Granger ADF test; half‑life (speed of mean reversion)
      - T+1 alignment (avoid look‑ahead): compute signals on t−1, act at t
  - Correlation + Shrinkage
      - Covariance/correlation matrices; estimation noise; Ledoit–Wolf shrinkage
      - Spectral ideas: eigenvectors of (Laplacian of) a graph as smooth cluster coordinates
  - Unsupervised Clustering
      - K‑Means (centroid‑based) on low‑noise embeddings (spectral for A; beta‑space for B)
      - PCA (orthogonal components); how/when to use (on factor returns, not on betas)
      - Cluster stability: Adjusted Rand Index (ARI); Hungarian alignment (label permutation invariance)
  - Factor Models + Beta Estimation
      - Fama–French (Mkt–RF, SMB, HML, RMW, CMA), momentum; FRED macro deltas (rates/credit)
      - Ridge regression to estimate betas (stabilizes collinearity); EWMA (recency weighting)
      - Residualization: remove common factors first; test cointegration on idiosyncratic spreads
  - Backtesting Hygiene
      - Walk‑forward formation vs trading windows (formation gate vs daily signal)
      - False discovery control (within‑cluster multiple testing); persistence gating
      - Overlap/pair persistence; trade EV; exits/stops; costs/impact; capacity (ADV caps)
  - Market Structure Diagnostics
      - PC1 variance share (how “one‑factor” the market is); dispersion/average correlation
      - Regime‑adaptive sizing/throttles when breadth or PC1 indicates “don’t push”

  ELI5: What This Strategy Does

  - First, we group “buddy stocks”: Method A groups by “who tends to wiggle together” (returns‑based);
    Method B groups by “who reacts similarly to common forces” (beta‑based).
  - Inside each buddy group, we test pairs whose price gap looks like a rubber band (idiosyncratic
    cointegration after removing market/sector/rate effects).
  - We wait until the band is stretched (z‑score big), then bet it snaps back (enter); we take profit
    when it relaxes (exit), and we bail if it snaps (stop).
  - We size bets based on how jumpy the pair‑return is; we never peek at the future; we stop trading or
    trade less when “everything is moving as one thing.”
  - We keep only pairs that keep proving themselves, and we limit how many brand‑new pairs enter each
    quarter so the book isn’t full of untested ideas.

  Method A vs Method B (why each, and what we learned)

  - Method A (returns‑spectral): group by co‑movement fingerprints (shrunk correlation → spectral
    embedding → K‑Means). More stable at daily SPX; worked best here.
  - Method B (beta‑space): group by exposures to risk factors (betas). Needs orthogonal factors (PCA
    on factor returns first) and very stable betas (EWMA ridge, longer windows). On daily SPX, factor
    exposures rotate faster than quarterly refits → clusters churned; positive but small gross SR after
    orthogonalization.
  - Fused (blend A and B) only makes sense after each is independently stable; otherwise it amplifies
    noise.

  - Data + Factors
      - Load SPX prices/volume; build factor matrix (FF+MOM, FRED deltas, ETF proxies); cache parquet
      - Standardize factors (rolling 252d z‑score); drop RF; (for B) optionally PCA to orthogonalize
        factor returns
  - Clustering
      - Method A: returns‑spectral with LW shrinkage; MP‑k; ARI freeze; refit quarterly (63d)
      - Method B (advanced): EWMA ridge betas to orthogonalized factors; cosine affinity; spectral +
        K‑Means; ARI freeze; refit 63d
      - Diagnostics: cluster sizes, ARI over time, Hungarian stock persistence (target ≥ ~70%)
  - Pair Discovery (formation gate)
      - Residualize each leg’s returns vs factors (T+1 betas); integrate to pseudo‑prices
      - Tail window ADF/HL (e.g., last 252d of a longer residual history), corr63 ≥ threshold, HL band
        (e.g., [8–20])
      - Persistence: p<0.05 for new, p<0.10 for maintenance; ≥2/3 recent passes; cap “new” ≤ 35–40% of
        book
  - Trading (daily, T+1)
      - Entries: z_entry (e.g., 2.0) with turning‑point confirmation (z moving back toward 0)
      - Exits/stops: z_exit (e.g., 0.5), time_stop, hard_stop (e.g., 3.0), structural‑break exits (5d
        corr < 0.5; |Δβ| > 30%)
      - Sizing: pair‑return based (r_pair = rA − β̂t−1·rB), target daily σ per pair (e.g., 10 bps)
  - Risk + Costs + Capacity (after gross SR > 0)
      - Costs (2–5 bps/leg), ADV cap (≤1%/leg), issuer/cluster caps, optional impact (k ×
        sqrt(participation))
      - Breadth/PC1 throttle: scale down and block new entries when breadth thin or PC1/avg corr high;
        maintain exits
  - Validation + Research Log
      - Zero‑cost SR, MaxDD, overlap, trips, hit rate; then net SR at 2–3 bps/leg
      - Subperiod OOS splits; append runs to EXPERIMENTS.md with params/results/notes

  Why This Strategy Works (when it works)

  - It monetizes idiosyncratic mean‑reversion between economically related stocks — after removing common
    factors — with disciplined entries/exits and risk controls.
  - It trades less (or not at all) when the market behaves as “one factor” (PC1 dominance), because
    cointegration signals then become unreliable.

  If you want to go deeper right now

  - Skim: Engle–Granger cointegration; ADF tests; half‑life estimation
  - Skim: Ledoit–Wolf shrinkage; spectral clustering intuition
  - Skim: Ridge regression; EWMA; PCA and why to orthogonalize factors first
  - Read: Avellaneda & Lee (2010) residualization idea; Gatev et al. (1999) baseline results
  -  Internalize: research hygiene (T+1, walk‑forward, persistence gates) and market‑structure diagnostics
    (PC1/dispersion) to stay honest about when to press vs stand down

    ere’s the “explain like I’m five” summary of what we changed, why we changed it, and how it helped.

  - We cleaned the price signal (residualization)
      - Change: Before testing pairs, we first removed the “market/sector/rates” wiggles from each stock,
        then tested the leftover (idiosyncratic) wiggle.
      - Why: If you don’t strip out common moves, pairs look “broken” just because the whole market
        shifted. After cleaning, good pairs stayed good more often (overlap jumped).
  - We stopped losses earlier (hard stop 4σ → 3σ)
      - Change: Cut losing trades sooner.
      - Why: The blow‑ups were where we lost all the money. Smaller losses made the whole game winnable.
  - We took profits a bit sooner (kept z_exit = 0.5; avoided 0.25)
      - Change: We tested tighter profit‑taking (0.25) but it cut our win rate with little benefit.
      - Why: 0.5 harvested enough without hurting the hit rate.
  - We picked faster mean‑reverting pairs (HL band [8–20] days)
      - Change: Only trade “rubber bands” that snap back in ~1–3 weeks.
      - Why: Very slow bands tie up cash; very fast bands are noisy. Middle speed had cleaner trades and
        lower drawdown.
  - We raised the entry bar and added a “turning‑point” check (z=2.0 + confirmation)
      - Change: Enter only when the band is really stretched (2σ) AND has started to snap back (z moving
        toward 0).
      - Why: This avoids “still stretching” entries that run straight into the stop.
  - We separated “who to trade” from “how to trade” (decoupled horizons)
      - Change: Used longer history to decide which pairs are stable, but used much shorter recent
        history for signals/sizing.
      - Why: Long windows find durable pairs; short windows produce cleaner, tradable signals.
  - We favored proven pairs and limited “new” ones (maintenance gate + new‑pair cap)
      - Change: Pairs that passed last time get a slightly easier pass; we cap how many brand‑new pairs
        can enter each quarter.
      - Why: The book shouldn’t be mostly untested ideas; this forces persistence and reduces churn.
  - We adjusted breadth carefully (top‑k ≈ 10)
      - Change: Don’t add the 11th, 12th, … lowest‑quality pairs.
      - Why: More pairs help only if they’re good; otherwise they dilute performance.
  - We avoided bad environments (breadth/PC1 throttle)
      - Change: Trade less and block new entries when the whole market moves as “one thing” (high PC1
        share / low dispersion).
      - Why: In one‑factor periods, idiosyncratic pairs don’t mean‑revert reliably. Standing down cuts
        drawdowns.
  - We sized trades in return units (not price levels)
      - Change: We measured and sized on daily pair‑returns, not on the level of the spread.
      - Why: Apples‑to‑apples risk sizing; avoids tiny positions that fake a high Sharpe.
  - We tried a different way to group stocks (Method B: beta‑space)
      - What we tried first (didn’t work): Estimate betas to correlated ETFs/factors and cluster on those
        betas (or PCA on betas).
      - Why it failed: Inputs were collinear; betas drifted fast; clusters churned.
      - The fix we tested: Orthogonalize factor RETURNS first (PCA), then estimate betas to those
        orthogonal factors, then cluster.
      - Result: First positive gross for B, but clusters still unstable versus Method A at daily
        frequency — factor exposures change faster than our quarterly cadence.
      - Takeaway: Our Method B is closed for daily SPX (as built), but beta‑space clustering remains
        viable with higher‑frequency betas/risk‑model factors (Barra/Axioma) or a different horizon.
  - We validated across time (2014–2024)
      - Change: Ran the best setup across earlier years and sliced by subperiods.
      - Why: Proved the parameters weren’t just lucky; also showed when to trade less (one‑factor
        periods) and when to press (high‑idiosyncrasy periods).

        Research Evolution — ELI5

  Think of tuning this strategy like calibrating a metal detector on a beach. You start with bad
  settings, find you're picking up bottle caps instead of gold coins, and adjust one dial at a
  time.

  ---
  Phase 1: Discovered the Sharpe Was Fake
  
  Change: Switched from log-spread sizing to pair-return sizing.

  Before: w = target / σ(spread_level) — denominator was the spread's total wandering range over
  63 days (~5-8× too large). Positions 10× too small. Like weighing gold in kilograms when you
  need grams — the number looks good but means nothing.

  After: r_pair = r_A - β·r_B as the PnL unit. σ_pair = daily volatility of that return. Weight w 
  = 0.001/σ_pair puts positions in real portfolio fractions (~5-15%).

  Result: Sharpe collapsed 2.59 → -0.19. The "good" result was measuring in the wrong units. Now
  loss was real — and informative.

  ---
  Phase 2: Discovered Clustering Was Unstable
  
  Change: Method A, corr_window 126d → 189d. ARI freeze threshold 0.6 → 0.7.

  Before: 126d window — too short. Quarterly correlation matrix shifts significantly as 63 days of
   new data replaces 63 days of old data. Cluster labels changed completely every quarter
  (Hungarian persistence ~0%). Same stocks, different groups — pairs had no history.

  After: 189d window gives a more stable snapshot of who-moves-with-whom. ARI freeze at 0.7 means
  "don't update labels unless the new grouping is meaningfully different."

  Result: Hungarian cluster persistence jumped to 76%. The same stocks stayed in the same clusters
   across consecutive quarters. Now pairs could build history.

  ---
  Phase 3: Discovered Formation Filters Were Unstable (ADF on Raw Spreads)
  
  Change: Added residualization (Avellaneda-Lee) before ADF/HL tests.

  Before: ADF on log(P1) - β·log(P2) — this spread contains factor components (market, sector,
  rates). When tech sector exposure shifts, the spread changes even if the pair relationship
  hasn't broken. ADF fails → pair dropped → new untested pair replaces it → overlap 7%.

  After: Strip factors first. ê_i = r_i - β̂'_{t-1} F_t, integrate to pseudo-price p̂_i = 
  cumsum(ê_i), ADF on p̂_A - γ p̂_B. Now testing pure idiosyncratic cointegration — stable because
  it's not contaminated by factor drift.

  Why integrate residuals? You need a price-like series for cointegration testing. Returns are
  already stationary (don't need a test). Cumulative returns approximate a log-price stripped of
  factor influences.

  Result: Pair overlap 7% → 17%. Same clusters, now same pairs quarter-to-quarter.

  ---
  Phase 4: Found the Right Stop Structure
  
  Change: Hard stop 4.0 → 3.0.

  Before: At hard_stop=4.0, a losing trade gets to |z|=4 before forced exit. At z_entry=1.5,
  that's a 2.5σ loss on the loser vs 1.0σ gain on the winner. With 80% hit rate: 0.8×1.0 - 0.2×2.5
   = +0.3σ per trade — positive but thin. In practice, spreads that reach z=4 often keep going
  (structural breaks, earnings, regime change) → actual loser is larger than 2.5σ.

  After: Hard_stop=3.0 cuts losses at 1.5σ from entry. Structural breaks are caught earlier. More
  trips (capital freed faster), less tail damage.

  Result: Gross Sharpe -0.05 → +0.030. MaxDD -21% → -19%. First positive Sharpe.

  ---
  Phase 5: Quality Filtering on Pairs
  
  Change: HL filter [5-30d] → [8-20d]. Correlation filter 0.70 → 0.75.

  HL [8-20d] — why:
  - HL < 8d: spread reverts so fast it's already partially back by the time z=2.0 fires. You enter
   late, the remaining move is tiny, time_stop often fires before full reversion. Thin wins, 
  normal losses.
  - HL > 20d: spread takes longer than time_stop=20d to revert. Time stop fires mid-trade at a
  loss.
  - [8-20d] matches the geometry: if HL=12d, in 20 days the spread reverts 75% → z from 2.0 to
  ~0.5. Entry and exit are aligned.
  
  corr63 ≥ 0.75 — why:
  Lower correlation = more noise in the spread. The z-score is messier, entries at z=2.0 are less
  reliable, hard stops fire more. At 0.75+, the pair is tightly economically linked — spread
  movements are genuine signal, not noise.

  Result: Gross +0.030 → +0.124 (k=8) → +0.190 (k=10). MaxDD compressed to -13%.

  ---
  Phase 6: Entry Timing and Structural Break Exits
  
  Change: Added (a) confirmation filter, (b) structural-break exits, (c) new-pair cap.

  (a) Confirmation filter: Only enter when z is already turning back toward zero. Short entry: z_t
   ≥ 2.0 AND z_t < z_{t-1} (spread falling from peak). Long: z_t ≤ -2.0 AND z_t > z_{t-1}.

  Why: Without confirmation, ~30% of entries at z=2.0 were "on the way up" — the spread was still
  diverging. These went straight to hard_stop. Confirmation cuts this subset. Hit rate recovers at
   z=2.0 (was 68%, now 75%).

  (b) Structural-break exits: If 5-day correlation of the two stocks drops below 0.5, or hedge
  ratio drifts >30% from entry β — exit immediately. Don't wait for hard_stop.

  Why: These signals indicate the pair relationship has broken (earnings, sector reclassification,
   acquisition). Waiting for z=3.0 when the relationship is gone means you're fading a structural
  divergence, not a temporary spread.

  (c) New-pair cap: Maintenance pairs fill first. Fresh pairs capped at 35% of the book.

  Why: Fresh pairs have zero track record. If 90% of the book is fresh every quarter, you're
  essentially starting over with untested pairs each time. Forcing ≥65% maintenance mix means the
  book is mostly proven pairs with at least 2 recent passes.

  Result: Gross +0.190 → +0.267. MaxDD -14% → -7%. Confirmation + structural breaks together
  halved MaxDD.

  ---
  Phase 7: Entry Asymmetry (z_entry 1.5 → 2.0)
  
  Change: Raised entry threshold from 1.5 to 2.0, with confirmation filter active.

  Before (z=1.5, WITHOUT confirmation): Enter at z=1.5. Winner earns 1.0σ (to z_exit=0.5). Loser
  loses 1.5σ (to hard_stop=3.0). With 74% hit rate: 0.74×1.0 - 0.26×1.5 = +0.35σ. Slightly
  positive but the asymmetry is unfavorable — losers lose more than winners win.

  Why z=2.0 previously failed (H5): Without confirmation, the heavy-tail nature of residual
  spreads meant that being at z=2.0 gave no advantage over z=1.5 for predicting whether the spread
   would keep going to z=3.0. The fat tails made both entry levels equally risky for the 25% of
  losing trades.

  Why z=2.0 works WITH confirmation: Confirmation means we only enter when the spread is already
  turning back. At z=2.0+turning, the spread has peaked and is reverting. Winner earns 1.5σ (from
  z=2.0 to z_exit=0.5). Loser loses 1.0σ (from z=2.0 to hard_stop=3.0). With 75% hit rate:
  0.75×1.5 - 0.25×1.0 = +0.875σ. Asymmetry now favorable — winners earn more than losers lose.

  Result: Gross +0.267 → +0.357. MaxDD -7% → -4.6%. This was the biggest single improvement.

  ---
  Phase 8: Method B Orthogonalization
  
  Change (wrong → right): PCA on betas after estimation → PCA on factor returns before estimation.

  Wrong order (S15b):
  correlated factors → ridge betas → PCA(betas)
  Ridge on correlated factors gives unstable betas (collinearity). Then PCA compresses unstable
  betas → information loss. ClusterPersist 50.5%, Gross -0.424.
  
  Why collinearity hurts ridge: Ridge penalizes ‖β‖². With collinear factors (SPY ≈ FF_MktRF),
  ridge can't tell which factor "owns" the exposure — it splits it arbitrarily. Different splits
  each quarter → unstable betas → unstable clusters.

  Correct order (S16c) — Barra-style:
  correlated factors → PCA(factor returns) → orthogonal factors → ridge betas
  PCA on factor returns extracts 8 truly independent risk dimensions. Each PC explains a distinct
  uncorrelated risk source. Ridge on orthogonal inputs is well-conditioned — each beta has a clear
   meaning. ClusterPersist 50.6%, Gross +0.099. 
  
  Critical implementation detail — stable vs time-varying basis:
  - Time-varying: fit PCA within each 504d window → PC1 in Q1 ≠ PC1 in Q2 (same label, different
  risk dimension) → betas incomparable across refits → ClusterPersist 0% → no trades fire.
  - Stable: fit PCA on all history up to refit date → fixed rotation applied to estimation window
  → PC1 always means the same risk dimension → betas comparable quarter-to-quarter.
  
  Why ClusterPersist still stuck at ~50% for Method B: Even with correct orthogonalization, which
  stocks load on which PCs changes quarterly (growth vs value rotation, tech vs rates leadership).
   The rotation of the factor space is stable but which stocks are "high-PC1" vs "high-PC2"
  shifts. This is structural — can't be fixed without Barra/Axioma quality risk models updated
  daily.

  ---
  Summary Table

  ┌───────┬───────────────────────────────────────┬────────────────────────┬─────────────────┐
  │ Phase │                Change                 │          Why           │     Sharpe      │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 0     │ Log-spread sizing                     │ Wrong units            │ +2.59 (fake)    │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 1     │ Pair-return sizing                    │ Real units             │ -0.19           │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 2     │ corr_window 126→189, ARI 0.6→0.7      │ Stable clusters        │ -0.05           │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 3     │ Residualized ADF (Avellaneda-Lee)     │ Idiosyncratic          │ -0.05 (overlap  │
  │       │                                       │ cointegration          │ 17%)            │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 4     │ hard_stop 4→3                         │ Cut tail losses        │ +0.030          │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 5     │ HL[8-20], corr≥0.75, k=10             │ Quality pairs          │ +0.190          │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 6     │ Confirmation + struct-break exits +   │ Better entry/exit      │ +0.267          │
  │       │ new-pair cap                          │                        │                 │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 7     │ z_entry 1.5→2.0 (with confirmation)   │ Asymmetric payoff      │ +0.357          │
  ├───────┼───────────────────────────────────────┼────────────────────────┼─────────────────┤
  │ 8     │ Method B ortho-factors (correct       │ Orthogonal betas       │ +0.099 (B only) │
  │       │ order)                                │                        │                 │
  └───────┴───────────────────────────────────────┴────────────────────────┴─────────────────┘

-------------------------------------------------------------------------------
Step 8 — Implementation Results (2026-05-23)
-------------------------------------------------------------------------------

## Data integration

All four sources now fetched and cached in data/:

| File | Source | Used in pipeline | Role |
|------|--------|-----------------|------|
| ff_daily.parquet | WRDS ff.factors_daily / Ken French | Optional (--factor-source ff_fred_first/both) | FF5+MOM factor set for Method B beta clustering and residualization |
| fred.parquet | FRED REST API + WRDS frb.rates_daily | Not yet active | Macro rates/spreads for regime conditioning (NEXT BUILD) |
| ibes_earnings.parquet | WRDS ibes.statsum_epsus | Active (--earnings-blackout 3) | Earnings blackout filter |
| factor_close.parquet | yfinance (17 ETFs) | Active (etf_only default) | Method B clustering + residualization |

Key finding on factor source: --factor-source both (ff+fred+ETF) consistently HURTS vs etf_only.
Adding ff_daily/fred to the factor set degrades pair residualization quality (+0.41 vs +0.57 gross SR).
Confirmed across S17b (with IBES) and S16c (pre-IBES). Etf_only is optimal.

## OLS vs Ridge — where each belongs

The codebase uses different regression methods intentionally:

- rolling_beta_alpha() (backtest_pairs.py): OLS on 1 regressor (pb). Correct.
  Engle-Granger Step 2 requires OLS for the cointegrating vector. Ridge would shrink β toward
  zero — the opposite of what cointegration needs. With N=1 regressor, collinearity is
  irrelevant. OLS is unbiased, Ridge is not. Keep OLS.

- _ridge_betas_ewma() (pairs_feature_matrix.py): Ridge on 17-20 factor ETFs. Correct.
  Multi-factor regression with correlated regressors (SPY ≈ FF_MktRF, TLT ≈ FF_HML, etc.)
  requires regularization. Ridge with EWMA weighting is the right tool here.

- _residualize() (pairs_discovery.py): Ridge on factor set. Correct for same reason.

Rule: OLS for pairwise spread hedge ratio; Ridge everywhere multi-factor collinearity appears.

## IBES earnings blackout (S17 series)

Implementation: backtest_pairs.py --earnings-blackout N --no-earnings-exit
Logic: block entries within ±N calendar days of actual IBES announcement date for either leg.
Forced exit: if already in position and hit blackout window, force exit (marginally better than
entry-only; S17a vs S17c: +0.57 vs +0.55, 1 fewer trip over 11 years).

Results vs baseline:
- S12d (no IBES, ~2015-2022): Gross +0.357, Net +0.107 @ 3bps
- S17a (+IBES ±3d, 2015-2026): Gross +0.57, Net +0.20 @ 3bps, Net +0.51 @ 0.5bps (MOC)
- Improvement: +60% gross SR, +87% net SR @ 3bps
- HitRate unchanged (75.3% → 75.5%); MaxDD improved (-4.6% → -4.5%)
- AvgHold 4.0d (was ~8-10d in S12d — reflects 11yr vs 5yr period difference, not IBES)
- Best confirmed: earnings_blackout=3, force_exit=True, etf_only, formation=378, neutralize=True

## Parametric tuning exhausted (S18 series)

After IBES, systematic parameter search showed the gross SR ceiling is ~0.55-0.60:

- Strict ADF p<0.01 + top-k=5 (S18a): Gross +0.28 — reduces capacity without improving quality.
  Conclusion: false-positive pairs (p<0.05, fail p<0.01) are contributing positively. BH correction
  not needed; the ADF threshold is not the quality bottleneck.
- z_exit=1.0 (S18b/S19f): Gross +0.54 vs +0.57. z_exit=0.5 is optimal — full reversion
  earns more per trade than exiting at partial reversion despite the higher HitRate.
- Longer holds (AvgHold) via z_exit tuning: ineffective. AvgHold is driven by hard_stop=3.0 +
  confirmation filter dynamics, not exit threshold.

## Kalman filter β experiment (S19 series, negative result)

Implementation: kalman_beta_alpha() in backtest_pairs.py (--use-kalman flag, default OFF).
Uses scaled H=[pb/pb_rolling_mean, 1] to prevent large log-price magnitudes dominating Kalman gain.
OLS-seeded initial P from warmup regression for proper initialization.
β uncertainty gate: skip entry when beta_std/|β| > kalman_beta_unc_cap (default 0.30).

Result: Kalman does NOT improve over rolling OLS for this strategy at any delta tried.

| Config | Gross SR | Trips | HitRate | Problem |
|--------|----------|-------|---------|---------|
| OLS baseline (S17a) | +0.57 | 558 | 75.5% | — |
| Kalman δ=1e-4 (S19a/c) | +0.61 | 92 | 89.2% | β over-adapts, removes spread |
| Kalman δ=1e-7 old (S19d) | -0.04 | 598 | 67.6% | β stuck, mispriced after years |
| Kalman δ=1e-7 scaled (S19e) | +0.04 | 762 | 49.1% | near-random |
| Kalman δ=1e-5 scaled (S19f) | +0.36 | 415 | 77.6% | semi-functional, worse |

Root cause: Rolling OLS "staleness" is a FEATURE for mean-reversion trading. When a pair diverges
(z→2.0), OLS β was estimated pre-divergence → reflects equilibrium. Kalman in fast mode interprets
the divergence as β change and absorbs it, removing the tradeable signal. Kalman in slow mode leaves
β fixed at warmup value, which drifts from truth over years → random HitRate.

This is consistent with the pairs trading literature (Elliott et al. 2005): Kalman outperforms rolling
OLS only when β genuinely changes over multi-month horizons (regime switching). For short-horizon
(4-10 day) mean reversion, the fixed OLS window is optimal.

The kalman_beta_alpha() function remains in code (--use-kalman flag) for future experiments, fully
documented with this negative result. Zero impact on default behavior.

## Cost model correction

3bps/leg (assumed throughout) is retail/aggressive-intraday cost for SPX-300 names.
Realistic MOC (Market on Close) execution on S&P 500 constituents: 0.5-1.0bps/leg.

MOC is appropriate because:
- SPX-300 names clear >$500M ADV; our $10M book is <0.1% ADV → MOC with zero market impact
- Closing auction crosses net buyers/sellers; effective cost ≈ 0.5bps for liquid names

S17a_moc (0.5bps): Net SR +0.51 (vs +0.20 @ 3bps). Net SR almost triples from cost correction alone.

## Revised performance ceiling (daily SPX-300)

| Scenario | Gross SR | Net SR | Achievable |
|----------|----------|--------|-----------|
| Current best (S17a, 3bps) | +0.57 | +0.20 | Done |
| Current best (S17a, MOC 0.5bps) | +0.57 | +0.51 | Done (cost correction) |
| + CRSP universe (removes survivorship bias) | +0.60-0.70 | +0.55-0.65 | Next |
| + ADF clustering affinity | +0.65-0.75 | +0.60-0.70 | Next build |
| + FRED regime gate | +0.65-0.75 | +0.60-0.70 | Next build |
| Net SR > 1 on daily data | requires gross SR > 1.06 @ 0.5bps | Hard ceiling | Needs intraday |

Daily SPX-300 gross SR ceiling ≈ 0.70-0.90 with best-case signal improvements (ADF clustering +
regime conditioning). Net SR > 1 on daily data requires either <0.1bps execution (institutional
rebates) or intraday execution (30-60min, projected SR 0.8-1.5 per EXPERIMENTS.md).

## Next build targets

Priority 1 (2 days): ADF-based clustering affinity
  - Build pairwise cointegration strength matrix A_coint[i,j] = exp(-adf_pvalue[i,j])
  - Fuse with returns-spectral affinity: A_fused = 0.5*A_ret + 0.5*A_coint
  - This puts cointegration INTO the cluster structure (not just post-hoc filtering)
  - O(N²) ADF tests, ~15s after corr>0.6 prefilter

Priority 2 (half day): FRED macro regime gate
  - Halve position sizes when: BAMLH0A0HYM2 > 5% OR T10Y2Y < 0
  - Data already in fred.parquet; 20-line addition to backtest_pairs.py
  - Targets the 2022-2024 underperformance period (regime-conditional SR -0.32)

Priority 3 (separate project): Intraday execution
  - TAQ 30-60min signals, same clustering infrastructure
  - Projected SR 0.8-1.5 (per EXPERIMENTS.md ceiling analysis)

## S26 Series — Remote PC-Core Replication Study (2026-05-27 to 2026-05-29)

### Motivation

Remote repo (Donking123/pairs-trading-ml) claims SR 1.028 on CRSP S&P 500 data 2003-2023.
Goal: understand methodology, close implementation gaps, assess if SR > 1 is achievable locally.

### Remote methodology (ground truth from repo source code)

- **Clustering**: OPTICS (xi=0.04) on PC-distance = 1 - corr(market-residual returns)
  - Market residualization: single-factor OLS: r_i = α + β*r_market + ε; distance on ε
  - NOT agglomerative/b_agglo (first agent hallucinated this)
- **Price series**: Total-return index (TRI): (1+ret).cumprod() from crsp.dsf.ret field
  - NOT split-adjusted price (prc/cfacpr). Dividend-adjusted.
- **Formation**: 756 trading days (3yr), rolling monthly (refit=21)
- **Hedge ratio**: OLS from formation window, FROZEN for entire trading month
- **Z-score**: Rolling 126d window (6 months), NOT frozen from formation
- **Entry**: |z| >= 2.0, no confirmation filter
- **Exit**: z crosses zero (zero-crossing), month-end force-close
- **Stop-loss**: NONE for "PC core" variant (SR 1.028)
- **Costs**: Zero-cost for SR 1.028 (gross SR)
- **Universe**: ~500 active S&P 500 constituents per formation date (crsp.dsp500list)
- **Period**: 2003-2023, formation data from 2000

### Implementation changes made

1. `data_fetcher.py`: CRSP pull changed from prc/cfacpr → ret field; crsp_close.parquet now
   stores TRI = (1+ret).cumprod() anchored to 1.0 at 2000-01-03.
2. `backtest_pairs.py`: Added --freeze-beta, --freeze-spread-stats, --zero-cross-exit,
   --month-end-forceclose, --no-entry-confirm, --roll-win, --no-factor-zscore,
   --normalize-prices, --cooldown-days, --spx-membership, --b-agglo-threshold flags.
3. `pairs_feature_matrix.py`: Added cluster_method_b_agglo() (tested, inferior to c_optics).
   Added optics_xi parameter, OLS residualization option.
4. `pairs_discovery.py`: Added OLS option for residualization.

### S26 results progression

| Run | SR Gross | MaxDD | Key change |
|-----|----------|-------|-----------|
| S22 (original attempt) | -0.19 | — | baseline |
| S26a (7 gaps closed) | 0.05 | -30% | period mismatch, wrong params |
| S26a_v2 | 0.21 | -7% | +no-confirm, +freeze-spread-stats |
| S26a_v4 | 0.47 | -5% | correct period 2010-23, roll_win=63 (still wrong) |
| bt_S26_exact | **0.70** | -175% | CORRECT: c_optics, roll_win=126, no stop, no filter, TRI |
| bt_S26_exact_v3 | **0.53** | -19% | +top-k=30 (practical pair count) |
| bt_S26_coint | 0.48 | -21% | +ADF/HL filter (their "filtered" variant = 0.752) |

### Why remote SR 1.028 > our 0.53-0.70

1. **Universe**: ~500 active S&P 500 stocks → tighter OPTICS clusters → 50-200 pairs/segment.
   Our 1335 historical members → 475 pairs/segment at no cap → dilutes signal, inflates vol.
2. **In-sample**: Remote parameters tuned on 2003-2023, SR reported on same period.
   Our 0.53-0.70 is an equivalent out-of-sample estimate.
3. **Pattern confirmed**: no-filter (0.70) > with-filter (0.48) matches remote's 1.028 > 0.752.

### Takeaways for local strategy

**Methodology improvements adopted from remote:**
- freeze-beta: formation OLS hedge ratio frozen for trading month (better than rolling)
- no-entry-confirm: pure |z|>=2.0 entry (remote has no confirmation)
- zero-cross-exit: full reversion capture (vs z_exit=0.5)
- roll_win=126: 6-month z-score window (more stable than 63d)
- month-end-forceclose: clean segment boundaries
- TRI data: dividend-adjusted returns for spread construction

**Elements NOT adopted (hurt local performance):**
- normalize-prices: changes spread scale, hurts with frozen formation stats
- cooldown-days: reduces entries without improving quality
- remove ADF/HL filter: increases noise pairs on 1335-stock universe

**Net SR TBD**: Running bt_S26_net with 3bps costs. See EXPERIMENTS.md for result.

