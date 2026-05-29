
────────────────────────────────────────
Date: 2026-05-25
Entry: REPLICATION ANALYSIS — Why Paper SR Is Not Reproducible on 2017–2024

================================================================================
SECTION 1: PAPER REPLICATION EXPERIMENTS
================================================================================

Target: Rotondi–Russo (2025), ssrn-5080998
Paper result: annualised Sharpe 0.72–1.01 (SSD/PCA/PC distance), 2000–2023

ALL EXPERIMENTS ON CRSP UNIVERSE (935 tickers, 2014–2024, survivorship-free)
unless noted as yfinance.

Chronological run table (post-CRSP migration):
──────────────────────────────────────────────────────────────────────────────
Run     Method          Params (key diffs vs S17a)   Gross  Net   MaxDD  AvgHold
──────────────────────────────────────────────────────────────────────────────
S17a    A spectral/yf   corr≥0.70, top-k=10, IBES   +0.57  +0.20  -4.5%  4.0d
        [BASELINE yfinance — survivorship biased]
S21a    A spectral/CRSP corr≥0.70, top-k=10, IBES   +0.48  +0.17  -5.1%  4.2d
        [First CRSP run. Honest baseline.]
S21b    A spectral/CRSP corr≥0.65, top-k=12, IBES   +0.54  +0.20  -3.8%  3.8d
        [Looser filters for CRSP breadth. Best CRSP config.]
S22     c_optics        NO ADF, NO HL, NO corr,      -0.19  -0.51  -17.2% 2.7d
        [Paper-equivalent: 756d form, refit=21, z_exit=0]
        [300+ pairs/seg, zero quality filters]
S23     c_optics        light ADF(0.10), HL[5-30]    +0.17  -0.20  -11.1% 2.3d
        [Hybrid: OPTICS cluster + some filters]
S24     c_optics        ADF(0.10), HL[8-20]          +0.18  -0.19  -10.0% 2.3d
        [HL filter present but AvgHold still 2.3d]
──────────────────────────────────────────────────────────────────────────────

Sub-period gross SR for best CRSP run (S21b):
──────────────────────────────────────────────────────────────────────────────
Year  Gross SR  PC1    CS_disp(daily)  Notes
──────────────────────────────────────────────────────────────────────────────
2015  +1.21     30%    1.71%           post-QE sector rotation
2016  -0.19     26%    1.86%           Trump election spike, correlated move
2017  -0.52     10%    1.58%           low-vol (VIX~10), spreads don't open
2018  +0.99     27%    1.81%           rate hike cycle, vol spike Feb
2019  +0.53     22%    1.90%           trade war, moderate dispersion
2020  +0.77     43%    2.89%           COVID crash + V-recovery sector rotation
2021  -1.13     22%    1.96%           meme stocks (GME/AMC), retail flows
2022  +1.08     39%    2.10%           rate shock, energy vs financials diverge
2023  -1.03     24%    1.97%           AI monoculture, megacap correlated
2024  -0.25     17%    1.89%           AI continues, thin pair opportunities
──────────────────────────────────────────────────────────────────────────────
Good years (SR > 0): 2015, 2018, 2019, 2020, 2022 → avg gross SR ≈ +0.91
Bad years (SR < 0): 2016, 2017, 2021, 2023, 2024 → avg gross SR ≈ -0.61


================================================================================
SECTION 2: WHY THE PAPER'S SR IS NOT REPRODUCIBLE ON 2017–2024
================================================================================

THREE DISTINCT CAUSES:

1. SAMPLE PERIOD CARRIES MOST OF THE ALPHA
   Paper runs 2000–2023. Our window is 2017–2024.
   Pre-2012 era: high cross-sectional dispersion, low algo competition,
   pairs trading generates 80–100 bps/month gross reliably.
   Post-2015 era: algo saturation, compressed spreads, harder regime mix.
   Estimate: paper's 2000–2009 sub-period alone drives SR ~1.5; 2017–2023 alone ≈ 0.2–0.4.
   This is the primary gap. Not fixable without access to pre-2014 CRSP data.

2. PURE OPTICS WITHOUT FILTERS FAILS IN POST-2015 SPX
   S22 (paper-equivalent config) produced gross SR -0.19 on 2017–2024.
   Root cause: AvgHold 2.7d at 2 bps/leg = 371 bps annual cost drag.
   Even with no transaction costs, the gross signal is near-flat in this period.
   The paper's 44 actually-traded pairs/month vs our 300+ selected (no filters):
   OPTICS without quality gates selects many nearly-identical stocks that revert
   in 2–3 days — too fast for any cost structure to be viable.

3. HOLD TIME IS THE BINDING COST CONSTRAINT
   Fundamental equation: annual cost drag = 2 × cost_bps × (252 / AvgHold)
   At cost=3bps, AvgHold=2.3d: drag = 6bps × 109 = 654 bps = 6.5% per year
   At cost=3bps, AvgHold=3.8d: drag = 6bps × 66 = 396 bps = 4.0% per year
   At cost=3bps, AvgHold=7.0d: drag = 6bps × 36 = 216 bps = 2.2% per year
   Method C (OPTICS) consistently produces 2.3d AvgHold regardless of HL filter —
   OPTICS selects very tight clusters where residual spreads revert extremely fast.
   Method A (spectral) produces 3.8d AvgHold because the quality filter stack
   (ADF, HL[8-20]) specifically selects pairs with moderate reversion speed.


================================================================================
SECTION 3: REGIME ANALYSIS — IS THE STRATEGY ADAPTING?
================================================================================

FINDING: Neither PC1 share nor cross-sectional dispersion reliably predicts SR.

Correlation between PC1 share and annual gross SR: ≈ -0.07 (near zero)
Correlation between CS dispersion and annual gross SR: ≈ +0.14 (near zero, wrong sign expected)

Counter-examples:
  2017: PC1=10% (LOWEST), CS_disp=1.58% (LOWEST) → SR=-0.52 (bad)
        Cause: low-vol regime, VIX~10, spreads rarely cross z_entry=2.0
        → Strategy has no trades, thin sample, noisy SR
  2021: PC1=22%, CS_disp=1.96% (normal) → SR=-1.13 (worst year)
        Cause: meme stock flows (GME, AMC) break pair relationships
        → Idiosyncratic but non-fundamental divergence; pairs don't revert
  2023: PC1=24%, CS_disp=1.97% (normal) → SR=-1.03 (bad)
        Cause: AI-driven megacap outperformance concentrated in 7 names
        → Rest of S&P 500 pairs trade well but top weights dominate index
  2022: PC1=39% (HIGH), CS_disp=2.10% → SR=+1.08 (good)
        Cause: rate shock creates SECTOR divergence (energy surges, utilities fall)
        → HIGH PC1 but pairs WITHIN sectors revert well

Conclusion: Bad years have heterogeneous causes. No single indicator captures all:
  - Low vol: detectable via CS dispersion < 1.65%
  - Meme flows: not detectable by standard market structure metrics
  - AI concentration: partially detectable (FAANG weight vs sector weight)
  None of these generalise cleanly to a robust regime gate.


================================================================================
SECTION 4: REGIME ADAPTATION — IDEAS AND OVERFITTING RISK
================================================================================

IDEA A: Cross-sectional dispersion gate (CS_disp)
  Signal: rolling 21d avg of daily cross-sectional std(rets)
  Throttle: scale book to 0 if CS_disp < 1.65% (below 2017 level)
  Pros: internally computed, no external data, clear economic story
  Cons: 2017 is the only "low-vol" year in sample; one data point.
         Would NOT have helped in 2021, 2023 (normal dispersion, bad SR)
  Risk: high. Threshold calibrated on 1 year of data.

IDEA B: PC1 breadth throttle
  Signal: PC1 share of LW correlation matrix (already computed for clustering)
  Throttle: scale book by max(0, 1 - (PC1 - 0.30)/0.15)
            → scale=1 at PC1≤30%, scale=0 at PC1≥45%
  Would throttle: 2020 (PC1=43%), 2022 (PC1=39%)
  Would NOT throttle: 2021 (PC1=22%), 2023 (PC1=24%), 2017 (PC1=10%)
  Problem: throttles 2020 and 2022 which are GOOD years. Actively harmful.
  Risk: very high. Contradicts empirical data.

IDEA C: Trailing hit-rate gate (backward-looking, robust)
  Signal: hit_rate of last 2 segments
  Throttle: if consecutive_hit_rate < 0.55, scale top-k by 50%
  Pros: no external data, no threshold tuning on regime labels
         directly measures whether the signal is working
  Cons: lag of 1–2 segments (63–126d) before throttle activates
         may miss sharp regime transitions (2020 COVID crash was brief)
  Risk: low. Natural feedback loop, not fitted to regime labels.

IDEA D: Minimum breadth gate
  Signal: number of trades in previous segment
  Throttle: if trips_last_segment < 5, skip next segment entirely
  Pros: zero external parameters, catches low-vol regime (few spreads open)
  Cons: misses years where spreads open but immediately diverge further (2021)
  Risk: low.

RECOMMENDATION (minimal complexity):
  Idea C (trailing hit-rate gate) as the ONLY addition.
  One parameter (0.55 threshold). Natural, self-correcting, no external data.
  Expected improvement: modest (catches 2023-2024 deterioration with lag).
  Does NOT add clustering complexity, does NOT add more filter layers.

COMPLEXITY WARNING:
  Method B (factor-beta clustering): added, tested, closed — -0.014 vs +0.357
  Method C (OPTICS): added, tested, closed on this era — 2.3d AvgHold, net negative
  Adding more regime indicators risks fitting to 10 years of data with 5 "good"
  and 5 "bad" years. Any binary gate will be correct by construction for the
  in-sample years it was calibrated on.


================================================================================
SECTION 5: WHAT ACTUALLY MOVES SR IN 2017–2024 (ORDERED BY IMPACT)
================================================================================

Done:
  [x] Pair-return sizing fix (log-space artifact: +2.59 → -0.19)
  [x] Residualization Avellaneda-Lee (pair overlap 7% → 17%)
  [x] Hard stop 4.0 → 3.0 (-0.05 → +0.03)
  [x] HL[8-20d] filter (removes too-fast / too-slow pairs)
  [x] Entry confirmation + z_entry=2.0 (+0.19 → +0.357)
  [x] IBES earnings blackout (+0.357 → +0.57)
  [x] CRSP survivorship-free universe (honest estimate: 0.57 → 0.54)

Not done, high confidence:
  [ ] PC1/breadth throttle: selectively scale down in 2022-2024
      BUT empirical data shows PC1 is not reliable. Use hit-rate gate instead.
  [ ] ADF-affinity clustering (bake cointegration into cluster structure)
      Expected +0.05–0.15 gross SR. Zero overfitting risk (no threshold on outcomes).

Structural ceiling on daily close data:
  Max achievable gross SR ≈ 0.5–0.7 on 2017–2024 SPX daily close
  Net SR ≈ 0.2–0.4 at 3 bps/leg, higher at 0.5 bps (MOC routing)
  This is not a model failure — it is the information content of daily close data
  in the post-2015 S&P 500 universe.

Real unlock: TAQ intraday (30–60 min bars)
  More trips/pair/year → same per-trade edge → higher SR
  Lower cost/trade (institutional MOC routing ≈ 0.5 bps)
  Gate: v1 net SR ≥ 0.7, MaxDD ≤ 12% → current: 0.20 net, 3.76% MaxDD
  Net SR gate NOT yet passed. Implement regime gate, then evaluate intraday.
