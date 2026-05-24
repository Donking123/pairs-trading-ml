# Phase 1 — Decision Log

Each entry: **decision · alternatives considered · rationale · status**.

For cross-phase decisions (paper vs proposal reconciliation, build philosophy), see
`../../notes/strategy-reconciliation.md` and `../../notes/concepts-walkthrough.md`.

---

## D1.1 — Build the pipeline as a "vertical slice" before deepening
- **Decision**: build the full pipeline end-to-end on the SSD metric only, then layer PC and factor-beta as 1-line swaps.
- **Alternatives**:
  - (a) Build each module fully (all 4 metrics in `distances.py`) before any backtest.
  - (b) Build SSD-only pipeline first (chosen).
- **Rationale**: alternative (a) means weeks of build before any number lands; bugs in
  one module mask bugs in another; first real Sharpe is months out. (b) gets a
  paper-comparable number in days; subsequent metrics reuse the validated engine.
- **Status**: ✅ executed.

## D1.2 — OPTICS xi=0.10 (locked)
- **Decision**: lock `OPTICS_XI = 0.10` in `src/config.py`.
- **Process**: tuned 2026-05-24 via `phases/phase1/notebooks/02_xi_tuning_sweep.py`:
  - Tried `xi ∈ {0.05, 0.10, 0.15}` on Dec 2023 (paper's reported eval date).
  - `xi=0.10` produced 47 clusters (paper: 48 ±5) and purity 0.871.
  - Validated on Dec 2010 (34 clusters) and Dec 2015 (33 clusters) — sensible values,
    not lucky outliers.
- **Bias discipline**: tuning targeted *cluster count* to match the paper, not
  *Sharpe*. Locked once chosen; not re-tuned after seeing trading performance.
- **Status**: ✅ locked in `src/config.py`.

## D1.3 — OLS hedge ratio (frozen for trading month)
- **Decision**: use OLS β estimated on the 3-year formation window, **frozen** for
  the trading month.
- **Alternatives**:
  - RLM (Tukey biweight) — promoted to Phase 3 robustness cell.
  - Kalman dynamic-β — Phase 4 optional extension.
- **Rationale**: matches paper's convention; Engle-Granger two-step (Phase 2) requires
  OLS for the ADF residual to be well-defined; RLM only changes outlier sensitivity
  on the formation window, which has 756 days — outliers are diluted anyway.
- **Status**: ✅ implemented in `src/spread.py::fit_hedge_ratio`.

## D1.4 — Equal-dollar position sizing
- **Decision**: at trade entry, hold $0.50 long the long leg and $0.50 short the short
  leg. γ enters only via the spread/z-score signal — not the sizing.
- **Alternatives**: γ-weighted shares (1 share A, γ shares B). Differ when γ ≠ 1.
- **Rationale**: paper convention (Gatev-Goetzmann 2006 lineage). Equal-dollar gives
  literally market-neutral net dollar exposure at entry; γ-weighted does not unless
  γ × P_B = P_A exactly.
- **Status**: ✅ implemented in `src/backtest.py::_equal_dollar_daily_return`.

## D1.5 — Equal-weight allocation across currently-open pairs
- **Decision**: daily portfolio return = mean of returns across pairs in position.
- **Alternatives**: weight by |entry-z|; fixed slots (10/20/50); volatility-targeted.
- **Rationale**: paper convention. Alternatives are explored as Phase 4 sensitivity.
- **Status**: ✅ implemented in `src/backtest.py::run_one_month` aggregation block.
- **Note**: this is suspected as one cause of the Sharpe gap (dilution of strong
  signals). Will be tested as a Phase 3 sensitivity.

## D1.6 — t+1 close-to-close execution
- **Decision**: signal at close[t] takes effect on day t+1's close-to-close return.
- **Alternatives**: open[t+1] execution (would need open-price data); same-day close.
- **Rationale**: realistic without requiring open prices we don't have in the panel;
  matches paper's stated "no look-ahead" but doesn't claim sub-second latency.
- **Status**: ✅ implemented (positions evolve at close in `run_one_month`).

## D1.7 — Option B (code-dependent) delisting fallback
- **Decision**: when `dlret` missing, infer from `dlstcd`:
  | code range | cause | fallback |
  |---|---|---|
  | 200–299 | M&A | 0% |
  | 300–499 | liquidation/dropped | −30% |
  | 500–599 | exchange-related/OTC | −5% |
  | 600+ | other | 0% |
- **Alternatives**:
  - Option A: flat −30% always.
  - Option C: drop the pair entirely from that month.
- **Rationale**: A over-penalises M&A (most missing-`dlret` cases); C is a hidden
  survivorship bias (excludes the worst events). B is the calibrated middle.
- **Status**: ✅ implemented in `src/backtest.py::_delisting_fallback_return`.
- **Empirical impact**: only 31 trades / 12,255 (0.3%) affected — small lever in
  practice, but the right default for realism.

## D1.8 — Force-close at month end (no carry-over)
- **Decision**: any position still open on the last trading day of the trading month
  is force-closed at that day's close.
- **Alternatives**: carry into next month with the *same pair selection*; carry until
  z hits zero regardless of month boundary.
- **Rationale**: paper convention; pair selection re-runs each month so the same pair
  may be re-selected (and the new month's trade starts from a fresh γ); carry-over
  complicates accounting.
- **Status**: ✅ implemented.
- **Empirical impact**: **88.4% of all trades end via force-close**. They have a
  mean return of −32 bps. This is the strategy's biggest drag and the primary
  Phase-2 target.

## D1.9 — No stop-loss in core (Phase 1)
- **Decision**: in the faithful core, do not stop trades out at any |z| level.
- **Alternatives**: 3.5σ stop (proposal default).
- **Rationale**: matches paper's reported Sharpe (computed without a stop); the
  realism variant (Phase 4) will report Sharpe with vs without 3.5σ stop as a
  comparison cell.
- **Status**: ✅ implemented as `STOP_LOSS_SIGMA = None` in `config.py`.

## D1.10 — Strict survivorship filter
- **Decision**: a stock is in the formation universe **only if** continuously a S&P
  500 constituent for the *entire* 3-year formation window.
- **Alternatives**: in-S&P-at-start; in-S&P-at-any-point.
- **Rationale**: rigorous survivorship-bias prevention. Matches Phase 0 design.
- **Empirical**: yields approximately 407 stocks per Dec-2023 window vs likely approximately 500 in the paper.
  Suspected as a Sharpe-gap contributor (smaller universe → fewer good pairs).
  Will sensitivity-test softening this in Phase 3 / Phase 4.
- **Status**: ✅ implemented in `src/panel.py::formation_window_panel`.

## D1.11 — Accept Sharpe gap rather than re-tune SSD
- **Decision**: CP1 Sharpe is 0.589 vs paper's 0.88 ±0.15 (gap −0.29). Move to
  Phase 2 instead of chasing 0.88 on SSD.
- **Alternatives**:
  - Tune force-close convention, allocation scheme, z-score window, etc.
  - Try different universe filters / softer survivorship.
- **Rationale**: the paper's *real* edge is the PC metric (Sharpe 1.01), not SSD.
  Time better spent on PC + cointegration filter (Phase 2) which directly attacks
  the bigger lever — the −32 bps force-close drag. Per-Sharpe arithmetic: halving
  that drag should lift Sharpe from 0.59 to ~0.9.
- **Documentation**: gap is honestly documented; not hidden. Likely causes in
  `phase1_complete_reference.ipynb` §8.4.
- **Status**: ✅ decided 2026-05-24.

---

## Open questions deferred to later phases

- Q1 — Does PC distance close the Sharpe gap on its own? (Phase 2)
- Q2 — Does the cointegration filter halve the force-close drag? (Phase 2)
- Q3 — Does softening the survivorship filter help? (Phase 3 sensitivity)
- Q4 — Does |entry-z|-weighted allocation outperform equal-weight? (Phase 3)
- Q5 — Does open-price execution (vs close-to-close) materially change Sharpe? (Phase 4)
