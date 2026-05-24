# Phase 2 — Decision Log

Decisions will be added as we build. Each entry: **decision · alternatives considered · rationale · status**.

**Resolution status:**
- D2.1 ✅ resolved 2026-05-24 (SPY-only)
- D2.2 ✅ resolved in Phase 1 reconciliation (comparison cell, not gate)
- D2.3 ✅ resolved 2026-05-24 (half-life [5, 60] days)
- D2.4 ✅ resolved 2026-05-24 (AR(1))
- D2.5 ✅ resolved 2026-05-24 (lower p-value of both directions)
- D2.7 ✅ resolved 2026-05-24 (metric-specific xi; `OPTICS_XI_PC = 0.04`) — added
  after building `pc_distance` and discovering the SSD-tuned xi underclusters PC.

All Phase 2 open decisions resolved.

---

## Decisions (resolved 2026-05-24, locked for Phase 2 build)

### D2.1 — Market adjustment for PC distance: SPY beta or Fama-French?
- **Default**: regress each stock's returns on SPY return → use residuals for the
  pairwise PC distance. Matches paper §3.
- **Alternative**: use Fama-French 3-factor (MKT, SMB, HML) residuals (we have FF
  factors in `data/ff_factors.parquet`).
- **Resolved 2026-05-24**: ✅ **(a) SPY-only** chosen as paper-faithful baseline.
  FF 3-factor logged as **a future robustness improvement** to test if PC cluster
  count is materially off from the paper's 109.
- **Status**: locked for the initial Phase 2 build; FF variant queued for Phase 3.

### D2.2 — Cointegration filter as gate or comparison cell?
- **Default per reconciliation #4**: report BOTH PC core (no filter) and PC + filter
  side by side. The cointegration filter is a *tested filter*, not a mandatory gate.
- **Confirmed** in Phase 1 strategy-reconciliation.

### D2.3 — Half-life bounds inside the filtered variant
- **Default per reconciliation #5**: `5 ≤ half-life ≤ 60` trading days.
- **Sensitivity** (post-build): test `[3, 90]` and `[10, 40]` as alternatives.
- **Resolved 2026-05-24**: ✅ **(a) [5, 60] days** chosen. Locked.
- **Status**: lock in `src/config.py` once `cointegration.py` is built.

### D2.4 — Half-life estimation method
- **Resolved 2026-05-24**: ✅ **(a) AR(1) discrete-time**. Fit
  `s_t = c + ρ · s_{t-1} + ε_t`, then `τ_{1/2} = -ln(2) / ln(ρ)`.
- **Rationale**: equivalent to OU under reasonable assumptions; simpler to implement
  and unit-test.
- **Alternative**: OU continuous-time — queued only if AR(1) shows pathological
  behaviour in unit tests.

### D2.5 — Engle-Granger test direction
- **Resolved 2026-05-24**: ✅ **(a) Lower p-value of both directions**. Run both
  A-on-B and B-on-A, take the smaller p-value as the test result.
- **Rationale**: literature convention; the small bias toward acceptance is
  documented; we report it honestly.
- **Alternative**: Johansen test queued as a robustness check (Phase 3).

### D2.7 — Metric-specific xi (locked 2026-05-24, post-build)
- **Issue discovered after building pc_distance**: applying SSD's locked `xi=0.10`
  to PC distance produces only 66 clusters on Dec 2023 (paper: 109). PC distance
  lives on a different numerical scale than SSD ([0, 2] vs [0, ~3000]), so the
  same steepness threshold has different effective behaviour.
- **Resolution**: introduce a **per-metric xi** in `src/config.py`:
  - `OPTICS_XI = 0.10` for SSD (Phase 1, unchanged)
  - `OPTICS_XI_PC = 0.04` for PC (Phase 2, newly locked)
- **Tuning process** (same discipline as Phase 1):
  - Sweep `{0.02, 0.03, 0.04, 0.05, 0.07, 0.10}` on Dec 2010 / Dec 2015 / Dec 2023.
  - All values >= 0.02 land in 64–84 clusters on Dec 2023 — function plateaus
    around 84, so xi=0.02 is at the cliff. Locked xi=0.04 firmly inside the
    stable region.
- **Result on Dec 2023**: 81 PC clusters (paper: 109 ±10 — **out of tolerance**),
  purity 0.937 (paper: 0.84 — *higher*, cleaner clusters).
- **Honest framing**: undercount is approximately consistent with our smaller
  universe (407 stocks vs the paper's likely ~500); PC/SSD cluster-count ratio
  ~1.79× matches paper's 2.27× direction but not magnitude. Documented openly;
  not papered over by overfitting xi below 0.02.
- **Status**: locked. Sweep script: `phases/phase2/notebooks/02_xi_tuning_pc.py`.
- **Phase 1 invariant**: this change is **purely additive** — Phase 1's `OPTICS_XI=0.10`
  is bit-identical, so Phase 1 results (SSD Sharpe 0.589) reproduce exactly when
  the upcoming 2×2 backtest grid runs the SSD-baseline cell.

---

## Decisions made (none yet — phase not started)

(Will fill in as we go.)
