# Phase 6 — Decisions (implementation-correctness review fixes)

Phase 6 implements the fixes from the 2026-06-10 code review of `src/`. Every fix sits
behind its own engine flag whose **default reproduces the Phase 5 engine bit-identically**
(verified: the full pre-Phase-6 test suite passes unchanged, plus
`tests/test_corrections_synthetic.py::test_inert_flags_are_bit_identical`). This lets us
run the corrections **one at a time** and attribute any Sharpe change to a specific fix —
same discipline as the Phase 3 robustness cells.

None of these flags was tuned on Sharpe. They are *correctness* changes, decided from
code inspection before any Phase 6 backtest was run; whatever they do to Sharpe, we report.

---

## D6.1 — MacKinnon p-values for the Engle-Granger test  (`coint_pvalue="mackinnon"`)

**Flaw.** `engle_granger` ran `adfuller()` on the Step-1 OLS *residuals* and used its
p-value. The ADF distribution assumes an **observed** series; residuals from an estimated
cointegrating regression are biased toward stationarity, so raw-ADF p-values on them are
anti-conservative — the 5% gate admits **more** pairs than a true 5% Engle-Granger test.
The correct distribution is MacKinnon's (implemented in
`statsmodels.tsa.stattools.coint`).

**Fix.** `pvalue_method="mackinnon"` uses `coint(A, B, trend="c", autolag="aic")` for the
p-value of each direction. γ, α, residuals, the half-life, and the min-p two-direction
convention (D2.5) are unchanged — only the p-value source differs. Synthetic check:
MacKinnon p > raw-ADF p on independent random walks; a planted cointegrated pair still
passes.

**Expected effect.** Fewer pairs pass the filter → filtered cells become more selective.
Direction on Sharpe unknown a priori (Phase 4 found the filter sheds alpha; a stricter
filter could shed more, or keep only truer reverters).

## D6.2 — Delisting corrections  (`delisting_fix=True`)

Four related flaws, bundled because they are all "the delisting event is handled
slightly wrong" and individually tiny:

1. **Code-map semantics shifted by one CRSP bucket.** The fallback map (used only when
   `dlret` is missing) labelled 300s "liquidation → −30%" and 500s "exchange-related →
   −5%". CRSP: 300s are *exchanges* (issue swapped in a reorg — roughly neutral), 400s
   are liquidations, and **500s are "dropped by the exchange"** — the bankruptcy /
   insufficient-capital class where `dlret` is most often missing and Shumway (1997)
   estimates ≈ **−30%**. Corrected map: 200–399 → 0%, 400–499 → −30%, 500 & 520–584 →
   −30%, 501–519 (moved to another exchange) → 0%. This matters most for the **v2/OOS
   pull**, where `wrds_pull.py` writes sentinel `dlstcd=500` — under the old map every
   missing-`dlret` OOS delisting got a lenient −5%.
2. **Compounding.** CRSP measures `dlret` *after* the last close, so the delist-day
   return is `(1+ret)·(1+dlret)−1`; the old engine *overwrote* `ret` with `dlret`.
3. **Weekend/holiday delist dates.** The engine applied the delisting return only when
   `dlstdt` exactly matched a panel date; otherwise the event silently never fired and
   the position accrued NaNs. Fixed by snapping `dlstdt` to the next trading day inside
   the month.
4. **NaN-z delist day.** The forced delisting close sat *behind* the `z is NaN → skip
   day` guard, so a NaN z on the delist day skipped the close. A delisting close is
   mechanical; with the fix it fires regardless of z.

**Expected effect.** Slightly **lower** (more honest) returns in months containing
missing-`dlret` delistings; main impact on the 2024–25 OOS numbers.

## D6.3 — Stop-loss cooldown  (`stop_cooldown=True`)

**Flaw.** After a stop-out at |z| ≥ 3.5, the entry rule sees |z| ≥ 2 still true the next
day and re-opens the **same** position, paying entry costs again. The stop therefore
mostly converts one position into close/reopen churn — which contaminates the Phase 4
finding that "the stop is net-negative (+0.11–0.14 Sharpe from dropping it)".

**Fix.** After a stop-loss exit the pair is *disarmed*: no re-entry until |z| has come
back inside the entry band (|z| < entry σ) once. Re-arming uses the same
look-ahead-safe z.

**Expected effect.** Only realism cells with a stop change. A fair re-test of the stop:
if the stop is *still* net-negative with the cooldown, the Phase 4 conclusion stands on
solid ground (and gets stronger); if not, the conclusion was an artifact.

## D6.4 — No entries on the month's last trading day  (`block_last_day_entry=True`)

**Flaw.** A signal on the final trading day opened a position at that day's close that
was force-closed at the *same close* (zero return, but charged entry+exit costs). The
aggregation layer then dropped the pair entirely (`days_open == 0`), so the costs
vanished from portfolio P&L, the Trade record was lost while `n_pairs_traded` still
counted it — and under carry-over the position was carried with its entry cost never
booked.

**Fix.** Don't enter on the last trading day. (The cleanest of the possible repairs:
the alternative — booking the degenerate round trip — just adds guaranteed-loss noise.)

**Expected effect.** Frictionless no-carry cells: no portfolio change (the degenerate
trades had zero P&L). Realism/carry cells: small but strictly-correct improvement in
cost accounting.

## D6.5 — Execution delay  (`execution_delay=1`)

**Flaw (honesty, not bias).** Fills happen at the same close the z-signal is computed
from. The first P&L day is t+1 (hence the "t+1 execution" label), but the *fill price*
is the signal-day close — you cannot observe a close and trade at it. Gatev et al.
(2006) report that delaying one day materially cuts pairs-trading returns, so this is
the most likely examiner challenge.

**Fix.** `execution_delay=1`: a signal observed at close t fills at close t+1 (entry
*and* exit; the recorded `entry_z` stays the signal-day dislocation). Delisting closes
are mechanical and never delayed; a signal still pending at month-end lapses (the
month-end force-close/carry rule applies as usual).

**Expected effect.** Sharpe **down** — the open question is by how much. If the edge
survives a one-day delay, the strategy is much more credible; if it dies, the
frictionless 1.03 was substantially a fill-at-signal-close artifact. Either result is
a finding.

## D6.6 — Trade the validated direction  (`use_coint_gamma=True`)

**Flaw.** The cointegration filter picks the better of A-on-B vs B-on-A (min p, D2.5)
and reports that direction's γ — but the engine then traded the A-on-B OLS γ
regardless. When B-on-A won, the traded spread was not the one whose stationarity the
filter validated.

**Fix.** With the flag on (filtered cells only), each kept pair trades the **winning
direction's** spread with the **filter's γ**. Carried positions keep their frozen
orientation and γ exactly as before.

**Expected effect.** Small; affects only filtered cells, only the ~half of pairs where
B-on-A wins. Internal consistency more than performance.

## D6.7 — NaN guard in portfolio aggregation  (always on)

**Flaw.** A NaN pair-day return (a data problem by definition) was silently skipped by
`.mean(skipna)` while `(ret != 0)` counted the NaN pair as "open" — masking exactly the
kind of bug D6.2(3) turned out to be.

**Fix.** Loud per-month warning naming the offending pairs, then explicit `fillna(0)`.
Numerically identical when no NaNs exist (all current cells), so this is always on, not
flagged. Also in this bucket: `filter_cointegrated_pairs`'s
`except (ValueError, Exception)` tightened to a plain `except Exception`.

---

## What Phase 6 deliberately does NOT change

* **Reporting conventions** (rf=0 Sharpe, fully-invested daily mean across open pairs,
  monthly compounding) — kept for comparability with Phases 1–5 and the paper. The
  evaluator (`02_evaluate_corrections.py`) additionally reports an **excess-of-rf
  Sharpe** as a reporting column, without touching the engine.
* **Multiple-testing correction** for the per-month ADF battery — noted as future work;
  D6.1 already tightens the same gate in a better-grounded way.
* **No re-tuning of any hyperparameter** (xi, thresholds, half-life bounds) — Phase 2
  lock stands.
