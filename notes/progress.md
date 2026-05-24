# Project Progress — QF621 Pairs Trading (ML Clustering)

**Last updated:** 2026-05-25 · **Status:** **Phase 2 COMPLETE — paper replicated + published to GitHub.** PC core Sharpe 1.028 vs paper 1.01 ✅. Ready for Phase 2.5 (factor-beta extension).

> Session handoff — **read this first to resume.** The project is a from-scratch
> replication of **Rotondi & Russo (2025)** clustering-based pairs trading, plus a
> **factor-beta clustering** extension, for the QF621 group project.

## 🌐 GitHub repo (published 2026-05-24)

- **URL:** https://github.com/Donking123/pairs-trading-ml
- **Visibility:** PUBLIC (was originally private, flipped to public same day after audit)
- **Working folder:** `~/Documents/repos/pairs-trading-ml/` — this is the folder wired to
  GitHub (origin/main); commit + push from here going forward.
- **Old working folder:** `~/Documents/Pairs Trading - Machine Learning/pairs-trading-ml/`
  still exists with the `data/` cache (140MB, gitignored). Don't commit from there.
- **License:** MIT (© 2026 Don)
- **Author email on commits:** `Donkingyappy1@users.noreply.github.com` (privacy-preserving)

### Collaborators
| User | Status | Permission |
|---|---|---|
| `Donking123` (Don) | Owner | admin |
| `deepakgarrepalli1998` | ✅ Accepted | write |
| `santapris` | ⏳ Invited 2026-05-24 14:04 UTC | write (pending) |
| `nglei1999` | ⏳ Invited 2026-05-24 15:03 UTC | write (pending) |

Re-check: `gh api "/repos/Donking123/pairs-trading-ml/collaborators" --jq '.[].login'`

---

## ▶ Quick resume — do this first

1. Open **`phases/phase1/README.md`** for the Phase 1 summary (what we built, why, headline results).
2. Open **`phases/phase1/notebooks/phase1_complete_reference.ipynb`** for the full walkthrough.
3. Open **`phases/phase1/notebooks/phase1_pnl_attribution.ipynb`** for the P&L attribution
   (bimodal duration finding — load-bearing for Phase 2 design).
4. Open **`phases/phase2/README.md`** for the Phase 2 plan and CP2 targets.
5. Skim this file + `notes/strategy-reconciliation.md` for cross-phase context.
6. Start **Phase 2**: build `src/distances.py::pc_distance` and `src/cointegration.py`.

> **New structure as of 2026-05-24:** each phase has its own folder under `phases/` with
> README, decisions log, notebooks, and results. See `phases/README.md` for the layout.

**Two key findings from the attribution that shape Phase 2:**
- **Bimodal pattern**: 11.4% of trades cleanly revert (mean +471 bps, median 14d hold);
  88.4% are force-closed at month-end (mean -32 bps). Strategy's entire net P&L
  (+30.41) = reversions (+65.58) + force-closes (-34.25) + delisting (-0.92).
- **Refined Phase 2 thesis**: Phase 2's job isn't just "switch to PC for better Sharpe."
  It's **"reduce the force-close drag by rejecting fragile pairs"** — the cointegration
  filter directly attacks the +88% trades that lose 32 bps each. A 50% reduction in
  force-close drag would push Sharpe from 0.59 to ~0.9.

---

## Reference docs

### Per-phase artefacts (under `phases/`)

| Doc | What it is |
|---|---|
| **`phases/README.md`** | Phase folder structure & navigation |
| **`phases/phase1/README.md`** | **Phase 1 summary — what we built, why, headline results** |
| **`phases/phase1/decisions.md`** | Phase 1 decision log (11 decisions made) |
| **`phases/phase1/notebooks/phase1_complete_reference.ipynb`** | Master walkthrough — concept + worked examples + real data + CP1 + Phase 2 roadmap |
| **`phases/phase1/notebooks/phase1_pnl_attribution.ipynb`** | Attribution + bimodal duration finding |
| **`phases/phase2/README.md`** | **Phase 2 plan — CP2 targets, build order, 2×2 scorecard** |
| **`phases/phase2/decisions.md`** | Phase 2 decision log (5 open decisions, none resolved yet) |

### Cross-phase docs (under `notes/`)

| Doc | What it is |
|---|---|
| `notes/strategy-reconciliation.md` | The 12 proposal-vs-paper decisions + paste-ready reworked Strategy A |
| `notes/concepts-walkthrough.md` | Topic-organised reference: phase plan, conventions, rolling window, synthetic test walkthrough, glossary |
| `notes/phase-0-data-spine.md` | Phase 0 reference note |
| `../ssrn-5080998.pdf` | Anchor paper — Rotondi & Russo (2025) |
| `../Pair_Trading_Project_Proposal_Updated.pdf` | The updated QF621 proposal (post-update 2026-05-24) |

---

## Phase status

| Phase | Status |
|---|---|
| 0 — Data spine (WRDS pull) | ✅ complete |
| 1 — SSD vertical slice | ✅ complete (CP1 partial — clustering ✅; Sharpe 0.589) |
| **2 — PC distance + cointegration filter** | **✅ COMPLETE — paper replicated (PC core 1.028 vs paper 1.01)** |
| 2.5 — Factor-beta clustering extension (first-class) | 🔵 **NEXT (QF621 contribution)** |
| 3 — Robustness cells: Hierarchical algo, RLM hedge ratio, stop-loss variants | ⬜ pending |
| 4 — Realism, Alpaca forward test, write-up | ⬜ pending |

---

## Phase 2 — final results (2026-05-24)

Full 4-cell 2×2 grid (SSD/PC × filter on/off) run end-to-end across 251 months:

| Cell | Ours Sharpe | Paper Target | Δ | Verdict |
|---|---:|---:|---:|---|
| **PC core** | **1.028** | **1.01 ±0.15** | **+0.018** | **✅ matches paper** |
| PC + cointegration filter | 0.752 | 0.80 ±0.15 | −0.048 | ✅ matches paper |
| SSD + cointegration filter | 0.731 | 0.75 ±0.15 | −0.019 | ✅ matches paper |
| SSD core | 0.589 | 0.88 ±0.15 | −0.291 | ❌ below (same as Phase 1; invariant ✓) |

**Phase 1 invariant check ✅**: `ssd_core` Sharpe = 0.589 reproduces Phase 1's number to
4 decimals (Δ −0.0005). Confirms the backtest extension is bit-identical when args
default to Phase 1 behavior.

### Bimodal lever moved as predicted

| Cell | force_close mean | Outliers \|rt\| > 50% |
|---|---:|---:|
| SSD core | −32 bps | 5 (2008-09 crisis) |
| **PC core** | **−11 bps (-65%)** | **3** |
| PC + filter | −18 bps | **0** ⭐ |

PC core slashes the force-close drag by 65% per trade — the exact mechanism Phase 1
attribution forecast. PC + filter eliminates *every* outlier trade in the 21-year
sample (0 |rt| > 50% trades vs SSD's 5).

### Headline = PC core (Sharpe 1.028)

For the QF621 writeup:
- **Headline strategy**: PC core (Sharpe 1.028, paper 1.01)
- **Realism variant**: PC + filter (Sharpe 0.752, paper 0.80) — eliminates outlier blowups
- **Phase 1 baseline**: SSD core (0.589) — documented gap to paper

Full deliverables in `phases/phase2/notebooks/phase2_complete_reference.ipynb` and
`phase2_pnl_attribution.ipynb` (both executed with plots baked in).

---

## Phase 1 — final results

Full 251-month SSD backtest 2003–2023 run on 2026-05-24. Numbers from
`results/ssd_core_monthly.parquet` + `notebooks/06_evaluate_cp1.py`:

| Metric | Ours | Paper target | Verdict |
|---|---:|---:|---|
| # SSD clusters (Dec 2023) | 47 | 48 ±5 | ✅ |
| Purity vs SIC | 0.871 | 0.81 ±0.05 | ✅ |
| GOOG/GOOGL co-cluster | ✓ | ✓ | ✅ |
| **Annualised Sharpe** | **0.589** | **0.88 ±0.15** | ❌ outside tolerance |
| Annualised return | 3.01% | ~5% | low |
| Annualised vol | 5.28% | ~5.6% | ✓ matches |
| Max drawdown | -14.3% | ~-15% | ✓ matches |
| Hit rate | 57.4% | ~57% | ✓ matches |
| Total trades | 12,255 | — | — |
| Exit: force-close | 88.4% | (similar) | not the cause of gap |

**CP1 partial pass:** clustering side matches paper (47 vs 48; purity 0.871 vs 0.81);
Sharpe is 0.29 below the lower tolerance band. The risk profile (vol, drawdown, hit
rate) matches; only the *return* numerator is short. Likely causes ranked in the
notebook §8.4. Conclusion: don't keep tweaking SSD; move to PC (paper's real edge).

## Phase 1 attribution findings (2026-05-24)

Full breakdown in `notebooks/phase1_pnl_attribution.ipynb`. Key numbers:

### Holding duration
- Mean **19.2 days**, median **22 days**, max 30 days (month-end cap)
- 25th / 75th percentile: 12 / 28 days

### Per-trade P&L
- Mean **+25 bps**, median **+19 bps**, std **4.14%**
- Skew −0.13, **excess kurtosis +31** (extreme fat tails)
- Best / worst: +64.2% / −63.0%

### The BIMODAL pattern by exit reason
| Exit reason | n | % | mean dur | mean P&L | win rate | Total |
|---|---:|---:|---:|---:|---:|---:|
| **reversion** | 1,392 | 11.4% | 14.5d | **+471 bps** | **99.2%** | **+65.58** |
| force_close | 10,832 | 88.4% | 19.9d | **−32 bps** | 48.2% | **−34.25** |
| delisting | 31 | 0.3% | 17.1d | −297 bps | 32% | −0.92 |
| **NET** | 12,255 | 100% | 19.2d | +25 bps | 54% | **+30.41** |

**Net P&L identity: +65.58 (rev) − 34.25 (FC) − 0.92 (delist) = +30.41**

The strategy's entire profit comes from the 11.4% of trades that fully revert. The
other 88.4% drag returns by 32 bps each. Reducing the force-close drag is the #1 lever.

### By duration bucket (where trades flip from profitable → losing)
| Bucket | n | share | mean P&L | win rate |
|---|---:|---:|---:|---:|
| 1–3 d | 830 | 6.8% | **+68 bps** | 58% |
| 4–7 d | 1,025 | 8.4% | **+99 bps** | 65% |
| 8–14 d | 2,069 | 16.9% | **+97 bps** | 62% |
| 15–21 d | 2,179 | 17.8% | **+53 bps** | 56% |
| **22–35 d** | 6,152 | 50.2% | **−28 bps** | 48% |

**Trades held longer than 21 days are net losing.** Half the sample is here.

### Direction & sector balance
- Direction symmetric: long_spread +17.0 / short_spread +13.4 (no hidden bias) ✓
- Top 3 sector pairs = 54.2% of P&L (moderate concentration)
- Mfg/Mfg 23.5%, Fin/Fin 17.0%, Fin/Mfg 13.7%

### Macro-regime concentration ⚠
- 2007–09 GFC = **39.3% of P&L** from 14% of trades
- 2010–19 expansion (calm decade) = **18.1% of P&L** from 46% of trades
- Crisis dependency is real — the strategy harvests dislocations

### Refined Phase 2 thesis
Phase 2's job is not just "switch to PC for higher Sharpe." It's **reduce the
force-close drag**. Two levers, both in Phase 2:
1. **Engle-Granger cointegration filter** → reject pairs that won't revert →
   shrink force-close population OR shift its mean from −32 bps toward neutral.
2. **PC distance** → finds more idiosyncratic-mean-reverting pairs → higher
   reversion rate among traded pairs.

If cointegration filter halves the force-close drag (−34 → −17), net per-trade
P&L jumps from +30 to +48 → Sharpe rises from 0.59 toward 0.9.

---

## What's built (`pairs-trading-ml/`)

```
config.py         ✅ paths + constants + OPTICS hyperparams (xi=0.10 locked) + exit/stop
wrds_pull.py      ✅ Phase 0 data pull (run; data cached)
distances.py      🔵 ssd_distance() built + tested; pc_distance pending (Phase 2)
clustering.py     ✅ cluster_optics, purity_index, clusters_to_pairs,
                     cluster_summary, sic_division — synthetic-tested + real-data run
panel.py          ✅ formation_window_panel, ticker_lookup, siccd_lookup,
                     total_return_price
spread.py         ✅ fit_hedge_ratio (OLS), spread_series, rolling_zscore (lookback-safe)
backtest.py       ✅ run_one_month, run_backtest, MonthResult, Trade,
                     equal-dollar P&L, Option-B delisting handling
performance.py    ✅ compute_metrics (Sharpe/Sortino/Calmar/MDD/hit-rate), format_metrics
cointegration.py  ⬜ not created (Phase 2)  ← NEXT
factors.py        ⬜ not created (Phase 2.5)
data/             ✅ 5 parquet panels cached (shared)
tests/            ✅ test_clustering / spread / performance _synthetic.py (18/18 pass)
notes/            ✅ phase-0-data-spine.md, strategy-reconciliation.md,
                     concepts-walkthrough.md, progress.md  (cross-phase)
phases/
  phase1/         ✅ README.md, decisions.md
    notebooks/    ✅ phase1_complete_reference.ipynb, phase1_pnl_attribution.ipynb,
                     01..07 (.py demos), _build_*.py generators
    results/      ✅ ssd_core_monthly.parquet (251), ssd_core_trades.parquet (12,255)
  phase2/         🔵 README.md (plan), decisions.md (open questions)
    notebooks/    ⬜ to be built
    results/      ⬜ to be populated
README / requirements.txt / .gitignore  ✅
```

---

## Phase 0 — complete

Data pulled from WRDS CRSP, cached as 5 parquet files in `data/`. Final universe =
**991 survivorship-bias-free ordinary US common stocks**, **288 months**, 6,037
trading days, 2000-01-03 → 2023-12-29. The 991 (vs ~1,070 raw) is fully explained —
79 dropped were foreign-incorporated / REITs / funds (correct per the 10/11 filter).
Detail in `notes/phase-0-data-spine.md`.

## Phase 1 — complete

All sub-components built, synthetic-tested, and run end-to-end on real CRSP data.
- ✅ **1a** — `distances.ssd_distance()` + `clustering.py` + `panel.py`.
  Dec 2023 → 47 clusters / 0.871 purity / GOOG=GOOGL ✓ (paper: 48 / 0.81).
  xi=0.10 locked in `config.py` (validated on Dec 2010 + 2015).
- ✅ **1b** — `spread.py`: OLS γ + spread + 6-month rolling z-score with strict
  look-ahead protection. Showcase pairs (GOOG/GOOGL γ=1.0052, MA/V γ=0.9528, etc).
- ✅ **1c** — `backtest.py`: rolling 3y/1m loop, t+1 close-to-close execution,
  Option-B delisting handling. 251 months run.
- ✅ **1d** — `performance.py`: Sharpe + Sortino + Calmar + MDD + hit-rate, all
  synthetic-tested.

**CP1 (Phase 1 gate):** clustering ✅ (47/0.871 in tolerance); Sharpe ❌ (0.589 vs
0.88 ± 0.15). Gap is **risk-numerator only** — vol/drawdown/hit-rate all match.
Likely causes: equal-weight allocation dilution, smaller universe than paper,
z-score window edge cases. Documented in
`notebooks/phase1_complete_reference.ipynb` §8.4.

## Phase 2 — preview

Build order:
1. `src/distances.py::pc_distance` — Partial correlation distance on market-adjusted
   returns. Paper's winner: Sharpe 1.01.
2. `src/cointegration.py` — Engle-Granger ADF test + half-life [5, 60] day filter.
3. Run full backtest 4 ways: SSD/PC × {with, without} cointegration filter.

Expected after Phase 2:

|  | no cointegration filter | with cointegration filter |
|---|---|---|
| **SSD** | 0.59 (done) | (Phase 2) |
| **PC** | (Phase 2 target ≈ 0.8–1.0) | (Phase 2 target ≈ 0.7–1.0) |

---

## Key decisions locked

- **Data window (Path B):** keep the 2000–2023 pull; first 3 years are formation, so
  the **trading window is 2003–2023 (251 monthly returns confirmed)**. No re-pull.
- **Metric build order:** SSD ✅ → PC (Phase 2) → factor-beta (Phase 2.5) → PCA (bonus).
- **Priorities:** P1 = replicate the paper. Phase 1 SSD partial-passed; Phase 2 PC
  is the real validation.
- **Clustering:** OPTICS core (xi=0.10, locked); HDBSCAN + hierarchical as
  robustness cells (Phase 3).
- **Cointegration:** a *tested filter*, not a mandatory gate — run & report both
  with / without (Phase 2). Half-life [5,60]-day filter inside the filtered variant.
- **Realism:** t+1 execution, bid/ask transaction costs, realistic delisting losses,
  ~35 bps borrow cost, 3.5σ stop-loss in the realism variant (Phase 4).
- **Delisting:** Option B — code-dependent fallback (M&A → 0%, bankruptcy → -30%,
  OTC → -5%). Implemented in `src/backtest.py::_delisting_fallback_return`.
- **Live deployment:** kept as framing — validated honestly via the backtest **+ an
  Alpaca paper-trade forward test** (Phase 4, after the replication core).
- **All 12 proposal-vs-paper reconciliation points are decided** + 4 proposal-driven
  extensions promoted (hierarchical, factor-beta first-class, RLM hedge ratio,
  stop-loss variants) — see `notes/strategy-reconciliation.md`.

## Open (non-code) items

- Paste the **reworded Strategy A** (ready in `strategy-reconciliation.md`) into the
  proposal; do one consolidated proposal pass.
- Optional: add the one-line "clustering" sentence to the Investment Thesis.

---

## Next steps (Phase 2, in order)

1. Build `src/distances.py::pc_distance` — partial correlation on market-adjusted
   returns. Add a `pc_distance_synthetic` test.
2. Build `src/cointegration.py::engle_granger` — OLS spread regression + ADF p-value
   + half-life estimate via Ornstein-Uhlenbeck.
3. Update `backtest.py` to accept a `metric: "ssd" | "pc"` arg and an optional
   `cointegration_filter: bool`.
4. Run all four backtests (SSD/PC × filter on/off) → save to `results/`.
5. Build `notebooks/phase2_complete_reference.ipynb` — same structure as Phase 1.
6. Check **CP2**: PC Sharpe within ±0.15 of paper's 1.01.
