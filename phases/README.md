# Phase Folders

Each subfolder here contains one **phase** of the QF621 pairs-trading replication.

| Phase | Folder | Status | Headline result |
|---|---|---|---|
| 1 — SSD vertical slice (baseline) | `phase1/` | ✅ complete | SSD Sharpe 0.589 (paper 0.88) — partial pass; gap diagnosed in attribution |
| **2 — PC distance + cointegration filter** | **`phase2/`** | **✅ COMPLETE** | **PC core Sharpe 1.028 vs paper 1.01 ✅ — paper replicated** |
| 2.5 — Factor-beta clustering extension | (future) `phase2_5/` | ⬜ planned | QF621 group project's contribution |
| 3 — Robustness cells (hierarchical, RLM, allocation alts) | (future) `phase3/` | ⬜ planned | Sensitivity / robustness |
| 4 — Realism + Alpaca forward test + writeup | (future) `phase4/` | ⬜ planned | Bid/ask costs, 35bps borrow, live paper trade |

---

## What's in each phase folder

Same structure across phases for easy comparison:

```
phaseN/
├── README.md           ← Phase summary: goal, build, headline results
├── decisions.md        ← Decision log (what we chose and why)
├── notebooks/
│   ├── phaseN_complete_reference.ipynb    ← Master walkthrough (concept + real data + results)
│   ├── phaseN_pnl_attribution.ipynb       ← P&L attribution + concentration analysis
│   ├── 01..NN_*.py     ← Runnable demo scripts
│   └── _build_*.py     ← Notebook generators (regenerate the .ipynb after edits)
└── results/            ← Parquet outputs from this phase's backtests
```

---

## Why this structure

- **Shared code lives in `src/`** at the project root — the library evolves over phases
  (e.g. Phase 2 adds `pc_distance`, `cointegration.py`). Older phases keep working
  because we don't remove anything, just extend.
- **Phase-specific work lives in `phases/phaseN/`** — notebooks (the *deliverables*),
  results parquets, and the decision log explaining WHY we built it this way.
- **Cross-phase docs live in `notes/`** at the project root — paper-vs-proposal
  reconciliation, conventions glossary, Phase 0 data spine details.

**Result:** to read what was done in Phase 1, open `phase1/README.md`. To re-run any
notebook, paths work without modification because `src/` is found via a "walk up to
project root" helper that's robust to any depth.

---

## How to compare phases

Open multiple phases' README.md and the master `_complete_reference.ipynb` notebooks
side by side. The headline scorecards are identical in structure across phases so the
comparison is mechanical:

```
phase1/notebooks/phase1_complete_reference.ipynb   §7 (Headline results)
phase2/notebooks/phase2_complete_reference.ipynb   §7 (Headline results, when built)
```
