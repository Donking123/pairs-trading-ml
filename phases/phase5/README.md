# Phase 5 — Position Carry-Over + Passive Execution

Two improvements over the force-close-every-month engine, motivated by the Phase-4 finding
that **88.4% of trades exit via force-close at a mean loss** — positions are guillotined at
month-end before their spreads can revert.

1. **Position carry-over** — a position still open at month-end is HELD into the next month
   (capped at `MAX_CARRY_MONTHS = 3`, tied to the 60-trading-day half-life ceiling) as long
   as the clustering still endorses the pair. Closes on reversion / stop / delisting / drop
   from the candidate set / cap.
2. **Passive execution** — shared `REALISM_PASSIVE` preset (limit orders,
   `spread_cost_multiplier = 0.5`), the Phase-4 cost-opt winner.

`carry_over=False` is bit-identical to the pre-Phase-5 engine (verified, Δ = 0.0).

## How to run

```bash
cd pairs-trading-ml

# 1) FAIL-FAST — the single most informative cell (carry, frictionless PC) vs F = 1.028
python phases/phase5/notebooks/01_run_carryover_grid.py --metrics pc --cells E

# 2) the decisive net-of-cost test (carry vs no-carry, passive)
python phases/phase5/notebooks/01_run_carryover_grid.py --metrics pc --cells B,D

# 3) full grid (E,B,D × pc,factor,ssd + PC sensitivity), background — each cell ~1-3h
nohup python phases/phase5/notebooks/01_run_carryover_grid.py --sens \
    > phases/phase5/results/grid.log 2>&1 &

# 4) evaluate (runs against whatever cells exist on disk)
python phases/phase5/notebooks/02_evaluate_phase5.py

# 5) frozen out-of-sample (needs data_through_2025/, same as Phase 4)
python phases/phase5/notebooks/03_forward_test_carry.py --data-dir data_through_2025
```

## Cells

| Cell | carry | execution | stop | vs |
|------|-------|-----------|------|----|
| F (reused) | off | frictionless | none | — baseline (phase1/2/2.5 cores) |
| **E** | on | frictionless | none | E vs F = signal effect |
| **B** | off | passive | none | — |
| **D** ⭐ | on | passive | none | D vs B = realistic operating point |
| Dstop | on | passive | 3.5σ | PC-only stop sensitivity |
| Dmkt | on | marketable | none | PC-only execution sensitivity |

## Result so far (see `decisions.md` D5.7)

**PC frictionless, carry (E) vs no-carry (F):** Sharpe **1.019 vs 1.028** (flat) — but
force-close **91.5%→62.8%**, mean hold **19.9→49.7d**, trades **−39%**, max-DD **−5.7%→−3.9%**.
Carry-over is a **turnover/cost lever, not a signal lever**; its payoff must show up net of
costs (cells D vs B). The 39% turnover cut is the reason to expect a net-of-cost win.

## Files

- `notebooks/01_run_carryover_grid.py` — runs the new cells, writes `*_monthly/_trades.parquet`
  + `carryover_grid_summary.csv`.
- `notebooks/02_evaluate_phase5.py` — E-vs-F and D-vs-B tables + carry diagnostics.
- `notebooks/03_forward_test_carry.py` — frozen E/D on 2024-2025.
- `decisions.md` — locked design decisions + results log.
