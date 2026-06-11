# Presentation — how to open & present

## Files
- **`QF621_pairs_trading_deck.pdf`** — **the deck. Open this and present.** 15 clean 16:9 slides, charts built from the real backtest output. Looks identical on any machine, no tooling needed.
- **`build_deck.py`** — the script that generates the PDF (pure matplotlib). Edit text/layout here and re-run.
- **`assets/`** — the three charts (equity curve, Sharpe bars, mechanism), generated from `phases/phase2/results/*.parquet`.
- `QF621_pairs_trading_deck.md` — an optional Marp source (kept for reference; the PDF is the real deliverable).

## To edit & rebuild
1. Edit slide text/layout in **`build_deck.py`**.
2. Regenerate charts if data changed: re-run the figure block (reads the result parquets directly).
3. Rebuild the deck:  `python3 report/build_deck.py`  → overwrites the PDF.

## Presenting from PDF
Open in Preview (macOS) → View ▸ Slideshow (⌥⌘P), or any PDF viewer in fullscreen. Arrow keys advance slides.

---

# 90-second talk track (per slide)

1. **Title** — "I rebuilt a 2025 quant paper from scratch and matched its headline Sharpe."
2. **Problem** — pair selection across 1,000 stocks is the hard part; the paper uses ML clustering to pick economically-related pairs.
3. **Data** — institutional CRSP data, *survivorship-bias-free*, 24 yrs; first 3 yrs are formation so everything reported is out-of-sample.
4. **Pipeline** — distance → cluster → cointegration filter → z-score signal → rolling monthly backtest, 251 times.
5–7. **Method** — SSD vs PC distance; OPTICS (no preset cluster count); cointegration + |z|>2 entry / z=0 exit.
8. **Result** — "Sharpe 1.028 vs paper 1.01; all four variants land where the paper says."
9. **Equity curve** — "PC core is the smoothest; SSD+filter ends higher but choppier — risk-adjusted, PC wins."
10. **Scorecard** — PC halves vol and drawdown.
11–12. **Why it works** — I attributed P&L first, found the bimodal pattern, *then* engineered the fix; prediction confirmed (drag −65%, zero tail blow-ups).
13. **Rigour** — modular, unit-tested, look-ahead-safe, hyperparameters frozen before seeing Sharpe.
14. **What it shows** — independent replication + quant intuition + engineering discipline.

---

# Likely questions & crisp answers

- **"Is this overfit?"** No — hyperparameters (`xi`, cointegration thresholds) were frozen on cluster-quality/economic criteria *before* looking at any Sharpe; results are fully out-of-sample on a 21-yr rolling window.
- **"Why is the Sharpe ~1 and not huge?"** It matches the published academic result on a realistic, survivorship-bias-free universe. The honesty *is* the point — vol, drawdown and hit-rate all match the paper too.
- **"What's partial-correlation distance?"** Correlation of two stocks' returns after regressing out the market — isolates *idiosyncratic* co-movement, so the pairs are genuinely market-neutral.
- **"What would you do with more time?"** Factor-beta clustering (my own extension), a robustness suite (hierarchical clustering, robust hedge ratios, stop-losses), and a live Alpaca paper-trade forward test.
- **"Where's the profit actually coming from?"** 11% of trades that cleanly revert; the strategy edge is selecting *those* and minimising the force-closed drag — which the attribution slide quantifies.
