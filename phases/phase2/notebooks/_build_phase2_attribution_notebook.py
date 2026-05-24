"""
Generator script — builds notebooks/phase2_pnl_attribution.ipynb.

Phase 2 attribution notebook: per-cell P&L distributions, regime-by-regime
Sharpe comparison, concentration metrics, and most importantly — DID THE
BIMODAL DURATION PATTERN SHIFT? (the load-bearing Phase 2 question).

Run: python phases/phase2/notebooks/_build_phase2_attribution_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


cells: list[nbf.NotebookNode] = []

# ════════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md(r"""# Phase 2 — P&L Attribution & Lever Verification

**Sister notebook to:** `phase2_complete_reference.ipynb`

Mirrors `phases/phase1/notebooks/phase1_pnl_attribution.ipynb` but applies the
same diagnostic battery to **all 4 cells** of the Phase 2 backtest grid.

The two questions this notebook answers:

1. **Where did each cell's return come from?** Trades, pairs, sectors, directions,
   regimes, durations.
2. **Did the Phase 2 lever actually move?** Specifically — did the cointegration
   filter shrink the force-close drag predicted by Phase 1's bimodal finding?

## Quick navigation
1. [Setup + cell loading](#1)
2. [Headline scorecard — all 4 cells side by side](#2)
3. [Trade-level distributions per cell](#3)
4. [**The bimodal pattern — lever check (most important)**](#4)
5. [Regime-by-regime Sharpe (did calm periods get better?)](#5)
6. [Sector + direction attribution](#6)
7. [Pareto / concentration metrics](#7)
8. [Cross-cell pair overlap (which pairs survive across cells?)](#8)
9. [Conclusion + Phase 3 implications](#9)"""))

# ════════════════════════════════════════════════════════════════════════════════
# 1. SETUP
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""<a id='1'></a>
## 1. Setup + cell loading

We load the 4 cells; sections gracefully report 'data not ready' if a parquet is missing."""))

cells.append(code("""from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# walk up to project root
_p = Path.cwd().resolve()
while _p != _p.parent:
    if (_p / 'src' / 'config.py').exists():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent
del _p

from src.clustering import sic_division
from src.config import PHASE2_DIR
from src.panel import load_crsp_daily, ticker_lookup, siccd_lookup
from src.performance import compute_metrics

sns.set_theme(style='whitegrid', context='notebook', font_scale=1.0)
plt.rcParams['figure.figsize'] = (11, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

CELLS = ['ssd_core', 'ssd_filtered', 'pc_core', 'pc_filtered']
CELL_COLORS = {
    'ssd_core':     '#3a76c4',
    'ssd_filtered': '#1f4d8a',
    'pc_core':      '#c47a3a',
    'pc_filtered':  '#8a4d1f',
}
PAPER_SHARPE = {'ssd_core': 0.88, 'ssd_filtered': 0.75, 'pc_core': 1.01, 'pc_filtered': 0.80}

RESULTS = PHASE2_DIR / 'results'
cell_monthly = {}
cell_trades = {}
for cell in CELLS:
    mp = RESULTS / f'{cell}_monthly.parquet'
    tp = RESULTS / f'{cell}_trades.parquet'
    if mp.exists():
        cell_monthly[cell] = pd.read_parquet(mp)
        cell_trades[cell] = pd.read_parquet(tp) if tp.exists() else pd.DataFrame()
        print(f'  ✅ {cell}: {len(cell_monthly[cell])} months, {len(cell_trades[cell])} trades')
    else:
        print(f'  ⚠ {cell}: not found')

if not cell_monthly:
    print('\\n⚠ No parquets available. Run phases/phase2/notebooks/04_run_full_backtest_grid.py first.')
else:
    crsp = load_crsp_daily()
    print('\\nDecorating trades with tickers + sectors …')
    for cell in cell_trades:
        t = cell_trades[cell]
        if len(t):
            permnos = list(set(t['permno_a']) | set(t['permno_b']))
            tk = ticker_lookup(permnos, crsp=crsp)
            sec = siccd_lookup(permnos, crsp=crsp).apply(sic_division)
            t['ticker_a'] = t['permno_a'].map(tk)
            t['ticker_b'] = t['permno_b'].map(tk)
            t['sector_a'] = t['permno_a'].map(sec)
            t['sector_b'] = t['permno_b'].map(sec)
            t['year'] = pd.to_datetime(t['entry_date']).dt.year
            t['duration_days'] = (pd.to_datetime(t['exit_date']) - pd.to_datetime(t['entry_date'])).dt.days
            t['pair_key'] = t.apply(lambda r: f'{r[\"ticker_a\"]}/{r[\"ticker_b\"]}'
                                    if pd.notna(r['ticker_a']) and pd.notna(r['ticker_b'])
                                    else f'{r[\"permno_a\"]}/{r[\"permno_b\"]}', axis=1)
    print('Done.')"""))

# ════════════════════════════════════════════════════════════════════════════════
# 2. HEADLINE SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""<a id='2'></a>
## 2. Headline scorecard — all 4 cells side by side"""))

cells.append(code("""if cell_monthly:
    rows = []
    for cell in CELLS:
        if cell not in cell_monthly:
            continue
        m = compute_metrics(cell_monthly[cell]['monthly_return'])
        rows.append({
            'cell': cell,
            'paper Sharpe': PAPER_SHARPE[cell],
            'ours Sharpe': m.sharpe,
            'Δ vs paper': m.sharpe - PAPER_SHARPE[cell],
            'ann return': m.ann_return,
            'ann vol': m.ann_vol,
            'max DD': m.max_drawdown,
            'hit rate': m.hit_rate,
            'n_trades': len(cell_trades.get(cell, [])),
        })
    sc = pd.DataFrame(rows).set_index('cell')
    for col in ['paper Sharpe', 'ours Sharpe', 'Δ vs paper']:
        sc[col] = sc[col].map(lambda x: f'{x:+.3f}')
    for col in ['ann return', 'ann vol', 'max DD', 'hit rate']:
        sc[col] = sc[col].map(lambda x: f'{x:+.2%}')
    print(sc.to_string())
else:
    print('No cells loaded.')"""))

# ════════════════════════════════════════════════════════════════════════════════
# 3. TRADE-LEVEL DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""<a id='3'></a>
## 3. Trade-level distributions per cell"""))

cells.append(code("""if cell_trades:
    rows = []
    for cell in CELLS:
        if cell not in cell_trades or len(cell_trades[cell]) == 0:
            continue
        rt = cell_trades[cell]['round_trip_return']
        rows.append({
            'cell': cell,
            'n': len(rt),
            'mean (bps)': rt.mean() * 10000,
            'median (bps)': rt.median() * 10000,
            'std': rt.std(),
            'skew': rt.skew(),
            'kurtosis (excess)': rt.kurtosis(),
            'hit rate': (rt > 0).mean(),
            'best': rt.max(),
            'worst': rt.min(),
            'sum (per-trade total)': rt.sum(),
        })
    dist = pd.DataFrame(rows).set_index('cell')
    print(dist.round(4).to_string())

    # Overlay histograms
    fig, ax = plt.subplots(figsize=(13, 5))
    for cell in CELLS:
        if cell in cell_trades and len(cell_trades[cell]):
            rt = cell_trades[cell]['round_trip_return']
            ax.hist(rt.clip(-0.15, 0.15) * 100, bins=60, alpha=0.45,
                    color=CELL_COLORS[cell], label=f'{cell} (n={len(rt):,})', edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel('Round-trip return (%, clipped at ±15%)')
    ax.set_ylabel('Count')
    ax.set_title('Trade return distribution — 4 cells overlaid')
    ax.legend()
    plt.tight_layout()
    plt.show()"""))

# ════════════════════════════════════════════════════════════════════════════════
# 4. BIMODAL PATTERN — THE LEVER CHECK
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md(r"""<a id='4'></a>
## 4. The bimodal pattern — lever check ⭐

Phase 1's identity (the load-bearing insight):

$$
\underbrace{+65.58}_{\text{reversion (11.4\%, +471 bps)}} + \underbrace{-34.25}_{\text{force\_close (88.4\%, -32 bps)}} + \underbrace{-0.92}_{\text{delisting}} = +30.41
$$

**Phase 2 hypothesis**: the cointegration filter shrinks `force_close` total by
rejecting "case 2" pairs (broken cointegrations like XOM/MPC, COP/CVX). If so:
- Force-close total moves toward zero (or even positive)
- Net per-trade total grows
- Sharpe rises (paper target 1.01 for PC core, 0.80 for PC + filter)"""))

cells.append(code("""if cell_trades:
    rows = []
    for cell in CELLS:
        if cell not in cell_trades or len(cell_trades[cell]) == 0:
            continue
        t = cell_trades[cell]
        rev = t.loc[t['exit_reason'] == 'reversion']
        fc  = t.loc[t['exit_reason'] == 'force_close']
        dl  = t.loc[t['exit_reason'] == 'delisting']
        rows.append({
            'cell': cell,
            'n_total': len(t),
            'n_reversion': len(rev),
            'pct_reversion (%)': len(rev) / len(t) * 100,
            'reversion total': rev['round_trip_return'].sum(),
            'reversion mean (bps)': rev['round_trip_return'].mean() * 10000 if len(rev) else 0,
            'force_close total': fc['round_trip_return'].sum(),
            'force_close mean (bps)': fc['round_trip_return'].mean() * 10000 if len(fc) else 0,
            'delisting total': dl['round_trip_return'].sum(),
            'NET (sum of per-trade)': t['round_trip_return'].sum(),
        })
    drag = pd.DataFrame(rows).set_index('cell')
    print(drag.round(4).to_string())

    # Visualise: stacked bar of decomposition
    if len(drag) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        # left: stacked decomposition
        x = np.arange(len(drag))
        for cell in drag.index:
            i = list(drag.index).index(cell)
            rev = drag.loc[cell, 'reversion total']
            fc  = drag.loc[cell, 'force_close total']
            dl  = drag.loc[cell, 'delisting total']
            axes[0].bar(i, rev, color='#2ca02c', label='reversion' if i == 0 else None)
            axes[0].bar(i, fc, color='#d62728', label='force_close' if i == 0 else None)
            axes[0].bar(i, dl, color='#9467bd', label='delisting' if i == 0 else None)
            axes[0].scatter(i, drag.loc[cell, 'NET (sum of per-trade)'], color='black', s=80, zorder=5,
                            label='NET' if i == 0 else None)
        axes[0].set_xticks(x); axes[0].set_xticklabels(drag.index, rotation=30, ha='right')
        axes[0].set_ylabel('Total per-trade P&L')
        axes[0].set_title('Bimodal decomposition: reversion + force_close + delisting = NET')
        axes[0].axhline(0, color='black', lw=0.5)
        axes[0].legend(loc='lower left', fontsize=9)

        # right: mean P&L per exit reason
        widths = 0.2
        offsets = [-1.5*widths, -0.5*widths, 0.5*widths, 1.5*widths]
        x_reason = np.arange(3)
        labels = ['reversion', 'force_close', 'delisting']
        keys = ['reversion mean (bps)', 'force_close mean (bps)', None]
        for i, cell in enumerate(drag.index):
            for j, k in enumerate(keys):
                if k is None:
                    val = (drag.loc[cell, 'delisting total'] /
                           max(drag.loc[cell, 'n_total'] - drag.loc[cell, 'n_reversion'] -
                               (drag.loc[cell, 'force_close total'] != 0) * 1, 1)) * 10000  # placeholder
                    continue
                axes[1].bar(x_reason[j] + offsets[i], drag.loc[cell, k], widths,
                            color=CELL_COLORS[cell], label=cell if j == 0 else None,
                            edgecolor='black', linewidth=0.3)
        axes[1].set_xticks(x_reason[:2]); axes[1].set_xticklabels(labels[:2])
        axes[1].set_ylabel('Mean per-trade return (bps)')
        axes[1].set_title('Mean per-trade P&L by exit reason — all 4 cells')
        axes[1].axhline(0, color='black', lw=0.5)
        axes[1].legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    # Lever verdict
    print('\\n' + '═' * 70)
    print('LEVER VERDICT — did the cointegration filter shrink force-close drag?')
    print('═' * 70)
    for pair in [('ssd_core', 'ssd_filtered'), ('pc_core', 'pc_filtered')]:
        if all(c in drag.index for c in pair):
            no_filt = drag.loc[pair[0], 'force_close total']
            yes_filt = drag.loc[pair[1], 'force_close total']
            change = yes_filt - no_filt
            sign = '✅ shrunk' if change > 0 else '❌ grew'
            print(f'  {pair[0]:<14} → {pair[1]:<14}  '
                  f'force_close total {no_filt:+.2f} → {yes_filt:+.2f}  '
                  f'(Δ={change:+.2f})  {sign}')

else:
    print('No trade data — cannot run lever check yet.')"""))

# ════════════════════════════════════════════════════════════════════════════════
# 5. REGIME ATTRIBUTION
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md(r"""<a id='5'></a>
## 5. Regime-by-regime Sharpe (did calm periods improve?)

Phase 1's finding: 61% of P&L came from GFC + COVID; 18% from the calm 2010-19
decade despite 46% of trades. Phase 2 should *partially* fix this — the
cointegration filter rejects pairs that drift quietly in calm regimes."""))

cells.append(code("""def regime(d):
    y = pd.Timestamp(d).year
    if y < 2007: return '2003-06 pre-crisis'
    if y < 2010: return '2007-09 GFC'
    if y < 2020: return '2010-19 expansion'
    if y < 2022: return '2020-21 COVID'
    return '2022-23 inflation'

if cell_monthly:
    rows = []
    for cell in CELLS:
        if cell not in cell_monthly:
            continue
        m = cell_monthly[cell].copy()
        m['regime'] = m.index.map(regime)
        for reg, grp in m.groupby('regime', observed=True):
            rets = grp['monthly_return']
            ann_ret = (1 + rets.mean()) ** 12 - 1 if len(rets) else 0
            ann_vol = rets.std(ddof=1) * np.sqrt(12) if len(rets) > 1 else float('nan')
            sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else float('nan')
            rows.append({
                'cell': cell, 'regime': reg, 'n_months': len(rets),
                'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
            })
    reg_df = pd.DataFrame(rows)
    sharpe_table = reg_df.pivot_table(index='regime', columns='cell', values='sharpe')
    sharpe_table = sharpe_table.reindex(['2003-06 pre-crisis', '2007-09 GFC',
                                          '2010-19 expansion', '2020-21 COVID', '2022-23 inflation'])
    print('Sharpe by regime × cell:')
    print(sharpe_table.round(3).to_string())

    if not sharpe_table.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sharpe_table.plot(kind='bar', ax=ax, color=[CELL_COLORS.get(c, 'gray') for c in sharpe_table.columns])
        ax.axhline(0, color='black', lw=0.6)
        ax.set_ylabel('Sharpe (in-regime)')
        ax.set_title('Regime-by-regime Sharpe — all 4 cells')
        ax.legend(title='cell')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()"""))

# ════════════════════════════════════════════════════════════════════════════════
# 6. SECTOR + DIRECTION
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""<a id='6'></a>
## 6. Sector + direction attribution"""))

cells.append(code("""if cell_trades:
    for cell in CELLS:
        if cell not in cell_trades or len(cell_trades[cell]) == 0:
            continue
        t = cell_trades[cell]
        t['sector_pair'] = t.apply(
            lambda r: '/'.join(sorted([str(r['sector_a']), str(r['sector_b'])])), axis=1)
        sec_agg = t.groupby('sector_pair').agg(
            n=('round_trip_return', 'size'),
            total=('round_trip_return', 'sum'),
        ).sort_values('total', ascending=False).head(8)
        top3 = sec_agg.head(3)['total'].sum() / t['round_trip_return'].sum() * 100
        print(f'\\n--- {cell} ---')
        print(f'  Top 3 sector pairs = {top3:.1f}% of total P&L (concentration)')
        print(f'  Top 8 sectors:')
        for sp, row in sec_agg.iterrows():
            print(f'    {sp:<50}  n={int(row[\"n\"]):>5,}  total={row[\"total\"]:+.2f}')

    # Direction balance check per cell
    print('\\nDirection balance:')
    for cell in CELLS:
        if cell in cell_trades and len(cell_trades[cell]):
            t = cell_trades[cell]
            dg = t.groupby('direction').agg(n=('round_trip_return', 'size'),
                                            total=('round_trip_return', 'sum'),
                                            winrate=('round_trip_return', lambda x: (x > 0).mean()))
            long_pnl  = dg.loc[+1, 'total'] if +1 in dg.index else 0
            short_pnl = dg.loc[-1, 'total'] if -1 in dg.index else 0
            print(f'  {cell:<14}  long {long_pnl:+.2f} / short {short_pnl:+.2f}')"""))

# ════════════════════════════════════════════════════════════════════════════════
# 7. CONCENTRATION
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""<a id='7'></a>
## 7. Pareto / concentration metrics"""))

cells.append(code("""def gini(values):
    arr = np.sort(np.abs(np.asarray(values, dtype=float)))
    arr = arr[~np.isnan(arr)]
    if arr.size == 0: return float('nan')
    n = len(arr); cum = np.cumsum(arr)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(arr) / (n * cum[-1]))

def hhi_eff_n(values):
    arr = np.abs(np.asarray(values, dtype=float))
    arr = arr[~np.isnan(arr)]
    if arr.sum() == 0: return float('nan')
    shares = arr / arr.sum()
    return 1 / (shares ** 2).sum()

if cell_trades:
    rows = []
    for cell in CELLS:
        if cell not in cell_trades or len(cell_trades[cell]) == 0:
            continue
        rt = cell_trades[cell]['round_trip_return']
        # Top-K share
        sorted_abs = rt.abs().sort_values(ascending=False).values
        total_abs = sorted_abs.sum()
        top1_share = sorted_abs[:max(1, int(len(rt) * 0.01))].sum() / total_abs * 100
        top5_share = sorted_abs[:max(1, int(len(rt) * 0.05))].sum() / total_abs * 100
        top10_share = sorted_abs[:max(1, int(len(rt) * 0.10))].sum() / total_abs * 100
        rows.append({
            'cell': cell,
            'n_trades': len(rt),
            'Gini (|rt|)': gini(rt.values),
            'effective n (HHI⁻¹)': hhi_eff_n(rt.values),
            'top 1% share of |rt|': top1_share / 100,
            'top 5% share of |rt|': top5_share / 100,
            'top 10% share of |rt|': top10_share / 100,
        })
    concentration = pd.DataFrame(rows).set_index('cell')
    print('Concentration metrics:')
    print(concentration.round(4).to_string())
    print()
    print('Interpretation: lower Gini, higher effective n = more diversified P&L.')
    print('Phase 1 SSD had Gini ~0.54 and effective n ~4674 (out of 12,255 trades).')"""))

# ════════════════════════════════════════════════════════════════════════════════
# 8. CROSS-CELL PAIR OVERLAP
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""<a id='8'></a>
## 8. Cross-cell pair overlap

Which pairs survive across all 4 cells? Which are unique to one method?"""))

cells.append(code("""if cell_trades and all(c in cell_trades for c in CELLS):
    # Build per-cell pair P&L
    pair_pnl = {}
    for cell in CELLS:
        t = cell_trades[cell]
        if len(t) == 0:
            continue
        pair_pnl[cell] = t.groupby('pair_key')['round_trip_return'].sum()

    if pair_pnl:
        # Set comparison: which pairs appear where?
        all_pairs = set()
        for cell, ser in pair_pnl.items():
            all_pairs |= set(ser.index)
        membership = pd.DataFrame(index=sorted(all_pairs), columns=CELLS, data=False)
        for cell in CELLS:
            if cell in pair_pnl:
                for p in pair_pnl[cell].index:
                    membership.loc[p, cell] = True

        # 4-way overlap
        in_all = membership.all(axis=1).sum()
        in_ssd_only = (membership['ssd_core'] & membership['ssd_filtered']
                       & ~membership['pc_core'] & ~membership['pc_filtered']).sum()
        in_pc_only  = (~membership['ssd_core'] & ~membership['ssd_filtered']
                       & membership['pc_core'] & membership['pc_filtered']).sum()
        print(f'Total unique pairs traded across all cells: {len(all_pairs):,}')
        print(f'  in all 4 cells          : {in_all:>5,}  (consensus pairs)')
        print(f'  SSD-only (both filters) : {in_ssd_only:>5,}  (only SSD methods trade)')
        print(f'  PC-only (both filters)  : {in_pc_only:>5,}  (only PC methods trade)')

        # Show 10 pairs with biggest cross-cell P&L difference (PC_filtered vs SSD_core)
        if 'ssd_core' in pair_pnl and 'pc_filtered' in pair_pnl:
            ssd = pair_pnl['ssd_core']
            pc_f = pair_pnl['pc_filtered']
            common = ssd.index.intersection(pc_f.index)
            diff = (pc_f.loc[common] - ssd.loc[common]).sort_values(ascending=False)
            print('\\nPairs where pc_filtered did MUCH better than ssd_core:')
            for p, d in diff.head(10).items():
                print(f'  {p:<24}  ssd={ssd[p]:+.4f}  pc_filt={pc_f[p]:+.4f}  Δ={d:+.4f}')
            print('\\nPairs where pc_filtered did MUCH worse than ssd_core:')
            for p, d in diff.tail(10).items():
                print(f'  {p:<24}  ssd={ssd[p]:+.4f}  pc_filt={pc_f[p]:+.4f}  Δ={d:+.4f}')"""))

# ════════════════════════════════════════════════════════════════════════════════
# 9. CONCLUSION
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md(r"""<a id='9'></a>
## 9. Conclusion + Phase 3 implications

Once the 4-cell grid has been run, read off:

1. **Did the bimodal lever move?** (Section 4 — force-close drag should shrink)
2. **Did the headline Sharpe lift?** (Section 2 — PC core should approach 1.01)
3. **Did calm periods improve?** (Section 5 — 2010-19 expansion regime should lift)
4. **Did concentration tighten or spread?** (Section 7 — lower Gini = healthier)
5. **Where do the methods agree vs disagree?** (Section 8 — pair-level overlap)

### Likely Phase 3 follow-ups depending on results

| If we see... | Then in Phase 3... |
|---|---|
| PC core ≈ 1.01 ✅ | The replication is closed. Focus Phase 3 on the factor-beta extension (Phase 2.5). |
| PC core still 0.6-0.8 | Test the architectural suspects: |entry-z|-weighted allocation; softer survivorship; γ-weighted position sizing. |
| Force-close drag did *not* shrink | Investigate the filter's specificity. Try wider half-life bounds; or use OU instead of AR(1) for less-biased half-life estimation. |
| Crisis regimes still dominate | This is a structural property of pairs trading. Document honestly; set forward-test expectations accordingly. |
| Concentration grew (Gini ↑) | Filter may be over-pruning. Consider the unfiltered PC core as the primary deliverable. |"""))

# ════════════════════════════════════════════════════════════════════════════════
# WRITE OUT
# ════════════════════════════════════════════════════════════════════════════════
nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'},
    },
)
out = Path(__file__).parent / 'phase2_pnl_attribution.ipynb'
with open(out, 'w') as f:
    nbf.write(nb, f)
n_md = sum(1 for c in cells if c.cell_type == "markdown")
n_code = sum(1 for c in cells if c.cell_type == "code")
print(f'✅ Wrote {out}')
print(f'   {len(cells)} cells ({n_md} markdown, {n_code} code)')
