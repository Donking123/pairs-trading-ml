"""
GG (no clustering) vs OPTICS clustering comparison.

Runs 5 configs on the same data/period:
  1. GG naive top-20     — Gatev-Goetzmann (2006) style, no sector awareness
  2. GG top-20, floor=10 — GG but excluding trivially-perfect pairs (dual-class etc.)
  3. GG top-100, floor=5 — broader GG universe
  4. OPTICS baseline     — Donking's clustering, no overlays
  5. OPTICS + 4 overlays — pris-clustering improvements

Usage:
    python3 compare_gg_vs_clustering.py [--start 2016-01-01] [--end 2023-12-31]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.backtest import run_backtest, load_delisting, load_regime_scale
from src.panel import load_crsp_daily, load_sp500_constituents


# ── helpers ───────────────────────────────────────────────────────────────────

def sharpe(m: pd.DataFrame) -> float:
    r = m["monthly_return"]
    return float((r.mean() * 12) / (r.std() * np.sqrt(12)))

def maxdd(m: pd.DataFrame) -> float:
    c = (1 + m["monthly_return"]).cumprod()
    return float((c / c.cummax() - 1).min())

def ann_ret(m: pd.DataFrame) -> float:
    return float((1 + m["monthly_return"]).prod() ** (12 / len(m)) - 1)

def cum_ret(m: pd.DataFrame) -> pd.Series:
    return (1 + m["monthly_return"]).cumprod() - 1


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end",   default="2023-12-31")
    parser.add_argument("--output", default="figures/gg_vs_clustering.png")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    crsp   = load_crsp_daily()
    cons   = load_sp500_constituents()
    delist = load_delisting()
    rs     = load_regime_scale()

    common = dict(
        start=args.start, end=args.end,
        crsp=crsp, constituents=cons, delisting_df=delist,
        formation_years=2, metric="ssd", verbose=False,
    )

    configs = [
        # ── GG variants (no clustering) ───────────────────────────────────────
        ("GG naive top-20\n(no clustering, no floor)",
         dict(clusterer="gg", gg_top_n=20, gg_dist_floor=0.0),
         "#e74c3c", "--"),

        ("GG top-20, floor=10\n(excludes dual-class shares)",
         dict(clusterer="gg", gg_top_n=20, gg_dist_floor=10.0),
         "#e67e22", "--"),

        ("GG top-100, floor=5\n(broader, no clustering)",
         dict(clusterer="gg", gg_top_n=100, gg_dist_floor=5.0),
         "#f39c12", ":"),

        # ── Clustering variants ───────────────────────────────────────────────
        ("OPTICS clustering\n(Donking baseline, no overlays)",
         dict(clusterer="optics", exit_sigma=0.0),
         "#3498db", "-"),

        ("OPTICS + 4 overlays\n(pris-clustering: entry confirm\n+ struct exits + HMM)",
         dict(clusterer="optics",
              exit_sigma=-0.5,
              entry_confirm=True,
              structural_exits=True,
              use_regime_scale=True, regime_scale=rs),
         "#27ae60", "-"),
    ]

    results = {}
    for label, kw, color, ls in configs:
        short = label.split("\n")[0]
        print(f"  Running: {short}...", flush=True)
        m, trades = run_backtest(**common, **kw)
        results[label] = {
            "monthly": m,
            "color": color,
            "ls": ls,
            "sr": sharpe(m),
            "dd": maxdd(m),
            "ret": ann_ret(m),
            "n_pairs": m["n_pairs_traded"].mean(),
            "n_trades": len(trades),
        }
        print(f"    SR={results[label]['sr']:+.3f}  DD={results[label]['dd']:.1%}  "
              f"ret={results[label]['ret']:+.2%}  avg_pairs={results[label]['n_pairs']:.0f}",
              flush=True)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            height_ratios=[2.2, 1])

    # Top: cumulative return
    ax_cum = fig.add_subplot(gs[0, :])
    for label, d in results.items():
        cum = cum_ret(d["monthly"])
        lw  = 2.5 if "overlays" in label else 1.5
        ax_cum.plot(cum.index, cum * 100, color=d["color"], linewidth=lw,
                    linestyle=d["ls"],
                    label=f"{label.replace(chr(10),' | ')}  (SR={d['sr']:+.2f}  DD={d['dd']:.1%}  avg_pairs={d['n_pairs']:.0f})")

    ax_cum.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_cum.set_ylabel("Cumulative return (%)")
    ax_cum.set_title(
        f"GG no-clustering vs OPTICS clustering — {args.start[:4]}–{args.end[:4]}\n"
        "Solid = clustering  |  Dashed = GG (no clustering)",
        fontsize=11
    )
    ax_cum.legend(fontsize=7, loc="upper left")
    ax_cum.grid(axis="y", alpha=0.3)

    labels_short = [lbl.split("\n")[0] for lbl in results]

    # Bottom left: Gross SR
    ax_sr = fig.add_subplot(gs[1, 0])
    colors = [d["color"] for d in results.values()]
    srs    = [d["sr"]    for d in results.values()]
    bars   = ax_sr.bar(range(len(labels_short)), srs, color=colors)
    ax_sr.set_xticks(range(len(labels_short)))
    ax_sr.set_xticklabels(labels_short, rotation=20, ha="right", fontsize=7)
    ax_sr.axhline(0, color="black", linewidth=0.5)
    for bar, v in zip(bars, srs):
        ax_sr.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=7)
    ax_sr.set_ylabel("Gross Sharpe"); ax_sr.set_title("Gross SR"); ax_sr.grid(axis="y", alpha=0.3)

    # Bottom middle: MaxDD
    ax_dd = fig.add_subplot(gs[1, 1])
    dds   = [d["dd"] * 100 for d in results.values()]
    bars  = ax_dd.bar(range(len(labels_short)), dds, color=colors)
    ax_dd.set_xticks(range(len(labels_short)))
    ax_dd.set_xticklabels(labels_short, rotation=20, ha="right", fontsize=7)
    for bar, v in zip(bars, dds):
        ax_dd.text(bar.get_x() + bar.get_width()/2, v - 0.3, f"{v:.1f}%",
                   ha="center", va="top", fontsize=7)
    ax_dd.set_ylabel("MaxDD (%)"); ax_dd.set_title("Max Drawdown"); ax_dd.grid(axis="y", alpha=0.3)

    # Bottom right: avg pairs traded per month
    ax_np = fig.add_subplot(gs[1, 2])
    nps   = [d["n_pairs"] for d in results.values()]
    bars  = ax_np.bar(range(len(labels_short)), nps, color=colors)
    ax_np.set_xticks(range(len(labels_short)))
    ax_np.set_xticklabels(labels_short, rotation=20, ha="right", fontsize=7)
    for bar, v in zip(bars, nps):
        ax_np.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.0f}",
                   ha="center", va="bottom", fontsize=7)
    ax_np.set_ylabel("Avg pairs traded/month"); ax_np.set_title("Pair count (diversification)"); ax_np.grid(axis="y", alpha=0.3)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure saved → {out}")
    print(f"\n{'Strategy':45s}  {'SR':>7}  {'DD':>7}  {'Ret':>7}  {'Pairs':>6}")
    print("-" * 80)
    for label, d in results.items():
        print(f"{label.split(chr(10))[0]:45s}  {d['sr']:+7.3f}  {d['dd']:7.1%}  {d['ret']:+7.2%}  {d['n_pairs']:6.0f}")


if __name__ == "__main__":
    main()
