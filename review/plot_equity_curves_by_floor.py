#!/usr/bin/env python3
"""
plot_equity_curves_by_floor.py — overlay the OOS equity curves at 50k/250k/1M
ADR share-volume floors (produced by ``run_walkforward.py --floors`` then
``run_walkforward_portfolio.py --by-floor review/output/equity_by_floor``).

Reads review/output/equity_by_floor/floor_<label>/equity_curve.csv for each
floor and draws growth-of-$1 on one axis. Output:
    review/output/equity_by_floor/equity_curves_by_floor.png

Usage:  python review/plot_equity_curves_by_floor.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _REPO_ROOT / "review" / "output" / "equity_by_floor"

# floor label -> line colour (grey/green/navy/amber, repo palette).
# "none" is the no-floor (full-universe) curve; skipped automatically if its
# equity_curve.csv was never written (run with --floors 0 ... to produce it).
FLOORS = [("none", "#777777"), ("50k", "#1b7837"), ("250k", "#1a2e44"), ("1M", "#d98300")]
GREY = "#777777"


def _stats(floor_dir: Path) -> dict:
    p = floor_dir / "portfolio_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=_DEFAULT_DIR,
                    help="root dir containing floor_<label>/equity_curve.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <dir>/equity_curves_by_floor.png)")
    args = ap.parse_args(argv)
    out = args.out or (args.dir / "equity_curves_by_floor.png")

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12.8, 7.2))

    for label, colour in FLOORS:
        csv = args.dir / f"floor_{label}" / "equity_curve.csv"
        if not csv.exists():
            print(f"WARN missing {csv}", file=sys.stderr)
            continue
        eq = pd.read_csv(csv, parse_dates=["date"])
        st = _stats(args.dir / f"floor_{label}")
        end = eq["equity"].iloc[-1]
        legend = (f"{label} floor  ·  {(end-1)*100:+.1f}%  ·  "
                  f"Sharpe {st.get('sharpe', float('nan')):.2f}  ·  "
                  f"MaxDD {st.get('max_drawdown', float('nan'))*100:.1f}%")
        ax.plot(eq["date"], eq["equity"], color=colour, lw=2.2, label=legend)
        ax.scatter([eq["date"].iloc[-1]], [end], color=colour, zorder=5, s=28)
        ax.annotate(f"{end:.3f}", xy=(eq["date"].iloc[-1], end),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=11, fontweight="bold", color=colour)

    ax.axhline(1.0, color=GREY, lw=1.0, ls="--")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.set_ylabel("equity (growth of $1)")
    ax.set_xlabel("")
    ax.set_title("OOS equity curves by ADR share-volume floor — 2021–2025\n"
                 "(single-fold walk-forward, ROCE-net, equal-weight, daily MTM)",
                 fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=11, title="liquidity floor")
    fig.text(0.01, 0.005,
             "Floor = minimum median ADR share-volume over the train window "
             "(2010–2020); a coarse tradeability proxy (cf. $-volume). "
             "Higher floor -> fewer, more liquid pairs.",
             fontsize=9, color=GREY)

    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
