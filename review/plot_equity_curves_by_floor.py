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

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _REPO_ROOT / "review" / "output" / "equity_by_floor"

# House style — kept in sync with presentation/make_charts.py so the deck reads
# as one coherent set (chart 05 is the reference look).
GREEN = "#1b7837"     # leading / best floor
NAVY = "#1a2e44"
AMBER = "#d98300"
GREY = "#777777"
INK = "#16222e"       # headings / primary text
SLATE = "#33495e"     # axis labels / subtitle
HAIR = "#9aa6b1"      # spines / baseline

# floor label -> line colour. "none" is the no-floor (full-universe) baseline;
# excluded by default to match the 3-floor exhibit (pass --include-none to add).
FLOORS = [("none", GREY), ("50k", GREEN), ("250k", NAVY), ("1M", AMBER)]


def _stats(floor_dir: Path) -> dict:
    p = floor_dir / "portfolio_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _area_gradient(ax, x, y, color, base, alpha=0.34):
    """Soft vertical gradient fill under a curve (matches chart 05's look)."""
    xn = mdates.date2num(pd.to_datetime(x))
    y = np.asarray(y, dtype=float)
    lo, hi = min(base, np.nanmin(y)), max(base, np.nanmax(y))
    grad = np.empty((256, 1, 4))
    grad[:, :, :3] = mcolors.to_rgb(color)
    grad[:, :, 3] = np.linspace(0.0, alpha, 256)[:, None]
    im = ax.imshow(grad, aspect="auto", origin="lower",
                   extent=[xn.min(), xn.max(), lo, hi], zorder=1.2)
    verts = np.column_stack([np.concatenate([xn, xn[::-1]]),
                             np.concatenate([y, np.full_like(y, base)])])
    poly = Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(poly)
    im.set_clip_path(poly)
    return im


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=_DEFAULT_DIR,
                    help="root dir containing floor_<label>/equity_curve.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG (default: <dir>/equity_curves_by_floor.png)")
    ap.add_argument("--include-none", action="store_true",
                    help="also draw the no-floor (full-universe) baseline curve")
    args = ap.parse_args(argv)
    out = args.out or (args.dir / "equity_curves_by_floor.png")

    floors = FLOORS if args.include_none else [f for f in FLOORS if f[0] != "none"]

    with plt.rc_context({"font.family": "Helvetica Neue",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(13.2, 7.0))
        fig.subplots_adjust(top=0.82, left=0.075, right=0.93, bottom=0.12)

        curves, last_date = [], None
        for label, colour in floors:
            csv = args.dir / f"floor_{label}" / "equity_curve.csv"
            if not csv.exists():
                print(f"WARN missing {csv}", file=sys.stderr)
                continue
            eq = pd.read_csv(csv, parse_dates=["date"])
            st = _stats(args.dir / f"floor_{label}")
            end = eq["equity"].iloc[-1]
            curves.append((label, colour, eq, st, end))
            last_date = eq["date"].iloc[-1]

        # gradient accent under the best-performing floor (echoes chart 05)
        if curves:
            best = max(curves, key=lambda c: c[4])
            _area_gradient(ax, best[2]["date"], best[2]["equity"], best[1], base=1.0)

        for label, colour, eq, st, end in curves:
            lead = curves and end == max(c[4] for c in curves)
            ax.plot(eq["date"], eq["equity"], color=colour,
                    lw=2.8 if lead else 2.0, zorder=4 if lead else 3,
                    solid_capstyle="round",
                    label=(f"{label} floor   ·   {(end-1)*100:+.1f}%   ·   "
                           f"Sharpe {st.get('sharpe', float('nan')):.2f}   ·   "
                           f"MaxDD {st.get('max_drawdown', float('nan'))*100:.1f}%"))
            ax.scatter([eq["date"].iloc[-1]], [end], color=colour, zorder=6,
                       s=42, edgecolor="white", linewidth=1.4)
            ax.annotate(f"\\${end:.3f}", xy=(eq["date"].iloc[-1], end),
                        xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=12.5, fontweight="bold", color=colour)

        ax.axhline(1.0, color=HAIR, lw=1.2, ls=(0, (4, 3)), zorder=2)

        # executive header (left-aligned bold title + italic so-what subtitle)
        ax.set_title("")
        ax.text(0.0, 1.155, "Growth of \\$1 by liquidity floor — out-of-sample, 2021–2025",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=21, fontweight="bold", color=INK)
        ax.text(0.0, 1.055,
                "Higher floor = fewer, more liquid pairs. The edge concentrates in the "
                "most permissive (50k) universe — a key tradeability caveat.",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=13, color=SLATE, style="italic")

        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v:.2f}"))
        ax.set_ylabel("equity (growth of \\$1)")
        ax.set_xlabel("")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if curves:
            x0 = min(c[2]["date"].iloc[0] for c in curves)
            ax.set_xlim(x0 - pd.Timedelta(days=20),
                        last_date + pd.Timedelta(days=25))
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(HAIR)
        ax.tick_params(labelsize=13, colors=SLATE)
        leg = ax.legend(loc="upper left", frameon=False, fontsize=11.5,
                        title="liquidity floor", handlelength=1.6)
        leg.get_title().set_color(SLATE)
        leg.get_title().set_fontweight("bold")

        fig.text(0.005, 0.005,
                 "Floor = minimum median ADR share-volume over the train window "
                 "(2010–2020); a coarse tradeability proxy (cf. \\$-volume). "
                 "Single-fold walk-forward, ROCE-net, equal-weight, daily MTM, net of costs.",
                 fontsize=9, color=GREY, ha="left", va="bottom")

        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
