#!/usr/bin/env python3
"""
make_charts.py — generate every chart for the ADR-strategy trader pitch deck.

Reads ONLY the latest, sign-fixed source files (see SRC paths below) and writes
publication-quality PNGs to presentation/charts/. Re-run any time the underlying
data is regenerated.

Headline numbers come from the corrected out-of-sample walk-forward run
(walkforward_20260614_161536). The pre-fix research grid is used ONLY for the
relative parameter-ranking chart (clearly labelled).

Usage:
    cd presentation && python make_charts.py
Dependencies: pandas, numpy, matplotlib (already in requirements.txt).
              fastparquet is used for the spread example (optional; that one
              chart is skipped with a warning if the parquet data is unavailable).
"""
from __future__ import annotations
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "datastream", "data")
PAIRS_JSON = os.path.join(ROOT, "config", "pairs", "asian_adr_pairs.json")
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# Auto-discover the LATEST generated run directories.
# This script was lifted from a sibling repo with hard-coded run-stamp folders
# (e.g. walkforward_minadv1m_20260615_213113) that don't exist here, so it would
# crash on import. Instead we resolve, at run time, the most recently written
# directory that actually contains the files each chart needs — so re-running
# the pipeline and then this script always picks up the freshest outputs.
# --------------------------------------------------------------------------- #
def _latest_dir_with(parent, required):
    """Most-recently-written subdir of `parent` that contains every file in
    `required`. Ranked by the newest mtime among those required files. None if
    no subdir qualifies (or `parent` is missing)."""
    if not os.path.isdir(parent):
        return None
    cands = []
    for name in os.listdir(parent):
        d = os.path.join(parent, name)
        if not os.path.isdir(d):
            continue
        paths = [os.path.join(d, f) for f in required]
        if all(os.path.exists(p) for p in paths):
            cands.append((max(os.path.getmtime(p) for p in paths), d))
    return max(cands)[1] if cands else None


# Walk-forward portfolio tearsheet — the headline source. Needs the full set
# written by run_walkforward.py + review/run_walkforward_portfolio.py.
_WF_REQUIRED = ["portfolio_stats.json", "summary.json", "distribution.json",
                "equity_curve.csv", "rolling_sharpe.csv", "trades_oos.csv"]
WF = _latest_dir_with(os.path.join(DATA, "walkforward_output"), _WF_REQUIRED)
if WF is None:
    sys.exit(
        "make_charts: no walk-forward run with a full portfolio tearsheet found under "
        f"{os.path.join(DATA, 'walkforward_output')}.\n"
        "  Generate one first:\n"
        "    python datastream/run_walkforward.py\n"
        "    python review/run_walkforward_portfolio.py"
    )

# Latest in-sample backtest run (optional; kept for provenance, not charted).
INSAMPLE = _latest_dir_with(os.path.join(DATA, "backtest"), ["summary.json"])

# Research grid for the parameter-ranking chart. run_research_permutations.py
# writes research_runs.csv either flat in data/backtest/ or inside a run subdir;
# accept either, else leave None and the chart skips itself.
def _find_research(parent):
    flat = os.path.join(parent, "research_runs.csv")
    if os.path.exists(flat):
        return parent
    return _latest_dir_with(parent, ["research_runs.csv"])

RESEARCH = _find_research(os.path.join(DATA, "backtest"))

# --------------------------------------------------------------------------- #
# House style
# --------------------------------------------------------------------------- #
GREEN = "#1b7837"     # strategy
GREEN_L = "#7fbf7b"
GREY = "#777777"      # cost-stress / secondary
RED = "#b2182b"       # losses / before
BLUE = "#2166ac"      # ADR leg
ORANGE = "#d6604d"
NAVY = "#1a2e44"
INK = "#16222e"       # primary text / headings
SLATE = "#33495e"     # axis labels / secondary text
HAIR = "#9aa6b1"      # spines, gridlines, rules

# A single house font (matches the polished walk-forward schematic) applied to
# every chart so the deck reads as one coherent set. Helvetica Neue ships with
# macOS; the fallbacks keep the script portable.
plt.rcParams.update({
    "figure.figsize": (12.8, 7.2),   # 16:9
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 21,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlepad": 14,
    "axes.labelsize": 16,
    "axes.labelcolor": SLATE,
    "axes.edgecolor": HAIR,
    "axes.linewidth": 1.0,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.5,
    "grid.color": HAIR,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
})

PCT = FuncFormatter(lambda x, _: f"{x*100:.0f}%")
PCT1 = FuncFormatter(lambda x, _: f"{x*100:.1f}%")


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


def _footnote(fig, text):
    fig.text(0.005, 0.005, text, fontsize=10, color=GREY, ha="left", va="bottom")


def _header(ax, title, subtitle):
    """Left-aligned executive header: bold INK headline above an italic muted
    'so-what' subtitle. Matches the polished walk-forward schematic (chart 04)
    and reads like an FT/Economist exhibit. Call after plotting; pair with
    `fig.subplots_adjust(top=...)` to reserve room above the axes."""
    ax.set_title("")  # suppress the default centred title
    ax.text(0.0, 1.155, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=22, fontweight="bold", color=INK)
    ax.text(0.0, 1.055, subtitle, transform=ax.transAxes, ha="left", va="top",
            fontsize=13.5, color=SLATE, style="italic")


def _area_gradient(ax, x, y, color, base, alpha=0.42):
    """Soft vertical gradient fill between curve `y` and the horizontal `base`,
    saturated at the curve edge and fading to transparent toward the base — a
    cleaner, more premium look than a flat alpha fill."""
    xn = mdates.date2num(pd.to_datetime(x))
    y = np.asarray(y, dtype=float)
    lo, hi = min(base, np.nanmin(y)), max(base, np.nanmax(y))
    grad = np.empty((256, 1, 4))
    grad[:, :, :3] = mcolors.to_rgb(color)
    above = np.nanmean(y) >= base
    ramp = np.linspace(0.0, alpha, 256) if above else np.linspace(alpha, 0.0, 256)
    grad[:, :, 3] = ramp[:, None]
    im = ax.imshow(grad, aspect="auto", origin="lower",
                   extent=[xn.min(), xn.max(), lo, hi], zorder=1.2)
    verts = np.column_stack([np.concatenate([xn, xn[::-1]]),
                             np.concatenate([y, np.full_like(y, base)])])
    poly = Polygon(verts, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(poly)
    im.set_clip_path(poly)
    return im


def _year_axis(ax):
    """Clean yearly x-ticks for the OOS date axis."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.01)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Source data (loaded once)
# --------------------------------------------------------------------------- #
PSTATS = _load_json(os.path.join(WF, "portfolio_stats.json"))
SUMMARY = _load_json(os.path.join(WF, "summary.json"))
DISTRIB = _load_json(os.path.join(WF, "distribution.json"))
EQ = pd.read_csv(os.path.join(WF, "equity_curve.csv"), parse_dates=["date"])
ROLL = pd.read_csv(os.path.join(WF, "rolling_sharpe.csv"), parse_dates=["date"])
TRADES = pd.read_csv(os.path.join(WF, "trades_oos.csv"),
                     parse_dates=["open_date", "close_date"],
                     dtype={"pair_id": str, "adr_ticker": str,
                            "underlying_ticker": str})
PAIRS = pd.DataFrame(_load_json(PAIRS_JSON))

_WF_NAME = os.path.basename(WF)
SRC_TAG = (f"Source: {_WF_NAME} "
           f"(out-of-sample walk-forward; {PSTATS.get('metric', 'roce_net')}, "
           f"{PSTATS.get('weight', 'equal')}-weight, daily MTM) unless noted.")


# --------------------------------------------------------------------------- #
# 00 — KPI strip
# --------------------------------------------------------------------------- #
def chart_kpi_strip():
    # `direction` flags metrics where the sign means good/bad performance, so a
    # negative value is shown in red rather than the (misleading) house green.
    kpis = [
        ("OOS Sharpe", f"{PSTATS['sharpe']:.2f}", "daily MTM, rf=0", PSTATS['sharpe']),
        ("Ann. return", f"{PSTATS['annualised_return']*100:.1f}%", "2021–2025",
         PSTATS['annualised_return']),
        ("Ann. vol", f"{PSTATS['annualised_vol']*100:.1f}%", "of daily P&L", None),
        ("Max drawdown", f"{PSTATS['max_drawdown']*100:.1f}%", "peak-to-trough", None),
        ("Total return", f"{PSTATS['total_return']*100:+.0f}%", "over ~5y OOS",
         PSTATS['total_return']),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 3.0))
    ax.axis("off")
    n = len(kpis)
    for i, (label, val, sub, direction) in enumerate(kpis):
        x = (i + 0.5) / n
        val_color = RED if (direction is not None and direction < 0) else GREEN
        ax.add_patch(Rectangle((i / n + 0.01, 0.05), 1 / n - 0.02, 0.9,
                               transform=ax.transAxes, facecolor="#f2f7f3",
                               edgecolor=GREEN, lw=1.5))
        ax.text(x, 0.66, val, transform=ax.transAxes, ha="center",
                va="center", fontsize=30, fontweight="bold", color=val_color)
        ax.text(x, 0.34, label, transform=ax.transAxes, ha="center",
                va="center", fontsize=15, fontweight="bold", color=NAVY)
        ax.text(x, 0.17, sub, transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color=GREY)
    fig.suptitle("ADR overnight pairs strategy — out-of-sample headline metrics",
                 fontsize=18, fontweight="bold", y=1.02)
    _save(fig, "00_kpi_strip.png")


# --------------------------------------------------------------------------- #
# 01 — Trade-lifecycle timeline schematic
# --------------------------------------------------------------------------- #
def chart_timeline():
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # session bands
    ax.add_patch(Rectangle((0.3, 4.2), 2.2, 1.0, facecolor="#cfe2f3", ec="none"))
    ax.text(1.4, 4.7, "US session\nDay D", ha="center", va="center", fontsize=13)
    ax.add_patch(Rectangle((4.0, 4.2), 2.2, 1.0, facecolor="#fce5cd", ec="none"))
    ax.text(5.1, 4.7, "Asia session\nDay D+1 open", ha="center", va="center", fontsize=13)
    ax.add_patch(Rectangle((7.5, 4.2), 2.2, 1.0, facecolor="#d9ead3", ec="none"))
    ax.text(8.6, 4.7, "Convergence\nDay K (≈5d median)", ha="center", va="center", fontsize=13)

    # overnight gap
    ax.annotate("", xy=(4.0, 4.7), xytext=(2.5, 4.7),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=2))
    ax.text(3.25, 5.05, "overnight\ngap", ha="center", va="center",
            fontsize=11, color=GREY, style="italic")

    steps = [
        (1.4, "1. ADR premium > μ + k₀σ\n$\\rightarrow$ SHORT the ADR\n(at US close)", BLUE),
        (5.1, "2. Spread still wide?\n$\\rightarrow$ BUY local share\n(at Asia open)\nNow market-neutral", GREEN),
        (8.6, "3. Spread reverts < μ\n$\\rightarrow$ COVER ADR + SELL local\n$\\rightarrow$ book ROCE / RUCE", NAVY),
    ]
    for x, txt, c in steps:
        ax.annotate("", xy=(x, 4.1), xytext=(x, 3.4),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.text(x, 2.1, txt, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=c, lw=1.8))

    ax.text(5.0, 0.5,
            "If the spread has already reverted by the Asia open, the ADR short is "
            f"covered immediately (overnight-abort, {SUMMARY['abort_rate']*100:.0f}% of entries).",
            ha="center", va="center", fontsize=12, color=GREY, style="italic")
    ax.set_title("How one trade works — short the overpriced ADR, hedge at the Asia open, "
                 "hold ~5 days to convergence", fontsize=17)
    _save(fig, "01_timeline_schematic.png")


# --------------------------------------------------------------------------- #
# Spread reconstruction (real pair) for chart 02
# --------------------------------------------------------------------------- #
def _read_parquet_cols(path, columns):
    for eng in ("pyarrow", "fastparquet"):
        try:
            return pd.read_parquet(path, columns=columns, engine=eng)
        except Exception:
            continue
    raise RuntimeError(f"could not read {path}")


def chart_spread_example():
    PAIR = "26981M__26030V"   # HKD / HKG, adr_ratio 0.05, liquid, many OOS trades
    adr_t, loc_t = PAIR.split("__")
    ratio = 0.05
    T, k0, kc = 90, 2.5, 0.0
    try:
        adr = _read_parquet_cols(os.path.join(DATA, "parquet", "adr", "adr_prices.parquet"),
                                 ["marketdate", "close", "adj_factor", "ticker"])
        glo = _read_parquet_cols(os.path.join(DATA, "parquet", "global", "global_prices.parquet"),
                                 ["marketdate", "close", "adj_factor", "ticker", "currency"])
        fx = _read_parquet_cols(os.path.join(DATA, "parquet", "fx", "fx_rates.parquet"),
                                ["date", "base_currency", "quote_currency", "mid"])
    except Exception as e:  # pragma: no cover
        warnings.warn(f"spread example skipped (parquet unreadable): {e}")
        return

    a = adr[adr["ticker"] == adr_t].copy()
    g = glo[glo["ticker"] == loc_t].copy()
    if a.empty or g.empty:
        warnings.warn("spread example skipped (ticker not found)")
        return
    cur = g["currency"].iloc[0]
    a["adj"] = a["close"] * a["adj_factor"]
    g["adj"] = g["close"] * g["adj_factor"]
    a = a.set_index("marketdate")["adj"].sort_index()
    g = g.set_index("marketdate")["adj"].sort_index()

    f = fx[(fx["base_currency"] == cur) & (fx["quote_currency"] == "USD")].copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.set_index("date")["mid"].sort_index()

    df = pd.DataFrame({"adr": a, "loc": g, "fx": f}).dropna()
    df["spread"] = df["adr"] - (df["loc"] * df["fx"]) / ratio
    df["mu"] = df["spread"].rolling(T).mean()
    df["sigma"] = df["spread"].rolling(T).std(ddof=1)
    df["upper"] = df["mu"] + k0 * df["sigma"]
    df["lower"] = df["mu"] + kc * df["sigma"]

    win = df.loc["2021-06-01":"2022-12-31"].dropna()
    if win.empty:
        warnings.warn("spread example skipped (empty window)")
        return

    fig, ax = plt.subplots()
    ax.plot(win.index, win["spread"], color=NAVY, lw=1.6, label="Spread $S_t$ (USD)")
    ax.plot(win.index, win["mu"], color=GREY, lw=1.3, ls="--",
            label=f"Rolling mean μ (T={T}d)")
    ax.plot(win.index, win["upper"], color=RED, lw=1.2,
            label=f"Entry band μ + {k0}σ")
    ax.plot(win.index, win["lower"], color=GREEN, lw=1.2,
            label=f"Exit band μ + {kc}σ")
    ax.fill_between(win.index, win["lower"], win["upper"], color=GREEN_L, alpha=0.12)

    # mark real trades on this pair in the window
    tr = TRADES[(TRADES["pair_id"] == PAIR) &
                (TRADES["close_reason"] != "overnight_abort") &
                (TRADES["open_date"] >= win.index.min()) &
                (TRADES["open_date"] <= win.index.max())]
    for _, r in tr.iterrows():
        if r["open_date"] in win.index:
            ax.scatter(r["open_date"], win.loc[r["open_date"], "spread"],
                       marker="v", color=RED, s=120, zorder=5)
        if r["close_date"] in win.index:
            ax.scatter(r["close_date"], win.loc[r["close_date"], "spread"],
                       marker="^", color=GREEN, s=120, zorder=5)
    ax.scatter([], [], marker="v", color=RED, s=120, label="Short ADR (entry)")
    ax.scatter([], [], marker="^", color=GREEN, s=120, label="Close (exit)")

    ax.set_title(f"The traded signal — fair-value spread for one pair ({cur} underlying)")
    ax.set_ylabel("Spread (USD per ADR)")
    ax.legend(loc="upper left", ncol=2, fontsize=12)
    _footnote(fig, "Spread = P_ADR − (P_local × FX)/adr_ratio, on split-adjusted prices. "
                   "Entry fires above the red band; trade closes back at the mean. Real OOS trades marked.")
    _save(fig, "02_spread_example.png")


# --------------------------------------------------------------------------- #
# 03 — Universe breakdown
# --------------------------------------------------------------------------- #
def chart_universe():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 7.0))
    byx = PAIRS["underlying_exchange"].value_counts()
    b1 = ax1.barh(byx.index[::-1], byx.values[::-1], color=GREEN)
    ax1.bar_label(b1, padding=4, fontsize=12, color=SLATE, fontweight="bold")
    ax1.set_title("By underlying exchange", fontsize=16, color=SLATE)
    ax1.set_xlabel("number of pairs")
    ax1.margins(x=0.12)
    ax1.grid(axis="y", visible=False)

    byc = PAIRS["underlying_currency"].value_counts()
    b2 = ax2.barh(byc.index[::-1], byc.values[::-1], color=BLUE)
    ax2.bar_label(b2, padding=4, fontsize=12, color=SLATE, fontweight="bold")
    ax2.set_title("By underlying currency", fontsize=16, color=SLATE)
    ax2.set_xlabel("number of pairs")
    ax2.margins(x=0.12)
    ax2.grid(axis="y", visible=False)

    fig.suptitle(f"Tradable universe — {len(PAIRS)} cointegration-screened Asian ADR pairs",
                 fontsize=20, fontweight="bold", color=INK, y=0.99)
    fig.subplots_adjust(top=0.84, wspace=0.25)
    _footnote(fig, "Source: config/pairs/asian_adr_pairs.json (screened ADF/PP p<0.05, liquidity & stale-price filters).")
    _save(fig, "03_universe_breakdown.png")


# --------------------------------------------------------------------------- #
# 04 — Walk-forward schematic
# --------------------------------------------------------------------------- #
def chart_walkforward():
    # Executive schematic (data-independent). Clean rounded blocks, a clear
    # "no look-ahead" cut-off, and a time ruler — styled for a C-suite deck.
    SLATE = "#33495e"     # selection window
    INK = "#16222e"       # headings / arrow
    AMBER = "#c8801b"     # the no-look-ahead cut-off
    X0, X1 = 2009.2, 2026.8
    with plt.rc_context({"font.family": "Helvetica Neue"}):
        fig, ax = plt.subplots(figsize=(13.2, 5.9))
        ax.set_xlim(X0, X1)
        ax.set_ylim(0, 3.5)
        ax.axis("off")

        bar_y, bar_h = 1.42, 0.88
        x_train = (2010.0, 2020.78)
        x_test = (2021.22, 2026.0)

        def rbox(x0, x1, fc):
            ax.add_patch(FancyBboxPatch(
                (x0, bar_y), x1 - x0, bar_h,
                boxstyle="round,pad=0,rounding_size=0.14", mutation_aspect=2.7,
                facecolor=fc, edgecolor="none", zorder=3))

        rbox(*x_train, SLATE)
        rbox(*x_test, GREEN)

        cxt, cxe = sum(x_train) / 2, sum(x_test) / 2
        ax.text(cxt, bar_y + bar_h * 0.63, "TRAIN", ha="center", va="center",
                color="white", fontsize=23, fontweight="bold", zorder=4)
        ax.text(cxt, bar_y + bar_h * 0.28, "2010 – 2020", ha="center", va="center",
                color="#d9e0e7", fontsize=15, fontweight="bold", zorder=4)
        ax.text(cxt, bar_y - 0.27, "Selection window  ·  screen & select cointegrated pairs",
                ha="center", va="top", color=SLATE, fontsize=12.5, zorder=4)

        ax.text(cxe, bar_y + bar_h * 0.63, "TEST", ha="center", va="center",
                color="white", fontsize=23, fontweight="bold", zorder=4)
        ax.text(cxe, bar_y + bar_h * 0.28, "2021 – 2025", ha="center", va="center",
                color="#e6f1ea", fontsize=15, fontweight="bold", zorder=4)
        ax.text(cxe, bar_y - 0.27, "Evaluation window  ·  trade out-of-sample",
                ha="center", va="top", color=GREEN, fontsize=12.5, fontweight="bold", zorder=4)

        # carry-forward arrow bridging the two windows
        xc = (x_train[1] + x_test[0]) / 2
        ax.add_patch(FancyArrowPatch(
            (2019.4, bar_y + bar_h + 0.12), (2022.9, bar_y + bar_h + 0.12),
            connectionstyle="arc3,rad=-0.28", arrowstyle="-|>",
            mutation_scale=26, lw=2.4, color=INK, zorder=5))
        ax.text(xc, bar_y + bar_h + 0.70, "selected pairs traded forward",
                ha="center", va="bottom", fontsize=12.5, color=INK, style="italic", zorder=5)

        # no-look-ahead cut-off divider
        ax.plot([xc, xc], [0.98, bar_y + bar_h + 0.02], color=AMBER, lw=2.2,
                ls=(0, (4, 3)), zorder=2)
        ax.scatter([xc], [0.98], s=70, marker="D", color=AMBER, zorder=6)
        ax.text(xc, 0.66, "selection cut-off — NO LOOK-AHEAD", ha="center", va="top",
                fontsize=12.5, fontweight="bold", color=AMBER)

        # time ruler
        ax.annotate("", xy=(X1 - 0.3, 0.98), xytext=(X0 + 0.3, 0.98),
                    arrowprops=dict(arrowstyle="-", color="#9aa6b1", lw=1.4))
        for yr in range(2010, 2027, 2):
            ax.plot([yr, yr], [0.93, 0.98], color="#9aa6b1", lw=1.2)
            ax.text(yr, 0.82, str(yr), ha="center", va="top", fontsize=11.5, color=GREY)

        # title + subtitle
        ax.text(0.5, 1.02, "Walk-forward design — built to prevent curve-fitting",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=23, fontweight="bold", color=INK)
        ax.text((X0 + X1) / 2, 3.18,
                "Pairs are chosen only from data the strategy could have seen; "
                "every reported number comes from the untouched 2021–2025 window.",
                ha="center", va="center", fontsize=13, color=GREY, style="italic")

        _save(fig, "04_walkforward_schematic.png")


# --------------------------------------------------------------------------- #
# 05 — Equity curve
# --------------------------------------------------------------------------- #
def chart_equity():
    with plt.rc_context({"font.family": "Helvetica Neue"}):
        fig, ax = plt.subplots(figsize=(13.2, 7.0))
        fig.subplots_adjust(top=0.84, left=0.075, right=0.965, bottom=0.12)
        end = EQ["equity"].iloc[-1]
        up = end >= 1.0
        series_color = GREEN if up else RED
        fill_color = GREEN if up else RED

        _area_gradient(ax, EQ["date"], EQ["equity"], fill_color, base=1.0)
        ax.plot(EQ["date"], EQ["equity"], color=series_color, lw=2.6, zorder=4,
                solid_capstyle="round")
        ax.axhline(1.0, color=HAIR, lw=1.2, ls=(0, (4, 3)), zorder=2)

        # end-point callout
        last_date = EQ["date"].iloc[-1]
        ax.scatter([last_date], [end], s=70, color=series_color, zorder=6,
                   edgecolor="white", linewidth=1.6)
        ax.annotate(f"\\${end:.2f}", xy=(last_date, end),
                    xytext=(-10, 14), textcoords="offset points",
                    ha="right", va="bottom", fontsize=20, fontweight="bold",
                    color=series_color)
        ax.annotate(f"{(end-1)*100:+.0f}% on \\$1", xy=(last_date, end),
                    xytext=(-10, 2), textcoords="offset points",
                    ha="right", va="top", fontsize=13, color=SLATE)

        # executive header
        _header(ax,
                "Growth of \\$1 — Out-of-Sample, 2021–2025",
                "Net of costs, daily mark-to-market on the untouched 2021–2025 window — "
                f"\\$1 compounds to \\${end:.2f}.")

        # KPI ribbon (top-left interior is empty for a rising curve)
        kpis = [("Total return", f"{PSTATS['total_return']*100:+.0f}%"),
                ("Annualised", f"{PSTATS['annualised_return']*100:+.1f}%"),
                ("Sharpe", f"{PSTATS['sharpe']:.2f}"),
                ("Max drawdown", f"{PSTATS['max_drawdown']*100:.1f}%")]
        x0, n, w, gap = 0.022, len(kpis), 0.158, 0.012
        for i, (lab, val) in enumerate(kpis):
            x = x0 + i * (w + gap)
            ax.add_patch(FancyBboxPatch(
                (x, 0.80), w, 0.155, transform=ax.transAxes,
                boxstyle="round,pad=0,rounding_size=0.018", mutation_aspect=1,
                facecolor="#f2f7f3", edgecolor=GREEN, lw=1.2, zorder=7))
            ax.text(x + w / 2, 0.905, val, transform=ax.transAxes, ha="center",
                    va="center", fontsize=17, fontweight="bold", color=INK, zorder=8)
            ax.text(x + w / 2, 0.832, lab, transform=ax.transAxes, ha="center",
                    va="center", fontsize=10.5, color=SLATE, zorder=8)

        ax.set_ylabel("Equity (Per \\$1 invested)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v:.1f}"))
        _year_axis(ax)
        # pad the right so the end-point callout never clips, and the left so
        # the first (2021) year tick is visible
        ax.set_xlim(EQ["date"].iloc[0] - pd.Timedelta(days=20),
                    last_date + pd.Timedelta(days=70))
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(HAIR)
        _footnote(fig, SRC_TAG + "  ROCE-net (unlevered), equal-weighted across concurrent positions.")
        _save(fig, "05_equity_curve.png")


# --------------------------------------------------------------------------- #
# 06 — Underwater drawdown
# --------------------------------------------------------------------------- #
def chart_drawdown():
    eq = EQ.copy()
    eq["dd"] = eq["equity"] / eq["equity"].cummax() - 1.0
    with plt.rc_context({"font.family": "Helvetica Neue"}):
        fig, ax = plt.subplots(figsize=(13.2, 7.0))
        fig.subplots_adjust(top=0.84, left=0.075, right=0.965, bottom=0.12)

        _area_gradient(ax, eq["date"], eq["dd"], RED, base=0.0, alpha=0.40)
        ax.plot(eq["date"], eq["dd"], color=RED, lw=2.0, zorder=4,
                solid_capstyle="round")
        ax.axhline(0, color=HAIR, lw=1.2, zorder=2)

        trough = eq.loc[eq["dd"].idxmin()]
        ax.scatter([trough["date"]], [trough["dd"]], s=70, color=RED, zorder=6,
                   edgecolor="white", linewidth=1.6)
        ax.annotate(f"max drawdown {trough['dd']*100:.1f}%",
                    xy=(trough["date"], trough["dd"]),
                    xytext=(14, 10), textcoords="offset points",
                    fontsize=14, fontweight="bold", color=RED,
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fbf3f3", ec=RED, lw=1.2),
                    arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))

        _header(ax,
                "Drawdown profile — Shallow and quick to recover",
                f"Worst peak-to-trough loss is just {trough['dd']*100:.1f}%; "
                f"Calmar (annual return ÷ |max drawdown|) = {PSTATS['calmar']:.2f}.")

        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylabel("Drawdown from peak")
        ax.set_ylim(eq["dd"].min() * 1.18, 0.004)
        _year_axis(ax)
        ax.set_xlim(eq["date"].iloc[0] - pd.Timedelta(days=20),
                    eq["date"].iloc[-1] + pd.Timedelta(days=20))
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(HAIR)
        _footnote(fig, SRC_TAG + f"  Calmar (ann.return/|MaxDD|) = {PSTATS['calmar']:.2f}.")
        _save(fig, "06_drawdown_underwater.png")


# --------------------------------------------------------------------------- #
# 07 — Rolling 252-day Sharpe
# --------------------------------------------------------------------------- #
def chart_rolling_sharpe():
    r = ROLL.dropna(subset=["rolling_sharpe_252d"])
    fig, ax = plt.subplots()
    ax.plot(r["date"], r["rolling_sharpe_252d"], color=NAVY, lw=2.0)
    ax.fill_between(r["date"], 0, r["rolling_sharpe_252d"],
                    where=r["rolling_sharpe_252d"] >= 0, color=GREEN_L, alpha=0.25)
    ax.fill_between(r["date"], 0, r["rolling_sharpe_252d"],
                    where=r["rolling_sharpe_252d"] < 0, color="#f4a582", alpha=0.4)
    ax.axhline(0, color=GREY, lw=1)
    ax.axhline(PSTATS["sharpe"], color=GREEN, lw=1.4, ls="--",
               label=f"full-period Sharpe {PSTATS['sharpe']:.2f}")
    ax.set_title("Rolling 1-year Sharpe — strength is concentrated in 2024–2025 (key caveat)")
    ax.set_ylabel("trailing 252-day Sharpe")
    ax.legend(loc="upper right")
    _footnote(fig, SRC_TAG + "  First 252 days are warm-up (no value).")
    _save(fig, "07_rolling_sharpe.png")


# --------------------------------------------------------------------------- #
# 08 — Per-trade return histogram
# --------------------------------------------------------------------------- #
def chart_trade_hist():
    fig, ax = plt.subplots()
    data = TRADES["ruce_net"].dropna()
    data = data[(data > -0.4) & (data < 0.6)]  # clip extreme tails for readability
    ax.hist(data, bins=80, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.3)
    med = TRADES["ruce_net"].median()
    ax.axvline(0, color=GREY, lw=1.3, ls="--")
    ax.axvline(med, color=GREEN, lw=2.2, label=f"median {med*100:+.2f}%")
    win = (TRADES["ruce_net"] > 0).mean()
    ax.xaxis.set_major_formatter(PCT)
    ax.set_title("Per-trade outcome distribution — small positive median, right-skewed tail")
    ax.set_xlabel("RUCE-net per trade (Reg-T margin, net of costs)")
    ax.set_ylabel("number of trades")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.95,
            f"n = {len(TRADES):,} trades\nwin rate {win*100:.0f}%\nmedian hold 5 days",
            transform=ax.transAxes, va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef3f8", ec=BLUE))
    _footnote(fig, SRC_TAG + "  x-axis clipped to [-40%,+60%] for readability; medians use full sample.")
    _save(fig, "08_ruce_hist.png")


# --------------------------------------------------------------------------- #
# 08b — Percentile fan
# --------------------------------------------------------------------------- #
def chart_percentile_fan():
    dist = {d["metric"]: d for d in DISTRIB}
    metrics = [("ROCE net per trade", "ROCE-net\n(unlevered)", GREY),
               ("RUCE net per trade", "RUCE-net\n(Reg-T 2× margin)", GREEN)]
    levels = ["p10", "p25", "median", "p75", "p90"]
    fig, ax = plt.subplots()
    for i, (key, label, c) in enumerate(metrics):
        d = dist[key]
        ys = [d[l] for l in levels]
        x = i
        ax.plot([x, x], [d["p10"], d["p90"]], color=c, lw=3, solid_capstyle="round")
        ax.plot([x, x], [d["p25"], d["p75"]], color=c, lw=10, alpha=0.5, solid_capstyle="round")
        ax.scatter([x], [d["median"]], color=c, s=160, zorder=5,
                   edgecolor="white", linewidth=1.5)
        ax.text(x + 0.08, d["median"], f"med {d['median']*100:+.2f}%",
                va="center", fontsize=13, fontweight="bold", color=c)
        ax.text(x + 0.08, d["p90"], f"p90 {d['p90']*100:+.1f}%", va="center", fontsize=11, color=GREY)
        ax.text(x + 0.08, d["p10"], f"p10 {d['p10']*100:.1f}%", va="center", fontsize=11, color=GREY)
    ax.axhline(0, color=GREY, lw=1, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_xlim(-0.5, 1.7)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Per-trade return percentiles — leverage widens both upside and risk")
    ax.set_ylabel("per-trade return")
    _footnote(fig, SRC_TAG + "  Box = p25–p75, whisker = p10–p90, dot = median (distribution.json).")
    _save(fig, "08b_percentile_fan.png")


# --------------------------------------------------------------------------- #
# 09 — Cost / break-even sensitivity
# --------------------------------------------------------------------------- #
def chart_cost_sensitivity():
    # Additional slippage s (per side per leg) on top of the Roll spread already
    # embedded in fills. Closed pair trade pays it on 4 fills:
    #   ROCE = local_ret + adr_ret           -> loses 4·s  (2 fills/leg, 2 legs)
    #   RUCE = local_ret + 2·adr_ret         -> loses 6·s  (adr leg cost levered 2×)
    bps = np.array([0, 10, 25, 50, 75, 100])
    s = bps / 10000.0
    roce0 = TRADES["roce_net"]
    ruce0 = TRADES["ruce_net"]
    roce_med = [(roce0 - 4 * x).median() for x in s]
    ruce_med = [(ruce0 - 6 * x).median() for x in s]

    fig, ax = plt.subplots()
    ax.plot(bps, np.array(ruce_med) * 100, "-o", color=GREEN, lw=2.4,
            label="RUCE-net (Reg-T 2× margin)")
    ax.plot(bps, np.array(roce_med) * 100, "-o", color=GREY, lw=2.2,
            label="ROCE-net (unlevered)")
    ax.axhline(0, color=RED, lw=1.4, ls="--")
    ax.set_title("Cost stress — median per-trade return vs added slippage")
    ax.set_xlabel("additional slippage per side, per leg (bps)")
    ax.set_ylabel("median per-trade return (%)")
    ax.legend(loc="upper right")
    # breakeven marker for RUCE
    rm = np.array(ruce_med)
    if (rm < 0).any():
        idx = np.argmax(rm < 0)
        be = np.interp(0, [rm[idx]*100, rm[idx-1]*100], [bps[idx], bps[idx-1]])
        ax.axvline(be, color=ORANGE, lw=1.6, ls=":")
        ax.text(be, ax.get_ylim()[1]*0.85, f" break-even ≈ {be:.0f} bps",
                color=ORANGE, fontsize=12, fontweight="bold")
    _footnote(fig, SRC_TAG + "  Roll spread (~2.3% round-trip) already in fills; this is INCREMENTAL slippage. "
                             "ROCE loses 4×bps, RUCE 6×bps (levered ADR leg).")
    _save(fig, "09_cost_sensitivity.png")


# --------------------------------------------------------------------------- #
# 10 — Leg attribution
# --------------------------------------------------------------------------- #
def chart_leg_attribution():
    closed = TRADES[TRADES["close_reason"] != "overnight_abort"]
    legs = [("ADR short leg", closed["adr_return"], BLUE),
            ("Local long leg", closed["local_return"], GREY)]
    names = [l[0] for l in legs]
    meds = [l[1].median() for l in legs]
    wins = [(l[1] > 0).mean() for l in legs]
    cols = [l[2] for l in legs]

    with plt.rc_context({"font.family": "Helvetica Neue"}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 7.0))
        fig.subplots_adjust(top=0.78, bottom=0.10, left=0.07, right=0.965, wspace=0.26)

        # figure-level executive header (matches _header on single-axis charts)
        fig.text(0.07, 0.955, "Where the alpha lives — the ADR short is the edge",
                 ha="left", va="top", fontsize=22, fontweight="bold", color=INK)
        fig.text(0.07, 0.888,
                 "The overpriced-ADR short carries the return and wins "
                 f"{wins[0]*100:.0f}% of the time; the local leg is a near-flat market hedge.",
                 ha="left", va="top", fontsize=13.5, color=SLATE, style="italic")

        # --- Panel A: median return per leg ---
        b1 = ax1.bar(names, [m * 100 for m in meds], color=cols, width=0.55,
                     zorder=3)
        ax1.axhline(0, color=HAIR, lw=1.2, zorder=2)
        ax1.set_title("Median return per leg", fontsize=16, color=SLATE,
                      fontweight="bold", pad=10)
        ax1.set_ylabel("median per-trade return (%)")
        span = max(abs(m) for m in meds) * 100
        ax1.set_ylim(-span * 1.45, span * 1.45)
        for b, m in zip(b1, meds):
            y = m * 100
            ax1.text(b.get_x() + b.get_width()/2,
                     y + (0.07 * span if m >= 0 else -0.07 * span),
                     f"{m*100:+.2f}%", ha="center",
                     va="bottom" if m >= 0 else "top",
                     fontsize=17, fontweight="bold",
                     color=GREEN if m >= 0 else RED)

        # --- Panel B: win rate per leg ---
        b2 = ax2.bar(names, [w * 100 for w in wins], color=cols, width=0.55,
                     zorder=3)
        ax2.axhline(50, color=RED, lw=1.6, ls=(0, (4, 3)), zorder=4)
        ax2.text(1.48, 50, "coin-flip 50%", ha="right", va="bottom",
                 fontsize=12, color=RED, fontweight="bold")
        ax2.set_title("Win rate per leg", fontsize=16, color=SLATE,
                      fontweight="bold", pad=10)
        ax2.set_ylabel("% of trades profitable")
        ax2.set_ylim(0, 100)
        for b, w in zip(b2, wins):
            ax2.text(b.get_x() + b.get_width()/2, w*100 + 2.2,
                     f"{w*100:.0f}%", ha="center", fontsize=17, fontweight="bold",
                     color=INK)

        for ax in (ax1, ax2):
            ax.grid(False)
            ax.margins(x=0.18)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(HAIR)
            ax.tick_params(labelsize=13)

        _footnote(fig, SRC_TAG + "  Closed (round-trip) trades only.")
        _save(fig, "10_leg_attribution.png")


# --------------------------------------------------------------------------- #
# 11 — Parameter sensitivity (PRE-FIX research grid — relative ranking only)
# --------------------------------------------------------------------------- #
def chart_param_sensitivity():
    if RESEARCH is None:
        warnings.warn("param sensitivity skipped (no research_runs.csv found; "
                      "run datastream/run_research_permutations.py to generate it)")
        return
    try:
        rr = pd.read_csv(os.path.join(RESEARCH, "research_runs.csv"))
    except FileNotFoundError:
        warnings.warn("param sensitivity skipped (research_runs.csv missing)")
        return
    neg_share = (rr["ruce_net_median"] < 0).mean()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 6.2), sharey=True)
    specs = [("k0", "Entry threshold k₀", BLUE),
             ("estimation_days", "Estimation window T (days)", GREEN),
             ("kc", "Exit threshold k_c", ORANGE)]
    for ax, (col, label, c) in zip(axes, specs):
        grp = rr.groupby(col)["ruce_net_median"].median()
        ax.bar([str(x) for x in grp.index], grp.values * 100, color=c, width=0.6)
        ax.axhline(0, color=GREY, lw=1.2)
        for i, v in enumerate(grp.values):
            ax.text(i, v*100 - 0.06, f"{v*100:.2f}%", ha="center", va="top",
                    fontsize=12, fontweight="bold", color=RED)
        ax.set_title(label, fontsize=15)
        ax.set_xlabel(col)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"median per-trade RUCE-net\nacross {len(rr):,} runs (%)")
    axes[0].set_ylim(min(rr.groupby("estimation_days")["ruce_net_median"].median().min(),
                        rr.groupby("k0")["ruce_net_median"].median().min()) * 100 * 1.35, 0.35)
    fig.suptitle("Parameter ranking (sign-fixed) — selectivity helps, but the MEDIAN trade is negative",
                 fontsize=16, fontweight="bold")
    _footnote(fig, f"Source: {os.path.basename(RESEARCH)} grid, {len(rr):,} sign-fixed train/test runs — "
                   f"{neg_share*100:.0f}% have a negative median trade. Ranking holds (k₀=2.5, T=90, k_c=0 least-bad); "
                   "the positive headline comes from the right-skewed MEAN, not the typical trade (see slides 14 & 16).")
    _save(fig, "11_param_sensitivity.png")


# --------------------------------------------------------------------------- #
# 13b — Anatomy of the 2024–25 returns
# --------------------------------------------------------------------------- #
def chart_anatomy():
    # Panel A: OOS return by calendar year (compounded daily returns)
    eq = EQ.copy()
    eq["yr"] = eq["date"].dt.year
    yrs, rets, wr = [], [], []
    for y, g in eq.groupby("yr"):
        yrs.append(int(y))
        rets.append((1 + g["daily_return"]).prod() - 1)
    # win rate per year from trades (by close year)
    tr = TRADES.copy()
    tr["yr"] = tr["close_date"].dt.year
    wr_map = tr.groupby("yr").apply(lambda d: (d["ruce_net"] > 0).mean())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 6.6))

    cols = [GREEN if r >= 0 else RED for r in rets]
    bars = axA.bar([str(y) for y in yrs], [r * 100 for r in rets], color=cols, width=0.65)
    axA.axhline(0, color=GREY, lw=1)
    # adaptive y-limits so small/negative bars stay legible
    rmax = max([r * 100 for r in rets] + [0.0])
    rmin = min([r * 100 for r in rets] + [0.0])
    span = max(rmax - rmin, 1.0)
    lo, hi = rmin - 0.25 * span, rmax + 0.18 * span
    win_y = lo + 0.05 * span
    for b, r, y in zip(bars, rets, yrs):
        axA.text(b.get_x() + b.get_width()/2, r*100 + (0.02 if r >= 0 else -0.02) * span,
                 f"{r*100:+.1f}%", ha="center", va="bottom" if r >= 0 else "top",
                 fontsize=14, fontweight="bold")
        w = wr_map.get(y, np.nan)
        if w == w:
            axA.text(b.get_x() + b.get_width()/2, win_y, f"win {w*100:.0f}%",
                     ha="center", fontsize=11, color=GREY)
    axA.set_title("OOS return by calendar year", fontsize=16)
    axA.set_ylabel("annual return (%)")
    axA.set_ylim(lo, hi)
    axA.grid(axis="x", visible=False)

    # Panel B: concentration of 2025 gains (cumulative contribution curve)
    g25 = tr[tr["yr"] == 2025].sort_values("ruce_net", ascending=False).reset_index(drop=True)
    pos = g25["ruce_net"].clip(lower=0)
    if len(g25) == 0 or pos.sum() == 0:
        warnings.warn("anatomy Panel B skipped (no positive 2025 trades)")
        axB.axis("off")
        axB.text(0.5, 0.5, "no positive 2025 trades", ha="center", va="center",
                 transform=axB.transAxes, color=GREY)
        fig.suptitle("Anatomy of the OOS returns — by calendar year",
                     fontsize=18, fontweight="bold")
        fig.subplots_adjust(top=0.86)
        _footnote(fig, SRC_TAG)
        _save(fig, "13b_anatomy_2024_25.png")
        return
    cum = pos.cumsum() / pos.sum()
    xfrac = (np.arange(1, len(g25) + 1)) / len(g25)
    axB.plot(xfrac * 100, cum * 100, color=BLUE, lw=2.6)
    axB.plot([0, 100], [0, 100], color=GREY, lw=1, ls="--", label="if returns were even")
    # mark the RUCE>10% cohort
    n_big = int((g25["ruce_net"] > 0.10).sum())
    x_big = n_big / len(g25) * 100
    y_big = cum.iloc[n_big - 1] * 100
    axB.scatter([x_big], [y_big], color=RED, s=120, zorder=5)
    axB.annotate(f"top {x_big:.0f}% of trades\n(RUCE>10%) = {y_big:.0f}% of gains",
                 xy=(x_big, y_big), xytext=(x_big + 8, y_big - 22),
                 fontsize=12.5, fontweight="bold", color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED))
    axB.set_title("…and 2025's gains come from a few big reversions", fontsize=15)
    axB.set_xlabel("% of 2025 trades (largest first)")
    axB.set_ylabel("cumulative % of gross-positive return")
    axB.set_xlim(0, 100); axB.set_ylim(0, 100)
    axB.legend(loc="lower right")
    big = tr[(tr["yr"] == 2025) & (tr["ruce_net"] > 0.10)]
    axB.text(0.03, 0.97,
             f"those {n_big} trades:\nADR leg +{big['adr_return'].median()*100:.1f}% (median)\n"
             f"local leg {big['local_return'].median()*100:+.1f}%\n~{big['duration_days'].median():.0f}-day hold",
             transform=axB.transAxes, va="top", fontsize=11.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="#eef3f8", ec=BLUE))

    fig.suptitle("Anatomy of the 2024–25 returns — episodic, ADR-leg, big-reversion driven",
                 fontsize=18, fontweight="bold")
    fig.subplots_adjust(top=0.86)
    _footnote(fig, SRC_TAG + "  Annual returns compound daily MTM; 2025 concentration from trades_oos (close-date year).")
    _save(fig, "13b_anatomy_2024_25.png")


# --------------------------------------------------------------------------- #
# A1 — Top contributors to the 2024–25 gains
# --------------------------------------------------------------------------- #
# ADR ISIN -> (display name, OTC/unsponsored flag) resolved via public lookups
# (boerse-frankfurt, onvista, finanzen.net, ariva, marketscreener). Names not in
# the project data; verify before external use. "?" = could not resolve by ISIN.
_NAME_BY_ISIN = {
    "US4385503036": ("Hong Kong & China Gas", True),
    "US65412C1080": ("Nihon Kohden", True),
    "US1777973058": ("City Developments", True),
    "US78392U1051": ("Ryohin Keikaku (MUJI)", True),
    "US8891091048": ("Japanese ADR (unresolved)", True),
    "US23340X1081": ("D&L Industries", True),
    "US4385813088": ("Hongkong Land", True),
    "US5738143081": ("Marui Group", True),
}


def chart_top_contributors():
    tr = TRADES.copy()
    tr["yr"] = tr["close_date"].dt.year
    g = tr[tr["yr"] >= 2024].copy()
    total = g["ruce_net"].sum()
    if total <= 0:
        warnings.warn(
            "top-contributors skipped: 2024-25 net RUCE is non-positive "
            f"({total:.4f}) under the liquidity filter, so 'share of gains' is "
            "undefined. The surge narrative does not hold for liquid ADRs.")
        return
    try:
        for eng in ("pyarrow", "fastparquet"):
            try:
                ref = pd.read_parquet(os.path.join(DATA, "parquet", "adr", "adr_reference.parquet"),
                                      engine=eng)
                break
            except Exception:
                continue
        iso = ref.drop_duplicates("adr_ticker").set_index("adr_ticker")["adr_isin"].to_dict()
    except Exception:
        warnings.warn("top-contributors skipped (reference unreadable)")
        return
    a = (g.groupby("adr_ticker")["ruce_net"]
         .agg(["sum", "size", "mean"]).reset_index().sort_values("sum", ascending=False))
    a["isin"] = a["adr_ticker"].map(iso)
    a["name"] = a["isin"].map(lambda i: _NAME_BY_ISIN.get(i, (f"{i}", None))[0])
    a["share"] = a["sum"] / total * 100
    top = a.head(8).iloc[::-1]  # reverse for horizontal bar (largest on top)

    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    labels = [f"{n}" for n in top["name"]]
    bars = ax.barh(labels, top["share"], color=BLUE)
    for b, sh, n, mn, isin in zip(bars, top["share"], top["size"], top["mean"], top["isin"]):
        ax.text(sh + 0.2, b.get_y() + b.get_height()/2,
                f"{sh:.0f}%   ({int(n)} trades · mean +{mn*100:.0f}%/trade)",
                va="center", fontsize=12)
    ax.set_xlabel("share of total 2024–25 gains (summed RUCE-net, %)")
    ax.set_xlim(0, max(top["share"]) * 1.5)
    ax.set_title("Top contributors to the 2024–25 surge — concentrated in unsponsored OTC ADRs")
    ax.grid(axis="y", visible=False)
    n_pos = int((a["sum"] > 0).sum())
    ax.text(0.97, 0.06,
            f"Top 8 ≈ {top['share'].sum():.0f}% of '24–25 gains · {n_pos} pairs positive overall\n"
            "Most are UNSPONSORED OTC ADRs $\\rightarrow$ big modeled convergences may not\n"
            "be capturable at size. Validate real bid/ask + impact before trading.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fbf3f3", ec=RED))
    _footnote(fig, SRC_TAG + "  Names resolved via public ISIN lookups (verify); not in project data.")
    _save(fig, "A1_top_contributors.png")


# --------------------------------------------------------------------------- #
# 12 — Integrity: before/after the sign fix
# --------------------------------------------------------------------------- #
def chart_integrity():
    # Figures from AMENDS.md (2026-06-14 ADR roll-cost sign fix).
    labels = ["Reported\n(buggy, in-sample)", "Corrected\nin-sample", "Corrected\nOUT-OF-SAMPLE"]
    sharpe = [5.66, -1.11, PSTATS["sharpe"]]
    cols = [RED, "#d6a000", GREEN]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, sharpe, color=cols, width=0.6)
    ax.axhline(0, color=GREY, lw=1)
    for b, v in zip(bars, sharpe):
        ax.text(b.get_x()+b.get_width()/2, v + (0.12 if v >= 0 else -0.28),
                f"{v:.2f}", ha="center", fontsize=16, fontweight="bold")
    ax.set_title("We found and fixed an ADR cost sign-error — and report the honest number")
    ax.set_ylabel("Sharpe ratio")
    _oos = PSTATS["sharpe"]
    ax.text(0.5, 0.92,
            "An ADR roll-cost sign bug had booked the bid/ask as a gain.\n"
            "Fixing it turned the in-sample Sharpe from +5.66 to -1.11. We headline\n"
            f"the corrected out-of-sample {_oos:.2f} — and show costs honestly throughout.",
            transform=ax.transAxes, ha="center", va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fbf3f3", ec=RED))
    _footnote(fig, f"Source: AMENDS.md (2026-06-14) for the in-sample bars; OOS = {_WF_NAME}.")
    _save(fig, "12_integrity_before_after.png")


# --------------------------------------------------------------------------- #
def main():
    print("Generating charts ->", OUT)
    chart_kpi_strip()
    chart_timeline()
    chart_spread_example()
    chart_universe()
    chart_walkforward()
    chart_equity()
    chart_drawdown()
    chart_rolling_sharpe()
    chart_trade_hist()
    chart_percentile_fan()
    chart_cost_sensitivity()
    chart_leg_attribution()
    chart_param_sensitivity()
    chart_anatomy()
    chart_top_contributors()
    chart_integrity()
    print("Done.")


if __name__ == "__main__":
    main()
