"""
Generate cumulative PNL, P&L breakdown, and year-by-year return charts.

Run from pairs-trading-ml/:
  python submission/report/build_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).resolve().parent.parent / "results"
ASSETS = Path(__file__).resolve().parent / "assets"

NAVY = "#1b4965"
BLUE = "#5fa8d3"
LIGHT = "#cae9ff"
INK = "#1d2733"
GRAY = "#5a6b7b"
GREEN = "#2e7d4f"
RED = "#b3402f"
ORANGE = "#d4812a"
PURPLE = "#7b4f9e"
PANEL = "#f2f7fb"


# ── 1. Cumulative PNL — IS + OOS stitched together ──

def build_cumulative_pnl():
    # IS data (2003-2020)
    is_cells = {
        "PC core": ("pc_core", NAVY),
        "PC + filter": ("pc_filtered", BLUE),
        "SSD + filter": ("ssd_filtered", ORANGE),
        "SSD core": ("ssd_core", GRAY),
    }
    # OOS data (2021-2025) — only PC and factor
    oos_cells = {
        "PC core": ("oos_pc_core", NAVY),
        "Factor-beta": ("oos_factor_core", PURPLE),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    oos_start = pd.Timestamp("2021-01-01")

    # Plot IS curves
    for label, (fname, color) in is_cells.items():
        df = pd.read_parquet(RESULTS / f"{fname}_monthly.parquet")
        cum = (1 + df["monthly_return"]).cumprod()
        lw = 2.8 if "PC core" in label else 1.6
        ax.plot(df.index, cum, label=f"{label} (IS)", color=color, linewidth=lw)

    # Plot OOS curves — stitch from end of IS
    pc_is = pd.read_parquet(RESULTS / "pc_core_monthly.parquet")
    pc_is_end_val = float((1 + pc_is["monthly_return"]).cumprod().iloc[-1])

    for label, (fname, color) in oos_cells.items():
        df = pd.read_parquet(RESULTS / f"{fname}_monthly.parquet")
        cum = (1 + df["monthly_return"]).cumprod()
        if label == "PC core":
            cum_stitched = cum * pc_is_end_val
            ax.plot(df.index, cum_stitched, color=color, linewidth=2.8, linestyle="--")
        else:
            # Factor starts at 1.0 for OOS (no IS baseline to stitch)
            ax.plot(df.index, cum, label=f"{label} (OOS only)", color=color,
                    linewidth=1.8, linestyle="--")

    # OOS shading
    ax.axvspan(oos_start, df.index[-1], alpha=0.07, color=RED, zorder=0)
    ax.axvline(oos_start, color=RED, linewidth=1.2, linestyle=":", alpha=0.6)

    # Annotations
    ax.annotate("IN-SAMPLE\n2003-2020", xy=(pd.Timestamp("2011-06-01"), 0.96),
                fontsize=11, color=NAVY, ha="center", fontstyle="italic")
    ax.annotate("OOS\n2021-2025", xy=(pd.Timestamp("2023-06-01"), 0.96),
                fontsize=11, color=RED, ha="center", fontstyle="italic")

    # Crisis/event annotations
    ax.annotate("GFC", xy=(pd.Timestamp("2008-09-01"), 1.0),
                xytext=(pd.Timestamp("2008-09-01"), 1.35),
                fontsize=9, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.annotate("COVID", xy=(pd.Timestamp("2020-03-01"), 1.0),
                xytext=(pd.Timestamp("2020-03-01"), 1.55),
                fontsize=9, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

    ax.set_ylabel("Growth of $1", fontsize=13, color=INK)
    ax.set_title("Cumulative Returns — In-Sample & Out-of-Sample",
                 fontsize=16, color=NAVY, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0.9)

    # Add dashed line legend note
    dash_patch = mpatches.Patch(color="none", label="-- dashed = OOS (2021-2025)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    out = ASSETS / "cumulative_pnl.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. P&L breakdown by exit reason ──

def build_pnl_breakdown():
    cells = {
        "SSD core": "ssd_core",
        "PC core": "pc_core",
        "PC + filter": "pc_filtered",
        "SSD + filter": "ssd_filtered",
    }

    rows = []
    for label, fname in cells.items():
        df = pd.read_parquet(RESULTS / f"{fname}_trades.parquet")
        total = len(df)
        for reason in ["reversion", "force_close", "stop"]:
            sub = df[df["exit_reason"] == reason]
            if len(sub) == 0:
                continue
            rows.append({
                "strategy": label,
                "exit_reason": reason,
                "count": len(sub),
                "pct_of_trades": len(sub) / total * 100,
                "mean_return_bps": sub["round_trip_return"].mean() * 10000,
                "hit_rate": (sub["round_trip_return"] > 0).mean() * 100,
            })

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    strategies = ["SSD core", "PC core", "PC + filter", "SSD + filter"]
    x = np.arange(len(strategies))
    bar_w = 0.35

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("pct_of_trades", "% of trades", "Trade Mix"),
        ("mean_return_bps", "Mean return (bps)", "Return per Trade"),
        ("hit_rate", "Hit rate (%)", "Win Rate"),
    ]):
        ax = axes[ax_idx]
        for i, reason in enumerate(["reversion", "force_close"]):
            vals = []
            for s in strategies:
                row = summary[(summary["strategy"] == s) & (summary["exit_reason"] == reason)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            color = GREEN if reason == "reversion" else RED
            label = "Reversion" if reason == "reversion" else "Force-close"
            ax.bar(x + i * bar_w, vals, bar_w, label=label, color=color, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY)
        ax.set_xticks(x + bar_w / 2)
        ax.set_xticklabels(strategies, fontsize=9, rotation=15)
        if metric == "mean_return_bps":
            ax.axhline(0, color="black", linewidth=0.5)
        if metric == "hit_rate":
            ax.axhline(50, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("P&L Breakdown by Exit Reason", fontsize=16, fontweight="bold",
                 color=NAVY, y=1.02)
    fig.tight_layout()
    out = ASSETS / "pnl_breakdown.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── 3. Year-by-year returns heatmap ──

def build_yearly_returns():
    cells = {
        "SSD core": "ssd_core",
        "SSD + filter": "ssd_filtered",
        "PC core": "pc_core",
        "PC + filter": "pc_filtered",
    }

    yearly = {}
    for label, fname in cells.items():
        df = pd.read_parquet(RESULTS / f"{fname}_monthly.parquet")
        df["year"] = df.index.year
        annual = df.groupby("year")["monthly_return"].apply(
            lambda x: float((1 + x).prod() - 1)
        ) * 100  # in %
        yearly[label] = annual

    # Add OOS
    for label, fname in [("PC core OOS", "oos_pc_core"), ("Factor OOS", "oos_factor_core")]:
        df = pd.read_parquet(RESULTS / f"{fname}_monthly.parquet")
        df["year"] = df.index.year
        annual = df.groupby("year")["monthly_return"].apply(
            lambda x: float((1 + x).prod() - 1)
        ) * 100
        yearly[label] = annual

    all_years = sorted(set().union(*[s.index for s in yearly.values()]))
    strategies = list(yearly.keys())

    data = np.full((len(strategies), len(all_years)), np.nan)
    for i, strat in enumerate(strategies):
        for j, year in enumerate(all_years):
            if year in yearly[strat].index:
                data[i, j] = yearly[strat][year]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    vmax = np.nanmax(np.abs(data))
    vmax = min(vmax, 15)  # cap for color scale
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels(all_years, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=12)

    # Add text annotations
    for i in range(len(strategies)):
        for j in range(len(all_years)):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.6 else INK
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if abs(val) > 5 else "normal")

    # OOS divider
    oos_start_idx = all_years.index(2021)
    ax.axvline(oos_start_idx - 0.5, color=RED, linewidth=2.5, linestyle="-")
    ax.text(oos_start_idx - 0.7, -0.8, "OOS -->", fontsize=10, color=RED,
            fontweight="bold", ha="left")
    ax.text(oos_start_idx - 1.3, -0.8, "<-- IS", fontsize=10, color=NAVY,
            fontweight="bold", ha="right")

    # IS/OOS row divider
    ax.axhline(3.5, color=GRAY, linewidth=1.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Annual Return (%)", fontsize=11)

    ax.set_title("Year-by-Year Returns (%) — All Strategies",
                 fontsize=16, color=NAVY, fontweight="bold", pad=15)

    fig.tight_layout()
    out = ASSETS / "yearly_returns.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    # Print summary findings
    print("\nKey findings:")
    pc_is = yearly["PC core"]
    print(f"  PC core best year:  {pc_is.idxmax()} ({pc_is.max():+.1f}%)")
    print(f"  PC core worst year: {pc_is.idxmin()} ({pc_is.min():+.1f}%)")
    pos = (pc_is > 0).sum()
    print(f"  PC core positive years: {pos}/{len(pc_is)} ({pos/len(pc_is)*100:.0f}%)")

    # GFC period
    gfc = pc_is.loc[2007:2009]
    print(f"  PC core GFC (2007-09): {gfc.sum():+.1f}% cumulative")

    # Calm years
    calm = pc_is.loc[2013:2019]
    print(f"  PC core calm (2013-19): {calm.sum():+.1f}% cumulative, avg {calm.mean():+.1f}%/yr")

    if "PC core OOS" in yearly:
        oos = yearly["PC core OOS"]
        print(f"  PC core OOS avg: {oos.mean():+.1f}%/yr")
        for y in oos.index:
            print(f"    {y}: {oos[y]:+.1f}%")


# ── 4. Year-by-year returns heatmap — filtered only ──

def build_yearly_returns_filtered():
    is_cells = {
        "SSD+filt IS": "ssd_filtered",
        "PC+filt IS": "pc_filtered",
        "Factor+filt IS": "factor_filtered",
    }
    oos_cells = {
        "SSD+filt OOS": "oos_ssd_filtered",
        "PC+filt OOS": "oos_pc_filtered",
        "Factor+filt OOS": "oos_factor_filtered",
    }

    yearly = {}
    for label, fname in {**is_cells, **oos_cells}.items():
        path = RESULTS / f"{fname}_monthly.parquet"
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        df = pd.read_parquet(path)
        df["year"] = df.index.year
        annual = df.groupby("year")["monthly_return"].apply(
            lambda x: float((1 + x).prod() - 1)
        ) * 100
        yearly[label] = annual

    if not yearly:
        print("  No filtered data found, skipping yearly_returns_filtered")
        return

    all_years = sorted(set().union(*[s.index for s in yearly.values()]))
    strategies = list(yearly.keys())

    data = np.full((len(strategies), len(all_years)), np.nan)
    for i, strat in enumerate(strategies):
        for j, year in enumerate(all_years):
            if year in yearly[strat].index:
                data[i, j] = yearly[strat][year]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    vmax = np.nanmax(np.abs(data))
    vmax = min(vmax, 15)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels(all_years, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=11)

    for i in range(len(strategies)):
        for j in range(len(all_years)):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.6 else INK
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if abs(val) > 5 else "normal")

    if 2021 in all_years:
        oos_start_idx = all_years.index(2021)
        ax.axvline(oos_start_idx - 0.5, color=RED, linewidth=2.5, linestyle="-")
        ax.text(oos_start_idx - 0.7, -0.8, "OOS -->", fontsize=10, color=RED,
                fontweight="bold", ha="left")
        ax.text(oos_start_idx - 1.3, -0.8, "<-- IS", fontsize=10, color=NAVY,
                fontweight="bold", ha="right")

    is_count = len([s for s in strategies if "IS" in s])
    if is_count > 0 and is_count < len(strategies):
        ax.axhline(is_count - 0.5, color=GRAY, linewidth=1.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Annual Return (%)", fontsize=11)

    ax.set_title("Year-by-Year Returns (%) — Filtered Strategies Only",
                 fontsize=16, color=NAVY, fontweight="bold", pad=15)

    fig.tight_layout()
    out = ASSETS / "yearly_returns_filtered.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    # Summary
    print("\nFiltered yearly summary:")
    for label in strategies:
        s = yearly[label]
        pos = (s > 0).sum()
        print(f"  {label}: {pos}/{len(s)} positive years ({pos/len(s)*100:.0f}%), avg {s.mean():+.1f}%/yr")


# ── 5. Cumulative PNL — filtered strategies only (IS + OOS) ──

def build_cumulative_pnl_filtered():
    is_cells = {
        "SSD + filter (IS)": ("ssd_filtered", ORANGE),
        "PC + filter (IS)": ("pc_filtered", BLUE),
        "Factor + filter (IS)": ("factor_filtered", PURPLE),
    }
    oos_cells = {
        "PC + filter (OOS)": ("oos_pc_filtered", BLUE),
        "Factor + filter (OOS)": ("oos_factor_filtered", PURPLE),
        "SSD + filter (OOS)": ("oos_ssd_filtered", ORANGE),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    oos_start = pd.Timestamp("2021-01-01")

    is_end_vals = {}
    for label, (fname, color) in is_cells.items():
        path = RESULTS / f"{fname}_monthly.parquet"
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        df = pd.read_parquet(path)
        cum = (1 + df["monthly_return"]).cumprod()
        is_end_vals[fname.replace("_filtered", "")] = float(cum.iloc[-1])
        lw = 2.5
        ax.plot(df.index, cum, label=label, color=color, linewidth=lw)

    for label, (fname, color) in oos_cells.items():
        path = RESULTS / f"{fname}_monthly.parquet"
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        df = pd.read_parquet(path)
        cum = (1 + df["monthly_return"]).cumprod()
        base_key = fname.replace("oos_", "").replace("_filtered", "")
        if base_key in is_end_vals:
            cum_stitched = cum * is_end_vals[base_key]
            ax.plot(df.index, cum_stitched, color=color, linewidth=2.5, linestyle="--")
        else:
            ax.plot(df.index, cum, label=label, color=color, linewidth=2.0, linestyle="--")

    ax.axvspan(oos_start, pd.Timestamp("2025-12-31"), alpha=0.07, color=RED, zorder=0)
    ax.axvline(oos_start, color=RED, linewidth=1.2, linestyle=":", alpha=0.6)
    ax.annotate("IN-SAMPLE\n2003-2020", xy=(pd.Timestamp("2011-06-01"), 0.96),
                fontsize=11, color=NAVY, ha="center", fontstyle="italic")
    ax.annotate("OOS\n2021-2025", xy=(pd.Timestamp("2023-06-01"), 0.96),
                fontsize=11, color=RED, ha="center", fontstyle="italic")

    ax.set_ylabel("Growth of $1", fontsize=13, color=INK)
    ax.set_title("Cumulative Returns — Filtered Strategies Only (IS + OOS)",
                 fontsize=16, color=NAVY, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0.9)

    fig.tight_layout()
    out = ASSETS / "cumulative_pnl_filtered.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── 5. P&L breakdown — filtered strategies only (IS + OOS) ──

def build_pnl_breakdown_filtered():
    cells_is = {
        "SSD+filt IS": "ssd_filtered",
        "PC+filt IS": "pc_filtered",
        "Factor+filt IS": "factor_filtered",
    }
    cells_oos = {
        "PC+filt OOS": "oos_pc_filtered",
        "Factor+filt OOS": "oos_factor_filtered",
        "SSD+filt OOS": "oos_ssd_filtered",
    }
    all_cells = {**cells_is, **cells_oos}

    rows = []
    strategies = []
    for label, fname in all_cells.items():
        path = RESULTS / f"{fname}_trades.parquet"
        if not path.exists():
            print(f"  SKIP {fname} trades (not found)")
            continue
        df = pd.read_parquet(path)
        total = len(df)
        if total == 0:
            continue
        strategies.append(label)
        for reason in ["reversion", "force_close", "stop"]:
            sub = df[df["exit_reason"] == reason]
            if len(sub) == 0:
                continue
            rows.append({
                "strategy": label,
                "exit_reason": reason,
                "count": len(sub),
                "pct_of_trades": len(sub) / total * 100,
                "mean_return_bps": sub["round_trip_return"].mean() * 10000,
                "hit_rate": (sub["round_trip_return"] > 0).mean() * 100,
            })

    if not rows:
        print("  No filtered trade data found, skipping pnl_breakdown_filtered")
        return

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(strategies))
    bar_w = 0.35

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("pct_of_trades", "% of trades", "Trade Mix"),
        ("mean_return_bps", "Mean return (bps)", "Return per Trade"),
        ("hit_rate", "Hit rate (%)", "Win Rate"),
    ]):
        ax = axes[ax_idx]
        for i, reason in enumerate(["reversion", "force_close"]):
            vals = []
            for s in strategies:
                row = summary[(summary["strategy"] == s) & (summary["exit_reason"] == reason)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            color = GREEN if reason == "reversion" else RED
            label = "Reversion" if reason == "reversion" else "Force-close"
            ax.bar(x + i * bar_w, vals, bar_w, label=label, color=color, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY)
        ax.set_xticks(x + bar_w / 2)
        ax.set_xticklabels(strategies, fontsize=8, rotation=20, ha="right")
        if metric == "mean_return_bps":
            ax.axhline(0, color="black", linewidth=0.5)
        if metric == "hit_rate":
            ax.axhline(50, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    is_count = len([s for s in strategies if "IS" in s])
    oos_count = len([s for s in strategies if "OOS" in s])
    if is_count > 0 and oos_count > 0:
        divider_x = is_count - 0.5
        for ax in axes:
            ax.axvline(divider_x, color=GRAY, linewidth=1.5, linestyle="--", alpha=0.6)

    fig.suptitle("P&L Breakdown — Filtered Strategies (IS vs OOS)",
                 fontsize=16, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    out = ASSETS / "pnl_breakdown_filtered.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    print("\nFiltered strategy summary:")
    for s in strategies:
        rev = summary[(summary["strategy"] == s) & (summary["exit_reason"] == "reversion")]
        fc = summary[(summary["strategy"] == s) & (summary["exit_reason"] == "force_close")]
        rev_bps = rev["mean_return_bps"].values[0] if len(rev) > 0 else 0
        fc_bps = fc["mean_return_bps"].values[0] if len(fc) > 0 else 0
        rev_pct = rev["pct_of_trades"].values[0] if len(rev) > 0 else 0
        print(f"  {s}: reversion {rev_pct:.1f}% @ {rev_bps:+.0f}bps, FC @ {fc_bps:+.0f}bps")


# ── 7. Carry-over charts (standalone) ──

def build_cumulative_pnl_carryover():
    is_cells = {
        "SSD + filter (IS)": ("carry_ssd_filtered", ORANGE),
        "PC + filter (IS)": ("carry_pc_filtered", BLUE),
        "Factor + filter (IS)": ("carry_factor_filtered", PURPLE),
    }
    oos_cells = {
        "PC + filter (OOS)": ("carry_oos_pc_filtered", BLUE),
        "Factor + filter (OOS)": ("carry_oos_factor_filtered", PURPLE),
        "SSD + filter (OOS)": ("carry_oos_ssd_filtered", ORANGE),
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    oos_start = pd.Timestamp("2021-01-01")

    is_end_vals = {}
    for label, (fname, color) in is_cells.items():
        path = RESULTS / f"{fname}_monthly.parquet"
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        df = pd.read_parquet(path)
        cum = (1 + df["monthly_return"]).cumprod()
        is_end_vals[fname.replace("carry_", "").replace("_filtered", "")] = float(cum.iloc[-1])
        ax.plot(df.index, cum, label=label, color=color, linewidth=2.5)

    for label, (fname, color) in oos_cells.items():
        path = RESULTS / f"{fname}_monthly.parquet"
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        df = pd.read_parquet(path)
        cum = (1 + df["monthly_return"]).cumprod()
        base_key = fname.replace("carry_oos_", "").replace("_filtered", "")
        if base_key in is_end_vals:
            cum_stitched = cum * is_end_vals[base_key]
            ax.plot(df.index, cum_stitched, color=color, linewidth=2.5, linestyle="--")
        else:
            ax.plot(df.index, cum, label=label, color=color, linewidth=2.0, linestyle="--")

    ax.axvspan(oos_start, pd.Timestamp("2025-12-31"), alpha=0.07, color=RED, zorder=0)
    ax.axvline(oos_start, color=RED, linewidth=1.2, linestyle=":", alpha=0.6)
    ax.annotate("IN-SAMPLE\n2003-2020", xy=(pd.Timestamp("2011-06-01"), 0.96),
                fontsize=11, color=NAVY, ha="center", fontstyle="italic")
    ax.annotate("OOS\n2021-2025", xy=(pd.Timestamp("2023-06-01"), 0.96),
                fontsize=11, color=RED, ha="center", fontstyle="italic")

    ax.set_ylabel("Growth of $1", fontsize=13, color=INK)
    ax.set_title("Cumulative Returns — Carry-Over Filtered Strategies (IS + OOS)",
                 fontsize=16, color=NAVY, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0.9)

    fig.tight_layout()
    out = ASSETS / "cumulative_pnl_carryover.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def build_pnl_breakdown_carryover():
    cells_is = {
        "SSD+filt IS": "carry_ssd_filtered",
        "PC+filt IS": "carry_pc_filtered",
        "Factor+filt IS": "carry_factor_filtered",
    }
    cells_oos = {
        "PC+filt OOS": "carry_oos_pc_filtered",
        "Factor+filt OOS": "carry_oos_factor_filtered",
        "SSD+filt OOS": "carry_oos_ssd_filtered",
    }
    all_cells = {**cells_is, **cells_oos}

    rows = []
    strategies = []
    for label, fname in all_cells.items():
        path = RESULTS / f"{fname}_trades.parquet"
        if not path.exists():
            print(f"  SKIP {fname} trades (not found)")
            continue
        df = pd.read_parquet(path)
        total = len(df)
        if total == 0:
            continue
        strategies.append(label)
        for reason in ["reversion", "force_close", "stop"]:
            sub = df[df["exit_reason"] == reason]
            if len(sub) == 0:
                continue
            rows.append({
                "strategy": label,
                "exit_reason": reason,
                "count": len(sub),
                "pct_of_trades": len(sub) / total * 100,
                "mean_return_bps": sub["round_trip_return"].mean() * 10000,
                "hit_rate": (sub["round_trip_return"] > 0).mean() * 100,
            })

    if not rows:
        print("  No carry-over trade data found, skipping")
        return

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(strategies))
    bar_w = 0.35

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("pct_of_trades", "% of trades", "Trade Mix"),
        ("mean_return_bps", "Mean return (bps)", "Return per Trade"),
        ("hit_rate", "Hit rate (%)", "Win Rate"),
    ]):
        ax = axes[ax_idx]
        for i, reason in enumerate(["reversion", "force_close"]):
            vals = []
            for s in strategies:
                row = summary[(summary["strategy"] == s) & (summary["exit_reason"] == reason)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            color = GREEN if reason == "reversion" else RED
            label = "Reversion" if reason == "reversion" else "Force-close"
            ax.bar(x + i * bar_w, vals, bar_w, label=label, color=color, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY)
        ax.set_xticks(x + bar_w / 2)
        ax.set_xticklabels(strategies, fontsize=8, rotation=20, ha="right")
        if metric == "mean_return_bps":
            ax.axhline(0, color="black", linewidth=0.5)
        if metric == "hit_rate":
            ax.axhline(50, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    is_count = len([s for s in strategies if "IS" in s])
    if is_count > 0 and is_count < len(strategies):
        for ax in axes:
            ax.axvline(is_count - 0.5, color=GRAY, linewidth=1.5, linestyle="--", alpha=0.6)

    fig.suptitle("P&L Breakdown — Carry-Over Filtered Strategies (IS vs OOS)",
                 fontsize=16, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    out = ASSETS / "pnl_breakdown_carryover.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def build_yearly_returns_carryover():
    is_cells = {
        "SSD+filt IS": "carry_ssd_filtered",
        "PC+filt IS": "carry_pc_filtered",
        "Factor+filt IS": "carry_factor_filtered",
    }
    oos_cells = {
        "SSD+filt OOS": "carry_oos_ssd_filtered",
        "PC+filt OOS": "carry_oos_pc_filtered",
        "Factor+filt OOS": "carry_oos_factor_filtered",
    }

    yearly = {}
    for label, fname in {**is_cells, **oos_cells}.items():
        path = RESULTS / f"{fname}_monthly.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["year"] = df.index.year
        annual = df.groupby("year")["monthly_return"].apply(
            lambda x: float((1 + x).prod() - 1)
        ) * 100
        yearly[label] = annual

    if not yearly:
        print("  No carry-over data found, skipping yearly_returns_carryover")
        return

    all_years = sorted(set().union(*[s.index for s in yearly.values()]))
    strategies = list(yearly.keys())

    data = np.full((len(strategies), len(all_years)), np.nan)
    for i, strat in enumerate(strategies):
        for j, year in enumerate(all_years):
            if year in yearly[strat].index:
                data[i, j] = yearly[strat][year]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    vmax = np.nanmax(np.abs(data))
    vmax = min(vmax, 15)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels(all_years, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=11)

    for i in range(len(strategies)):
        for j in range(len(all_years)):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.6 else INK
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if abs(val) > 5 else "normal")

    if 2021 in all_years:
        oos_start_idx = all_years.index(2021)
        ax.axvline(oos_start_idx - 0.5, color=RED, linewidth=2.5, linestyle="-")
        ax.text(oos_start_idx - 0.7, -0.8, "OOS -->", fontsize=10, color=RED,
                fontweight="bold", ha="left")
        ax.text(oos_start_idx - 1.3, -0.8, "<-- IS", fontsize=10, color=NAVY,
                fontweight="bold", ha="right")

    is_count = len([s for s in strategies if "IS" in s])
    if is_count > 0 and is_count < len(strategies):
        ax.axhline(is_count - 0.5, color=GRAY, linewidth=1.5, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Annual Return (%)", fontsize=11)

    ax.set_title("Year-by-Year Returns (%) — Carry-Over Filtered Strategies",
                 fontsize=16, color=NAVY, fontweight="bold", pad=15)

    fig.tight_layout()
    out = ASSETS / "yearly_returns_carryover.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── 8. Pair overlap Venn-style bar chart ──

def build_pair_overlap():
    fnames = {"SSD+filt": "ssd_filtered", "PC+filt": "pc_filtered", "Factor+filt": "factor_filtered"}
    pairs_by = {}
    for label, fname in fnames.items():
        df = pd.read_parquet(RESULTS / f"{fname}_trades.parquet")
        df["pair"] = df.apply(lambda r: tuple(sorted([r["permno_a"], r["permno_b"]])), axis=1)
        pairs_by[label] = set(df["pair"].unique())

    ssd, pc, fac = pairs_by["SSD+filt"], pairs_by["PC+filt"], pairs_by["Factor+filt"]
    categories = [
        ("SSD only", len(ssd - pc - fac)),
        ("PC only", len(pc - ssd - fac)),
        ("Factor only", len(fac - ssd - pc)),
        ("SSD & PC", len((ssd & pc) - fac)),
        ("SSD & Factor", len((ssd & fac) - pc)),
        ("PC & Factor", len((pc & fac) - ssd)),
        ("All three", len(ssd & pc & fac)),
    ]
    labels_v = [c[0] for c in categories]
    vals = [c[1] for c in categories]
    colors_v = [ORANGE, BLUE, PURPLE, "#e8a040", "#b87030", "#6080b0", GREEN]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    bars = ax.barh(range(len(labels_v)), vals, color=colors_v, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_yticks(range(len(labels_v)))
    ax.set_yticklabels(labels_v, fontsize=12)
    ax.set_xlabel("Number of unique pairs", fontsize=13)
    ax.set_title("Pair Selection Overlap Across Metrics",
                 fontsize=16, color=NAVY, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    total = len(ssd | pc | fac)
    ax.text(0.98, 0.02, f"Total unique pairs: {total}", transform=ax.transAxes,
            fontsize=11, ha="right", color=GRAY)

    fig.tight_layout()
    out = ASSETS / "pair_overlap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── 9. Biggest winners & losers ──

def build_top_trades():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.panel import load_crsp_daily
    crsp = load_crsp_daily()
    ticker_map = dict(zip(crsp.groupby("permno").last().reset_index()["permno"],
                          crsp.groupby("permno").last().reset_index()["ticker"]))

    fnames = {
        "SSD+filt": "ssd_filtered",
        "PC+filt": "pc_filtered",
        "Factor+filt": "factor_filtered",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    for ax, (label, fname) in zip(axes, fnames.items()):
        df = pd.read_parquet(RESULTS / f"{fname}_trades.parquet")
        top3 = df.nlargest(3, "round_trip_return")
        bot3 = df.nsmallest(3, "round_trip_return")
        combined = pd.concat([top3, bot3])

        names = []
        rets = []
        colors_bar = []
        for _, r in combined.iterrows():
            ta = ticker_map.get(r["permno_a"], str(r["permno_a"]))
            tb = ticker_map.get(r["permno_b"], str(r["permno_b"]))
            yr = r["entry_date"].strftime("%y")
            names.append(f"{ta}/{tb} '{yr}")
            rets.append(r["round_trip_return"] * 100)
            colors_bar.append(GREEN if r["round_trip_return"] > 0 else RED)

        y = np.arange(len(names))
        bars = ax.barh(y, rets, color=colors_bar, alpha=0.8, edgecolor="white", height=0.6)
        for bar, ret in zip(bars, rets):
            # Place labels inside the bars to avoid overlap with y-axis labels
            if ret >= 0:
                ax.text(ret - 1, bar.get_y() + bar.get_height() / 2,
                        f"{ret:+.1f}%", va="center", ha="right",
                        fontsize=9, fontweight="bold", color="white")
            else:
                ax.text(ret + 1, bar.get_y() + bar.get_height() / 2,
                        f"{ret:+.1f}%", va="center", ha="left",
                        fontsize=9, fontweight="bold", color="white")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(label, fontsize=15, fontweight="bold", color=NAVY)
        ax.set_xlabel("Return (%)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()
        xmax = max(abs(r) for r in rets) * 1.25
        ax.set_xlim(-xmax, xmax)

    fig.suptitle("Top 3 Winners & Losers by Metric (IS, Filtered)",
                 fontsize=16, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout(w_pad=3)
    out = ASSETS / "top_trades.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


# ── 10. Sector composition by metric ──

def build_sector_composition():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.panel import load_crsp_daily
    crsp = load_crsp_daily()
    sic_map = dict(zip(crsp.groupby("permno").last().reset_index()["permno"],
                       crsp.groupby("permno").last().reset_index()["siccd"]))

    def sic_sector(sic):
        if pd.isna(sic): return "Other"
        sic = int(sic)
        if 6000 <= sic < 6800: return "Finance"
        if 2000 <= sic < 4000: return "Mfg"
        if 4800 <= sic < 5000: return "Telecom"
        if 5200 <= sic < 6000: return "Retail"
        if 1000 <= sic < 1500: return "Mining/Oil"
        if 7000 <= sic < 9000: return "Services"
        if 4000 <= sic < 4800: return "Transport"
        return "Other"

    fnames = {"SSD+filt": "ssd_filtered", "PC+filt": "pc_filtered", "Factor+filt": "factor_filtered"}
    metrics_data = {}

    for label, fname in fnames.items():
        df = pd.read_parquet(RESULTS / f"{fname}_trades.parquet")
        same, cross = 0, 0
        sector_counts = {}
        for _, r in df.iterrows():
            sa = sic_sector(sic_map.get(r["permno_a"]))
            sb = sic_sector(sic_map.get(r["permno_b"]))
            if sa == sb:
                same += 1
                sector_counts[sa] = sector_counts.get(sa, 0) + 1
            else:
                cross += 1
        metrics_data[label] = {
            "same_pct": same / len(df) * 100,
            "cross_pct": cross / len(df) * 100,
            "sectors": sector_counts,
            "total": len(df),
        }

    # Stacked bar: same-sector breakdown by sector + cross-sector
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("white")

    # Left: sector breakdown
    ax = axes[0]
    all_sectors = ["Mfg", "Finance", "Telecom", "Retail", "Mining/Oil", "Services", "Other"]
    sector_colors = [NAVY, ORANGE, BLUE, GREEN, PURPLE, "#d4812a", GRAY]
    x = np.arange(len(fnames))
    bottom = np.zeros(len(fnames))

    for sec, col in zip(all_sectors, sector_colors):
        vals = []
        for label in fnames:
            total = metrics_data[label]["total"]
            count = metrics_data[label]["sectors"].get(sec, 0)
            vals.append(count / total * 100)
        ax.bar(x, vals, bottom=bottom, label=sec, color=col, alpha=0.85, width=0.5)
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(x[i], bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(list(fnames.keys()), fontsize=12)
    ax.set_ylabel("% of trades (same-sector)", fontsize=12)
    ax.set_title("Same-Sector Trade Composition", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: same vs cross pie-style bars
    ax2 = axes[1]
    labels_m = list(fnames.keys())
    same_pcts = [metrics_data[l]["same_pct"] for l in fnames]
    cross_pcts = [metrics_data[l]["cross_pct"] for l in fnames]
    y = np.arange(len(labels_m))
    ax2.barh(y, same_pcts, color=NAVY, alpha=0.8, label="Same-sector", height=0.4)
    ax2.barh(y + 0.4, cross_pcts, color=RED, alpha=0.7, label="Cross-sector", height=0.4)
    for i in range(len(labels_m)):
        ax2.text(same_pcts[i] + 1, y[i], f"{same_pcts[i]:.0f}%", va="center", fontsize=10, fontweight="bold")
        ax2.text(cross_pcts[i] + 1, y[i] + 0.4, f"{cross_pcts[i]:.0f}%", va="center", fontsize=10, color=RED)
    ax2.set_yticks(y + 0.2)
    ax2.set_yticklabels(labels_m, fontsize=12)
    ax2.set_xlabel("% of trades", fontsize=12)
    ax2.set_title("Same vs Cross-Sector", fontsize=14, fontweight="bold", color=NAVY)
    ax2.legend(fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    fig.suptitle("How Each Metric Clusters — Sector Composition of Traded Pairs",
                 fontsize=16, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    out = ASSETS / "sector_composition.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    build_cumulative_pnl()
    build_pnl_breakdown()
    build_yearly_returns()
    build_yearly_returns_filtered()
    build_cumulative_pnl_filtered()
    build_pnl_breakdown_filtered()
    build_cumulative_pnl_carryover()
    build_pnl_breakdown_carryover()
    build_yearly_returns_carryover()
    build_pair_overlap()
    build_top_trades()
    build_sector_composition()
    print("\nDone. Charts saved to submission/report/assets/")
