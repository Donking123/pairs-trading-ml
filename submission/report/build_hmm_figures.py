"""
Generate HMM-related figures for the deck.

Figures produced:
  assets/is_oos_hmm_equity.png  — Slide 7: equity curves IS+OOS for PC core,
                                   PC+filter, PC+filter+HMM
  assets/oos_bar.png            — Slide 10: OOS Sharpe bar chart across variants

Run from pairs-trading-ml/:
  python submission/report/build_hmm_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent.parent
RESULTS = Path(__file__).resolve().parent.parent / "results"
ASSETS  = Path(__file__).resolve().parent / "assets"

sys.path.insert(0, str(ROOT))
from src.performance import compute_metrics

NAVY   = "#1b4965"
BLUE   = "#5fa8d3"
ORANGE = "#d4812a"
GREEN  = "#2e7d4f"
RED    = "#b3402f"
GRAY   = "#5a6b7b"
INK    = "#1d2733"
PANEL  = "#f2f7fb"

IS_END  = pd.Timestamp("2020-12-31")
OOS_CUT = pd.Timestamp("2021-01-01")


def _sharpe(r: pd.Series) -> float:
    return float((r.mean() * 12) / (r.std() * 12 ** 0.5))


def _cum(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def build_is_oos_hmm_equity() -> None:
    """Equity curves: IS (2003-2020) + OOS (2021-2024) for three variants."""

    # IS series
    is_core    = pd.read_parquet(RESULTS / "pc_core_monthly.parquet")["monthly_return"]
    is_filt    = pd.read_parquet(RESULTS / "pc_filtered_monthly.parquet")["monthly_return"]
    is_hmm     = pd.read_parquet(RESULTS / "pc_filtered_hmm_is_monthly.parquet")["monthly_return"]

    # OOS — clip to same 47-month window as HMM OOS
    oos_hmm_raw = pd.read_parquet(RESULTS / "pc_filtered_hmm_oos_monthly.parquet")["monthly_return"]
    oos_start   = oos_hmm_raw.index[0]
    oos_end     = oos_hmm_raw.index[-1]

    oos_core_raw = pd.read_parquet(RESULTS / "oos_pc_core_monthly.parquet")["monthly_return"]
    oos_filt_raw = pd.read_parquet(RESULTS / "oos_pc_filtered_monthly.parquet")["monthly_return"]
    oos_core = oos_core_raw.loc[oos_start:oos_end]
    oos_filt = oos_filt_raw.loc[oos_start:oos_end]
    oos_hmm  = oos_hmm_raw

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")

    variants = [
        ("PC core",                  is_core, oos_core, GRAY,   1.4, "--"),
        ("PC + stationarity filter", is_filt, oos_filt, BLUE,   1.6, "-"),
        ("PC + filter + HMM",        is_hmm,  oos_hmm,  NAVY,   2.2, "-"),
    ]

    for label, is_r, oos_r, color, lw, ls in variants:
        is_sr  = _sharpe(is_r)
        oos_sr = _sharpe(oos_r)

        # IS equity
        is_cum = _cum(is_r)
        ax.plot(is_cum.index, is_cum, color=color, lw=lw, ls=ls,
                label=f"{label}  (IS SR={is_sr:.2f} · OOS SR={oos_sr:.2f})")

        # OOS equity — stitch from IS end value
        is_end_val = float(is_cum.iloc[-1])
        oos_cum = _cum(oos_r) * is_end_val
        ax.plot(oos_cum.index, oos_cum, color=color, lw=lw, ls="--", alpha=0.85)

    # OOS shading
    ax.axvspan(OOS_CUT, oos_end, alpha=0.06, color=RED, zorder=0)
    ax.axvline(OOS_CUT, color=RED, lw=1.2, ls=":", alpha=0.55)

    # Annotations
    ax.text(pd.Timestamp("2011-06-01"), ax.get_ylim()[0] * 1.02 if ax.get_ylim()[0] > 0 else 0.93,
            "IN-SAMPLE  2003–2020", fontsize=10, color=NAVY, ha="center", style="italic")
    ax.text(pd.Timestamp("2023-01-01"), ax.get_ylim()[0] * 1.02 if ax.get_ylim()[0] > 0 else 0.93,
            "OOS  2021–2024", fontsize=10, color=RED, ha="center", style="italic")

    ax.set_ylabel("Growth of $1", fontsize=12, color=INK)
    ax.set_title("Equity curves — IS & OOS, three variants",
                 fontsize=14, color=NAVY, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0.88)

    fig.tight_layout()
    out = ASSETS / "is_oos_hmm_equity.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def build_oos_bar() -> None:
    """OOS Sharpe bar chart: three variants on the same 47-month window."""

    oos_hmm_raw = pd.read_parquet(RESULTS / "pc_filtered_hmm_oos_monthly.parquet")["monthly_return"]
    oos_start, oos_end = oos_hmm_raw.index[0], oos_hmm_raw.index[-1]

    oos_core_raw = pd.read_parquet(RESULTS / "oos_pc_core_monthly.parquet")["monthly_return"]
    oos_filt_raw = pd.read_parquet(RESULTS / "oos_pc_filtered_monthly.parquet")["monthly_return"]

    series = {
        "PC core\n(no filter)":          oos_core_raw.loc[oos_start:oos_end],
        "PC + stationarity\nfilter":      oos_filt_raw.loc[oos_start:oos_end],
        "PC + filter\n+ HMM":            oos_hmm_raw,
    }

    labels  = list(series.keys())
    sharpes = [_sharpe(r) for r in series.values()]
    maxdds  = [float((_cum(r) / _cum(r).cummax() - 1).min()) for r in series.values()]
    colors  = [GRAY, BLUE, NAVY]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("white")

    # Sharpe bars
    ax = axes[0]
    bars = ax.bar(range(len(labels)), sharpes, color=colors, alpha=0.85,
                  width=0.5, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("OOS Sharpe (47 months)", fontsize=11)
    ax.set_title("OOS Sharpe — 2021–2024", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_ylim(0, max(sharpes) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # MaxDD bars
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(labels)), [abs(d) for d in maxdds],
                    color=colors, alpha=0.85, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars2, maxdds):
        ax2.text(bar.get_x() + bar.get_width() / 2, abs(v) + 0.001,
                 f"{v:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("OOS Max Drawdown (abs)", fontsize=11)
    ax2.set_title("OOS Max Drawdown — 2021–2024", fontsize=13, fontweight="bold", color=NAVY)
    ax2.set_ylim(0, max(abs(d) for d in maxdds) * 1.3)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Out-of-sample validation (47 months) — same window across all variants",
                 fontsize=12, color=GRAY, y=1.01)
    fig.tight_layout()
    out = ASSETS / "oos_bar.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    build_is_oos_hmm_equity()
    build_oos_bar()
    print("Done.")
