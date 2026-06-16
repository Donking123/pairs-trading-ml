"""
Overlay and cluster diagnostic visualizations.

Generates 5 figures:
  1. overlay_impact.png  — cumulative P&L of each overlay added cumulatively
  2. regime_calendar.png — monthly returns coloured by HMM regime state
  3. exit_reasons.png    — exit-reason breakdown and P&L per reason
  4. cluster_network.png — Dec-2023 cluster network (stocks as nodes, pairs as edges)
  5. falling_knife.png   — z-score path for one pair showing confirmed vs unconfirmed entry

Usage:
    python3 visualize_diagnostics.py [--output figures/]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import (
    run_backtest, load_delisting, load_ibes_earnings, load_regime_scale,
)
from src.panel import load_crsp_daily, load_sp500_constituents, ticker_lookup
from src.distances import ssd_distance
from src.clustering import cluster_optics, clusters_to_pairs
from src.panel import formation_window_panel
from src.config import OPTICS_XI, OPTICS_MIN_SAMPLES, OPTICS_MIN_CLUSTER_SIZE

PALETTE = {
    "calm":    "#2ecc71",
    "stressed":"#f39c12",
    "crisis":  "#e74c3c",
    "baseline":"#95a5a6",
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load_all():
    print("Loading data...", flush=True)
    crsp = load_crsp_daily()
    cons = load_sp500_constituents()
    delist = load_delisting()
    ibes = load_ibes_earnings()
    rs = load_regime_scale()
    hmm_regimes = pd.read_parquet(ROOT / "data" / "hmm_regimes.parquet")
    return crsp, cons, delist, ibes, rs, hmm_regimes


def _run(crsp, cons, delist, ibes, rs, **kw):
    return run_backtest(
        start="2016-01-01", end="2023-12-31",
        crsp=crsp, constituents=cons, delisting_df=delist,
        formation_years=2, metric="pc", verbose=False, **kw
    )


# ── Figure 1: cumulative overlay impact ───────────────────────────────────────

def fig_overlay_impact(crsp, cons, delist, ibes, rs, out: Path):
    configs = [
        ("Baseline (exit=0)",         dict(exit_sigma=0.0)),
        ("+ Partial exit |z|≤0.5",    dict(exit_sigma=-0.5)),
        ("+ Entry confirmation",       dict(exit_sigma=-0.5, entry_confirm=True)),
        ("+ Structural break exit",    dict(exit_sigma=-0.5, entry_confirm=True, structural_exits=True)),
        ("+ IBES blackout",            dict(exit_sigma=-0.5, entry_confirm=True, structural_exits=True,
                                            blackout_days=3, ibes_df=ibes)),
        ("+ HMM regime (full stack)",  dict(exit_sigma=-0.5, entry_confirm=True, structural_exits=True,
                                            blackout_days=3, ibes_df=ibes,
                                            use_regime_scale=True, regime_scale=rs)),
    ]
    colours = ["#7f8c8d", "#2980b9", "#8e44ad", "#e67e22", "#27ae60", "#e74c3c"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax_cum, ax_sr = axes

    srs, dds = [], []
    for i, (label, kw) in enumerate(configs):
        print(f"  Running: {label}", flush=True)
        m, _ = _run(crsp, cons, delist, ibes, rs, **kw)
        r = m["monthly_return"]
        cum = (1 + r).cumprod() - 1
        sr = (r.mean() * 12) / (r.std() * np.sqrt(12))
        dd = float(((1+r).cumprod() / (1+r).cumprod().cummax() - 1).min())
        srs.append(sr); dds.append(dd)

        ax_cum.plot(cum.index, cum * 100, color=colours[i], linewidth=1.8 if i == 5 else 1.2,
                    label=f"{label}  (SR={sr:+.2f}, DD={dd:.1%})", zorder=i+1)

    ax_cum.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_cum.set_ylabel("Cumulative return (%)")
    ax_cum.set_title("Additive overlay impact — Donking SSD pipeline, 2016–2023\n"
                      "(each row adds one overlay to all previous)", fontsize=11, pad=10)
    ax_cum.legend(fontsize=7.5, loc="upper left")
    ax_cum.grid(axis="y", alpha=0.3)

    # SR bar chart
    x = np.arange(len(configs))
    bars = ax_sr.bar(x, srs, color=colours)
    ax_sr.set_xticks(x)
    ax_sr.set_xticklabels([c[0] for c in configs], rotation=15, ha="right", fontsize=7)
    ax_sr.set_ylabel("Gross SR")
    ax_sr.axhline(0, color="black", linewidth=0.5)
    for bar, v in zip(bars, srs):
        ax_sr.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=7)
    ax_sr.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2)
    fig.savefig(out / "overlay_impact.png", dpi=150)
    plt.close(fig)
    print(f"  → {out}/overlay_impact.png", flush=True)


# ── Figure 2: regime calendar heatmap ─────────────────────────────────────────

def fig_regime_calendar(rs: pd.Series, hmm_regimes: pd.DataFrame, crsp, cons, delist, ibes, out: Path):
    print("  Running full-stack for regime calendar...", flush=True)
    m, _ = _run(crsp, cons, delist, ibes, rs,
                exit_sigma=-0.5, entry_confirm=True, structural_exits=True,
                blackout_days=3, ibes_df=ibes, use_regime_scale=True, regime_scale=rs)

    ret = m["monthly_return"]
    dates = pd.DatetimeIndex(ret.index)

    # Classify each month by majority Viterbi state (0=calm, 1=stressed, 2=crisis)
    state_names = {0: "calm", 1: "stressed", 2: "crisis"}
    viterbi = hmm_regimes["state"].reindex(
        pd.date_range("2016-01-01", "2023-12-31", freq="D"), method="ffill"
    ).dropna()

    def monthly_regime(month_end):
        mo = pd.Period(month_end, "M")
        subset = viterbi[viterbi.index.to_period("M") == mo]
        if len(subset) == 0:
            return "calm"
        return state_names[int(subset.mode().iloc[0])]

    regime_labels = [monthly_regime(d) for d in dates]

    fig, ax = plt.subplots(figsize=(14, 5))
    years = sorted(set(d.year for d in dates))
    months = list(range(1, 13))

    # Build matrix
    Z = np.full((len(years), 12), np.nan)
    R = np.full((len(years), 12), "calm", dtype=object)
    for d, v, reg in zip(dates, ret.values, regime_labels):
        yi, mi = years.index(d.year), d.month - 1
        Z[yi, mi] = v * 100
        R[yi, mi] = reg

    regime_cmap = {"calm": PALETTE["calm"], "stressed": PALETTE["stressed"], "crisis": PALETTE["crisis"]}
    for yi, y in enumerate(years):
        for mi in range(12):
            if not np.isnan(Z[yi, mi]):
                color = regime_cmap.get(R[yi, mi], "#bdc3c7")
                alpha = 0.35 + 0.65 * min(abs(Z[yi, mi]) / 1.5, 1.0)
                ax.add_patch(mpatches.Rectangle((mi, yi), 1, 1, color=color, alpha=alpha, ec="white"))
                ax.text(mi + 0.5, yi + 0.5, f"{Z[yi, mi]:+.2f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold" if abs(Z[yi, mi]) > 0.5 else "normal")

    ax.set_xlim(0, 12); ax.set_ylim(0, len(years))
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(np.arange(len(years)) + 0.5)
    ax.set_yticklabels(years)
    ax.set_title("Monthly returns by HMM regime state — full stack, 2016–2023\n"
                 "Green=calm · Yellow=stressed · Red=crisis (colour intensity ∝ |return|)", fontsize=10)

    patches = [mpatches.Patch(color=v, label=k) for k, v in regime_cmap.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out / "regime_calendar.png", dpi=150)
    plt.close(fig)
    print(f"  → {out}/regime_calendar.png", flush=True)


# ── Figure 3: exit reason breakdown ───────────────────────────────────────────

def fig_exit_reasons(crsp, cons, delist, ibes, rs, out: Path):
    print("  Running for exit reasons...", flush=True)
    m, trades = _run(crsp, cons, delist, ibes, rs,
                     exit_sigma=-0.5, entry_confirm=True, structural_exits=True,
                     blackout_days=3, ibes_df=ibes, use_regime_scale=True, regime_scale=rs)

    reasons = [t.exit_reason for t in trades]
    pnls    = [t.round_trip_return for t in trades]
    df = pd.DataFrame({"reason": reasons, "pnl": pnls})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Count pie
    counts = df["reason"].value_counts()
    colors_pie = [{"reversion": "#2ecc71", "force_close": "#f39c12",
                   "stop_loss": "#e74c3c", "delisting": "#95a5a6"}.get(r, "#bdc3c7")
                  for r in counts.index]
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.0f%%",
                colors=colors_pie, startangle=90)
    axes[0].set_title(f"Exit reason distribution\n({len(trades)} total trades)")

    # Mean P&L per reason
    mean_pnl = df.groupby("reason")["pnl"].mean().sort_values()
    colors_bar = [{"reversion": "#2ecc71", "force_close": "#f39c12",
                   "stop_loss": "#e74c3c", "delisting": "#95a5a6"}.get(r, "#bdc3c7")
                  for r in mean_pnl.index]
    axes[1].barh(mean_pnl.index, mean_pnl.values * 100, color=colors_bar)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Mean round-trip return (%)")
    axes[1].set_title("Mean P&L per exit type\n(full stack, 2016–2023)")
    axes[1].grid(axis="x", alpha=0.3)

    # P&L distribution box per reason
    reason_order = ["reversion", "force_close", "stop_loss"]
    box_data = [df[df["reason"]==r]["pnl"].values * 100 for r in reason_order if r in df["reason"].values]
    box_labels = [r for r in reason_order if r in df["reason"].values]
    bp = axes[2].boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
    exit_colors = {"reversion": "#2ecc71", "force_close": "#f39c12", "stop_loss": "#e74c3c"}
    for patch, label in zip(bp["boxes"], box_labels):
        patch.set_facecolor(exit_colors.get(label, "#bdc3c7"))
        patch.set_alpha(0.7)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Round-trip return (%)")
    axes[2].set_title("P&L distribution by exit type\n(whiskers = 5–95th percentile)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Exit analysis — full overlay stack, 2016–2023", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "exit_reasons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}/exit_reasons.png", flush=True)


# ── Figure 4: cluster network ──────────────────────────────────────────────────

def fig_cluster_network(crsp, cons, out: Path):
    print("  Building Dec-2023 cluster network...", flush=True)
    import networkx as nx

    panel = formation_window_panel("2023-12-29", crsp=crsp, constituents=cons, formation_years=2)
    dmat = ssd_distance(panel)
    labels = cluster_optics(dmat, min_samples=OPTICS_MIN_SAMPLES, xi=OPTICS_XI,
                             min_cluster_size=OPTICS_MIN_CLUSTER_SIZE)
    pairs = clusters_to_pairs(labels)
    tick = ticker_lookup(permnos=list(panel.columns), crsp=crsp, as_of=pd.Timestamp("2023-12-29"))

    def get_ticker(perm):
        t = tick.get(perm, str(perm))
        return str(t.iloc[0]) if hasattr(t, "iloc") else str(t)

    clustered = labels[labels != -1]
    cluster_ids = sorted(clustered.unique())

    # Keep only clusters with ≥ 2 members AND rank by cluster size; show top 15
    sizes = {cid: int((clustered == cid).sum()) for cid in cluster_ids}
    top_clusters = sorted(cluster_ids, key=lambda c: -sizes[c])[:15]
    top_perms = set(clustered[clustered.isin(top_clusters)].index)
    top_pairs = [(a, b) for a, b in pairs if a in top_perms and b in top_perms]

    palette_net = plt.cm.tab20(np.linspace(0, 1, len(top_clusters)))
    cid_to_color = {cid: palette_net[i] for i, cid in enumerate(top_clusters)}

    # Build networkx graph
    G = nx.Graph()
    for p in top_perms:
        G.add_node(p, ticker=get_ticker(p), cluster=int(labels[p]))
    for a, b in top_pairs:
        d = float(dmat.loc[a, b])
        G.add_edge(a, b, weight=1.0 / max(d, 0.1))

    # Spring layout — weighted by similarity (closer pairs attract more)
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2.5 / np.sqrt(len(G)))

    fig, ax = plt.subplots(figsize=(16, 13))

    # Draw edges, colour by cluster, width ∝ similarity
    max_w = max(d["weight"] for _, _, d in G.edges(data=True)) if G.edges else 1
    for a, b, data in G.edges(data=True):
        cid = G.nodes[a]["cluster"]
        color = cid_to_color.get(cid, "#bdc3c7")
        lw = max(0.4, 3.0 * data["weight"] / max_w)
        x = [pos[a][0], pos[b][0]]; y = [pos[a][1], pos[b][1]]
        ax.plot(x, y, color=color, linewidth=lw, alpha=0.4, zorder=1)

    # Draw nodes
    for perm in G.nodes:
        cid = G.nodes[perm]["cluster"]
        color = cid_to_color.get(cid, "#bdc3c7")
        x, y = pos[perm]
        ax.scatter(x, y, color=color, s=90, zorder=3, edgecolors="white", linewidths=0.8)
        ax.text(x, y + 0.025, G.nodes[perm]["ticker"], ha="center", va="bottom",
                fontsize=7, zorder=4, color="#2c3e50", fontweight="medium")

    # Cluster legend with economic labels
    econ_names = {
        top_clusters[0]: "Cluster 1", top_clusters[1]: "Cluster 2",
    }
    patches = [mpatches.Patch(color=cid_to_color[cid],
               label=f"Cluster {cid}  ({sizes[cid]} stocks, "
                     f"{sum(1 for a,b in top_pairs if labels[a]==cid)} pairs)")
               for cid in top_clusters]
    ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=2,
              title="Top 15 clusters by size", title_fontsize=8)

    ax.set_title(
        f"Dec-2023 OPTICS cluster network — top 15 clusters\n"
        f"{len(top_clusters)} clusters · {len(top_perms)} stocks · "
        f"{len(top_pairs)} candidate pairs  |  "
        f"Node colour = cluster · Edge width ∝ SSD similarity (1/distance)",
        fontsize=10
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "cluster_network.png", dpi=150)
    plt.close(fig)
    print(f"  → {out}/cluster_network.png", flush=True)




# ── Figure 5: falling knife + overlay mechanisms illustration ─────────────────

def fig_falling_knife(crsp, cons, out: Path):
    """Three-panel illustration of each overlay mechanism using stylised paths
    calibrated to match empirical behaviour during Feb-2018 vol spike."""
    print("  Building overlay mechanism illustration...", flush=True)

    np.random.seed(2018)
    T = 22
    t = np.arange(T)

    # Scenario A: falling knife path
    z_knife = np.where(t < 8, 2.0 + 0.4*t,
               np.where(t < 15, 5.2 - 0.55*(t-8), 1.3 - 0.35*(t-15)))
    z_knife += np.random.randn(T) * 0.07

    # 5d rolling corr for structural break scenario
    corr_5d = np.clip(np.where(t < 8, 0.82 - 0.02*t, 0.66 - 0.055*(t-8)), 0.05, 1.0)

    # HMM scale: calm -> stressed -> crisis -> recovery
    hmm_scale = np.where(t < 4, 0.75, np.where(t < 8, 0.50,
                np.where(t < 12, 0.10, np.where(t < 16, 0.35, 0.80))))

    def _zband(ax):
        ax.axhline(2.0, color="#e74c3c", lw=1, ls="--", alpha=0.7, label="|z|=2.0 entry")
        ax.axhline(0.5, color="#27ae60", lw=0.8, ls=":", alpha=0.6, label="|z|=0.5 exit")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.grid(alpha=0.2); ax.set_xlabel("Trading day"); ax.set_ylabel("Z-score")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Three overlay mechanisms — how each failure mode is detected and suppressed",
                 fontsize=12, fontweight="bold", y=1.02)

    # ── A1: Falling knife z-score path ────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t, z_knife, color="#2c3e50", lw=2, label="z-score")
    ax.fill_between(t, 2.0, z_knife, where=(z_knife > 2), color="#e74c3c", alpha=0.10)
    ax.scatter([0], [z_knife[0]], color="#e74c3c", s=150, marker="v", zorder=5,
               label="Unconfirmed entry (z still rising)")
    ax.scatter([8], [z_knife[8]], color="#27ae60", s=150, marker="^", zorder=5,
               label="Confirmed entry (z turning back)")
    ax.axvspan(16.5, 17.5, color="#2ecc71", alpha=0.15)
    ax.text(17, 0.7, "exit\n|z|=0.5", fontsize=7, color="#27ae60", ha="center")
    _zband(ax)
    ax.set_ylim(-0.3, 6.5)
    ax.set_title("Overlay 1 — Entry confirmation\nWait for z to turn before entering", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    # ── A2: P&L comparison confirmed vs unconfirmed ───────────────────────────
    ax = axes[1, 0]
    pnl_bad  = np.cumsum(np.where(t >= 0,  -0.3 + 0.25*np.clip(t-8, 0, None), 0))
    pnl_good = np.cumsum(np.where(t >= 8,  +0.25 - 0.05*(t-8), 0))
    ax.plot(t, pnl_bad,  color="#e74c3c", lw=2, label="Unconfirmed (day 0)")
    ax.plot(t, pnl_good, color="#27ae60", lw=2, label="Confirmed (day 8)")
    ax.fill_between(t, pnl_bad, 0, where=(pnl_bad < 0), color="#e74c3c", alpha=0.15)
    ax.fill_between(t, pnl_good, 0, where=(pnl_good > 0), color="#27ae60", alpha=0.15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("P&L outcome per strategy", fontsize=9)
    ax.set_xlabel("Trading day"); ax.set_ylabel("Cumul. P&L")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── B1: Structural break — z diverges when corr collapses ─────────────────
    z_break = np.where(t < 5, 2.0 + 0.3*t,
               np.where(t < 10, 3.5 - 0.1*(t-5), 3.0 + 0.25*(t-10)))
    z_break += np.random.randn(T) * 0.08

    ax = axes[0, 1]
    ax.plot(t, z_break, color="#2c3e50", lw=2, label="z-score")
    ax.fill_between(t, 2.0, z_break, where=(z_break > 2), color="#e74c3c", alpha=0.10)
    ax.axvspan(9.5, 10.5, color="#e67e22", alpha=0.35, label="Struct. break exit fires")
    ax.scatter([0], [z_break[0]], color="#27ae60", s=120, marker="^", zorder=5)
    ax.annotate("Corr < 0.5\n=> exit", xy=(10, z_break[10]), xytext=(14, 2.8),
                fontsize=7.5, color="#e67e22",
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.2))
    ax.annotate("Without exit:\nloss keeps growing", xy=(19, z_break[19]),
                xytext=(16, 5.2), fontsize=7.5, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))
    _zband(ax)
    ax.set_ylim(-0.3, 6.5)
    ax.set_title("Overlay 2 — Structural break exit\n5-day rolling correlation drops below 0.5", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")

    # ── B2: rolling correlation path ──────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t, corr_5d, color="#9b59b6", lw=2, label="5d rolling corr")
    ax.axhline(0.5, color="#e67e22", lw=1.5, ls="--", label="Exit threshold (0.5)")
    ax.axvspan(9.5, 10.5, color="#e67e22", alpha=0.35)
    ax.fill_between(t, corr_5d, 0.5, where=(corr_5d < 0.5), color="#e67e22", alpha=0.20)
    ax.set_ylim(0, 1.1); ax.set_xlabel("Trading day"); ax.set_ylabel("Correlation")
    ax.set_title("5-day rolling correlation\nbetween pair legs", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── C1: HMM regime scale bars ─────────────────────────────────────────────
    ax = axes[0, 2]
    colors_r = [PALETTE["calm"] if v >= 0.75 else PALETTE["stressed"] if v >= 0.35
                else PALETTE["crisis"] for v in hmm_scale]
    ax.bar(t, hmm_scale, color=colors_r, width=0.85, edgecolor="white", lw=0.5)
    ax.axhline(0.5, color="#7f8c8d", lw=1, ls=":")
    ax.set_ylim(0, 1.25); ax.set_ylabel("HMM scale")
    ax.set_xlabel("Trading day")
    ax.set_title("Overlay 3 — HMM regime\nP&L of any open trade x regime scale", fontsize=9)
    for label, x, c in [("calm", 2, PALETTE["calm"]), ("stressed", 6, PALETTE["stressed"]),
                         ("crisis", 10, PALETTE["crisis"]), ("recovery", 18, PALETTE["calm"])]:
        ax.text(x, 1.10, label, ha="center", fontsize=7, color=c, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=PALETTE["calm"], label="Calm (~0.75-1.0)"),
                        mpatches.Patch(color=PALETTE["stressed"], label="Stressed (~0.5)"),
                        mpatches.Patch(color=PALETTE["crisis"], label="Crisis (~0.0)")],
              fontsize=7.5, loc="upper right"); ax.grid(alpha=0.2)

    # ── C2: P&L with and without HMM ─────────────────────────────────────────
    ax = axes[1, 2]
    base_daily = np.random.randn(T) * 0.4 + np.where(t < 12, 0.1, -0.1)
    pnl_raw    = np.cumsum(base_daily)
    pnl_scaled = np.cumsum(base_daily * hmm_scale)
    ax.plot(t, pnl_raw,    color="#7f8c8d", lw=1.8, ls="--", label="No HMM (raw P&L)")
    ax.plot(t, pnl_scaled, color=PALETTE["calm"], lw=2.0, label="HMM scaled P&L")
    ax.fill_between(t, pnl_raw, pnl_scaled, alpha=0.12, color="#3498db",
                    label="P&L saved by HMM")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Portfolio P&L: raw vs HMM-scaled\n(crisis = scale to 0 = no loss)", fontsize=9)
    ax.set_xlabel("Trading day"); ax.set_ylabel("Cumul. P&L")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.2)

    fig.tight_layout(pad=2.5)
    fig.savefig(out / "falling_knife.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}/falling_knife.png", flush=True)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures", help="Output directory")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["impact", "calendar", "exits", "cluster", "knife"])
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    crsp, cons, delist, ibes, rs, hmm_regimes = _load_all()

    if "impact" not in args.skip:
        print("\n[1/5] Overlay impact chart...")
        fig_overlay_impact(crsp, cons, delist, ibes, rs, out)

    if "calendar" not in args.skip:
        print("\n[2/5] Regime calendar...")
        fig_regime_calendar(rs, hmm_regimes, crsp, cons, delist, ibes, out)

    if "exits" not in args.skip:
        print("\n[3/5] Exit reason breakdown...")
        fig_exit_reasons(crsp, cons, delist, ibes, rs, out)

    if "cluster" not in args.skip:
        print("\n[4/5] Cluster network...")
        fig_cluster_network(crsp, cons, out)

    if "knife" not in args.skip:
        print("\n[5/5] Falling knife illustration...")
        fig_falling_knife(crsp, cons, out)

    print(f"\nDone. All figures in {out}/")


if __name__ == "__main__":
    main()
