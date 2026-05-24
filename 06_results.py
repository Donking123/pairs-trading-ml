"""
06_results.py
─────────────
Generates all plots and summary tables for the project report.

Outputs in data/results/:
  cumulative_returns.png    — equity curve vs S&P 500
  monthly_heatmap.png       — monthly P&L heatmap
  cluster_size_dist.png     — cluster size distribution over time
  factor_beta_heatmap.png   — mean factor betas per cluster (sample window)
  pairs_summary_chart.png   — pairs per window over time
  performance_table.csv     — final performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from config import DATA_RAW, DATA_PROC, DATA_RES

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "figure.facecolor":"white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "font.size":       11,
})
BLUE   = "#185FA5"
GRAY   = "#888780"
CORAL  = "#993C1D"


# ── 1. Equity curve ────────────────────────────────────────────────────────────
def plot_equity_curve():
    pnl = pd.read_parquet(DATA_RES / "daily_pnl.parquet")["pnl"]
    cum = (1 + pnl).cumprod()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum.index, cum.values, color=BLUE, linewidth=1.5, label="Strategy")
    ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--")

    # Try to overlay S&P 500 if we have SPY returns
    spy_path = DATA_RAW / "factor_returns.parquet"
    if spy_path.exists():
        factors = pd.read_parquet(spy_path)
        if "Market" in factors.columns:
            spy = factors["Market"].reindex(pnl.index).fillna(0)
            spy_cum = (1 + spy).cumprod()
            ax.plot(spy_cum.index, spy_cum.values,
                    color=GRAY, linewidth=1.0, linestyle="--",
                    label="S&P 500 (SPY)", alpha=0.7)

    ax.set_title("Strategy cumulative return (net of costs)")
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    out = DATA_RES / "cumulative_returns.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 2. Monthly P&L heatmap ─────────────────────────────────────────────────────
def plot_monthly_heatmap():
    pnl = pd.read_parquet(DATA_RES / "daily_pnl.parquet")["pnl"]

    monthly = (
        pnl
        .resample("ME")
        .apply(lambda r: (1 + r).prod() - 1)
        .rename_axis("date")
        .reset_index()
    )
    monthly["year"]  = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    pivot = monthly.pivot(index="year", columns="month", values="pnl") * 100
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(pivot) + 1.5))
    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn", center=0,
        annot=True, fmt=".1f",
        linewidths=0.4, linecolor="#e0e0e0",
        cbar_kws={"label": "Monthly return (%)"},
    )
    ax.set_title("Monthly returns (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")

    out = DATA_RES / "monthly_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 3. Cluster size distribution ───────────────────────────────────────────────
def plot_cluster_sizes():
    diag_path = DATA_PROC / "cluster_diagnostics.csv"
    if not diag_path.exists():
        print("  Skipping cluster size plot (diagnostics not found).")
        return

    diag = pd.read_csv(diag_path, index_col="window")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    axes[0].plot(range(len(diag)), diag["mean_size"].values,
                 color=BLUE, linewidth=1.5)
    axes[0].axhline(8,  color=CORAL, linestyle="--", linewidth=0.8, label="Min target (8)")
    axes[0].axhline(20, color=GRAY,  linestyle="--", linewidth=0.8, label="Max target (20)")
    axes[0].set_title("Mean cluster size per window")
    axes[0].set_xlabel("Window index")
    axes[0].set_ylabel("Stocks per cluster")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(range(len(diag)), diag["n_clusters"].values,
                 color=BLUE, linewidth=1.5)
    axes[1].set_title("Number of clusters per window")
    axes[1].set_xlabel("Window index")
    axes[1].set_ylabel("Clusters")

    out = DATA_RES / "cluster_size_dist.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 4. Factor beta heatmap (sample window) ────────────────────────────────────
def plot_beta_heatmap():
    """Shows mean factor betas per cluster for the most recent formation window."""
    beta_files   = sorted((DATA_PROC / "betas").glob("betas_*.parquet"))
    cluster_files= sorted((DATA_PROC / "clusters").glob("clusters_*.parquet"))
    if not beta_files or not cluster_files:
        print("  Skipping beta heatmap (files not found).")
        return

    # Use the last window
    beta_df    = pd.read_parquet(beta_files[-1])
    cluster_df = pd.read_parquet(cluster_files[-1])
    cluster_df = cluster_df[cluster_df["cluster_id"] >= 0]

    merged = beta_df.merge(
        cluster_df.set_index("permno")["cluster_id"],
        left_index=True, right_index=True, how="inner"
    )

    mean_betas = merged.groupby("cluster_id").mean()
    if mean_betas.empty or mean_betas.shape[1] < 2:
        print("  Skipping beta heatmap (insufficient data).")
        return

    fig, ax = plt.subplots(figsize=(max(10, mean_betas.shape[1] * 0.7),
                                     max(4,  mean_betas.shape[0] * 0.5 + 1)))
    sns.heatmap(
        mean_betas, ax=ax,
        cmap="coolwarm", center=0,
        annot=True, fmt=".2f",
        linewidths=0.3, linecolor="#e0e0e0",
        cbar_kws={"label": "Mean robust beta"},
    )
    ax.set_title("Mean factor betas per cluster (most recent formation window)")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Cluster ID")

    out = DATA_RES / "factor_beta_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 5. Pairs per window ────────────────────────────────────────────────────────
def plot_pairs_per_window():
    summary_path = DATA_PROC / "pairs_summary.csv"
    if not summary_path.exists():
        print("  Skipping pairs summary chart (file not found).")
        return

    summary = pd.read_csv(summary_path, index_col="window")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(len(summary)), summary["n_pairs"].values, color=BLUE, alpha=0.8)
    ax.set_title("Cointegrating pairs per formation window")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Number of pairs")

    out = DATA_RES / "pairs_summary_chart.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 6. Print final performance table ──────────────────────────────────────────
def print_performance():
    perf_path = DATA_RES / "performance.csv"
    if not perf_path.exists():
        print("  Performance CSV not found.")
        return
    perf = pd.read_csv(perf_path)
    print("\n── Final performance (net of costs) ─────────────────────")
    for col in perf.columns:
        print(f"  {col:<30}: {perf.iloc[0][col]}")


# ── 7. Cost sensitivity analysis ──────────────────────────────────────────────
def plot_cost_sensitivity():
    """
    Shows how Sharpe ratio and annualised return change across different
    transaction cost assumptions. Useful for the report to show the strategy
    is robust to the exact cost estimate used.

    Cost scenarios reflect:
      0 bps  — gross P&L (no costs, theoretical upper bound)
      3 bps  — IBKR Lite (commission-free) + bid-ask spread only
      5 bps  — IBKR Pro Fixed ($0.005/share) + bid-ask spread  ← our assumption
      10 bps — conservative/older academic literature assumption
      20 bps — very conservative / small account with min $1 commissions
    """
    pnl_path = DATA_RES / "daily_pnl.parquet"
    tlog_path = DATA_RES / "trade_log.parquet"
    if not pnl_path.exists() or not tlog_path.exists():
        print("  Skipping cost sensitivity (P&L or trade log not found).")
        return

    base_pnl   = pd.read_parquet(pnl_path)["pnl"]
    trade_log  = pd.read_parquet(tlog_path)
    n_trades   = len(trade_log)
    n_days     = len(base_pnl)

    # The base P&L was computed with ROUND_TRIP_BPS=5 and SHORT_BORROW_BPS=25
    # To simulate other cost assumptions, we adjust daily:
    #   cost_delta_per_day = (new_rt_bps - 5) / 10000 * (trades_per_day / active_pairs)
    # Approximation: treat the cost adjustment as a flat daily P&L add/subtract
    # based on the difference in round-trip costs × trade frequency

    from config import ROUND_TRIP_BPS
    base_rt  = ROUND_TRIP_BPS   # bps used in the actual backtest
    # Estimate daily cost impact: trades per day × cost_one_way / avg_active_pairs
    # From performance: n_trades over n_days, avg ~30 active pairs
    avg_active   = 30
    trades_per_day = n_trades / n_days

    scenarios = [0, 3, 5, 10, 20]
    sharpes   = []
    returns   = []

    for rt_bps in scenarios:
        delta_per_trade  = (rt_bps - base_rt) / 10_000     # change vs base
        daily_cost_delta = trades_per_day * delta_per_trade / avg_active
        adj_pnl = base_pnl - daily_cost_delta
        ann     = 252
        ret_ann = adj_pnl.mean() * ann * 100
        vol_ann = adj_pnl.std() * (ann ** 0.5) * 100
        sharpe  = (ret_ann / vol_ann) if vol_ann > 0 else 0
        sharpes.append(round(sharpe, 3))
        returns.append(round(ret_ann, 2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Sharpe chart
    colors = [CORAL if s < 0 else BLUE for s in sharpes]
    axes[0].bar(scenarios, sharpes, color=colors, alpha=0.85, width=1.8)
    axes[0].axhline(0, color=GRAY, linewidth=1.0, linestyle="--")
    axes[0].set_title("Sharpe ratio vs transaction cost assumption")
    axes[0].set_xlabel("Round-trip cost (bps)")
    axes[0].set_ylabel("Sharpe ratio")
    axes[0].set_xticks(scenarios)
    labels = [f"{b} bps" + (" ← IBKR Pro" if b == 5 else
              " ← GC only" if b == 3 else
              " ← No cost" if b == 0 else
              " ← Academic" if b == 10 else
              " ← Conservative") for b in scenarios]
    for i, (b, s) in enumerate(zip(scenarios, sharpes)):
        axes[0].text(b, s + (0.02 if s >= 0 else -0.05),
                     f"{s:.2f}", ha="center", va="bottom" if s >= 0 else "top",
                     fontsize=9)

    # Return chart
    colors2 = [CORAL if r < 0 else BLUE for r in returns]
    axes[1].bar(scenarios, returns, color=colors2, alpha=0.85, width=1.8)
    axes[1].axhline(0, color=GRAY, linewidth=1.0, linestyle="--")
    axes[1].set_title("Annualised return vs transaction cost assumption")
    axes[1].set_xlabel("Round-trip cost (bps)")
    axes[1].set_ylabel("Annualised return (%)")
    axes[1].set_xticks(scenarios)
    for i, (b, r) in enumerate(zip(scenarios, returns)):
        axes[1].text(b, r + (0.05 if r >= 0 else -0.1),
                     f"{r:.1f}%", ha="center", va="bottom" if r >= 0 else "top",
                     fontsize=9)

    # Reference line for IBKR assumption
    for ax in axes:
        ax.axvline(5, color="#185FA5", linewidth=1.0, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Cost sensitivity  |  IBKR Pro Fixed: $0.005/share + bid-ask ≈ 5 bps\n"
        "Sources: interactivebrokers.com/en/pricing/commissions-stocks.php  "
        "|  interactivebrokers.com/en/pricing/short-sale-cost.php",
        fontsize=9, color=GRAY
    )
    fig.tight_layout()
    out = DATA_RES / "cost_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating output charts and tables…\n")
    plot_equity_curve()
    plot_monthly_heatmap()
    plot_cluster_sizes()
    plot_beta_heatmap()
    plot_pairs_per_window()
    print_performance()
    plot_cost_sensitivity()
    print(f"\nAll outputs saved to: {DATA_RES}")
