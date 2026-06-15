#!/usr/bin/env python3
"""
run_sweep_liquidity.py — OOS Sharpe vs an ADR **share-volume** liquidity floor.

Adapted from ``presentation/sweep_liquidity.py`` (the "Mine" tree) to this repo's
walk-forward + portfolio scripts. Re-evaluates the single-fold walk-forward
(train 2010–2020 -> trade 2021–2025; T=90, k0=2.5, kc=0.0, H=90; ROCE-net,
equal-weight, daily MTM) at a series of minimum median-daily-share-volume
thresholds on the ADR leg, and records the OOS portfolio Sharpe (plus return /
drawdown / trade & pair counts) at each floor.

How the floor is applied
------------------------
The reference script re-ran the *whole* walk-forward per threshold via a
``run_fold(..., min_shares=th)`` argument. This repo's ``run_walkforward.run_fold``
has no such parameter, so we take the equivalent-but-cheaper route:

    1. Run the single-fold walk-forward ONCE (no floor) -> approved pairs + all
       OOS trades. The Team OOS discipline is preserved: we drop trades whose
       window straddles a corporate action (``spans_adj_change``), exactly as
       ``run_walkforward.main`` does.
    2. Compute each ADR ticker's median daily share volume over the TRAIN window
       [train_start, split] only (no post-split data — same discipline as the
       reference's selection-time screen).
    3. For each threshold ``th`` keep only the pairs whose ADR median train-window
       volume exceeds ``th`` and the trades belonging to them, then build the OOS
       portfolio tearsheet for that filtered set.

This is identical to re-screening with the floor because ``run_pipeline`` screens
each pair independently (cointegration/liquidity on that pair's own series) and
``backtest_pair`` trades each pair independently — removing other ADRs from the
universe never changes a surviving pair's approval or its trades.

NOTE: the floor is on raw **shares/day**, not dollars. Share counts mix names of
very different prices, so this is a coarser tradeability proxy than $-volume.

Outputs (written to review/output/):
    liquidity_sweep_shares.csv
    liquidity_sweep_shares.png   (only if matplotlib is available)

Usage:  python review/run_sweep_liquidity.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("sweep_liquidity")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"
_REVIEW = _REPO_ROOT / "review"
_OUT_DIR = _REVIEW / "output"

GREEN, RED, NAVY, GREY = "#1b7837", "#b2182b", "#1a2e44", "#777777"


def _load_module(name: str, path: Path):
    """Import a sibling script by path (no package install needed)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_wf = _load_module("ds_walkforward", _DATASTREAM / "run_walkforward.py")
_wfp = _load_module("rev_wf_portfolio", _REVIEW / "run_walkforward_portfolio.py")

# walk-forward defaults (match the headline run / run_walkforward CLI defaults)
TRAIN_START = date(2010, 1, 1)
SPLIT = date(2020, 12, 31)
TEST_END = date(2025, 12, 31)

THRESHOLDS = [0, 1_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]


def fmt_floor(v: int) -> str:
    if v == 0:
        return "none"
    if v >= 1_000_000:
        return f"{v // 1_000_000}M"
    if v >= 1_000:
        return f"{v // 1_000}k"
    return str(v)


def load_data():
    adr = _wf._screen._load_or_die(_DATASTREAM / "data/parquet/adr/adr_prices.parquet", "ADR prices")
    ref = _wf._screen._load_or_die(_DATASTREAM / "data/parquet/adr/adr_reference.parquet", "ADR reference")
    glb = _wf._screen._load_or_die(_DATASTREAM / "data/parquet/global/global_prices.parquet", "Global prices")
    fx = _wf._screen._load_or_die(_DATASTREAM / "data/parquet/fx/fx_rates.parquet", "FX rates")
    fx = fx.copy()
    fx["date"] = pd.to_datetime(fx["date"]).dt.normalize()
    return adr, ref, glb, fx


def run_base_walkforward(adr, ref, glb, fx, cfg, keep_adj_change: bool):
    """Single-fold walk-forward, no floor. Returns ``(trades, approved_dicts)``
    where ``trades`` already has corporate-action straddles dropped (unless
    ``keep_adj_change``), matching run_walkforward.main's OOS reporting."""
    folds = _wf.build_folds(TRAIN_START, SPLIT, TEST_END, 1)
    all_trades, approved = [], []
    for k, (ts, te, ee) in enumerate(folds):
        trades, appr = _wf.run_fold(k, ts, te, ee, adr, ref, glb, fx, cfg)
        if not keep_adj_change:
            n_before = len(trades)
            trades = [t for t in trades if not t.spans_adj_change]
            log.info("fold %d: dropped %d trades spanning a corporate action",
                     k, n_before - len(trades))
        all_trades.extend(trades)
        approved.extend(appr)
    return all_trades, approved


def train_window_median_volume(adr: pd.DataFrame) -> pd.Series:
    """Median daily ADR share volume per ticker over the TRAIN window only."""
    win = adr[(adr["marketdate"] >= pd.Timestamp(TRAIN_START)) &
              (adr["marketdate"] <= pd.Timestamp(SPLIT))]
    return win.groupby("ticker")["volume"].median()


def portfolio_stats_for(trades: list, cfg: dict, metric: str) -> dict:
    """Build the OOS portfolio tearsheet for a list of Trade objects by reusing
    run_walkforward_portfolio.main on a temporary trades_oos.csv (full MTM)."""
    if not trades:
        return {}
    report = _wf._bt.build_report(trades, [], cfg)
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        pd.DataFrame(report["trades"]).to_csv(dd / "trades_oos.csv", index=False)
        (dd / "summary.json").write_text(json.dumps(report["summary"], default=str))
        rc = _wfp.main(["--wf-dir", str(dd), "--metric", metric])
        if rc != 0:
            return {}
        return json.loads((dd / "portfolio_stats.json").read_text())


def run_threshold(th: int, all_trades: list, approved: list,
                  med_vol: pd.Series, cfg: dict, metric: str) -> dict:
    """Filter the base run to ADRs whose train-window median volume exceeds `th`,
    then compute the OOS portfolio stats for that subset."""
    if th > 0:
        eligible = set(med_vol[med_vol > th].index)
        trades = [t for t in all_trades if t.adr_ticker in eligible]
        n_pairs = sum(1 for p in approved if p["adr_ticker"] in eligible)
    else:
        trades = list(all_trades)
        n_pairs = len(approved)

    row = {
        "share_floor": th,
        "n_pairs_selected": n_pairs,
        "n_trades": len(trades),
        "n_closed": sum(1 for t in trades if not t.was_aborted),
    }
    if not trades:
        row.update(sharpe=np.nan, sharpe_nw=np.nan, annualised_return=np.nan,
                   total_return=np.nan, max_drawdown=np.nan, median_roce_net=np.nan)
        return row

    report = _wf._bt.build_report(trades, [], cfg)
    row["median_roce_net"] = report["summary"].get("median_roce_net")
    stats = portfolio_stats_for(trades, cfg, metric)
    for k in ("sharpe", "sharpe_nw", "annualised_return", "total_return", "max_drawdown"):
        row[k] = stats.get(k, np.nan)
    return row


def plot(df: pd.DataFrame, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping PNG (CSV still written)")
        return False

    x = np.arange(len(df))
    sharpe = df["sharpe"].values
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    cols = [GREEN if (s == s and s >= 0) else RED for s in sharpe]
    ax.plot(x, sharpe, color=NAVY, lw=2.0, zorder=1)
    ax.scatter(x, sharpe, c=cols, s=140, zorder=3, edgecolor="white", linewidth=1.5)
    ax.axhline(0, color=GREY, lw=1.2, ls="--")
    for xi, s, n, p in zip(x, sharpe, df["n_trades"], df["n_pairs_selected"]):
        if np.isnan(s):
            continue
        ax.annotate(f"{s:.2f}", (xi, s), textcoords="offset points",
                    xytext=(0, 12 if s >= 0 else -18), ha="center",
                    fontsize=13, fontweight="bold",
                    color=GREEN if s >= 0 else RED)
        ax.annotate(f"{int(p)} pairs\n{int(n)} trades", (xi, s),
                    textcoords="offset points", xytext=(0, -32 if s >= 0 else 20),
                    ha="center", fontsize=9, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_floor(v) for v in df["share_floor"]])
    ax.set_xlabel("minimum median ADR share-volume floor (shares/day, train window)")
    ax.set_ylabel("OOS portfolio Sharpe (ROCE-net, equal-weight, daily MTM)")
    ax.set_title("Edge vs liquidity — OOS Sharpe vs the ADR share-volume floor",
                 fontsize=18, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.text(0.005, 0.005,
            "Single-fold walk-forward (2010–2020 select -> 2021–2025 trade). "
            "Floor is on raw shares/day, a coarse tradeability proxy (cf. $-volume).",
            transform=ax.transAxes, fontsize=10, color=GREY)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep OOS Sharpe vs an ADR median share-volume liquidity floor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metric", default="roce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"],
                   help="per-trade return column compounded by the portfolio engine")
    p.add_argument("--cost-bps", type=float, default=0.0, dest="cost_bps",
                   help="incremental per-side-per-leg cost (bps) passed to the "
                        "walk-forward backtest; 0 isolates the liquidity effect "
                        "(matches the reference sweep)")
    p.add_argument("--keep-adj-change", action="store_true", dest="keep_adj_change",
                   help="keep (rather than drop) trades straddling a corporate "
                        "action; default drops them, matching run_walkforward.main")
    p.add_argument("--out-dir", type=Path, default=_OUT_DIR,
                   help="directory for the CSV/PNG outputs")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = dict(estimation_days=90, holding_days=90, k0=2.5, kc=0.0,
               cointegration_alpha=0.05, min_history_days=504,
               min_non_zero_return_pct=0.50, max_zero_return_pct_adr=0.50,
               max_overnight_gap=4, cost_bps=args.cost_bps)

    adr, ref, glb, fx = load_data()

    log.info("running base single-fold walk-forward once (no floor) ...")
    all_trades, approved = run_base_walkforward(adr, ref, glb, fx, cfg, args.keep_adj_change)
    log.info("base run: %d approved pairs, %d OOS trades", len(approved), len(all_trades))
    if not all_trades:
        log.error("base walk-forward produced no trades; aborting")
        return 1

    med_vol = train_window_median_volume(adr)

    rows = []
    for th in THRESHOLDS:
        log.info("===== share floor %s (%d) =====", fmt_floor(th), th)
        rows.append(run_threshold(th, all_trades, approved, med_vol, cfg, args.metric))
    df = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "liquidity_sweep_shares.csv"
    png_path = args.out_dir / "liquidity_sweep_shares.png"
    df.to_csv(csv_path, index=False)
    plotted = plot(df, png_path)

    log.info("=" * 72)
    log.info("LIQUIDITY SWEEP (metric=%s, equal-weight, daily MTM)", args.metric)
    log.info("=" * 72)
    log.info("\n%s", df.to_string(index=False))
    log.info("=" * 72)
    log.info("wrote %s", csv_path)
    if plotted:
        log.info("wrote %s", png_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
