#!/usr/bin/env python3
"""
run_regime.py
=============

Tests whether the strategy's edge is regime-dependent by splitting OOS
trades by market volatility regime. If most of the edge concentrates in
high-vol periods (spread widening events), the strategy is effectively a
volatility bet — not a structural arbitrage — and may dry up as markets
normalise.

Regime proxy (no external data required):
    Rolling 20-day realised annualised volatility of an equal-weight
    local equity basket constructed from the underlying local prices in
    the pair registry. Trades are assigned the regime prevailing at their
    ``open_date``.

Regime buckets: **low / medium / high** tertile of the vol proxy over
the full OOS period.

Optionally, supply ``--vix-csv`` as a CSV with columns ``date,vix`` to
use VIX directly instead of the internal proxy.

Also reports RUCE-net by **calendar year** so sub-period consistency can
be assessed visually.

Output: ``regime_breakdown.csv`` + ``regime_summary.json``.

Usage
-----
::

    # No flags needed: auto-discovers the latest trades file and uses the
    # default pairs registry + global prices parquet.
    python review/run_regime.py

    # Or point at a specific run / override any default:
    python review/run_regime.py \\
        --trades        datastream/data/walkforward_output/walkforward_XXXX/trades_oos.csv \\
        --global-prices datastream/data/parquet/global/global_prices.parquet \\
        --pairs         config/pairs/asian_adr_pairs.json \\
        --out-dir       datastream/data/walkforward_output/walkforward_XXXX

    # Use VIX directly (CSV must have columns: date, vix):
    python review/run_regime.py --vix-csv /path/to/vix.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("regime")

TRADING_DAYS = 252

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"

# Default locations searched when --trades / --pairs / --global-prices are omitted.
_PAIRS_DEFAULT = _REPO_ROOT / "config/pairs/asian_adr_pairs.json"
_GLOBAL_PRICES_DEFAULT = _DATASTREAM / "data/parquet/global/global_prices.parquet"
# Output roots that may contain walkforward_* run folders with a trades file.
_OUTPUT_ROOTS = [
    _DATASTREAM / "data" / "walkforward_output",
    _REPO_ROOT / "review" / "output",
    _REPO_ROOT / "research" / "output",
]
# Trades filenames in order of preference.
_TRADES_NAMES = ["trades_oos.csv", "trades.csv", "trades.parquet"]


def _find_latest_trades() -> Optional[Path]:
    """Auto-discover the most recently modified trades file under the known
    output roots so ``--trades`` can be omitted. Returns None if none found."""
    candidates: list[Path] = []
    for root in _OUTPUT_ROOTS:
        if not root.exists():
            continue
        for name in _TRADES_NAMES:
            candidates.extend(root.rglob(name))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_screen = _load_module("ds_screening_regime", _DATASTREAM / "run_asian_adr_screening.py")


# -----------------------------------------------------------------------------
# Volatility proxy
# -----------------------------------------------------------------------------
def build_vol_proxy(
    global_prices: pd.DataFrame,
    pairs_data: list[dict],
    window: int = 20,
) -> pd.Series:
    """Rolling annualised realised vol of an equal-weight local equity basket.
    Returns a daily series aligned to trading dates."""
    tickers = list({p["underlying_ticker"] for p in pairs_data})
    rets: dict[str, pd.Series] = {}
    for ticker in tickers:
        px = (global_prices[global_prices["ticker"] == ticker]
              .set_index("marketdate")["close"]
              .sort_index())
        if len(px) > window + 5:
            rets[ticker] = px.pct_change()
    if not rets:
        return pd.Series(dtype=float, name="vol_proxy")
    df = pd.DataFrame(rets).dropna(how="all")
    eq_ret = df.mean(axis=1)
    vol = eq_ret.rolling(window, min_periods=window // 2).std(ddof=1) * np.sqrt(TRADING_DAYS)
    vol.name = "vol_proxy"
    log.info("vol proxy: %d tickers, rolling %d-day window, %d dates",
             len(rets), window, len(vol.dropna()))
    return vol


# -----------------------------------------------------------------------------
# Regime assignment
# -----------------------------------------------------------------------------
def assign_regimes(
    vol_series: pd.Series,
    n_buckets: int = 3,
) -> pd.Series:
    """Assign each date to a regime bucket (low/medium/high) based on
    percentile of the vol proxy over the full series."""
    labels = {1: "low", 2: "medium", 3: "high"} if n_buckets == 3 else None
    buckets = pd.qcut(vol_series.dropna(), q=n_buckets,
                      labels=[labels[i + 1] for i in range(n_buckets)] if labels else None,
                      duplicates="drop")
    return buckets.reindex(vol_series.index)


# -----------------------------------------------------------------------------
# Per-regime stats
# -----------------------------------------------------------------------------
def regime_stats(subset: pd.DataFrame, metric: str) -> dict:
    """Summary statistics for a regime slice of the trades dataframe."""
    if subset.empty:
        return {"n_trades": 0}
    closed = subset[~subset.get("was_aborted", False).astype(bool)]
    vals = closed[metric].to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"n_trades": len(subset), "n_closed": 0}
    return {
        "n_trades": len(subset),
        "n_closed": len(closed),
        "median": round(float(np.median(vals)), 6),
        "mean": round(float(np.mean(vals)), 6),
        "std": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else None,
        "pct_profitable": round(float((vals > 0).mean()), 4),
        "p25": round(float(np.quantile(vals, 0.25)), 6),
        "p75": round(float(np.quantile(vals, 0.75)), 6),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regime-dependence test: split OOS trades by vol regime and year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades", type=Path, default=None,
                   help="trades_oos.csv / trades.csv / trades.parquet "
                        "(auto-discovers latest under datastream/data/walkforward_output if omitted)")
    p.add_argument("--pairs", type=Path, default=_PAIRS_DEFAULT)
    p.add_argument("--global-prices", type=Path, default=_GLOBAL_PRICES_DEFAULT)
    p.add_argument("--vix-csv", type=Path, default=None,
                   help="optional CSV with columns 'date,vix'; overrides internal proxy")
    p.add_argument("--vol-window", type=int, default=20,
                   help="rolling window (days) for realised vol proxy")
    p.add_argument("--metric", default="roce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"])
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "review" / "output"
                   / f"regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # --- resolve trades path (auto-discover latest if not supplied) -----------
    if args.trades is None:
        args.trades = _find_latest_trades()
        if args.trades is None:
            log.error("no trades file found under %s — pass --trades explicitly",
                      ", ".join(str(r) for r in _OUTPUT_ROOTS))
            return 2
        log.info("auto-discovered trades file: %s", args.trades)
    if not args.trades.exists():
        log.error("trades file not found: %s", args.trades)
        return 2

    # --- load trades ----------------------------------------------------------
    if args.trades.suffix == ".parquet":
        trades = pd.read_parquet(args.trades)
    else:
        trades = pd.read_csv(args.trades)
    trades["open_date"]  = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    log.info("loaded %d trades from %s", len(trades), args.trades)

    # --- build vol proxy ------------------------------------------------------
    if args.vix_csv is not None and args.vix_csv.exists():
        vix_df = pd.read_csv(args.vix_csv, parse_dates=["date"])
        vol_series = vix_df.set_index("date")["vix"].sort_index()
        vol_series.index = pd.to_datetime(vol_series.index).normalize()
        vol_series.name = "vol_proxy"
        vol_label = "VIX"
        log.info("using VIX from %s: %d observations", args.vix_csv, len(vol_series))
    else:
        if not args.pairs.exists():
            log.error("pairs file not found: %s — pass --pairs or --vix-csv", args.pairs)
            return 2
        if not args.global_prices.exists():
            log.error("global prices not found: %s", args.global_prices)
            return 2
        pairs_data = json.loads(args.pairs.read_text())
        global_prices = _screen._load_or_die(args.global_prices, "global prices")
        global_prices["marketdate"] = pd.to_datetime(global_prices["marketdate"]).dt.normalize()
        vol_series = build_vol_proxy(global_prices, pairs_data, args.vol_window)
        vol_series.index = pd.to_datetime(vol_series.index).normalize()
        vol_label = f"realised_vol_{args.vol_window}d"

    # --- assign regime to each trade by open_date ----------------------------
    regimes = assign_regimes(vol_series.dropna())
    # forward-fill so that non-trading dates still get a regime label
    regime_all = regimes.reindex(
        pd.date_range(vol_series.index.min(), vol_series.index.max(), freq="D")
    ).ffill()

    open_dates = trades["open_date"].dt.normalize()
    trades["vol_regime"] = open_dates.map(regime_all).astype(str)
    trades["vol_level"] = open_dates.map(vol_series)  # actual vol value for diagnostics
    trades["year"] = trades["open_date"].dt.year

    # --- per-regime stats -----------------------------------------------------
    regime_order = ["low", "medium", "high"]
    regime_records = []
    for regime in regime_order:
        sub = trades[trades["vol_regime"] == regime]
        stats = regime_stats(sub, args.metric)
        regime_records.append({"regime": regime, "vol_label": vol_label, **stats})
        log.info("  regime=%-6s  n=%4d  median=%-8s  pct_profitable=%s",
                 regime, stats.get("n_trades", 0),
                 f"{stats['median']:.4f}" if stats.get("median") is not None else "n/a",
                 f"{stats.get('pct_profitable', 0):.2f}" if stats.get("pct_profitable") is not None else "n/a")

    # --- per-year stats -------------------------------------------------------
    year_records = []
    for yr in sorted(trades["year"].dropna().unique()):
        sub = trades[trades["year"] == yr]
        stats = regime_stats(sub, args.metric)
        year_records.append({"year": int(yr), **stats})

    # --- monotonicity check (regime dependence flag) --------------------------
    medians = [r.get("median") for r in regime_records if r.get("median") is not None]
    n_valid_regimes = sum(1 for r in regime_records if r.get("n_closed", 0) >= 5)
    regime_monotone = (len(medians) >= 2 and
                       all(medians[i] <= medians[i + 1] for i in range(len(medians) - 1)))

    summary = {
        "metric": args.metric,
        "vol_proxy": vol_label,
        "vol_window_days": args.vol_window,
        "regime_monotone_increasing": regime_monotone,
        "regime_dependence_flagged": regime_monotone and len(medians) >= 2 and (
            (medians[-1] - medians[0]) / abs(medians[0]) > 0.5 if medians[0] != 0 else False
        ),
        "regimes": regime_records,
        "by_year": year_records,
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regime_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # tidy CSV: one row per regime + year
    pd.DataFrame(regime_records).to_csv(out_dir / "regime_breakdown.csv", index=False)
    pd.DataFrame(year_records).to_csv(out_dir / "regime_by_year.csv", index=False)

    # also save trades enriched with regime labels for external plotting
    trades_out = trades[["pair_id", "open_date", "close_date", args.metric,
                         "vol_regime", "vol_level", "year"]
                        if "pair_id" in trades.columns else
                        ["open_date", "close_date", args.metric,
                         "vol_regime", "vol_level", "year"]].copy()
    trades_out.to_csv(out_dir / "regime_trades.csv", index=False)

    log.info("=" * 64)
    log.info("REGIME BREAKDOWN  (metric=%s, proxy=%s)", args.metric, vol_label)
    log.info("=" * 64)
    for r in regime_records:
        log.info("  %-8s  n=%-4d  median=%-8s  pct_profitable=%s",
                 r["regime"], r.get("n_trades", 0),
                 f"{r['median']:.4f}" if r.get("median") is not None else "n/a",
                 f"{r.get('pct_profitable', 0):.2f}" if r.get("pct_profitable") is not None else "n/a")
    if summary["regime_dependence_flagged"]:
        log.warning("  *** edge is concentrated in high-vol regime — "
                    "strategy may be a disguised vol bet ***")
    else:
        log.info("  regime spread is modest — no strong vol-regime dependence detected")
    log.info("  YEAR BREAKDOWN")
    for r in year_records:
        log.info("    %d  n=%-4d  median=%s",
                 r["year"], r.get("n_trades", 0),
                 f"{r['median']:.4f}" if r.get("median") is not None else "n/a")
    log.info("=" * 64)
    log.info("wrote regime_breakdown.csv + regime_summary.json to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
