#!/usr/bin/env python3
"""
run_data_quality.py
===================

Checks the price data for stale-price contamination: a situation where
zero-return days (price identical to the prior day) inflate the apparent
mean-reversion of the spread, causing ADF/PP tests to flag a pair as
cointegrated for a spurious reason (bid-ask bounce, not genuine parity
convergence).

For each approved pair it reports:

1. **Max stale-price run**: longest consecutive run of identical prices
   for each leg. Runs > 5 days are flagged.

2. **Zero-return spread fraction**: fraction of total absolute spread
   change that occurs on days where at least one leg has a zero return.
   Values > 50% indicate that spread movements are driven by stale-leg
   reversals, not genuine price convergence.

3. **Trade-level stale exposure**: fraction of closed round-trips that
   were opened or closed on a stale-price day (requires ``--trades``).

Output: ``data_quality.csv`` (one row per pair) + ``data_quality_summary.json``.

Usage
-----
::

    python research/run_data_quality.py \\
        --pairs         config/pairs/asian_adr_pairs.json \\
        --adr-prices    datastream/data/parquet/adr/adr_prices.parquet \\
        --global-prices datastream/data/parquet/global/global_prices.parquet \\
        --fx-rates      datastream/data/parquet/fx/fx_rates.parquet \\
        --trades        research/output/walkforward_XXXX/trades_oos.csv \\
        --out-dir       research/output/walkforward_XXXX
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
log = logging.getLogger("data_quality")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_screen = _load_module("ds_screening_dq", _DATASTREAM / "run_asian_adr_screening.py")


# -----------------------------------------------------------------------------
# Per-leg stale-price diagnostics
# -----------------------------------------------------------------------------
def max_stale_run(prices: pd.Series) -> int:
    """Longest consecutive run of identical prices (stale feed indicator)."""
    if len(prices) < 2:
        return 0
    stale = prices.eq(prices.shift(1)).fillna(False)
    cur = max_run = 0
    for s in stale:
        cur = cur + 1 if s else 0
        if cur > max_run:
            max_run = cur
    return max_run


def zero_return_spread_fraction(
    spread: pd.Series,
    adr_zero: pd.Series,
    loc_zero: pd.Series,
) -> float:
    """Fraction of |spread change| attributable to days where either leg
    has zero return. High values flag stale-price-driven mean-reversion."""
    delta = spread.diff().abs().dropna()
    either = (adr_zero | loc_zero).reindex(delta.index, fill_value=False)
    total = float(delta.sum())
    if total == 0.0:
        return float("nan")
    return float(delta[either].sum() / total)


def analyse_pair(
    pair_id: str,
    adr_ticker: str,
    underlying_ticker: str,
    underlying_currency: str,
    adr_ratio: float,
    adr_prices: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_pivot: pd.DataFrame,
) -> dict:
    base = {"pair_id": pair_id, "adr_ticker": adr_ticker,
            "underlying_ticker": underlying_ticker}

    adr_px = (adr_prices[adr_prices["ticker"] == adr_ticker]
              .set_index("marketdate")["close"].sort_index())
    loc_px = (global_prices[global_prices["ticker"] == underlying_ticker]
              .set_index("marketdate")["close"].sort_index())

    if adr_px.empty or loc_px.empty:
        return {**base, "status": "missing_prices"}
    if underlying_currency not in fx_pivot.columns:
        return {**base, "status": "missing_fx"}

    fx_s = fx_pivot[underlying_currency]
    combined = pd.concat(
        [adr_px.rename("adr"), loc_px.rename("loc"), fx_s.rename("fx")],
        axis=1
    ).dropna()

    if len(combined) < 10:
        return {**base, "status": "insufficient_overlap", "n_days": len(combined)}

    adr_ret = combined["adr"].pct_change()
    loc_ret = combined["loc"].pct_change()
    spread = combined["adr"] - (combined["loc"] * combined["fx"]) / adr_ratio

    adr_zero = adr_ret == 0
    loc_zero = loc_ret == 0
    adr_zero_pct = float(adr_zero.sum() / max(adr_zero.notna().sum(), 1))
    loc_zero_pct = float(loc_zero.sum() / max(loc_zero.notna().sum(), 1))
    adr_run = max_stale_run(combined["adr"])
    loc_run = max_stale_run(combined["loc"])
    zsf = zero_return_spread_fraction(spread, adr_zero, loc_zero)

    flagged = adr_run > 5 or loc_run > 5 or (not np.isnan(zsf) and zsf > 0.50)
    return {
        **base,
        "status": "ok",
        "n_days": len(combined),
        "adr_zero_return_pct": round(adr_zero_pct, 4),
        "loc_zero_return_pct": round(loc_zero_pct, 4),
        "adr_max_stale_run_days": adr_run,
        "loc_max_stale_run_days": loc_run,
        "zero_return_spread_fraction": round(zsf, 4) if not np.isnan(zsf) else None,
        "flagged": flagged,
        "flag_reason": (
            (["adr_stale_run>5"] if adr_run > 5 else []) +
            (["loc_stale_run>5"] if loc_run > 5 else []) +
            (["zsf>0.50"] if not np.isnan(zsf) and zsf > 0.50 else [])
        ) or None,
    }


# -----------------------------------------------------------------------------
# Trade-level stale exposure
# -----------------------------------------------------------------------------
def stale_trade_exposure(
    trades: pd.DataFrame,
    adr_prices: pd.DataFrame,
    global_prices: pd.DataFrame,
    pairs_data: list[dict],
) -> dict:
    """Fraction of closed round-trips opened/closed on a stale-price day."""
    if trades.empty or "was_aborted" not in trades.columns:
        return {}
    closed = trades[~trades["was_aborted"].astype(bool)].copy()
    if closed.empty:
        return {}

    pair_map = {p["pair_id"]: p for p in pairs_data}

    # pre-compute zero-return date sets per ticker (fast vectorised pass)
    adr_zero: dict[str, set] = {}
    for ticker in adr_prices["ticker"].unique():
        px = (adr_prices[adr_prices["ticker"] == ticker]
              .set_index("marketdate")["close"].sort_index())
        zero_dates = set(px[px.pct_change() == 0].index.normalize().date)
        adr_zero[ticker] = zero_dates

    loc_zero: dict[str, set] = {}
    for ticker in global_prices["ticker"].unique():
        px = (global_prices[global_prices["ticker"] == ticker]
              .set_index("marketdate")["close"].sort_index())
        zero_dates = set(px[px.pct_change() == 0].index.normalize().date)
        loc_zero[ticker] = zero_dates

    def _stale(pid: str, dt) -> bool:
        p = pair_map.get(pid)
        if p is None:
            return False
        d = pd.Timestamp(dt).date()
        return (d in adr_zero.get(p["adr_ticker"], set()) or
                d in loc_zero.get(p["underlying_ticker"], set()))

    closed["open_stale"]  = closed.apply(lambda r: _stale(r["pair_id"], r["open_date"]),  axis=1)
    closed["close_stale"] = closed.apply(lambda r: _stale(r["pair_id"], r["close_date"]), axis=1)

    total = len(closed)
    so = int(closed["open_stale"].sum())
    sc = int(closed["close_stale"].sum())
    return {
        "total_closed_trades": total,
        "stale_open_count": so,
        "stale_close_count": sc,
        "stale_open_pct": round(so / total, 4) if total else None,
        "stale_close_pct": round(sc / total, 4) if total else None,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stale-price contamination check for approved ADR pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs", type=Path,
                   default=_REPO_ROOT / "config/pairs/asian_adr_pairs.json")
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--trades", type=Path, default=None,
                   help="optional trades CSV to compute stale trade-entry exposure")
    p.add_argument("--stale-run-threshold", type=int, default=5,
                   help="flag pairs with stale runs longer than this many days")
    p.add_argument("--zsf-threshold", type=float, default=0.50,
                   help="flag pairs where zero-return spread fraction exceeds this")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "review" / "output"
                   / f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not args.pairs.exists():
        log.error("pairs file not found: %s", args.pairs)
        return 2

    pairs_data = json.loads(args.pairs.read_text())
    log.info("loaded %d pairs from %s", len(pairs_data), args.pairs)

    adr_prices    = _screen._load_or_die(args.adr_prices,    "ADR prices")
    global_prices = _screen._load_or_die(args.global_prices, "global prices")
    fx_rates      = _screen._load_or_die(args.fx_rates,      "FX rates")
    adr_prices["marketdate"]    = pd.to_datetime(adr_prices["marketdate"]).dt.normalize()
    global_prices["marketdate"] = pd.to_datetime(global_prices["marketdate"]).dt.normalize()
    fx_rates["date"]            = pd.to_datetime(fx_rates["date"]).dt.normalize()

    fx_pivot = fx_rates.pivot_table(index="date", columns="base_currency",
                                    values="mid", aggfunc="last")

    records = []
    for i, p in enumerate(pairs_data, 1):
        rec = analyse_pair(
            pair_id=p["pair_id"],
            adr_ticker=p["adr_ticker"],
            underlying_ticker=p["underlying_ticker"],
            underlying_currency=p["underlying_currency"],
            adr_ratio=float(p["adr_ratio"]),
            adr_prices=adr_prices,
            global_prices=global_prices,
            fx_pivot=fx_pivot,
        )
        records.append(rec)
        if i % 10 == 0 or i == len(pairs_data):
            log.info("  analysed %d/%d pairs", i, len(pairs_data))

    df = pd.DataFrame(records)
    flagged_df = df[df.get("flagged", pd.Series(False, index=df.index)).fillna(False)]
    n_flagged = len(flagged_df)

    summary: dict = {
        "n_pairs_analysed": len(records),
        "n_pairs_flagged": n_flagged,
        "flagged_pair_ids": list(flagged_df["pair_id"]) if "pair_id" in flagged_df else [],
        "stale_run_threshold": args.stale_run_threshold,
        "zsf_threshold": args.zsf_threshold,
    }

    # --- optional trade-level stale exposure ----------------------------------
    if args.trades is not None and args.trades.exists():
        if args.trades.suffix == ".parquet":
            trades = pd.read_parquet(args.trades)
        else:
            trades = pd.read_csv(args.trades)
        log.info("computing stale trade exposure for %d trades ...", len(trades))
        stale_exp = stale_trade_exposure(trades, adr_prices, global_prices, pairs_data)
        summary["trade_stale_exposure"] = stale_exp
        log.info("stale trade exposure: open=%.1f%%  close=%.1f%%",
                 (stale_exp.get("stale_open_pct") or 0) * 100,
                 (stale_exp.get("stale_close_pct") or 0) * 100)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "data_quality.csv", index=False)
    (out_dir / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    log.info("=" * 64)
    log.info("DATA QUALITY SUMMARY")
    log.info("=" * 64)
    log.info("  pairs analysed          %d", summary["n_pairs_analysed"])
    log.info("  pairs flagged           %d  (%.1f%%)",
             n_flagged, 100 * n_flagged / max(len(records), 1))
    if n_flagged:
        log.warning("  flagged pairs: %s", ", ".join(summary["flagged_pair_ids"][:10]))
    if "trade_stale_exposure" in summary:
        se = summary["trade_stale_exposure"]
        log.info("  stale open pct          %.2f%%",
                 (se.get("stale_open_pct") or 0) * 100)
        log.info("  stale close pct         %.2f%%",
                 (se.get("stale_close_pct") or 0) * 100)
    log.info("=" * 64)
    log.info("wrote data_quality.csv + data_quality_summary.json to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
