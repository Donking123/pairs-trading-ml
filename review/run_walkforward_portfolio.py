#!/usr/bin/env python3
"""
run_walkforward_portfolio.py
============================

Portfolio-level tearsheet for an **out-of-sample walk-forward run**
(``datastream/run_walkforward.py``).  Produces the *same metrics* as
``review/portfolio.py`` — daily equity curve plus annualised return / vol /
Sharpe / Newey-West Sharpe / Sortino / max-drawdown / Calmar / hit-rate — but is
purpose-built to read a walk-forward output directory.

Why a separate script
---------------------
``review/portfolio.py`` builds its mark-to-market price panel from the **pair
registry** (``config/pairs/asian_adr_pairs.json``), which is screened on the
*full* history.  The walk-forward, by design, **re-screens pairs on each train
window**, so it trades pairs that are not in the full-history registry.  Those
pairs have no panel entry, so ``portfolio.py`` silently drops them — typically
~15% of OOS trades go unmarked.

This script closes that gap: it builds the panel from the pairs **actually
present in ``trades_oos.csv``**, resolving each pair's ``adr_ratio`` /
``underlying_currency`` / exchange from ``adr_reference.parquet`` instead of the
registry.  Result: ~100% MTM coverage and an honest OOS Sharpe over *all* the
walk-forward's trades.

All the heavy lifting (daily return construction, Newey-West Sharpe, the stats
block) is imported verbatim from ``review/portfolio.py`` so the two stay in
lock-step — this file only changes *where the trades and the price panel come
from*.

Usage
-----
::

    # auto-discover the latest walk-forward run and write the tearsheet into it
    python review/run_walkforward_portfolio.py

    # point at a specific walk-forward output directory
    python review/run_walkforward_portfolio.py \
        --wf-dir datastream/data/walkforward_output/wf_verify_20260613_000238 \
        --metric ruce_net

Outputs (written into the walk-forward run dir by default):
``equity_curve.csv``, ``rolling_sharpe.csv``, ``portfolio_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("wf_portfolio")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"
_WF_ROOT = _DATASTREAM / "data" / "walkforward_output"


# -----------------------------------------------------------------------------
# Reuse portfolio.py's metric engine + run_backtest's panel builder verbatim.
# -----------------------------------------------------------------------------
def _load_module(name: str, path: Path):
    """Import a sibling script by path (no package install needed)."""
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pf = _load_module("review_portfolio", Path(__file__).resolve().parent / "run_portfolio.py")
_bt = _load_module("ds_run_backtest", _DATASTREAM / "run_backtest.py")
# the adr_ratio estimator (_bulk_estimate_ratios) now lives in run_walkforward
_screen = _load_module("ds_walkforward_wfp", _DATASTREAM / "run_walkforward.py")


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file, falling back to fastparquet — some files in this
    repo trip pyarrow with 'Repetition level histogram size mismatch' (OSError).
    Mirrors run_backtest.load_panel's reader."""
    try:
        return pd.read_parquet(path)
    except OSError:
        return pd.read_parquet(path, engine="fastparquet")


# -----------------------------------------------------------------------------
# Resolve the walk-forward run / trades file
# -----------------------------------------------------------------------------
def find_latest_wf_dir() -> Optional[Path]:
    """Most recently modified walk-forward run dir (one containing trades_oos.csv)."""
    if not _WF_ROOT.exists():
        return None
    candidates = [p.parent for p in _WF_ROOT.rglob("trades_oos.csv")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "trades_oos.csv").stat().st_mtime)


# -----------------------------------------------------------------------------
# Build the MTM price panel from the pairs the walk-forward actually traded.
# -----------------------------------------------------------------------------
def build_panel_from_trades(
    trades: pd.DataFrame,
    adr_reference: Path,
    adr_prices: Path,
    global_prices: Path,
    fx_rates: Path,
    registry: Optional[Path] = None,
) -> dict:
    """Construct the per-pair price panel for every pair in ``trades`` (not just
    the registry), so the walk-forward's OOS-selected pairs are all covered.

    Each pair's ``adr_ratio`` / ``underlying_currency`` / ``underlying_exchange``
    is resolved from ``adr_reference.parquet`` (keyed on adr+underlying ticker).
    Pairs whose reference ``adr_ratio`` is null have it **estimated from price
    medians** using the screening pipeline's own estimator
    (``_bulk_estimate_ratios``), exactly as ``run_walkforward.py`` did when it
    selected them — so the ratios reconcile. The registry, if present, is a
    last-resort fallback. The panel is then built with ``run_backtest.load_panel``.
    """
    ref = _read_parquet(adr_reference)
    # one row per (adr, underlying); keep the most recent mapping if duplicated
    if "enddate" in ref.columns:
        ref = ref.sort_values("enddate")
    ref = ref.drop_duplicates(["adr_ticker", "underlying_ticker"], keep="last")
    ref_lookup = ref.set_index(["adr_ticker", "underlying_ticker"])

    # registry fallback (any pair the reference misses entirely)
    reg_lookup: dict = {}
    if registry is not None and registry.exists():
        for r in json.loads(registry.read_text()):
            reg_lookup[r["pair_id"]] = r

    want = trades[["pair_id", "adr_ticker", "underlying_ticker"]].drop_duplicates()

    # First pass: pull exchange/currency/ratio from the reference (or registry).
    rows: list[dict] = []
    for _, row in want.iterrows():
        pid, adr_t, und_t = row["pair_id"], row["adr_ticker"], row["underlying_ticker"]
        exch = curr = None
        ratio = float("nan")
        if (adr_t, und_t) in ref_lookup.index:
            rr = ref_lookup.loc[(adr_t, und_t)]
            exch, curr = rr["underlying_exchange"], rr["underlying_currency"]
            try:
                ratio = float(rr["adr_ratio"])
            except (TypeError, ValueError):
                ratio = float("nan")
        if (curr is None or pd.isna(ratio) or ratio <= 0) and pid in reg_lookup:
            rg = reg_lookup[pid]
            exch = exch or rg.get("underlying_exchange")
            curr = curr or rg.get("underlying_currency")
            try:
                rr2 = float(rg["adr_ratio"])
                if pd.isna(ratio) or ratio <= 0:
                    ratio = rr2
            except (KeyError, TypeError, ValueError):
                pass
        rows.append({"pair_id": pid, "adr_ticker": adr_t, "underlying_ticker": und_t,
                     "underlying_exchange": exch, "underlying_currency": curr,
                     "adr_ratio": ratio})
    resolved = pd.DataFrame(rows)

    # Second pass: estimate ratios for null-ratio pairs that have a currency,
    # using the screening pipeline's own price-median estimator (same snap rules).
    need = resolved[(resolved["adr_ratio"].isna() | (resolved["adr_ratio"] <= 0))
                    & resolved["underlying_currency"].notna()]
    if not need.empty:
        log.info("estimating adr_ratio from price medians for %d null-ratio pairs "
                 "(screening's estimator) ...", len(need))
        adr_df = _read_parquet(adr_prices)
        glb_df = _read_parquet(global_prices)
        fx_df = _read_parquet(fx_rates)
        est = _screen._bulk_estimate_ratios(
            need[["adr_ticker", "underlying_ticker", "underlying_currency"]],
            adr_df, glb_df, fx_df,
        )
        for idx, r in need.iterrows():
            e = est.get((r["adr_ticker"], r["underlying_ticker"]), float("nan"))
            if pd.notna(e) and e > 0:
                resolved.at[idx, "adr_ratio"] = float(e)

    pairs: list = []
    skipped: list[str] = []
    for _, r in resolved.iterrows():
        if r["underlying_currency"] is None or pd.isna(r["adr_ratio"]) or r["adr_ratio"] <= 0:
            skipped.append(r["pair_id"])
            continue
        pairs.append(_bt.Pair(
            pair_id=r["pair_id"],
            adr_ticker=r["adr_ticker"],
            underlying_ticker=r["underlying_ticker"],
            underlying_exchange=r["underlying_exchange"] or "",
            underlying_currency=r["underlying_currency"],
            adr_ratio=float(r["adr_ratio"]),
        ))

    log.info("resolved %d/%d traded pairs for the panel (%d unresolved)",
             len(pairs), len(want), len(skipped))
    if skipped:
        log.warning("could not resolve adr_ratio/currency for %d pairs: %s",
                    len(skipped), ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""))
    return _bt.load_panel(adr_prices, global_prices, fx_rates, pairs)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Portfolio-level OOS tearsheet from a run_walkforward.py output dir "
                    "(same metrics as review/portfolio.py, full MTM coverage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wf-dir", type=Path, default=None,
                   help="walk-forward run directory containing trades_oos.csv "
                        "(auto-discovers the latest under "
                        "datastream/data/walkforward_output if omitted)")
    p.add_argument("--by-floor", type=Path, default=None,
                   help="root dir holding floor_<label>/trades_oos.csv subdirs "
                        "(from 'run_walkforward.py --floors'); build a persistent "
                        "equity_curve.csv + portfolio_stats.json in each and print a "
                        "per-floor summary table. Overrides --wf-dir/--trades.")
    p.add_argument("--trades", type=Path, default=None,
                   help="explicit trades_oos.csv path (overrides --wf-dir)")
    p.add_argument("--metric", default="roce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"],
                   help="per-trade return column to compound (see portfolio.py)")
    p.add_argument("--weight", default="equal", choices=["equal", "capital"])
    p.add_argument("--max-concurrent", type=int, default=20,
                   help="capital-budget cap (only used with --weight capital)")
    p.add_argument("--rolling-window", type=int, default=252,
                   help="calendar-day window for rolling Sharpe output")
    p.add_argument("--rf", type=float, default=0.0,
                   help="annual risk-free rate; Sharpe/Sortino/NW-Sharpe use excess return")
    p.add_argument("--nw-lags", type=int, default=None,
                   help="Bartlett lag truncation for the Newey-West Sharpe "
                        "(default: Newey-West 1994 automatic bandwidth)")
    p.add_argument("--no-mtm", dest="mtm", action="store_false", default=True,
                   help="use legacy close-date lumped returns instead of mark-to-market")
    p.add_argument("--adr-reference", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_reference.parquet")
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--pairs", type=Path,
                   default=_REPO_ROOT / "config/pairs/asian_adr_pairs.json",
                   help="registry, used only as a fallback for pairs the "
                        "adr_reference can't resolve")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output dir for equity_curve.csv + portfolio_stats.json "
                        "(default: the walk-forward run dir itself)")
    return p.parse_args(argv)


def _floor_sort_key(name: str) -> int:
    """Numeric ordering for a ``floor_<label>`` dir name (none < 50k < 250k < 1M)."""
    lab = name.replace("floor_", "")
    if lab == "none":
        return 0
    mult = 1
    if lab.endswith("M"):
        mult, lab = 1_000_000, lab[:-1]
    elif lab.endswith("k"):
        mult, lab = 1_000, lab[:-1]
    try:
        return int(float(lab) * mult)
    except ValueError:
        return 0


def run_portfolio_for_dir(
    args: argparse.Namespace, wf_dir: Path, trades_path: Path, out_dir: Path,
) -> Optional[dict]:
    """Build the OOS portfolio tearsheet for a single walk-forward dir: load the
    trades, construct the daily return series (full-coverage MTM or lumped), and
    write ``equity_curve.csv`` / ``rolling_sharpe.csv`` / ``portfolio_stats.json``
    into ``out_dir``. Returns the portfolio-stats dict (or ``None`` on failure)."""
    if not trades_path.exists():
        log.error("trades file not found: %s", trades_path)
        return None

    # surface the OOS context recorded by run_walkforward.py, if present
    wf_summary = {}
    summ_path = wf_dir / "summary.json"
    if summ_path.exists():
        try:
            wf_summary = json.loads(summ_path.read_text())
        except json.JSONDecodeError:
            pass

    # --- load trades (reuse portfolio.py's loader) ----------------------------
    trades = _pf.load_trades(trades_path)
    log.info("loaded %d trades from %s", len(trades), trades_path)

    # --- daily return series: mark-to-market (full coverage) or lumped --------
    ret = None
    ret_mode = "lumped_close_date"
    if args.mtm:
        missing = [p for p in (args.adr_reference, args.adr_prices,
                               args.global_prices, args.fx_rates) if not p.exists()]
        if missing:
            log.warning("MTM data missing (%s) — falling back to lumped close-date returns",
                        ", ".join(map(str, missing)))
        else:
            try:
                panel = build_panel_from_trades(
                    trades, args.adr_reference, args.adr_prices,
                    args.global_prices, args.fx_rates, registry=args.pairs,
                )
                ret = _pf.daily_returns_mtm(trades, panel, args.metric,
                                            args.weight, args.max_concurrent)
                ret_mode = "mark_to_market"
            except Exception as exc:
                log.warning("MTM build failed (%s) — falling back to lumped returns", exc)
                ret = None
    if ret is None or ret.empty:
        ret = _pf.daily_returns(trades, args.metric, args.weight, args.max_concurrent)
        ret_mode = "lumped_close_date"

    # --- stats (reuse portfolio.py verbatim) ----------------------------------
    stats = _pf.performance_stats(ret, rf=args.rf, nw_lags=args.nw_lags)
    stats["metric"] = args.metric
    stats["weight"] = args.weight
    stats["return_mode"] = ret_mode
    stats["evaluation"] = wf_summary.get("evaluation", "out_of_sample_walkforward")
    stats["n_folds"] = wf_summary.get("n_folds")
    stats["n_trades_total"] = int(len(trades))
    for k in ("mtm_trades_marked", "mtm_trades_lumped",
              "mtm_trades_unmatched", "mtm_coverage", "mtm_local_timing"):
        if k in ret.attrs:
            stats[k] = ret.attrs[k]

    out_dir.mkdir(parents=True, exist_ok=True)

    if not ret.empty:
        equity = (1.0 + ret).cumprod()
        pd.DataFrame({"date": ret.index, "daily_return": ret.values,
                      "equity": equity.values}).to_csv(
            out_dir / "equity_curve.csv", index=False)

        rs = _pf.rolling_sharpe(ret, args.rolling_window)
        if not rs.dropna().empty:
            pd.DataFrame({"date": rs.index, rs.name: rs.values}).to_csv(
                out_dir / "rolling_sharpe.csv", index=False)
            log.info("wrote rolling_sharpe.csv (%d-day window)", args.rolling_window)

    (out_dir / "portfolio_stats.json").write_text(
        json.dumps(stats, indent=2, default=str))

    log.info("=" * 60)
    log.info("OOS WALK-FORWARD PORTFOLIO (metric=%s, weight=%s)", args.metric, args.weight)
    log.info("=" * 60)
    for k, v in stats.items():
        log.info("  %-22s %s", k, v)
    log.info("=" * 60)
    log.info("wrote equity_curve.csv + portfolio_stats.json to %s", out_dir)
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # --- by-floor mode: one persistent equity curve per liquidity floor -------
    # Iterate the floor_<label>/ subdirs written by run_walkforward.py --floors
    # and print a per-floor summary table (folded in from run_walkforward_by_floor.py).
    if args.by_floor is not None:
        root = args.by_floor
        floor_dirs = sorted(
            (p.parent for p in root.glob("floor_*/trades_oos.csv")),
            key=lambda p: _floor_sort_key(p.name),
        )
        if not floor_dirs:
            log.error("no floor_*/trades_oos.csv under %s — run "
                      "'run_walkforward.py --floors' first", root)
            return 2
        rows: list[dict] = []
        for sub in floor_dirs:
            log.info("===== %s =====", sub.name)
            stats = run_portfolio_for_dir(args, sub, sub / "trades_oos.csv", sub)
            if stats:
                rows.append({
                    "share_floor": sub.name.replace("floor_", ""),
                    "n_trades_total": stats.get("n_trades_total"),
                    "sharpe": stats.get("sharpe"),
                    "annualised_return": stats.get("annualised_return"),
                    "max_drawdown": stats.get("max_drawdown"),
                    "mtm_coverage": stats.get("mtm_coverage"),
                })
        if rows:
            log.info("=" * 72)
            log.info("EQUITY CURVES BY FLOOR (metric=%s)", args.metric)
            log.info("\n%s", pd.DataFrame(rows).to_string(index=False))
            log.info("=" * 72)
            log.info("equity curves under %s", root)
        return 0

    # --- single-dir mode: resolve the trades file + run dir -------------------
    if args.trades is not None:
        trades_path = args.trades
        wf_dir = trades_path.parent
    else:
        wf_dir = args.wf_dir or find_latest_wf_dir()
        if wf_dir is None:
            log.error("no walk-forward run found under %s — pass --wf-dir or --trades",
                      _WF_ROOT)
            return 2
        trades_path = wf_dir / "trades_oos.csv"

    out_dir = args.out_dir or wf_dir
    log.info("walk-forward run dir: %s", wf_dir)
    stats = run_portfolio_for_dir(args, wf_dir, trades_path, out_dir)
    return 0 if stats is not None else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
