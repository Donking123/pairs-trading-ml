#!/usr/bin/env python3
"""
run_walkforward.py
==================

Out-of-sample (walk-forward) evaluation of the Hong & Susmel (2013) Asian ADR
pairs strategy.

Why this exists
---------------
The one-shot pipeline selects pairs (cointegration + liquidity) over a history
and then ``run_backtest.py`` trades that *same* history. Pairs are therefore
chosen with knowledge of the very prices they are then traded on — an in-sample
look-ahead bias that makes the headline ROCE/RUCE optimistic.

This driver removes that bias. For each fold it:

    1. SELECT  pairs using only data in the TRAIN window  [train_start, split]
       (re-runs the existing screening pipeline with ``as_of = split``).
    2. TRADE   those pairs only in the TEST window         (split, test_end]
       (rolling stats still warm up on pre-split history, but no trade is
        opened or closed before ``split`` — so every recorded trade is OOS).
    3. AGGREGATE the test-window trades across folds into a single OOS result.

A single fold ("holdout") is the default. Pass ``--folds N`` for an expanding
walk-forward: the train window grows, the test window rolls forward.

This script imports the screening and backtest logic verbatim from
``datastream/`` — it adds no new strategy logic, only the train/test discipline.

Usage
-----
::

    python datastream/run_walkforward.py \
        --adr-prices    datastream/data/parquet/adr/adr_prices.parquet \
        --adr-reference datastream/data/parquet/adr/adr_reference.parquet \
        --global-prices datastream/data/parquet/global/global_prices.parquet \
        --fx-rates      datastream/data/parquet/fx/fx_rates.parquet \
        --train-start   2010-01-01 \
        --split         2020-12-31 \
        --test-end      2025-12-31 \
        --out-dir       research/output/walkforward_$(date +%Y%m%d_%H%M%S)

For an expanding walk-forward with 3 OOS folds between ``--split`` and
``--test-end``::

    python datastream/run_walkforward.py ... --folds 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("walkforward")


# -----------------------------------------------------------------------------
# Import the existing datastream/ scripts as modules (they live one dir up and
# are not a package, so load them by path).
# -----------------------------------------------------------------------------
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


_screen = _load_module("ds_screening", _DATASTREAM / "run_asian_adr_screening.py")
_bt = _load_module("ds_backtest", _DATASTREAM / "run_backtest.py")


# -----------------------------------------------------------------------------
# Fold construction
# -----------------------------------------------------------------------------
def build_folds(
    train_start: date,
    split: date,
    test_end: date,
    n_folds: int,
) -> list[tuple[date, date, date]]:
    """
    Return a list of ``(train_start, train_end, test_end)`` tuples.

    Single fold (n_folds=1): one holdout — train on [train_start, split],
    test on (split, test_end].

    Expanding walk-forward (n_folds>1): the test span (split, test_end] is cut
    into ``n_folds`` equal calendar slices. Fold k trains on everything up to
    the start of test-slice k and tests on that slice — so the training window
    expands and the OOS window rolls forward without ever overlapping training.
    """
    if test_end <= split:
        raise ValueError("--test-end must be after --split")
    if split <= train_start:
        raise ValueError("--split must be after --train-start")

    if n_folds <= 1:
        return [(train_start, split, test_end)]

    total_days = (test_end - split).days
    step = total_days / n_folds
    folds: list[tuple[date, date, date]] = []
    for k in range(n_folds):
        slice_start = split + timedelta(days=round(k * step))
        slice_end = split + timedelta(days=round((k + 1) * step))
        # train up to the day before this OOS slice begins
        folds.append((train_start, slice_start, slice_end))
    return folds


# -----------------------------------------------------------------------------
# Single fold
# -----------------------------------------------------------------------------
def run_fold(
    fold_idx: int,
    train_start: date,
    train_end: date,
    test_end: date,
    adr_prices: pd.DataFrame,
    adr_reference: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    cfg: dict,
) -> tuple[list, list[dict]]:
    """
    Select pairs on [train_start, train_end], then backtest them on
    (train_end, test_end]. Returns ``(trades, approved_pair_dicts)``.
    """
    log.info(
        "fold %d: TRAIN [%s .. %s]  ->  TEST (%s .. %s]",
        fold_idx, train_start, train_end, train_end, test_end,
    )

    # --- 1. SELECT on the train window only (as_of = train_end) ---------------
    # The screening pipeline already trims each pair's history to <= as_of, so
    # cointegration/liquidity decisions use no post-split data.
    train_adr = adr_prices[adr_prices["marketdate"] >= pd.Timestamp(train_start)]
    train_glb = global_prices[global_prices["marketdate"] >= pd.Timestamp(train_start)]
    approved, _ = _screen.run_pipeline(
        train_adr, adr_reference, train_glb, fx_rates, train_end, cfg,
    )
    log.info("fold %d: %d pairs approved on train window", fold_idx, len(approved))
    if not approved:
        return [], []

    approved_dicts = [_screen.asdict(p) for p in approved]

    # --- 2. Build the price panel and TRADE the test window only -------------
    pairs = [
        _bt.Pair(
            pair_id=p.pair_id,
            adr_ticker=p.adr_ticker,
            underlying_ticker=p.underlying_ticker,
            underlying_exchange=p.underlying_exchange,
            underlying_currency=p.underlying_currency,
            adr_ratio=float(p.adr_ratio),
            estimation_days=cfg["estimation_days"],
            holding_days=cfg["holding_days"],
            k0=cfg["k0"],
            kc=cfg["kc"],
            roll_spread_adr=float(p.roll_spread_adr),
            roll_spread_local=float(p.roll_spread_local),
        )
        for p in approved
    ]

    # load_panel reads from disk; we already have the frames in memory, so build
    # the panel directly to avoid re-reading. Mirror load_panel's join logic.
    panel = _build_panel(adr_prices, global_prices, fx_rates, pairs)

    trades: list = []
    test_start = train_end + timedelta(days=1)
    for pair in pairs:
        df = panel.get(pair.pair_id)
        if df is None:
            continue
        pair_trades = _bt.backtest_pair(
            pair, df, start=test_start, end=test_end,
            max_overnight_gap_days=cfg.get("max_overnight_gap", 4),
        )
        trades.extend(pair_trades)

    log.info("fold %d: %d OOS trades", fold_idx, len(trades))
    return trades, approved_dicts


def _build_panel(
    adr_prices: pd.DataFrame,
    global_prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    pairs: list,
) -> dict[str, pd.DataFrame]:
    """In-memory equivalent of run_backtest.load_panel (no disk re-read).
    Mirrors load_panel: applies cumadjfactor adjustment and includes local_open_usd."""
    fx_pivot = fx_rates.pivot_table(
        index="date", columns="base_currency", values="mid", aggfunc="last"
    )
    fx_pivot.index = pd.to_datetime(fx_pivot.index).normalize()

    panel: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        adr_raw = adr_prices[adr_prices["ticker"] == pair.adr_ticker].set_index("marketdate")
        loc_raw = global_prices[global_prices["ticker"] == pair.underlying_ticker].set_index("marketdate")
        if adr_raw.empty or loc_raw.empty:
            continue
        if pair.underlying_currency not in fx_pivot.columns:
            continue
        fx_series = fx_pivot[[pair.underlying_currency]].rename(
            columns={pair.underlying_currency: "fx_mid"}
        )

        # apply price-return adjustment (adj = raw * adj_factor)
        adr_slice = _bt._adj_prices(adr_raw, "close").rename("adr_close").to_frame()
        loc_close_adj = _bt._adj_prices(loc_raw, "close").rename("local_close")

        df = (adr_slice
              .join(loc_close_adj, how="inner")
              .join(fx_series, how="inner")
              .dropna())
        if df.empty:
            continue
        df = df.sort_index()
        df["local_close_usd"] = df["local_close"] * df["fx_mid"]
        df["spread"] = df["adr_close"] - df["local_close_usd"] / pair.adr_ratio

        if "open" in loc_raw.columns:
            loc_open_adj = _bt._adj_prices(loc_raw, "open").rename("local_open")
            df = df.join(loc_open_adj, how="left")
            df["local_open_usd"] = df["local_open"] * df["fx_mid"]

        panel[pair.pair_id] = df
    return panel


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward (out-of-sample) backtest of the Asian ADR pairs strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--adr-reference", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_reference.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--out-dir", type=Path,
                   default=_DATASTREAM / "data" / "walkforward_output"
                   / f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    p.add_argument("--train-start", type=_parse_date, default=_parse_date("2010-01-01"),
                   help="first date used for pair selection")
    p.add_argument("--split", type=_parse_date, default=_parse_date("2020-12-31"),
                   help="train/test boundary: selection uses <= split, trading uses > split")
    p.add_argument("--test-end", type=_parse_date, default=_parse_date("2025-12-31"),
                   help="last date traded out-of-sample")
    p.add_argument("--folds", type=int, default=1,
                   help="1 = single holdout; >1 = expanding walk-forward")

    # strategy params (passed through to screening + backtest)
    # Defaults are the optimum from datastream/run_research_permutations.py
    # (T=90, k0=2.50, kc=0.0, H=90): k0=2.50 and kc=0.0 are the robust signals;
    # T=90 vs T=60 is within noise.
    p.add_argument("--estimation-days", type=int, default=90, help="T")
    p.add_argument("--holding-days", type=int, default=90, help="H")
    p.add_argument("--k0", type=float, default=2.5)
    p.add_argument("--kc", type=float, default=0.0)
    p.add_argument("--cointegration-alpha", type=float, default=0.05)
    p.add_argument("--min-history-days", type=int, default=504)
    p.add_argument("--min-non-zero-return-pct", type=float, default=0.50)
    p.add_argument("--max-zero-return-pct-adr", type=float, default=0.50)
    p.add_argument("--max-overnight-gap", type=int, default=4,
                   dest="max_overnight_gap",
                   help="skip two-bar fills where next joint bar is more than this "
                        "many calendar days after the ADR short day")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    adr_prices    = _screen._load_or_die(args.adr_prices, "ADR prices")
    adr_reference = _screen._load_or_die(args.adr_reference, "ADR reference")
    global_prices = _screen._load_or_die(args.global_prices, "Global prices")
    fx_rates      = _screen._load_or_die(args.fx_rates, "FX rates")
    # normalise FX date column once (screening filters by base_currency on the fly)
    fx_rates = fx_rates.copy()
    fx_rates["date"] = pd.to_datetime(fx_rates["date"]).dt.normalize()

    cfg = {
        "estimation_days": args.estimation_days,
        "holding_days": args.holding_days,
        "k0": args.k0,
        "kc": args.kc,
        "cointegration_alpha": args.cointegration_alpha,
        "min_history_days": args.min_history_days,
        "min_non_zero_return_pct": args.min_non_zero_return_pct,
        "max_zero_return_pct_adr": args.max_zero_return_pct_adr,
        "max_overnight_gap": args.max_overnight_gap,
    }

    folds = build_folds(args.train_start, args.split, args.test_end, args.folds)
    log.info("running %d fold(s)", len(folds))

    all_trades: list = []
    fold_records: list[dict] = []
    for k, (tr_start, tr_end, te_end) in enumerate(folds):
        trades, approved = run_fold(
            k, tr_start, tr_end, te_end,
            adr_prices, adr_reference, global_prices, fx_rates, cfg,
        )
        all_trades.extend(trades)
        closed = [t for t in trades if not t.was_aborted]
        fold_records.append({
            "fold": k,
            "train_start": tr_start.isoformat(),
            "train_end": tr_end.isoformat(),
            "test_end": te_end.isoformat(),
            "n_pairs_selected": len(approved),
            "n_trades": len(trades),
            "n_closed": len(closed),
            "median_ruce_net": (round(float(np.median([t.ruce_net for t in closed])), 6)
                                if closed else None),
            "median_roce_net": (round(float(np.median([t.roce_net for t in closed])), 6)
                                if closed else None),
        })

    # --- aggregate OOS report (reuse the backtest's report builder) -----------
    report = _bt.build_report(all_trades, [], cfg)
    report["summary"]["evaluation"] = "out_of_sample_walkforward"
    report["summary"]["n_folds"] = len(folds)
    report["folds"] = fold_records

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(report["summary"], indent=2, default=str))
    (args.out_dir / "distribution.json").write_text(
        json.dumps(report["distribution"], indent=2, default=str))
    (args.out_dir / "folds.json").write_text(
        json.dumps(fold_records, indent=2, default=str))
    trades_df = pd.DataFrame(report["trades"])
    if not trades_df.empty:
        trades_df.to_csv(args.out_dir / "trades_oos.csv", index=False)

    log.info("=" * 72)
    log.info("WALK-FORWARD (OUT-OF-SAMPLE) SUMMARY")
    log.info("=" * 72)
    for key, val in report["summary"].items():
        if key != "config":
            log.info("  %-22s %s", key, val)
    log.info("=" * 72)
    log.info("PER-FOLD")
    for fr in fold_records:
        log.info("  fold %s: %s pairs, %s trades, median RUCE-net=%s",
                 fr["fold"], fr["n_pairs_selected"], fr["n_trades"],
                 fr["median_ruce_net"])
    log.info("=" * 72)
    log.info("DISTRIBUTION (paper Table 7-B format)\n%s",
             _bt.format_distribution_table(report["distribution"]))
    log.info("results in %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
