"""Backtest runner CLI (architecture §13.1).

The module-based counterpart to ``datastream/run_backtest.py`` — but instead of
re-implementing the strategy, it drives the **real** components through
``BacktestEngine``, so the run validates production code. Loads prices/pairs from
the Parquet cache, runs the event-driven replay, and writes the tearsheet.

Run::

    python -m asian_adr.runners.backtest_runner \
        --start 2017-01-01 --end 2019-12-31 --reproduce

``--reproduce`` overrides each pair's ``approved_date`` to the start date so a
selected pair-set is backtested over its full sample history (rather than only
after its production selection date, which is the default point-in-time mode).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime
from pathlib import Path

from asian_adr.backtest import BacktestEngine

from .config import RunnerConfig

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest the Asian ADR strategy on the real components.")
    p.add_argument("--config", default=None, help="TOML config path")
    p.add_argument("--adr-prices", default=None)
    p.add_argument("--global-prices", default=None)
    p.add_argument("--fx-rates", default=None)
    p.add_argument("--pairs", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--k0", type=float, default=None)
    p.add_argument("--kc", type=float, default=None)
    p.add_argument("--T", dest="estimation_days", type=int, default=None)
    p.add_argument("--H", dest="holding_days", type=int, default=None)
    p.add_argument("--aum", type=float, default=None)
    p.add_argument("--notional", type=float, default=None)
    p.add_argument(
        "--reproduce", action="store_true",
        help="override approved_date to --start (backtest selected pairs over history)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")

    config = RunnerConfig.from_toml(args.config)
    # CLI overrides win over config/defaults.
    for attr in ("k0", "kc", "estimation_days", "holding_days"):
        if getattr(args, attr) is not None:
            setattr(config, attr, getattr(args, attr))
    if args.aum is not None:
        config.aum_usd = args.aum
    if args.notional is not None:
        config.notional_per_leg = args.notional

    adr = args.adr_prices or config.adr_prices
    glb = args.global_prices or config.global_prices
    fx = args.fx_rates or config.fx_rates
    pairs = args.pairs or config.pairs_json

    extra = {}
    if args.reproduce:
        extra["approved_date"] = args.start.isoformat()
        extra["expiry_date"] = args.end.isoformat()

    bt_config = config.backtest_config(extra_overrides=extra)
    engine = BacktestEngine.from_parquet(
        bt_config, adr_prices=adr, global_prices=glb, fx_rates=fx, pairs_json=pairs,
    )

    out_dir = args.out_dir or (
        Path(config.backtest_out_dir)
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger.info("backtest %s → %s, out=%s", args.start, args.end, out_dir)
    results = asyncio.run(engine.run_and_report(args.start, args.end, str(out_dir)))
    logger.info("done: %d round-trips → %s", len(results), out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
