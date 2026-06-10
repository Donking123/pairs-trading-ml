#!/usr/bin/env python3
"""
portfolio.py
============

Turn a per-trade backtest output into **portfolio-level** performance: a daily
equity curve and the headline risk/return statistics a strategy writeup needs
(annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar, hit rate).

The per-trade ROCE/RUCE distributions that ``run_backtest.py`` /
``run_walkforward.py`` produce answer "how good is the average trade?" — they do
*not* answer "how does the book do over time?" This script bridges that gap.

Model
-----
Each closed round-trip contributes a return realised on its ``close_date``. We
treat each trade as one unit of risk capital (equal-weight) and form a daily
portfolio return as the mean of all trades closing that day:

    r_portfolio(d) = mean( metric_net of trades with close_date == d )

Days with no closes have zero return (capital idle). The equity curve is the
cumulative product of (1 + daily return). This is the simplest defensible
aggregation; it deliberately does not model leverage, position sizing, or
overlapping-capital constraints — switch ``--weight`` to ``capital`` for a
crude capital-cap variant (caps the number of concurrent unit positions).

Usage
-----
::

    python research/portfolio.py \
        --trades research/output/walkforward_XXXX/trades_oos.csv \
        --metric ruce_net \
        --out-dir research/output/walkforward_XXXX

Accepts either the ``trades_oos.csv`` from the walk-forward driver or the
``trades.csv`` / ``trades.parquet`` from ``datastream/run_backtest.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("portfolio")

TRADING_DAYS = 252


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trades file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["close_date"] = pd.to_datetime(df["close_date"])
    if "open_date" in df.columns:
        df["open_date"] = pd.to_datetime(df["open_date"])
    return df


# -----------------------------------------------------------------------------
# Daily portfolio return series
# -----------------------------------------------------------------------------
def daily_returns(
    trades: pd.DataFrame,
    metric: str,
    weight: str = "equal",
    max_concurrent: int = 20,
) -> pd.Series:
    """
    Build a daily portfolio return series from per-trade holding-period returns.

    RUCE/ROCE are holding-period returns (avg ~6 days), not daily returns. To
    convert to a daily portfolio return we divide each day's closed-trade P&L by
    a capital base so the result is interpretable as a fraction-of-book return.

    ``equal``   : capital base = max_concurrent slots (fixed). Daily return =
                  sum(metric of trades closing today) / max_concurrent.
                  Equivalent to assuming the book always has max_concurrent
                  equal-sized positions deployed.

    ``capital`` : capital base = actual open-position count on each day
                  (estimated from open_date/close_date). Tighter model of a
                  book that scales with deployed capital.
    """
    if metric not in trades.columns:
        raise KeyError(f"metric '{metric}' not in trades columns: {list(trades.columns)}")

    closed = trades[~trades.get("was_aborted", False).astype(bool)].copy()
    if closed.empty:
        return pd.Series(dtype=float)

    full_range = pd.date_range(
        closed["close_date"].min(), closed["close_date"].max(), freq="D"
    )

    # sum of holding-period returns closing each day
    daily_pnl = (
        closed.groupby("close_date")[metric].sum()
        .reindex(full_range, fill_value=0.0)
    )

    if weight == "capital" and "open_date" in closed.columns:
        # count concurrent open positions per calendar day
        open_counts = pd.Series(0.0, index=full_range)
        for _, row in closed.iterrows():
            span = pd.date_range(row["open_date"], row["close_date"], freq="D")
            open_counts.loc[open_counts.index.isin(span)] += 1
        # use actual open count as capital base (floor at 1 to avoid div/0)
        capital_base = open_counts.clip(lower=1.0)
        ret = (daily_pnl / capital_base).fillna(0.0)
    else:
        # fixed capital base: book always assumed to have max_concurrent slots
        ret = daily_pnl / max_concurrent

    ret.index.name = "date"
    return ret


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
def performance_stats(ret: pd.Series) -> dict:
    if ret.empty:
        return {"n_days": 0}

    equity = (1.0 + ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    n_days = len(ret)
    years = n_days / 365.25

    ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    ann_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ret.std(ddof=1) > 0 else float("nan")

    downside = ret[ret < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else float("nan")
    sortino = float(ret.mean() * TRADING_DAYS / dvol) if dvol and dvol > 0 else float("nan")

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else float("nan")

    nonzero = ret[ret != 0]
    hit_rate = float((nonzero > 0).mean()) if len(nonzero) else float("nan")

    return {
        "n_days": int(n_days),
        "n_active_days": int(len(nonzero)),
        "total_return": round(total_return, 6),
        "annualised_return": round(ann_return, 6),
        "annualised_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "calmar": round(calmar, 4),
        "hit_rate": round(hit_rate, 4),
        "final_equity": round(float(equity.iloc[-1]), 6),
    }


# -----------------------------------------------------------------------------
# Rolling Sharpe (sub-period consistency check)
# -----------------------------------------------------------------------------
def rolling_sharpe(ret: pd.Series, window: int = 252) -> pd.Series:
    """Annualised Sharpe in a rolling calendar-day window.
    Declining or inconsistent rolling Sharpe flags regime dependence."""
    if ret.empty:
        return pd.Series(dtype=float, name=f"rolling_sharpe_{window}d")
    min_p = max(window // 4, 20)
    mu = ret.rolling(window, min_periods=min_p).mean()
    sd = ret.rolling(window, min_periods=min_p).std(ddof=1)
    rs = (mu / sd * np.sqrt(TRADING_DAYS)).where(sd > 0)
    rs.name = f"rolling_sharpe_{window}d"
    return rs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Portfolio-level tearsheet (equity curve + Sharpe/maxDD) from per-trade output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades", type=Path, required=True,
                   help="trades_oos.csv / trades.csv / trades.parquet")
    p.add_argument("--metric", default="ruce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"],
                   help="per-trade return column to compound")
    p.add_argument("--weight", default="equal", choices=["equal", "capital"])
    p.add_argument("--max-concurrent", type=int, default=20,
                   help="capital-budget cap (only used with --weight capital)")
    p.add_argument("--rolling-window", type=int, default=252,
                   help="calendar-day window for rolling Sharpe output")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for equity_curve.csv + portfolio_stats.json "
                        "(default: alongside --trades)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    trades = load_trades(args.trades)
    log.info("loaded %d trades from %s", len(trades), args.trades)

    ret = daily_returns(trades, args.metric, args.weight, args.max_concurrent)
    stats = performance_stats(ret)
    stats["metric"] = args.metric
    stats["weight"] = args.weight

    out_dir = args.out_dir or args.trades.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ret.empty:
        equity = (1.0 + ret).cumprod()
        curve = pd.DataFrame({"date": ret.index, "daily_return": ret.values,
                              "equity": equity.values})
        curve.to_csv(out_dir / "equity_curve.csv", index=False)

        rs = rolling_sharpe(ret, args.rolling_window)
        if not rs.dropna().empty:
            rs_df = pd.DataFrame({"date": rs.index, rs.name: rs.values})
            rs_df.to_csv(out_dir / "rolling_sharpe.csv", index=False)
            log.info("wrote rolling_sharpe.csv (%d-day window) to %s",
                     args.rolling_window, out_dir)

    (out_dir / "portfolio_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    log.info("=" * 60)
    log.info("PORTFOLIO STATS (metric=%s, weight=%s)", args.metric, args.weight)
    log.info("=" * 60)
    for k, v in stats.items():
        log.info("  %-20s %s", k, v)
    log.info("=" * 60)
    log.info("wrote equity_curve.csv + portfolio_stats.json to %s", out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
