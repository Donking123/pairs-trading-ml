#!/usr/bin/env python3
"""
portfolio.py
============

Turn a per-trade backtest output into portfolio-level performance: a daily
equity curve and the headline risk/return statistics a strategy writeup needs
(annualised return, volatility, Sharpe, Sortino, max drawdown, Calmar, hit rate).

Weight modes
------------
amortised (default)
    Each trade's holding-period return is spread uniformly over every calendar
    day it is open: daily_contribution = metric / duration_days. Capital base
    per day = actual number of concurrent open positions on that day. This gives
    a realistic daily P&L series with no artificial spikes on close dates.

    Limitations that remain: (a) linear amortisation overstates early-period
    P&L for mean-reversion trades that typically earn most at convergence;
    (b) cross-pair correlation is not modelled, so vol and Sharpe are still
    optimistic vs a fully correlated book.

equal (legacy)
    Sums all closing-day holding-period returns and divides by --max-concurrent
    (fixed, default 20). Severely inflates Sharpe: all P&L is booked on the
    single close day, and zero-return days suppress volatility.

capital (legacy)
    Same as equal but divides by the actual concurrent open-position count on
    each day. Still books all P&L on close day.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASTREAM = _REPO_ROOT / "datastream"
_OUTPUT_ROOTS = [
    _DATASTREAM / "data" / "walkforward_output",
    _REPO_ROOT / "review" / "output",
    _REPO_ROOT / "research" / "output",
    _REPO_ROOT / "data" / "backtest",
]
_TRADES_NAMES = ["trades_oos.csv", "trades.csv", "trades.parquet"]


def find_latest_trades() -> Optional[Path]:
    candidates: list[Path] = []
    for root in _OUTPUT_ROOTS:
        if not root.exists():
            continue
        for name in _TRADES_NAMES:
            candidates.extend(root.rglob(name))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
    for col in ("open_date", "close_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# -----------------------------------------------------------------------------
# Daily portfolio return — amortised (primary)
# -----------------------------------------------------------------------------
def daily_returns_amortised(
    trades: pd.DataFrame,
    metric: str,
    capital_slots: Optional[int] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Pro-rate each trade's holding-period return uniformly over the days it is
    open. Capital base per day = observed concurrent open positions.

    Returns
    -------
    ret : pd.Series
        Daily portfolio return (fraction of deployed capital).
    concurrent : pd.Series
        Number of open positions on each calendar day.
    """
    if metric not in trades.columns:
        raise KeyError(f"metric '{metric}' not in trades: {list(trades.columns)}")

    closed = trades[~trades.get("was_aborted", pd.Series(False, index=trades.index)).astype(bool)].copy()
    if closed.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    closed["open_date"] = pd.to_datetime(closed["open_date"]).dt.normalize()
    closed["close_date"] = pd.to_datetime(closed["close_date"]).dt.normalize()

    date_range = pd.date_range(closed["open_date"].min(), closed["close_date"].max(), freq="D")
    n = len(date_range)
    date_idx: dict[pd.Timestamp, int] = {d: i for i, d in enumerate(date_range)}

    pnl = np.zeros(n)
    count = np.zeros(n, dtype=np.int32)

    for row in closed[["open_date", "close_date", "duration_days", metric]].itertuples(index=False):
        dur = max(int(row.duration_days), 1)
        daily_r = getattr(row, metric) / dur
        o = date_idx.get(row.open_date)
        c = date_idx.get(row.close_date)
        if o is None or c is None:
            continue
        pnl[o : c + 1] += daily_r
        count[o : c + 1] += 1

    concurrent = pd.Series(count, index=date_range, name="concurrent_positions")
    # Fixed book size = peak observed concurrent positions (or user override).
    # Days with fewer open trades have idle capital — correctly earns 0 on those slots.
    # This is the most optimistic capital model (minimum required capital).
    # Use capital_slots=total_pairs (e.g. 224) for a more conservative view.
    cap_base = capital_slots if capital_slots else max(int(concurrent.max()), 1)
    ret = pd.Series(pnl / cap_base, index=date_range, name="daily_return")
    ret.index.name = "date"
    return ret, concurrent


# -----------------------------------------------------------------------------
# Daily portfolio return — legacy modes
# -----------------------------------------------------------------------------
def daily_returns_legacy(
    trades: pd.DataFrame,
    metric: str,
    weight: str = "equal",
    max_concurrent: int = 20,
) -> pd.Series:
    """Original implementation retained for comparison. Books all P&L on the
    close date — inflates Sharpe via zero-day vol suppression."""
    if metric not in trades.columns:
        raise KeyError(f"metric '{metric}' not in trades: {list(trades.columns)}")

    aborted_col = trades.get("was_aborted", pd.Series(False, index=trades.index))
    closed = trades[~aborted_col.astype(bool)].copy()
    if closed.empty:
        return pd.Series(dtype=float)

    full_range = pd.date_range(
        closed["close_date"].min(), closed["close_date"].max(), freq="D"
    )
    daily_pnl = (
        closed.groupby("close_date")[metric].sum()
        .reindex(full_range, fill_value=0.0)
    )

    if weight == "capital" and "open_date" in closed.columns:
        open_counts = pd.Series(0.0, index=full_range)
        for _, row in closed.iterrows():
            span = pd.date_range(row["open_date"], row["close_date"], freq="D")
            open_counts.loc[open_counts.index.isin(span)] += 1
        cap = open_counts.clip(lower=1.0)
        ret = (daily_pnl / cap).fillna(0.0)
    else:
        ret = daily_pnl / max_concurrent

    ret.index.name = "date"
    return ret


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
def performance_stats(ret: pd.Series, concurrent: Optional[pd.Series] = None) -> dict:
    if ret.empty:
        return {"n_days": 0}

    equity = (1.0 + ret).cumprod()
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

    # Hit rate: fraction of *active* days (≥1 position open) with positive return.
    # Excludes idle days so we measure when the book is actually deployed.
    if concurrent is not None:
        active = ret[concurrent > 0]
    else:
        active = ret[ret != 0]
    hit_rate = float((active > 0).mean()) if len(active) else float("nan")

    avg_concurrent = float(concurrent.mean()) if concurrent is not None else float("nan")
    max_concurrent = int(concurrent.max()) if concurrent is not None else None

    return {
        "n_days": int(n_days),
        "n_active_days": int(len(active)),
        "avg_concurrent_positions": round(avg_concurrent, 1),
        "max_concurrent_positions": max_concurrent,
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
# Rolling Sharpe
# -----------------------------------------------------------------------------
def rolling_sharpe(ret: pd.Series, window: int = 252) -> pd.Series:
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
        description="Portfolio-level tearsheet from per-trade backtest output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades", type=Path, default=None)
    p.add_argument("--metric", default="ruce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"])
    p.add_argument("--weight", default="amortised",
                   choices=["amortised", "equal", "capital"],
                   help="amortised=pro-rated MTM (recommended); "
                        "equal/capital=legacy close-day booking")
    p.add_argument("--max-concurrent", type=int, default=20,
                   help="capital-budget cap for legacy --weight equal mode only")
    p.add_argument("--capital-slots", type=int, default=None,
                   help="override book size for --weight amortised (default: max observed "
                        "concurrent positions). Set to total pairs (e.g. 224) for a "
                        "conservative view where most capital is idle.")
    p.add_argument("--rolling-window", type=int, default=252)
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "review" / "output"
                   / f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.trades is None:
        args.trades = find_latest_trades()
        if args.trades is None:
            log.error("no trades file found — pass --trades explicitly")
            return 2
        log.info("auto-discovered trades file: %s", args.trades)

    trades = load_trades(args.trades)
    log.info("loaded %d trades from %s", len(trades), args.trades)

    if args.weight == "amortised":
        if "open_date" not in trades.columns or "duration_days" not in trades.columns:
            log.error("--weight amortised requires open_date and duration_days columns; "
                      "use --weight equal for older trade files")
            return 2
        ret, concurrent = daily_returns_amortised(trades, args.metric, args.capital_slots)
        stats = performance_stats(ret, concurrent)
    else:
        if args.weight == "equal":
            log.warning("--weight equal books all P&L on the close date — "
                        "Sharpe will be inflated. Use --weight amortised for realistic stats.")
        ret = daily_returns_legacy(trades, args.metric, args.weight, args.max_concurrent)
        concurrent = None
        stats = performance_stats(ret, concurrent=None)

    stats["metric"] = args.metric
    stats["weight"] = args.weight

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not ret.empty:
        equity = (1.0 + ret).cumprod()
        curve_data: dict = {"date": ret.index, "daily_return": ret.values, "equity": equity.values}
        if concurrent is not None:
            curve_data["concurrent_positions"] = concurrent.reindex(ret.index, fill_value=0).values
        pd.DataFrame(curve_data).to_csv(args.out_dir / "equity_curve.csv", index=False)

        rs = rolling_sharpe(ret, args.rolling_window)
        if not rs.dropna().empty:
            pd.DataFrame({"date": rs.index, rs.name: rs.values}).to_csv(
                args.out_dir / "rolling_sharpe.csv", index=False)

    (args.out_dir / "portfolio_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    log.info("=" * 60)
    log.info("PORTFOLIO STATS  (metric=%s  weight=%s)", args.metric, args.weight)
    log.info("=" * 60)
    for k, v in stats.items():
        log.info("  %-28s %s", k, v)
    log.info("=" * 60)
    if args.weight == "amortised":
        log.info("CAVEATS (amortised mode):")
        log.info("  (1) Linear P&L accrual: each trade's return is spread uniformly across")
        log.info("      its holding period. Real mean-reversion trades earn most of their")
        log.info("      return at convergence, not linearly — so daily vol is understated")
        log.info("      and Sharpe is STILL INFLATED vs true mark-to-market.")
        log.info("  (2) Cross-pair correlation not modelled: concurrent ADR pairs share")
        log.info("      Asian market beta and FX moves — true portfolio vol is higher.")
        log.info("  (3) The reliable metrics are the per-trade distributions: median")
        log.info("      RUCE-net, win rate, and duration. Use those for strategy evaluation.")
    log.info("results written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
