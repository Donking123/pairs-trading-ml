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

    # No flags needed: auto-discovers the latest trades file.
    python review/portfolio.py

    # Or point at a specific run:
    python review/portfolio.py \
        --trades datastream/data/walkforward_output/walkforward_XXXX/trades_oos.csv \
        --metric ruce_net \
        --out-dir datastream/data/walkforward_output/walkforward_XXXX

Accepts either the ``trades_oos.csv`` from the walk-forward driver or the
``trades.csv`` / ``trades.parquet`` from ``datastream/run_backtest.py``.
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
# Output roots that may contain walkforward_* run folders with a trades file.
_OUTPUT_ROOTS = [
    _DATASTREAM / "data" / "walkforward_output",
    _REPO_ROOT / "review" / "output",
    _REPO_ROOT / "research" / "output",
]
# Trades filenames in order of preference.
_TRADES_NAMES = ["trades_oos.csv", "trades.csv", "trades.parquet"]


def find_latest_trades() -> Optional[Path]:
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


# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trades file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # Force ticker/id columns to string: numeric-looking tickers (e.g.
        # "932939") are otherwise inferred as int64 on read and then fail to
        # match the parquet's string `ticker` column, so the MTM panel reports
        # "missing price series" and silently drops the pair. This only bit
        # small all-numeric pair subsets (e.g. a high liquidity floor leaving a
        # handful of numeric tickers); mixed sets stayed object-typed by luck.
        # Reading as str also preserves any leading zeros. dtype keys absent
        # from the file are ignored by pandas.
        df = pd.read_csv(path, dtype={"pair_id": str, "adr_ticker": str,
                                      "underlying_ticker": str})
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
# Daily mark-to-market portfolio return series (honest Sharpe)
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


def build_panel(
    adr_prices: Path,
    global_prices: Path,
    fx_rates: Path,
    pairs: Path,
) -> dict:
    """Per-pair daily price panel (adr_close, local_close_usd, ...) using the
    *exact* construction the backtest uses, so MTM returns reconcile to the
    per-trade roce/ruce. Reuses ``run_backtest.load_panel`` / ``load_pairs``."""
    rb = _load_module("ds_run_backtest", _DATASTREAM / "run_backtest.py")
    pair_objs = rb.load_pairs(pairs)
    return rb.load_panel(adr_prices, global_prices, fx_rates, pair_objs)


def daily_returns_mtm(
    trades: pd.DataFrame,
    panel: dict,
    metric: str,
    weight: str = "equal",
    max_concurrent: int = 20,
) -> pd.Series:
    """
    Build a **true daily mark-to-market** portfolio return series.

    Unlike :func:`daily_returns`, which dumps each trade's whole holding-period
    return onto its ``close_date`` (scattering correlated bets across different
    days and hiding cross-pair correlation), this marks every open position to
    market each trading day from the price panel. Pairs that move together on the
    same day therefore add to portfolio variance — producing an honest, lower
    Sharpe rather than one inflated by a phantom diversification benefit.

    For each trade, the daily P&L (as a fraction of one capital unit) is

        r_adr(d) = -w_adr * (adr_close[d]      - adr_close[d-1])      / adr_open_price
        r_loc(d) =          (local_open[d+1]   - local_open[d])       / local_open_price

    where ``w_adr`` is 2 for ruce (which double-counts the ADR leg) and 1 for
    roce. A small per-trade residual (the realised ``metric`` minus the raw
    price-move sum — i.e. roll, borrow and effective-price endpoint effects) is
    spread evenly across the holding days so each trade's daily series sums
    exactly to its realised net return. The capital base mirrors
    :func:`daily_returns`.

    Synchronous local-leg marks
    ---------------------------
    An Asian ADR's two legs never print at the same instant: the local close[d]
    is ~15h BEFORE the ADR close[d] (local ~15:00 local ≈ 01:00 ET; ADR 16:00 ET
    ≈ 06:00 next-day local). Pairing ``adr_close[d]`` with ``local_close[d]``
    would mark a stale local leg, and the spread would mean-revert partly because
    the local *catches up the next day* — an untradeable smoothing that
    understates daily vol and inflates Sharpe.

    To keep the legs time-aligned the local leg is marked at
    ``local_open_usd[d+1]`` — the soonest tradeable local price after the ADR
    close, and the one the backtest's *fills* already use. The local accrues from
    the 2nd bar (k>=1): it is bought at the D+1 open, and its first synchronous
    mark (vs adr_close[D+1]) is the D+2 open.
    """
    if metric not in trades.columns:
        raise KeyError(f"metric '{metric}' not in trades columns: {list(trades.columns)}")

    adr_w = 2.0 if metric.startswith("ruce") else 1.0
    closed = trades[~trades.get("was_aborted", False).astype(bool)].copy()
    if closed.empty:
        return pd.Series(dtype=float)

    # Synchronous local marks: pair adr_close[d] with the NEXT local open
    # (local_open_usd[d+1]); build the shifted series once per pair. Fall back to
    # the stale close on the last bar / when open prices are unavailable.
    sync_local: dict = {}
    for pid, df in panel.items():
        lo = (df["local_open_usd"] if "local_open_usd" in df.columns
              else df["local_close_usd"])
        sync_local[pid] = lo.shift(-1).fillna(df["local_close_usd"])

    pnl: dict = {}        # date -> summed daily P&L across open positions
    open_cnt: dict = {}   # date -> number of positions open that day
    matched = unmatched = lumped = 0

    def _add(d, v):
        pnl[d] = pnl.get(d, 0.0) + v

    for _, t in closed.iterrows():
        df = panel.get(t["pair_id"])
        if df is None:
            unmatched += 1
            continue
        o, c = pd.Timestamp(t["open_date"]), pd.Timestamp(t["close_date"])
        win = df.loc[(df.index >= o) & (df.index <= c)]
        for d in win.index:                       # capital is tied up every bar held
            open_cnt[d] = open_cnt.get(d, 0.0) + 1.0
        if len(win) < 2:
            # cannot mark to market (single bar) — lump realised return on close
            _add(c, float(t[metric]))
            lumped += 1
            continue
        a = win["adr_close"].to_numpy()
        # time-aligned local: local_open_usd[d+1] (contemporaneous with adr_close[d])
        l = sync_local[t["pair_id"]].loc[win.index].to_numpy()
        wdates = win.index
        adr_open = float(t["adr_open_price"]) or float(a[0])
        loc_open = float(t["local_open_price_usd"]) or float(l[min(1, len(l) - 1)])
        raw = np.zeros(len(win))
        for k in range(1, len(win)):
            r_adr = -adr_w * (a[k] - a[k - 1]) / adr_open if adr_open else 0.0
            # local bought at the D+1 open, so it accrues from the 2nd bar (k>=1)
            r_loc = (l[k] - l[k - 1]) / loc_open if (k >= 1 and loc_open > 0) else 0.0
            raw[k] = r_adr + r_loc
        # reconcile to realised net metric (absorbs roll/borrow/endpoint effects)
        adj = (float(t[metric]) - raw.sum()) / (len(win) - 1)
        for k in range(1, len(win)):
            _add(wdates[k], raw[k] + adj)
        matched += 1

    if not pnl:
        return pd.Series(dtype=float)

    s = pd.Series(pnl).sort_index()
    # genuine trading-day index = union of panel dates within the active span
    all_dates = sorted(set().union(*[df.index for df in panel.values()]))
    idx = pd.DatetimeIndex([d for d in all_dates if s.index.min() <= d <= s.index.max()])
    daily_pnl = s.reindex(idx, fill_value=0.0)

    if weight == "capital":
        # Floor the capital base at `max_concurrent` reserved slots. A bare
        # floor of 1.0 let a single open position on a thin day become the whole
        # book — a +20% trade became a +20% portfolio day — which compounded to
        # absurd equity. Flooring at the nominal book size means under-deployed
        # days hold idle cash (as in the `equal` base) while still scaling up
        # when concurrency exceeds the reserve.
        floor = float(max(max_concurrent, 1))
        base = pd.Series(open_cnt).reindex(idx, fill_value=0.0).clip(lower=floor)
        ret = (daily_pnl / base).fillna(0.0)
    else:
        ret = daily_pnl / max_concurrent

    ret.index.name = "date"
    ret.attrs["mtm_trades_marked"] = matched
    ret.attrs["mtm_trades_lumped"] = lumped
    ret.attrs["mtm_trades_unmatched"] = unmatched
    ret.attrs["mtm_local_timing"] = "synchronous_open_d+1"
    total = matched + lumped + unmatched
    ret.attrs["mtm_coverage"] = round((matched + lumped) / total, 4) if total else 0.0
    log.info("MTM daily returns [%s local timing]: %d trades marked, %d lumped "
             "(single-bar), %d unmatched pairs (%.0f%% coverage); %d trading days",
             ret.attrs["mtm_local_timing"], matched, lumped, unmatched,
             100 * ret.attrs["mtm_coverage"], len(ret))
    return ret


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
def _nw_auto_lags(n: int) -> int:
    """Newey-West (1994) automatic Bartlett bandwidth: floor(4*(n/100)^(2/9))."""
    return max(1, min(n - 1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))))


def _newey_west_lrv(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of a 1-D series.

        LRV = g0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * g_k

    where g_k is the lag-k autocovariance (1/n normalisation). With positive
    autocorrelation the LRV exceeds the i.i.d. variance g0, so the resulting
    annual Sharpe is lower than the naive sqrt(252) figure — the honest measure
    when daily P&L is serially dependent (overlapping positions, slow drift)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return float("nan")
    lags = max(0, min(lags, n - 1))
    g0 = float(x @ x) / n
    lrv = g0
    for k in range(1, lags + 1):
        gk = float(x[:-k] @ x[k:]) / n
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    return lrv


def performance_stats(ret: pd.Series, rf: float = 0.0,
                      nw_lags: Optional[int] = None) -> dict:
    """Headline portfolio stats.

    ``rf``      : annual risk-free rate. The Sharpe/Sortino/NW-Sharpe numerators
                  use the *excess* daily return ``ret - rf/252``. Default 0.0
                  reproduces the gross-of-rf figures exactly.
    ``nw_lags`` : Bartlett-kernel lag truncation for the Newey-West long-run
                  variance behind ``sharpe_nw``. ``None`` uses the Newey-West
                  (1994) automatic bandwidth. Raise it (e.g. 252) to let
                  multi-month serial dependence enter the variance.
    """
    if ret.empty:
        return {"n_days": 0}

    equity = (1.0 + ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    n_days = len(ret)
    # Derive the holding span from the actual calendar dates of the return index
    # so annualisation is correct regardless of how `ret` is indexed. The MTM
    # path indexes on trading days (~252/yr); dividing n_days by 365.25 treated
    # a ~14.6y trading-day series as ~10y and inflated annualised return/Calmar.
    span_days = (ret.index[-1] - ret.index[0]).days if n_days > 1 else 0
    years = span_days / 365.25 if span_days > 0 else n_days / 365.25

    rf_daily = rf / TRADING_DAYS
    excess = ret - rf_daily
    std = float(ret.std(ddof=1))

    ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    ann_vol = float(std * np.sqrt(TRADING_DAYS))
    sharpe = float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")

    # Newey-West (serial-correlation-adjusted) annual Sharpe.
    lags = _nw_auto_lags(n_days) if nw_lags is None else max(0, min(nw_lags, n_days - 1))
    lrv = _newey_west_lrv(excess.to_numpy(), lags)
    sharpe_nw = (float(excess.mean()) * np.sqrt(TRADING_DAYS) / np.sqrt(lrv)
                 if lrv and lrv > 0 else float("nan"))

    downside = excess[excess < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else float("nan")
    sortino = float(excess.mean() * TRADING_DAYS / dvol) if dvol and dvol > 0 else float("nan")

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
        "rf": round(float(rf), 6),
        "sharpe": round(sharpe, 4),
        "sharpe_nw": round(sharpe_nw, 4),
        "nw_lags": int(lags),
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
    p.add_argument("--trades", type=Path, default=None,
                   help="trades_oos.csv / trades.csv / trades.parquet "
                        "(auto-discovers latest under datastream/data/walkforward_output if omitted)")
    p.add_argument("--metric", default="roce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"],
                   help="per-trade return column to compound. Default roce_net "
                        "(= local_ret + adr_ret) is the economic pair P&L; ruce "
                        "double-counts the ADR leg (local_ret + 2*adr_ret) and "
                        "inflates the return/Calmar level.")
    p.add_argument("--weight", default="equal", choices=["equal", "capital"])
    p.add_argument("--max-concurrent", type=int, default=20,
                   help="capital-budget cap (only used with --weight capital)")
    p.add_argument("--rolling-window", type=int, default=252,
                   help="calendar-day window for rolling Sharpe output")
    p.add_argument("--rf", type=float, default=0.0,
                   help="annual risk-free rate; Sharpe/Sortino/NW-Sharpe use "
                        "excess return (ret - rf/252). Default 0.0 = gross of rf.")
    p.add_argument("--nw-lags", type=int, default=None,
                   help="Bartlett lag truncation for the Newey-West serial-"
                        "correlation-adjusted Sharpe (sharpe_nw). Default: "
                        "Newey-West (1994) automatic bandwidth. Set e.g. 252 to "
                        "capture multi-month dependence in annual risk.")
    # --- mark-to-market daily returns (honest Sharpe; default on) -------------
    p.add_argument("--mtm", dest="mtm", action="store_true", default=True,
                   help="mark every open position to market each day from the "
                        "price panel so cross-pair correlation enters portfolio "
                        "vol (default; needs the price parquets + pairs registry)")
    p.add_argument("--no-mtm", dest="mtm", action="store_false",
                   help="use the legacy close-date lumped returns (overstates "
                        "Sharpe by hiding cross-pair correlation)")
    p.add_argument("--pairs", type=Path,
                   default=_REPO_ROOT / "config/pairs/asian_adr_pairs.json")
    p.add_argument("--adr-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/adr/adr_prices.parquet")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "review" / "output"
                   / f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                   help="directory for equity_curve.csv + portfolio_stats.json "
                        "(default: review/output/portfolio_<timestamp>)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # --- resolve trades path (auto-discover latest if not supplied) -----------
    if args.trades is None:
        args.trades = find_latest_trades()
        if args.trades is None:
            log.error("no trades file found under %s — pass --trades explicitly",
                      ", ".join(str(r) for r in _OUTPUT_ROOTS))
            return 2
        log.info("auto-discovered trades file: %s", args.trades)

    trades = load_trades(args.trades)
    log.info("loaded %d trades from %s", len(trades), args.trades)

    # --- daily return series: mark-to-market (honest) or legacy lumped --------
    ret = None
    ret_mode = "lumped_close_date"
    if args.mtm:
        missing = [p for p in (args.adr_prices, args.global_prices,
                               args.fx_rates, args.pairs) if not p.exists()]
        if missing:
            log.warning("MTM data missing (%s) — falling back to lumped close-date "
                        "returns; pass --no-mtm to silence", ", ".join(map(str, missing)))
        else:
            try:
                panel = build_panel(args.adr_prices, args.global_prices,
                                    args.fx_rates, args.pairs)
                ret = daily_returns_mtm(trades, panel, args.metric,
                                        args.weight, args.max_concurrent)
                ret_mode = "mark_to_market"
            except Exception as exc:
                log.warning("MTM build failed (%s) — falling back to lumped "
                            "close-date returns", exc)
                ret = None
    if ret is None or ret.empty:
        ret = daily_returns(trades, args.metric, args.weight, args.max_concurrent)
        ret_mode = "lumped_close_date"

    stats = performance_stats(ret, rf=args.rf, nw_lags=args.nw_lags)
    stats["metric"] = args.metric
    stats["weight"] = args.weight
    stats["return_mode"] = ret_mode
    for k in ("mtm_trades_marked", "mtm_trades_unmatched", "mtm_coverage",
              "mtm_local_timing"):
        if k in ret.attrs:
            stats[k] = ret.attrs[k]

    out_dir = args.out_dir
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
