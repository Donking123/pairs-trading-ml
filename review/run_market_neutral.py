#!/usr/bin/env python3
"""
run_market_neutral.py
=====================

Tests whether the strategy's edge is genuinely market-neutral or whether
it captures a hidden long-EM-equity or long-local-currency beta.

Two OLS regressions on the strategy's daily portfolio returns:

1. **Local equity regression**: daily portfolio return vs. an equal-weight
   daily return of all underlying local stocks in the pair registry. A
   significant positive beta means the strategy is effectively long EM
   equities in disguise.

2. **FX regression**: daily portfolio return vs. an equal-weight daily
   return of all underlying local currencies vs. USD. A positive beta
   flags unhedged FX exposure.

Both proxies are built internally from the same price parquet files —
no external market data is required. Alternatively, supply
``--market-returns`` as a CSV with columns ``date,return`` (e.g. MSCI EM
daily returns from a data vendor).

A significant positive intercept (alpha) after stripping market and FX
beta is evidence of genuine market-neutral edge.

Usage
-----
::

    # No flags needed: auto-discovers the latest trades file and uses the
    # default pairs registry + global prices + fx parquet.
    python review/run_market_neutral.py

    # Or point at a specific run / override any default:
    python review/run_market_neutral.py \\
        --trades    datastream/data/walkforward_output/walkforward_XXXX/trades_oos.csv \\
        --global-prices datastream/data/parquet/global/global_prices.parquet \\
        --fx-rates  datastream/data/parquet/fx/fx_rates.parquet \\
        --pairs     config/pairs/asian_adr_pairs.json \\
        --out-dir   datastream/data/walkforward_output/walkforward_XXXX

    # Use an external benchmark instead of the internal proxy:
    python review/run_market_neutral.py \\
        --market-returns /path/to/msci_em_daily.csv
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
log = logging.getLogger("market_neutral")

TRADING_DAYS = 252

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


_port = _load_module("research_portfolio", _REPO_ROOT / "review" / "portfolio.py")
_screen = _load_module("ds_screening_mn", _DATASTREAM / "run_asian_adr_screening.py")


# -----------------------------------------------------------------------------
# Market and FX proxies built from price parquets
# -----------------------------------------------------------------------------
def build_local_equity_proxy(
    global_prices: pd.DataFrame,
    pairs_data: list[dict],
) -> pd.Series:
    """Equal-weight daily return of all underlying local stocks.
    Serves as a proxy for the EM local-equity market factor."""
    tickers = list({p["underlying_ticker"] for p in pairs_data})
    rets: dict[str, pd.Series] = {}
    for ticker in tickers:
        px = (global_prices[global_prices["ticker"] == ticker]
              .set_index("marketdate")["close"]
              .sort_index())
        if len(px) > 5:
            rets[ticker] = px.pct_change(fill_method=None)
    if not rets:
        return pd.Series(dtype=float, name="local_equity_proxy")
    df = pd.DataFrame(rets).dropna(how="all")
    proxy = df.mean(axis=1)
    proxy.name = "local_equity_proxy"
    log.info("local equity proxy: %d tickers, %d dates", len(rets), len(df))
    return proxy


def build_fx_proxy(fx_rates: pd.DataFrame, pairs_data: list[dict]) -> pd.Series:
    """Equal-weight daily return of all underlying local currencies vs. USD.
    Positive FX beta would mean the strategy is implicitly long local ccys."""
    ccys = list({p["underlying_currency"] for p in pairs_data
                 if p.get("underlying_currency") and p["underlying_currency"] != "USD"})
    fx_wide = (fx_rates
               .pivot_table(index="date", columns="base_currency", values="mid", aggfunc="last")
               .sort_index())
    rets: dict[str, pd.Series] = {}
    for ccy in ccys:
        if ccy in fx_wide.columns:
            rets[ccy] = fx_wide[ccy].pct_change(fill_method=None)
    if not rets:
        return pd.Series(dtype=float, name="fx_proxy")
    df = pd.DataFrame(rets).dropna(how="all")
    proxy = df.mean(axis=1)
    proxy.name = "fx_proxy"
    log.info("FX proxy: %d currencies, %d dates", len(rets), len(df))
    return proxy


# -----------------------------------------------------------------------------
# OLS regression (no scipy dependency — pure numpy)
# -----------------------------------------------------------------------------
def ols_regression(y: np.ndarray, x: np.ndarray, label: str = "market") -> dict:
    """OLS: y = alpha + beta*x. Returns coefficients, R², t-statistics.
    Uses numpy lstsq; manually computes SEs from residual variance."""
    n = len(y)
    if n < 10 or n != len(x):
        return {"error": f"insufficient overlap (n={n})", "regressor": label}
    X = np.column_stack([np.ones(n), x])
    try:
        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"error": "lstsq failed", "regressor": label}
    alpha_d, beta = float(params[0]), float(params[1])
    y_hat = X @ params
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mse = ss_res / (n - 2) if n > 2 else float("nan")
    t_alpha = t_beta = float("nan")
    try:
        cov_mat = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov_mat), 0.0))
        t_alpha = alpha_d / se[0] if se[0] > 0 else float("nan")
        t_beta = beta / se[1] if se[1] > 0 else float("nan")
    except np.linalg.LinAlgError:
        pass
    corr = float(np.corrcoef(y, x)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return {
        "regressor": label,
        "n_obs": n,
        "alpha_daily": round(alpha_d, 8),
        "alpha_annualised": round(alpha_d * TRADING_DAYS, 6),
        "beta": round(beta, 6),
        "r_squared": round(r2, 6),
        "t_stat_alpha": round(t_alpha, 4),
        "t_stat_beta": round(t_beta, 4),
        "correlation": round(corr, 6),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Market-neutrality test: regress strategy daily returns vs equity/FX proxies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trades", type=Path, default=None,
                   help="trades_oos.csv / trades.csv / trades.parquet "
                        "(auto-discovers latest under datastream/data/walkforward_output if omitted)")
    p.add_argument("--pairs", type=Path,
                   default=_REPO_ROOT / "config/pairs/asian_adr_pairs.json")
    p.add_argument("--global-prices", type=Path,
                   default=_DATASTREAM / "data/parquet/global/global_prices.parquet")
    p.add_argument("--fx-rates", type=Path,
                   default=_DATASTREAM / "data/parquet/fx/fx_rates.parquet")
    p.add_argument("--market-returns", type=Path, default=None,
                   help="optional external benchmark CSV with columns 'date,return'; "
                        "overrides the internal local-equity proxy")
    p.add_argument("--metric", default="ruce_net",
                   choices=["roce", "ruce", "roce_net", "ruce_net"])
    p.add_argument("--max-concurrent", type=int, default=20,
                   help="capital slots for equal-weight daily-return construction")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO_ROOT / "review" / "output"
                   / f"market_neutral_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # --- resolve trades path (auto-discover latest if not supplied) -----------
    if args.trades is None:
        args.trades = _port.find_latest_trades()
        if args.trades is None:
            log.error("no trades file found under the default walkforward output "
                      "roots — pass --trades explicitly")
            return 2
        log.info("auto-discovered trades file: %s", args.trades)

    # --- build strategy daily returns -----------------------------------------
    trades = _port.load_trades(args.trades)
    log.info("loaded %d trades from %s", len(trades), args.trades)
    port_ret = _port.daily_returns(trades, args.metric, "equal", args.max_concurrent)
    if port_ret.empty:
        log.error("no portfolio returns; nothing to regress")
        return 2
    port_ret.index = pd.to_datetime(port_ret.index)

    # --- market proxy ---------------------------------------------------------
    if args.market_returns is not None:
        mkt_df = pd.read_csv(args.market_returns, parse_dates=["date"])
        mkt = mkt_df.set_index("date")["return"].sort_index()
        mkt.index = pd.to_datetime(mkt.index)
        mkt_label = "external_benchmark"
        log.info("using external market returns: %d observations", len(mkt))
    else:
        if not args.pairs.exists():
            log.error("pairs file not found: %s — pass --pairs or --market-returns", args.pairs)
            return 2
        if not args.global_prices.exists():
            log.error("global prices not found: %s", args.global_prices)
            return 2
        pairs_data = json.loads(args.pairs.read_text())
        global_prices = _screen._load_or_die(args.global_prices, "global prices")
        global_prices["marketdate"] = pd.to_datetime(global_prices["marketdate"]).dt.normalize()
        mkt = build_local_equity_proxy(global_prices, pairs_data)
        mkt.index = pd.to_datetime(mkt.index)
        mkt_label = "local_equity_proxy"

    # --- FX proxy (from internal data) ----------------------------------------
    fx_proxy = pd.Series(dtype=float)
    if args.fx_rates.exists() and args.pairs.exists():
        try:
            pairs_data_fx = json.loads(args.pairs.read_text())
            fx_rates = _screen._load_or_die(args.fx_rates, "FX rates")
            fx_rates["date"] = pd.to_datetime(fx_rates["date"]).dt.normalize()
            fx_proxy = build_fx_proxy(fx_rates, pairs_data_fx)
            fx_proxy.index = pd.to_datetime(fx_proxy.index)
        except Exception as exc:
            log.warning("could not build FX proxy: %s", exc)

    # --- align and regress ----------------------------------------------------
    aligned_mkt = (pd.concat([port_ret.rename("portfolio"), mkt.rename("market")],
                              axis=1, join="inner")
                   .dropna())
    log.info("aligned %d common dates for market regression", len(aligned_mkt))
    result_mkt = ols_regression(
        aligned_mkt["portfolio"].to_numpy(),
        aligned_mkt["market"].to_numpy(),
        label=mkt_label,
    )

    result_fx: dict = {}
    if not fx_proxy.empty:
        aligned_fx = (pd.concat([port_ret.rename("portfolio"), fx_proxy],
                                 axis=1, join="inner")
                      .dropna())
        if len(aligned_fx) >= 10:
            result_fx = ols_regression(
                aligned_fx["portfolio"].to_numpy(),
                aligned_fx[fx_proxy.name].to_numpy(),
                label="fx_proxy",
            )
            log.info("FX regression: beta=%.4f  R²=%.4f  t(alpha)=%.2f",
                     result_fx.get("beta", float("nan")),
                     result_fx.get("r_squared", float("nan")),
                     result_fx.get("t_stat_alpha", float("nan")))

    report = {
        "metric": args.metric,
        "market_regression": result_mkt,
        "fx_regression": result_fx if result_fx else None,
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "market_neutral.json").write_text(json.dumps(report, indent=2, default=str))

    # save aligned data for external plotting
    reg_df = aligned_mkt.copy()
    if not fx_proxy.empty and len(aligned_fx) >= 10:
        reg_df = reg_df.join(aligned_fx[fx_proxy.name], how="left")
    reg_df.to_csv(out_dir / "market_neutral_data.csv")

    log.info("=" * 64)
    log.info("MARKET-NEUTRALITY REGRESSION  (%s vs %s)", args.metric, mkt_label)
    log.info("=" * 64)
    log.info("  n_obs                  %d", result_mkt.get("n_obs", 0))
    log.info("  alpha (daily)          %.6f", result_mkt.get("alpha_daily", float("nan")))
    log.info("  alpha (annualised)     %.4f", result_mkt.get("alpha_annualised", float("nan")))
    log.info("  beta                   %.4f", result_mkt.get("beta", float("nan")))
    log.info("  R²                     %.4f", result_mkt.get("r_squared", float("nan")))
    log.info("  t-stat alpha           %.2f", result_mkt.get("t_stat_alpha", float("nan")))
    log.info("  t-stat beta            %.2f", result_mkt.get("t_stat_beta", float("nan")))
    log.info("  correlation            %.4f", result_mkt.get("correlation", float("nan")))
    beta = result_mkt.get("beta", 0.0) or 0.0
    if abs(beta) > 0.2:
        log.warning("  *** beta=%.3f is non-trivial — strategy has equity market exposure ***",
                    beta)
    else:
        log.info("  beta=%.3f (|β|<0.2) — strategy appears market-neutral", beta)
    log.info("=" * 64)
    log.info("wrote market_neutral.json to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
