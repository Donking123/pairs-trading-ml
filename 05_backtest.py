"""
05_backtest.py  —  Gatev (2006) pairs trading backtest
───────────────────────────────────────────────────────
Key design decisions:

  1. Normalised price series: both formation and trading windows anchored
     to 1.0 at the start of the formation window, so they are continuous
     across the boundary.

  2. Rolling z-score (20-day lookback): adapts to any level-shift in the
     spread between formation and trading periods. Formation stats are used
     as fallback for the first 20 days only.

  3. Entry threshold ±2.5σ (vs ±2.0σ default): trades only the most extreme
     spread deviations, reducing trade count and cost drag.

  4. 5-day cooldown after exit: prevents rapid re-entry as spread oscillates
     around the threshold, which was the main source of overtrading.

  5. MAX_PAIRS = 50 per window, sorted by cointegration p-value.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_RAW, DATA_PROC, DATA_RES,
    EXIT_ZSCORE, STOPLOSS_ZSCORE,
    ROUND_TRIP_BPS, SHORT_BORROW_BPS,
)

PAIRS_DIR = DATA_PROC / "pairs"

MAX_PAIRS          = 20     # top 20 pairs per window (paper's natural SSD count ~22)
ENTRY_ZSCORE       = 2.0    # paper default (was 2.5 — higher threshold was killing trade count)
ROLLING_WINDOW     = 63     # 3-month window — filters noise better than 10d (paper uses 126d)
COOLDOWN_DAYS      = 5      # min days between trades per pair


# ── Normalised price series ───────────────────────────────────────────────────
def normalised_prices(
    ret_formation: pd.Series,
    ret_trading:   pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Price series anchored to 1.0 at start of formation window.
    Trading prices continue from the formation end level so the
    spread computed in formation is directly comparable in trading.
    """
    ret_f = ret_formation.astype("float64").fillna(0.0)
    ret_t = ret_trading.astype("float64").fillna(0.0)
    price_f = (1 + ret_f).cumprod()
    base    = float(price_f.iloc[-1]) if len(price_f) > 0 else 1.0
    price_t = base * (1 + ret_t).cumprod()
    return price_f, price_t


# ── Simulate one pair ─────────────────────────────────────────────────────────
def simulate_pair(
    pair:      dict,
    ret_t_a:   pd.Series,
    ret_t_b:   pd.Series,
    price_t_a: pd.Series,
    price_t_b: pd.Series,
) -> tuple[pd.Series, list]:
    """
    Simulate one pair over its trading window.
    Uses rolling z-score, raised entry threshold, and cooldown between trades.
    """
    hedge       = float(pair["hedge_ratio"])
    spread_mean = float(pair["spread_mean"])
    spread_std  = float(pair["spread_std"])

    if spread_std < 1e-8:
        return pd.Series(dtype=float), []

    spread = price_t_a - hedge * price_t_b

    # Rolling z-score — adapts to level shifts between formation and trading
    roll_mean = spread.rolling(window=ROLLING_WINDOW, min_periods=5).mean().fillna(spread_mean)
    roll_std  = spread.rolling(window=ROLLING_WINDOW, min_periods=5).std().fillna(spread_std)
    roll_std  = roll_std.where(roll_std > 1e-8, spread_std)
    zscore    = (spread - roll_mean) / roll_std
    zscore    = zscore.replace([np.inf, -np.inf], np.nan)

    ret_a = ret_t_a.astype("float64").fillna(0.0)
    ret_b = ret_t_b.astype("float64").fillna(0.0)

    daily_pnl  = pd.Series(0.0, index=zscore.index)
    trades     = []
    position   = 0
    entry_date = None
    entry_z    = None
    last_exit  = None   # tracks cooldown

    cost_one_way   = ROUND_TRIP_BPS / 10_000 / 2
    borrow_per_day = SHORT_BORROW_BPS / 10_000 / 252

    for i in range(len(zscore)):
        z    = zscore.iloc[i]
        date = zscore.index[i]

        if pd.isna(z):
            continue

        r_a = float(ret_a.iloc[i])
        r_b = float(ret_b.iloc[i])

        if position == 0:
            # Cooldown: skip re-entry too soon after last exit
            in_cooldown = (
                last_exit is not None and
                (date - last_exit).days < COOLDOWN_DAYS
            )
            if in_cooldown:
                continue

            if z > ENTRY_ZSCORE:
                position   = -1   # spread too wide: short A, long B
                entry_date = date
                entry_z    = z
                daily_pnl.iloc[i] -= cost_one_way

            elif z < -ENTRY_ZSCORE:
                position   = +1   # spread too narrow: long A, short B
                entry_date = date
                entry_z    = z
                daily_pnl.iloc[i] -= cost_one_way

        else:
            # Daily P&L: dollar-neutral, $1 each leg → 0.5 per $1 GMV
            if position == +1:
                pnl = (r_a - r_b) * 0.5
            else:
                pnl = (r_b - r_a) * 0.5

            pnl -= borrow_per_day
            daily_pnl.iloc[i] += pnl

            exit_profit   = (position == +1 and z >= -EXIT_ZSCORE) or \
                            (position == -1 and z <=  EXIT_ZSCORE)
            exit_stoploss = abs(z) >= STOPLOSS_ZSCORE

            if exit_profit or exit_stoploss:
                daily_pnl.iloc[i] -= cost_one_way
                trades.append({
                    "permno_a":   pair["permno_a"],
                    "permno_b":   pair["permno_b"],
                    "cluster_id": pair["cluster_id"],
                    "entry_date": entry_date,
                    "exit_date":  date,
                    "position":   position,
                    "entry_z":    round(entry_z, 3),
                    "exit_z":     round(float(z), 3),
                    "stop_loss":  bool(exit_stoploss),
                    "trade_pnl":  round(float(daily_pnl.loc[entry_date:date].sum()), 6),
                })
                position   = 0
                entry_date = None
                entry_z    = None
                last_exit  = date

    return daily_pnl, trades


# ── One window ────────────────────────────────────────────────────────────────
def run_window(
    pairs_file:    Path,
    stock_returns: pd.DataFrame,
    window_index:  pd.DataFrame,
) -> tuple[pd.Series, list]:

    date_str = pairs_file.stem.replace("pairs_", "")
    f_start  = pd.Timestamp(date_str[:8])
    f_end    = pd.Timestamp(date_str[9:])

    mask = (window_index["formation_start"] == f_start) & \
           (window_index["formation_end"]   == f_end)
    row  = window_index[mask]
    if row.empty:
        return pd.Series(dtype=float), []

    t_start = row.iloc[0]["trading_start"]
    t_end   = row.iloc[0]["trading_end"]

    pairs_df = pd.read_parquet(pairs_file)
    if pairs_df.empty:
        return pd.Series(dtype=float), []

    # Quality filter: require minimum spread volatility to ensure
    # enough movement to overcome transaction costs
    pairs_df = pairs_df[pairs_df["spread_std"] > 0.01]

    # Rank by composite quality score:
    #   - lower p-value = stronger cointegration
    #   - lower half-life = faster mean-reversion = more return per $ deployed
    #   - higher spread_std = more spread movement to capture
    # Normalise each to [0,1] then combine
    if len(pairs_df) > 0:
        pairs_df = pairs_df.copy()
        # Rank score: lower p and hl = better; higher spread_std = better
        pairs_df["rank_p"]   = pairs_df["pvalue"].rank(pct=True)
        pairs_df["rank_hl"]  = pairs_df["half_life"].rank(pct=True)
        pairs_df["rank_std"] = pairs_df["spread_std"].rank(pct=True, ascending=False)
        pairs_df["quality"]  = (1 - pairs_df["rank_p"] + 
                                1 - pairs_df["rank_hl"] + 
                                pairs_df["rank_std"]) / 3
        pairs_df = pairs_df.sort_values("quality", ascending=False).head(MAX_PAIRS)

        # Inverse half-life weights: faster reverting pairs get more capital
        hl    = pairs_df["half_life"].values
        weights = (1.0 / hl) / (1.0 / hl).sum()
        pairs_df["weight"] = weights

    pairs    = pairs_df.to_dict("records") if len(pairs_df) > 0 else []

    ret_f = stock_returns.loc[f_start:f_end]
    ret_t = stock_returns.loc[t_start:t_end]

    all_pnl      = []
    all_trades   = []
    pair_weights = []

    for pair in pairs:
        pa, pb = pair["permno_a"], pair["permno_b"]
        if pa not in ret_t.columns or pb not in ret_t.columns:
            continue
        if pa not in ret_f.columns or pb not in ret_f.columns:
            continue

        _, price_t_a = normalised_prices(ret_f[pa], ret_t[pa])
        _, price_t_b = normalised_prices(ret_f[pb], ret_t[pb])

        pnl, trades = simulate_pair(
            pair,
            ret_t[pa], ret_t[pb],
            price_t_a, price_t_b,
        )
        if not pnl.empty:
            all_pnl.append(pnl)
            pair_weights.append(float(pair.get("weight", 1.0)))
        all_trades.extend(trades)

    if not all_pnl:
        return pd.Series(dtype=float), []

    # Weighted average: inverse half-life weights
    pnl_df  = pd.concat(all_pnl, axis=1)
    if len(pair_weights) == pnl_df.shape[1]:
        w = np.array(pair_weights)
        w = w / w.sum()
        combined = pnl_df.mul(w, axis=1).sum(axis=1)
    else:
        combined = pnl_df.mean(axis=1)
    return combined, all_trades


# ── Performance metrics ───────────────────────────────────────────────────────
def compute_performance(pnl: pd.Series) -> dict:
    ann     = 252
    ret_ann = pnl.mean() * ann
    vol_ann = pnl.std() * np.sqrt(ann)
    sharpe  = ret_ann / vol_ann if vol_ann > 0 else 0.0
    cum     = (1 + pnl).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    return {
        "annualised_return_pct": round(ret_ann * 100, 2),
        "annualised_vol_pct":    round(vol_ann * 100, 2),
        "sharpe_ratio":          round(sharpe, 3),
        "max_drawdown_pct":      round(dd.min() * 100, 2),
        "total_return_pct":      round((cum.iloc[-1] - 1) * 100, 2),
        "n_trading_days":        len(pnl),
        "pct_days_positive":     round((pnl > 0).mean() * 100, 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def strip_tz(df):
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_convert(None) if idx.tz is not None else idx
        return df

    stock_returns = strip_tz(pd.read_parquet(DATA_RAW / "stock_returns.parquet"))
    stock_returns = stock_returns.astype("float64")

    window_index = pd.read_parquet(DATA_PROC / "window_index.parquet")
    for col in ["formation_start", "formation_end", "trading_start", "trading_end"]:
        window_index[col] = pd.to_datetime(window_index[col])

    pairs_files = sorted(PAIRS_DIR.glob("pairs_*.parquet"))
    if not pairs_files:
        raise FileNotFoundError(
            f"No pair files in {PAIRS_DIR}. Run 04_cointegration.py first."
        )

    print(f"Running backtest across {len(pairs_files)} windows…")
    print(f"  Entry: ±{ENTRY_ZSCORE}σ  |  Exit: ±{EXIT_ZSCORE}σ  |  "
          f"Stop: ±{STOPLOSS_ZSCORE}σ  |  Cooldown: {COOLDOWN_DAYS}d  |  "
          f"Max pairs: {MAX_PAIRS}\n")

    all_pnl_series = []
    all_trades     = []

    for pf in tqdm(pairs_files, desc="  Windows"):
        pnl, trades = run_window(pf, stock_returns, window_index)
        if not pnl.empty:
            all_pnl_series.append(pnl)
        all_trades.extend(trades)

    if not all_pnl_series:
        print("No P&L generated.")
        exit(1)

    daily_pnl_full = (
        pd.concat(all_pnl_series)
        .groupby(level=0)
        .mean()
        .sort_index()
    )

    daily_pnl_full.to_frame("pnl").to_parquet(DATA_RES / "daily_pnl.parquet")

    if all_trades:
        trade_log = pd.DataFrame(all_trades)
        trade_log.to_parquet(DATA_RES / "trade_log.parquet", index=False)
        n_trades  = len(trade_log)
        win_rate  = (trade_log["trade_pnl"] > 0).mean() * 100
        stop_rate = trade_log["stop_loss"].mean() * 100
        avg_hold  = (trade_log["exit_date"] - trade_log["entry_date"]).dt.days.mean()
    else:
        n_trades = win_rate = stop_rate = avg_hold = 0

    perf = compute_performance(daily_pnl_full)
    perf.update({
        "n_trades":             n_trades,
        "win_rate_pct":         round(win_rate, 1),
        "stop_loss_rate_pct":   round(stop_rate, 1),
        "avg_holding_days":     round(avg_hold, 1),
        "max_pairs_per_window": MAX_PAIRS,
        "entry_zscore":         ENTRY_ZSCORE,
    })

    pd.DataFrame([perf]).to_csv(DATA_RES / "performance.csv", index=False)

    print("\n── Performance (net of costs) ──────────────────────────")
    for k, v in perf.items():
        print(f"  {k:<28}: {v}")
    print(f"\n  Saved to: {DATA_RES}")
    print("\nNext: python 06_results.py")
