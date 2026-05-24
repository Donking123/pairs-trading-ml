"""
Performance metrics  [Pipeline Stage 5].

Takes the monthly return series produced by backtest.run_backtest and computes the
standard set of risk-adjusted return measures used in the paper:

    Sharpe   = annualised excess return / annualised vol      [paper headline]
    Sortino  = annualised excess return / annualised downside vol
    Calmar   = annualised return / |max drawdown|
    plus    : total return, ann_return, ann_vol, max_drawdown, hit_rate, win_loss

Conventions (paper-faithful):
  * Monthly returns; annualisation factor = 12.
  * Risk-free rate defaults to 0 (matching the paper's gross-return Sharpe).
  * Downside vol uses returns below MAR (default 0), normalised to monthly std.
  * Sample std with ddof=1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class PerformanceMetrics:
    """All headline metrics computed from a monthly return series."""
    n_months: int
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float          # negative number (e.g., -0.18 = 18% drawdown)
    max_drawdown_start: pd.Timestamp | None
    max_drawdown_end: pd.Timestamp | None
    hit_rate: float              # fraction of months with return > 0
    best_month: float
    worst_month: float
    avg_win: float               # mean of positive months
    avg_loss: float              # mean of negative months
    win_loss_ratio: float        # |avg_win / avg_loss|

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("max_drawdown_start", "max_drawdown_end"):
            if d[k] is not None:
                d[k] = pd.Timestamp(d[k]).isoformat()
        return d


def _annualise_return(monthly_returns: pd.Series) -> float:
    total = float((1 + monthly_returns).prod() - 1)
    n = len(monthly_returns)
    if n == 0:
        return float("nan")
    return float((1 + total) ** (MONTHS_PER_YEAR / n) - 1)


def _annualise_vol(monthly_returns: pd.Series) -> float:
    return float(monthly_returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def _max_drawdown(monthly_returns: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Return (max_drawdown, peak_date, trough_date). MDD is a NON-positive float."""
    if len(monthly_returns) == 0:
        return 0.0, None, None
    cum = (1 + monthly_returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    trough = drawdown.idxmin()
    mdd = float(drawdown.loc[trough])
    if mdd >= 0:
        return 0.0, None, None
    peak = cum.loc[:trough].idxmax()
    return mdd, peak, trough


def compute_metrics(
    monthly_returns: pd.Series,
    rf_rate_annual: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full performance battery on a monthly return series.

    Parameters
    ----------
    monthly_returns : Series
        Indexed by month-end timestamps; values are arithmetic returns
        (e.g. 0.012 = +1.2% that month).
    rf_rate_annual : float
        Annualised risk-free rate. Default 0 = paper-faithful gross Sharpe.
    """
    if not isinstance(monthly_returns, pd.Series):
        raise TypeError("monthly_returns must be a pandas Series")
    monthly_returns = monthly_returns.dropna()
    n = len(monthly_returns)
    if n < 2:
        raise ValueError(f"need >= 2 months to compute metrics, got {n}")

    rf_monthly = (1 + rf_rate_annual) ** (1 / MONTHS_PER_YEAR) - 1
    excess = monthly_returns - rf_monthly

    total = float((1 + monthly_returns).prod() - 1)
    ann_return = _annualise_return(monthly_returns)
    ann_vol = _annualise_vol(monthly_returns)

    # Sharpe: annualised excess return / annualised vol. Use a small float
    # tolerance instead of strict >0 — pandas .std() on a constant series can
    # return ~1e-18 instead of exactly 0, which would give a bogus huge Sharpe.
    _ZERO_VOL = 1e-12
    sharpe = (
        float(excess.mean() * MONTHS_PER_YEAR / ann_vol)
        if ann_vol > _ZERO_VOL else float("nan")
    )

    # Sortino: same numerator, downside vol denominator
    downside = excess.clip(upper=0)
    downside_vol = float(np.sqrt((downside ** 2).mean()) * np.sqrt(MONTHS_PER_YEAR))
    sortino = (
        float(excess.mean() * MONTHS_PER_YEAR / downside_vol)
        if downside_vol > _ZERO_VOL else float("nan")
    )

    mdd, mdd_start, mdd_end = _max_drawdown(monthly_returns)
    calmar = float(ann_return / abs(mdd)) if mdd != 0 else float("nan")

    pos = monthly_returns[monthly_returns > 0]
    neg = monthly_returns[monthly_returns < 0]
    avg_win = float(pos.mean()) if len(pos) else 0.0
    avg_loss = float(neg.mean()) if len(neg) else 0.0
    win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else float("nan")

    return PerformanceMetrics(
        n_months=n,
        total_return=total,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        max_drawdown_start=mdd_start,
        max_drawdown_end=mdd_end,
        hit_rate=float((monthly_returns > 0).mean()),
        best_month=float(monthly_returns.max()),
        worst_month=float(monthly_returns.min()),
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
    )


def format_metrics(m: PerformanceMetrics) -> str:
    """Pretty-print metrics for human consumption."""
    lines = [
        f"  n_months         : {m.n_months}",
        f"  total return     : {m.total_return:+.1%}",
        f"  annualised ret   : {m.ann_return:+.2%}",
        f"  annualised vol   : {m.ann_vol:.2%}",
        f"  Sharpe (rf=0)    : {m.sharpe:.3f}    (paper target: 0.88)",
        f"  Sortino          : {m.sortino:.3f}",
        f"  Calmar           : {m.calmar:.3f}",
        f"  max drawdown     : {m.max_drawdown:.1%}",
    ]
    if m.max_drawdown_start and m.max_drawdown_end:
        lines.append(
            f"  drawdown period  : {m.max_drawdown_start.date()} → {m.max_drawdown_end.date()}"
        )
    lines.extend([
        f"  hit rate         : {m.hit_rate:.1%}",
        f"  best month       : {m.best_month:+.2%}",
        f"  worst month      : {m.worst_month:+.2%}",
        f"  avg win          : {m.avg_win:+.2%}",
        f"  avg loss         : {m.avg_loss:+.2%}",
        f"  win/loss ratio   : {m.win_loss_ratio:.2f}",
    ])
    return "\n".join(lines)
