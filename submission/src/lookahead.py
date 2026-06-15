"""
Lookahead-bias detector  [Phase 4b].

Black-box test for lookahead bias (per the QF621 prof's algorithm): run the backtest
over X→Y and again over X→Y' (Y' < Y), then check that every overlapping date's target
positions are byte-identical. If cutting off the future changes a past position, the
strategy is reaching into the future — that is lookahead bias, by definition. No
code-reading or paper trading required.

We represent the daily target position as a per-PAIR signed value in {-1, 0, +1}
(+1 = long spread = long A / short B), reconstructed from the `Trade` records each run
already produces. A trade holds its `direction` on trading days in [entry_date, exit_date)
(flat at exit). The same deterministic reconstruction is applied to both runs, so the test
is about whether the two runs AGREE on overlapping dates — see decisions D4b.2/D4b.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _pair_id(permno_a: int, permno_b: int) -> str:
    """Stable pair key. clusters_to_pairs emits sorted (a<b) pairs, so this is unique."""
    return f"{int(permno_a)}_{int(permno_b)}"


def reconstruct_daily_positions(
    trades: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reconstruct the daily per-pair signed target-position panel from trade records.

    Parameters
    ----------
    trades : DataFrame
        Trade records with columns: permno_a, permno_b, direction, entry_date, exit_date.
        (The `*_trades.parquet` schema produced by every backtest cell.)
    trading_days : DatetimeIndex
        The trading calendar to lay positions onto (sorted, unique).

    Returns
    -------
    DataFrame
        index = trading_days, columns = pair ids, values in {-1, 0, +1}. A trade holds
        `direction` on days in [entry_date, exit_date); 0 everywhere else.
    """
    trading_days = pd.DatetimeIndex(trading_days).sort_values().unique()
    pairs = sorted({_pair_id(a, b) for a, b in zip(trades["permno_a"], trades["permno_b"])})
    panel = pd.DataFrame(0, index=trading_days, columns=pairs, dtype="int8")

    for row in trades.itertuples(index=False):
        pid = _pair_id(row.permno_a, row.permno_b)
        entry = pd.Timestamp(row.entry_date)
        exit_ = pd.Timestamp(row.exit_date)
        # held on [entry, exit): flat at exit and after
        mask = (trading_days >= entry) & (trading_days < exit_)
        panel.loc[mask, pid] = int(row.direction)

    return panel


@dataclass
class LookaheadResult:
    """Outcome of one full-vs-truncated comparison."""

    cut_date: pd.Timestamp
    n_overlap_days: int
    n_pairs_compared: int
    n_mismatched_cells: int
    n_mismatched_days: int
    mismatches: pd.DataFrame   # columns: date, pair, full_pos, short_pos (empty if clean)

    @property
    def passed(self) -> bool:
        return self.n_mismatched_cells == 0

    def summary(self) -> str:
        verdict = "PASS ✅ no lookahead bias" if self.passed else "FAIL ❌ lookahead bias"
        return (
            f"cut {self.cut_date.date()}: {verdict} — "
            f"{self.n_overlap_days} overlap days × {self.n_pairs_compared} pairs, "
            f"{self.n_mismatched_cells} mismatched cells on {self.n_mismatched_days} days"
        )


def compare_position_panels(
    full_positions: pd.DataFrame,
    short_positions: pd.DataFrame,
    cut_date: pd.Timestamp,
) -> LookaheadResult:
    """Compare two daily position panels on their overlapping dates (< cut_date).

    The truncated run's final month force-closes at its own end; to compare only
    genuinely-overlapping behaviour we restrict to dates strictly BEFORE `cut_date`
    (our months are self-contained, so this is the clean overlap region).

    Parameters
    ----------
    full_positions, short_positions : DataFrame
        Output of `reconstruct_daily_positions` for the long and short runs.
    cut_date : Timestamp
        The truncated run's end date (Y').

    Returns
    -------
    LookaheadResult
    """
    cut_date = pd.Timestamp(cut_date)

    # overlap dates = dates present in BOTH, strictly before the cut
    common_dates = full_positions.index.intersection(short_positions.index)
    common_dates = common_dates[common_dates < cut_date]

    # union of pairs (a pair appearing in only one run is itself a mismatch → fill 0)
    all_pairs = full_positions.columns.union(short_positions.columns)
    f = full_positions.reindex(index=common_dates, columns=all_pairs, fill_value=0)
    s = short_positions.reindex(index=common_dates, columns=all_pairs, fill_value=0)

    diff = f.ne(s)
    n_cells = int(diff.to_numpy().sum())

    if n_cells:
        stacked = diff.stack()
        bad = stacked[stacked].index
        rows = [
            {"date": d, "pair": p, "full_pos": int(f.at[d, p]), "short_pos": int(s.at[d, p])}
            for d, p in bad
        ]
        mismatches = pd.DataFrame(rows)
        n_days = int(diff.any(axis=1).sum())
    else:
        mismatches = pd.DataFrame(columns=["date", "pair", "full_pos", "short_pos"])
        n_days = 0

    return LookaheadResult(
        cut_date=cut_date,
        n_overlap_days=len(common_dates),
        n_pairs_compared=len(all_pairs),
        n_mismatched_cells=n_cells,
        n_mismatched_days=n_days,
        mismatches=mismatches,
    )
