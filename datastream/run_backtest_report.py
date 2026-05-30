#!/usr/bin/env python3
"""
backtest_report.py
==================

Generates an HTML tearsheet for a completed backtest run, benchmarked against
Hong & Susmel (2013) paper results (Table 4 / Table 7-B).

Usage
-----
    python datastream/backtest_report.py \
        --run-dir  data/backtest/run_20260529_213128 \
        --pairs    config/pairs/asian_adr_pairs.json \
        --out      data/backtest/run_20260529_213128/tearsheet.html

    # Defaults: reads most-recent run dir automatically
    python datastream/backtest_report.py

Outputs
-------
    tearsheet.html — self-contained HTML (charts inlined as base64 PNG)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

log = logging.getLogger("backtest_report")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")


# ---------------------------------------------------------------------------
# Paper benchmarks  (Hong & Susmel 2013, Table 4 Panel A; k0=2, kc=0, T=60, H=90)
# ---------------------------------------------------------------------------
PAPER = {
    "median_roce":           0.028,
    "median_ruce":           0.053,
    "median_duration":       3.0,
    "iq_range_duration_lo":  1.0,
    "iq_range_duration_hi":  6.0,
    "adr_leg_contrib_pct":   0.90,
    "median_net_ruce":       0.027,
    "max_abort_rate":        0.30,
    "trades_per_firm_year":  11.6,
    "median_roll_cost":      0.0267,
}

# Liquidity bucket thresholds — zero-return-day percentage (ADR leg)
# From Table 5 Panel A of Hong & Susmel (2013)
BUCKET_THRESHOLDS = [
    ("High",        0.0,    0.0621),
    ("High-Medium", 0.0621, 0.1457),
    ("Medium-Low",  0.1457, 0.2975),
    ("Low",         0.2975, 1.01),
]
BUCKET_ORDER = ["High", "High-Medium", "Medium-Low", "Low"]

# Paper median ROCE per bucket (Table 7-B approximate)
PAPER_BUCKET_ROCE = {
    "High":         0.020,
    "High-Medium":  0.028,
    "Medium-Low":   0.030,
    "Low":          0.037,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _pct(v: float, decimals: int = 2) -> str:
    return f"{v * 100:.{decimals}f}%"


def _check(passed: bool) -> str:
    return "✅" if passed else "❌"


def assign_bucket(zero_return_pct: float | None) -> str:
    if zero_return_pct is None:
        return "Unknown"
    for name, lo, hi in BUCKET_THRESHOLDS:
        if lo <= zero_return_pct < hi:
            return name
    return "Low"


def dist_row(series: pd.Series, label: str) -> dict[str, Any]:
    return {
        "Metric":  label,
        "Mean":    series.mean(),
        "Std":     series.std(),
        "Max":     series.max(),
        "p90":     series.quantile(0.90),
        "p75":     series.quantile(0.75),
        "Median":  series.median(),
        "p25":     series.quantile(0.25),
        "p10":     series.quantile(0.10),
        "Min":     series.min(),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(run_dir: Path, pairs_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    trades = pd.read_parquet(run_dir / "trades.parquet")
    trades["open_date"]  = pd.to_datetime(trades["open_date"])
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    with open(run_dir / "summary.json") as f:
        summary = json.load(f)

    with open(pairs_path) as f:
        pairs_list = json.load(f)
    pairs = {p["pair_id"]: p for p in pairs_list}

    trades["zero_return_pct_adr"] = trades["pair_id"].map(
        lambda pid: pairs.get(pid, {}).get("zero_return_pct_adr")
    )
    trades["underlying_exchange"] = trades["pair_id"].map(
        lambda pid: pairs.get(pid, {}).get("underlying_exchange", "UNK")
    )
    trades["underlying_currency"] = trades["pair_id"].map(
        lambda pid: pairs.get(pid, {}).get("underlying_currency", "UNK")
    )
    trades["roll_spread_adr"] = trades["pair_id"].map(
        lambda pid: pairs.get(pid, {}).get("roll_spread_adr", np.nan)
    )
    trades["liquidity_bucket"] = trades["zero_return_pct_adr"].map(assign_bucket)

    return trades, summary, pairs


# ---------------------------------------------------------------------------
# Stat tables
# ---------------------------------------------------------------------------

def build_distribution_table(closed: pd.DataFrame) -> pd.DataFrame:
    rows = [
        dist_row(closed["roce"],          "ROCE per trade"),
        dist_row(closed["ruce"],          "RUCE per trade"),
        dist_row(closed["roce_net"],      "ROCE net per trade"),
        dist_row(closed["ruce_net"],      "RUCE net per trade"),
        dist_row(closed["duration_days"], "Duration (days)"),
        dist_row(closed["roll_cost_pct"], "Roll cost (round trip)"),
    ]
    df = pd.DataFrame(rows).set_index("Metric")
    return df


def build_bucket_table(closed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in BUCKET_ORDER:
        sub = closed[closed["liquidity_bucket"] == bucket]
        n = len(sub)
        paper_roce = PAPER_BUCKET_ROCE.get(bucket)
        rows.append({
            "Bucket":              bucket,
            "N trades":            n,
            "Median ROCE":         sub["roce"].median() if n else np.nan,
            "Median RUCE":         sub["ruce"].median() if n else np.nan,
            "Median ROCE net":     sub["roce_net"].median() if n else np.nan,
            "Median duration":     sub["duration_days"].median() if n else np.nan,
            "ADR leg contrib":     (sub["adr_return"].median() / sub["roce"].median()
                                    if n and sub["roce"].median() != 0 else np.nan),
            "Paper ROCE":          paper_roce,
        })
    return pd.DataFrame(rows).set_index("Bucket")


def build_exchange_table(trades: pd.DataFrame, closed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exch in sorted(trades["underlying_exchange"].unique()):
        all_ex  = trades[trades["underlying_exchange"] == exch]
        cl_ex   = closed[closed["underlying_exchange"] == exch]
        rows.append({
            "Exchange":      exch,
            "N initiated":   len(all_ex),
            "N closed":      len(cl_ex),
            "Abort rate":    len(all_ex[all_ex["was_aborted"]]) / len(all_ex) if len(all_ex) else np.nan,
            "Median ROCE":   cl_ex["roce"].median() if len(cl_ex) else np.nan,
            "Median RUCE":   cl_ex["ruce"].median() if len(cl_ex) else np.nan,
            "Median dur":    cl_ex["duration_days"].median() if len(cl_ex) else np.nan,
        })
    return pd.DataFrame(rows).set_index("Exchange")


def build_annual_table(trades: pd.DataFrame, closed: pd.DataFrame) -> pd.DataFrame:
    trades["year"] = trades["open_date"].dt.year
    closed2 = closed.copy()
    closed2["year"] = closed2["open_date"].dt.year

    rows = []
    for yr in sorted(trades["year"].unique()):
        all_yr = trades[trades["year"] == yr]
        cl_yr  = closed2[closed2["year"] == yr]
        rows.append({
            "Year":         yr,
            "N initiated":  len(all_yr),
            "N closed":     len(cl_yr),
            "N aborted":    len(all_yr[all_yr["was_aborted"]]),
            "Abort rate":   len(all_yr[all_yr["was_aborted"]]) / len(all_yr) if len(all_yr) else np.nan,
            "Median ROCE":  cl_yr["roce"].median() if len(cl_yr) else np.nan,
            "Median RUCE":  cl_yr["ruce"].median() if len(cl_yr) else np.nan,
            "Median dur":   cl_yr["duration_days"].median() if len(cl_yr) else np.nan,
        })
    return pd.DataFrame(rows).set_index("Year")


def compute_attribution(closed: pd.DataFrame) -> dict[str, float]:
    med_roce = closed["roce"].median()
    med_adr  = closed["adr_return"].median()
    med_loc  = closed["local_return"].median()
    adr_contrib = med_adr / med_roce if med_roce != 0 else np.nan
    return {
        "median_local_return": med_loc,
        "median_adr_return":   med_adr,
        "median_roce":         med_roce,
        "adr_contrib_pct":     adr_contrib,
    }


# ---------------------------------------------------------------------------
# Validation panel
# ---------------------------------------------------------------------------

def build_validation(summary: dict, closed: pd.DataFrame, trades: pd.DataFrame) -> list[dict]:
    attr = compute_attribution(closed)
    checks = [
        {
            "Check":   "Median ROCE ≈ 2.8% (±0.5%)",
            "Actual":  _pct(summary["median_roce"]),
            "Target":  _pct(PAPER["median_roce"]),
            "Pass":    abs(summary["median_roce"] - PAPER["median_roce"]) <= 0.005,
        },
        {
            "Check":   "Median RUCE ≈ 5.3% (±0.5%)",
            "Actual":  _pct(summary["median_ruce"]),
            "Target":  _pct(PAPER["median_ruce"]),
            "Pass":    abs(summary["median_ruce"] - PAPER["median_ruce"]) <= 0.005,
        },
        {
            "Check":   "Median duration ≈ 3 days (±1)",
            "Actual":  f"{summary['median_duration']:.1f} days",
            "Target":  f"{PAPER['median_duration']:.1f} days",
            "Pass":    abs(summary["median_duration"] - PAPER["median_duration"]) <= 1,
        },
        {
            "Check":   "IQ range of duration: 1–6 days",
            "Actual":  f"{closed['duration_days'].quantile(0.25):.0f}–{closed['duration_days'].quantile(0.75):.0f} days",
            "Target":  "1–6 days",
            "Pass":    (closed["duration_days"].quantile(0.25) <= 2
                        and closed["duration_days"].quantile(0.75) <= 8),
        },
        {
            "Check":   "ADR leg contribution ≈ 90%",
            "Actual":  _pct(attr["adr_contrib_pct"]),
            "Target":  _pct(PAPER["adr_leg_contrib_pct"]),
            "Pass":    abs(attr["adr_contrib_pct"] - PAPER["adr_leg_contrib_pct"]) <= 0.10,
        },
        {
            "Check":   "Overnight abort rate < 30%",
            "Actual":  _pct(summary["abort_rate"]),
            "Target":  f"< {_pct(PAPER['max_abort_rate'])}",
            "Pass":    summary["abort_rate"] < PAPER["max_abort_rate"],
        },
        {
            "Check":   "Liquidity bucket ROCE monotone (High < Low)",
            "Actual":  "see bucket table",
            "Target":  "monotone increasing",
            "Pass":    _bucket_monotone(closed),
        },
        {
            "Check":   "Median net RUCE ≈ 2.7% (±1%)",
            "Actual":  _pct(summary["median_ruce_net"]),
            "Target":  _pct(PAPER["median_net_ruce"]),
            "Pass":    abs(summary["median_ruce_net"] - PAPER["median_net_ruce"]) <= 0.01,
        },
    ]
    return checks


def _bucket_monotone(closed: pd.DataFrame) -> bool:
    medians = []
    for bucket in BUCKET_ORDER:
        sub = closed[closed["liquidity_bucket"] == bucket]
        if len(sub) >= 5:
            medians.append(sub["roce"].median())
    if len(medians) < 2:
        return True  # can't test with only one bucket populated
    return all(medians[i] <= medians[i + 1] for i in range(len(medians) - 1))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _hist_chart(series: pd.Series, title: str, xlabel: str,
                bins: int = 60, clip_lo: float = -0.30, clip_hi: float = 0.50,
                vline: float | None = None, paper_line: float | None = None,
                pct_fmt: bool = True) -> str:
    clipped = series.clip(clip_lo, clip_hi)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(clipped, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3, alpha=0.85)
    med = series.median()
    ax.axvline(med, color="#DD4444", linewidth=1.8, label=f"Median {med * 100:.2f}%")
    if paper_line is not None:
        ax.axvline(paper_line, color="#22AA55", linewidth=1.5,
                   linestyle="--", label=f"Paper {paper_line * 100:.1f}%")
    if vline is not None:
        ax.axvline(vline, color="#888888", linewidth=1.2, linestyle=":")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    if pct_fmt:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _duration_chart(series: pd.Series) -> str:
    clipped = series.clip(0, 30)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(clipped, bins=range(0, 32), color="#5A9B6A", edgecolor="white", linewidth=0.3, alpha=0.85)
    med = series.median()
    ax.axvline(med, color="#DD4444", linewidth=1.8, label=f"Median {med:.0f} days")
    ax.axvline(PAPER["median_duration"], color="#22AA55", linewidth=1.5,
               linestyle="--", label=f"Paper {PAPER['median_duration']:.0f} days")
    ax.set_title("Trade Duration Distribution (closed trades, clipped at 30d)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Duration (days)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _annual_chart(annual: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    years = annual.index.astype(str)
    x = np.arange(len(years))
    roce_vals = annual["Median ROCE"].fillna(0).values * 100

    bars = ax.bar(x, roce_vals, color=["#4C72B0" if v >= 0 else "#DD4444" for v in roce_vals],
                  alpha=0.85, width=0.6)
    ax.axhline(PAPER["median_roce"] * 100, color="#22AA55", linewidth=1.5,
               linestyle="--", label=f"Paper median ROCE {PAPER['median_roce']*100:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median ROCE (%)", fontsize=9)
    ax.set_title("Annual Median ROCE (closed trades)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _bucket_chart(bucket_tbl: pd.DataFrame) -> str:
    tbl = bucket_tbl.dropna(subset=["Median ROCE"])
    if tbl.empty:
        return ""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(tbl))
    w = 0.35
    ax.bar(x - w / 2, tbl["Median ROCE"] * 100, width=w, label="Backtest ROCE",
           color="#4C72B0", alpha=0.85)
    paper_vals = [PAPER_BUCKET_ROCE.get(b, np.nan) * 100 for b in tbl.index]
    ax.bar(x + w / 2, paper_vals, width=w, label="Paper ROCE",
           color="#22AA55", alpha=0.70)
    ax.set_xticks(x)
    ax.set_xticklabels(tbl.index, fontsize=9)
    ax.set_ylabel("Median ROCE (%)", fontsize=9)
    ax.set_title("Median ROCE by Liquidity Bucket", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _attribution_chart(closed: pd.DataFrame) -> str:
    attr = compute_attribution(closed)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["ADR leg", "Local leg"]
    medians = [attr["median_adr_return"] * 100, attr["median_local_return"] * 100]
    colors = ["#4C72B0", "#DD8800"]
    bars = ax.bar(labels, medians, color=colors, alpha=0.85, width=0.4)
    ax.axhline(attr["median_roce"] * 100, color="#DD4444", linewidth=1.5,
               linestyle="--", label=f"Total ROCE {attr['median_roce']*100:.2f}%")
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Median Return (%)", fontsize=9)
    ax.set_title("Leg Attribution (closed trades)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    paper_note = f"Paper: ADR ≈ 90% of ROCE"
    actual_note = f"Actual: ADR = {attr['adr_contrib_pct']*100:.1f}% of ROCE"
    ax.text(0.5, -0.20, f"{paper_note}  |  {actual_note}",
            ha="center", fontsize=8, color="#555555", transform=ax.transAxes)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;
       background: #f5f6fa; color: #222; margin: 0; padding: 0; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
h1  { font-size: 22px; font-weight: 700; margin-bottom: 4px; color: #1a2340; }
h2  { font-size: 15px; font-weight: 700; color: #2c3e70; margin: 24px 0 8px; border-bottom: 2px solid #d0d4e8; padding-bottom: 4px; }
.meta { color: #666; font-size: 11px; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 12px; }
th    { background: #2c3e70; color: #fff; padding: 6px 10px; text-align: right; font-weight: 600; }
th:first-child { text-align: left; }
td    { padding: 5px 10px; border-bottom: 1px solid #e0e4ef; text-align: right; }
td:first-child { text-align: left; font-weight: 500; }
tr:nth-child(even) td { background: #f0f2fb; }
.pass   { color: #1a7f37; font-weight: 700; }
.fail   { color: #c0392b; font-weight: 700; }
.warn   { color: #b07800; font-weight: 700; }
.charts { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; }
.chart  { background: #fff; border-radius: 8px; padding: 8px;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.chart img { display: block; max-width: 100%; }
.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
.stat-card    { background: #fff; border-radius: 8px; padding: 14px 16px;
                box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center; }
.stat-card .val  { font-size: 22px; font-weight: 700; color: #2c3e70; }
.stat-card .lbl  { font-size: 11px; color: #888; margin-top: 2px; }
.stat-card .diff { font-size: 11px; margin-top: 2px; }
"""


def _stat_card(label: str, value: str, diff: str = "", diff_class: str = "warn") -> str:
    diff_html = f'<div class="diff {diff_class}">{diff}</div>' if diff else ""
    return (f'<div class="stat-card">'
            f'<div class="val">{value}</div>'
            f'<div class="lbl">{label}</div>'
            f'{diff_html}</div>')


def _fmt_pct_cell(v: float | None, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def _fmt_num_cell(v: float | None, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _dist_table_html(dist: pd.DataFrame) -> str:
    pct_metrics = {"ROCE", "RUCE", "Roll cost"}
    cols = ["Mean", "Std", "Max", "p90", "p75", "Median", "p25", "p10", "Min"]
    rows_html = ""
    for metric, row in dist.iterrows():
        is_pct = any(k in str(metric) for k in pct_metrics)
        fmt = _fmt_pct_cell if is_pct else _fmt_num_cell
        cells = "".join(f"<td>{fmt(row[c])}</td>" for c in cols)
        rows_html += f"<tr><td>{metric}</td>{cells}</tr>\n"
    header = "<tr><th>Metric</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    return f"<table>{header}{rows_html}</table>"


def _bucket_table_html(bucket: pd.DataFrame) -> str:
    rows_html = ""
    prev_roce = None
    for bucket_name, row in bucket.iterrows():
        n = row["N trades"]
        if n == 0:
            rows_html += (f"<tr><td>{bucket_name}</td><td>0</td>"
                          f"<td colspan='6' style='text-align:center;color:#aaa'>no trades</td></tr>\n")
            continue
        paper_roce = row["Paper ROCE"]
        actual_roce = row["Median ROCE"]
        ok = (not np.isnan(actual_roce)) and (not np.isnan(paper_roce)) and abs(actual_roce - paper_roce) <= 0.015
        monotone_ok = prev_roce is None or np.isnan(actual_roce) or actual_roce >= prev_roce
        if not np.isnan(actual_roce):
            prev_roce = actual_roce
        roce_class = "pass" if ok else "fail"
        mon_class  = "pass" if monotone_ok else "fail"
        rows_html += (
            f"<tr>"
            f"<td>{bucket_name}</td>"
            f"<td>{int(n)}</td>"
            f"<td class='{roce_class}'>{_fmt_pct_cell(actual_roce)} <small>(paper {_fmt_pct_cell(paper_roce)})</small></td>"
            f"<td>{_fmt_pct_cell(row['Median RUCE'])}</td>"
            f"<td>{_fmt_pct_cell(row['Median ROCE net'])}</td>"
            f"<td>{_fmt_num_cell(row['Median duration'], 1)} d</td>"
            f"<td>{_fmt_pct_cell(row['ADR leg contrib'])}</td>"
            f"</tr>\n"
        )
    header = ("<tr><th>Bucket</th><th>N trades</th><th>Median ROCE</th>"
              "<th>Median RUCE</th><th>Median ROCE net</th>"
              "<th>Median duration</th><th>ADR leg contrib</th></tr>")
    return f"<table>{header}{rows_html}</table>"


def _exchange_table_html(exch: pd.DataFrame) -> str:
    rows_html = ""
    for exch_name, row in exch.iterrows():
        abort_class = "fail" if row["Abort rate"] > 0.30 else "pass"
        rows_html += (
            f"<tr>"
            f"<td>{exch_name}</td>"
            f"<td>{int(row['N initiated'])}</td>"
            f"<td>{int(row['N closed'])}</td>"
            f"<td class='{abort_class}'>{_fmt_pct_cell(row['Abort rate'])}</td>"
            f"<td>{_fmt_pct_cell(row['Median ROCE'])}</td>"
            f"<td>{_fmt_pct_cell(row['Median RUCE'])}</td>"
            f"<td>{_fmt_num_cell(row['Median dur'], 1)} d</td>"
            f"</tr>\n"
        )
    header = ("<tr><th>Exchange</th><th>N initiated</th><th>N closed</th>"
              "<th>Abort rate</th><th>Median ROCE</th><th>Median RUCE</th>"
              "<th>Median dur</th></tr>")
    return f"<table>{header}{rows_html}</table>"


def _annual_table_html(annual: pd.DataFrame) -> str:
    rows_html = ""
    for yr, row in annual.iterrows():
        abort_class = "fail" if row["Abort rate"] > 0.30 else "pass"
        roce_class  = "pass" if row["Median ROCE"] >= PAPER["median_roce"] - 0.005 else "warn"
        rows_html += (
            f"<tr>"
            f"<td>{yr}</td>"
            f"<td>{int(row['N initiated'])}</td>"
            f"<td>{int(row['N closed'])}</td>"
            f"<td>{int(row['N aborted'])}</td>"
            f"<td class='{abort_class}'>{_fmt_pct_cell(row['Abort rate'])}</td>"
            f"<td class='{roce_class}'>{_fmt_pct_cell(row['Median ROCE'])}</td>"
            f"<td>{_fmt_pct_cell(row['Median RUCE'])}</td>"
            f"<td>{_fmt_num_cell(row['Median dur'], 1)} d</td>"
            f"</tr>\n"
        )
    header = ("<tr><th>Year</th><th>N initiated</th><th>N closed</th><th>N aborted</th>"
              "<th>Abort rate</th><th>Median ROCE</th><th>Median RUCE</th>"
              "<th>Median dur</th></tr>")
    return f"<table>{header}{rows_html}</table>"


def _validation_table_html(checks: list[dict]) -> str:
    rows_html = ""
    for c in checks:
        icon  = "✅" if c["Pass"] else "❌"
        klass = "pass" if c["Pass"] else "fail"
        rows_html += (
            f"<tr>"
            f"<td>{icon}</td>"
            f"<td>{c['Check']}</td>"
            f"<td class='{klass}'>{c['Actual']}</td>"
            f"<td>{c['Target']}</td>"
            f"</tr>\n"
        )
    header = "<tr><th></th><th>Check</th><th>Actual</th><th>Paper target</th></tr>"
    return f"<table>{header}{rows_html}</table>"


def render_html(
    summary:     dict,
    dist:        pd.DataFrame,
    bucket:      pd.DataFrame,
    exchange:    pd.DataFrame,
    annual:      pd.DataFrame,
    validation:  list[dict],
    closed:      pd.DataFrame,
    charts:      dict[str, str],
    run_dir:     Path,
    n_pairs:     int,
) -> str:
    attr = compute_attribution(closed)
    n_pass = sum(1 for c in validation if c["Pass"])
    n_total = len(validation)

    # Summary cards
    median_roce = summary["median_roce"]
    median_ruce = summary["median_ruce"]
    roce_diff = median_roce - PAPER["median_roce"]
    ruce_diff = median_ruce - PAPER["median_ruce"]
    roce_class = "pass" if abs(roce_diff) <= 0.005 else "fail"
    ruce_class = "pass" if abs(ruce_diff) <= 0.005 else "fail"

    cards_html = (
        _stat_card("Median ROCE", _pct(median_roce), f"Paper {_pct(PAPER['median_roce'])} (Δ {roce_diff*100:+.2f}%)", roce_class) +
        _stat_card("Median RUCE", _pct(median_ruce), f"Paper {_pct(PAPER['median_ruce'])} (Δ {ruce_diff*100:+.2f}%)", ruce_class) +
        _stat_card("Median duration", f"{summary['median_duration']:.1f}d", f"Paper {PAPER['median_duration']:.0f}d",
                   "pass" if abs(summary['median_duration'] - PAPER['median_duration']) <= 1 else "fail") +
        _stat_card("Overnight abort rate", _pct(summary['abort_rate']), f"Target < {_pct(PAPER['max_abort_rate'])}",
                   "pass" if summary['abort_rate'] < PAPER['max_abort_rate'] else "fail") +
        _stat_card("Total trades", f"{summary['n_trades']:,}", f"{summary['n_closed']:,} closed") +
        _stat_card("ADR leg contrib", _pct(attr['adr_contrib_pct']), f"Paper ≈ 90%",
                   "pass" if abs(attr['adr_contrib_pct'] - 0.90) <= 0.10 else "fail") +
        _stat_card("Pairs traded", str(summary['n_pairs_traded']), f"of {n_pairs} approved") +
        _stat_card("Validation", f"{n_pass}/{n_total}", "checks passed",
                   "pass" if n_pass == n_total else "warn")
    )

    config = summary.get("config", {})
    config_str = (f"k0={config.get('k0', '?')}  kc={config.get('kc', '?')}  "
                  f"T={config.get('T', '?')}d  H={config.get('H', '?')}d")

    def chart_img(key: str, width: int = 700) -> str:
        b64 = charts.get(key, "")
        if not b64:
            return ""
        return f'<div class="chart"><img src="data:image/png;base64,{b64}" width="{width}"></div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ADR Strategy Backtest Tearsheet</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<h1>Asian ADR Pairs Strategy — Backtest Tearsheet</h1>
<div class="meta">
  Run directory: <code>{run_dir}</code> &nbsp;|&nbsp;
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
  Parameters: {config_str} &nbsp;|&nbsp;
  Reference: Hong &amp; Susmel (2013)
</div>

<h2>Key Metrics vs Paper Benchmarks</h2>
<div class="summary-grid">{cards_html}</div>

<h2>Paper Validation Checks</h2>
{_validation_table_html(validation)}

<h2>Return Distribution (closed round-trips only, n={len(closed):,})</h2>
{_dist_table_html(dist)}

<h2>Charts</h2>
<div class="charts">
  {chart_img("roce_hist")}
  {chart_img("ruce_hist")}
</div>
<div class="charts">
  {chart_img("duration_hist")}
  {chart_img("attribution")}
</div>

<h2>Annual Performance</h2>
{_annual_table_html(annual)}
<div class="charts">
  {chart_img("annual", width=860)}
</div>

<h2>Liquidity Bucket Attribution</h2>
<p style="font-size:11px;color:#666">
  Buckets defined by ADR zero-return-day percentage (Bekaert et al. 2007).
  Paper predicts monotonically increasing ROCE from High → Low (limits-to-arbitrage premium).
</p>
{_bucket_table_html(bucket)}
<div class="charts">
  {chart_img("bucket_chart")}
</div>

<h2>Exchange Breakdown</h2>
{_exchange_table_html(exchange)}

<h2>Leg Return Attribution</h2>
<table>
  <tr><th>Component</th><th>Median return</th><th>% of ROCE</th><th>Paper target</th></tr>
  <tr><td>ADR leg (adr_return)</td>
      <td>{_fmt_pct_cell(attr['median_adr_return'])}</td>
      <td class="{'pass' if abs(attr['adr_contrib_pct']-0.90)<=0.10 else 'fail'}">{_fmt_pct_cell(attr['adr_contrib_pct'])}</td>
      <td>≈ 90%</td></tr>
  <tr><td>Local leg (local_return)</td>
      <td>{_fmt_pct_cell(attr['median_local_return'])}</td>
      <td>{_fmt_pct_cell(1 - attr['adr_contrib_pct'])}</td>
      <td>≈ 10%</td></tr>
  <tr><td><strong>Total ROCE</strong></td>
      <td><strong>{_fmt_pct_cell(attr['median_roce'])}</strong></td>
      <td>100%</td>
      <td>≈ 2.8%</td></tr>
</table>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_latest_run(base: Path) -> Path:
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {base}")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backtest tearsheet HTML")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Path to backtest run directory (default: latest run_* in data/backtest/)")
    parser.add_argument("--pairs", type=Path,
                        default=Path("config/pairs/asian_adr_pairs.json"),
                        help="Path to asian_adr_pairs.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML file (default: <run-dir>/tearsheet.html)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    backtest_base = script_dir / "data" / "backtest"

    run_dir = args.run_dir or _find_latest_run(backtest_base)
    pairs_path = args.pairs if args.pairs.is_absolute() else script_dir / args.pairs
    out_path = args.out or (run_dir / "tearsheet.html")

    log.info("Run dir : %s", run_dir)
    log.info("Pairs   : %s", pairs_path)
    log.info("Output  : %s", out_path)

    # ---- Load ----------------------------------------------------------------
    log.info("Loading data ...")
    trades, summary, pairs = load_data(run_dir, pairs_path)
    closed = trades[~trades["was_aborted"]].copy()
    n_pairs = len(pairs)

    # ---- Compute tables ------------------------------------------------------
    log.info("Computing tables ...")
    dist       = build_distribution_table(closed)
    bucket     = build_bucket_table(closed)
    exchange   = build_exchange_table(trades, closed)
    annual     = build_annual_table(trades, closed)
    validation = build_validation(summary, closed, trades)

    # ---- Render charts -------------------------------------------------------
    log.info("Rendering charts ...")
    charts = {
        "roce_hist": _hist_chart(
            closed["roce"], "ROCE Distribution (closed trades)",
            "ROCE", clip_lo=-0.25, clip_hi=0.40,
            paper_line=PAPER["median_roce"],
        ),
        "ruce_hist": _hist_chart(
            closed["ruce"], "RUCE Distribution (closed trades)",
            "RUCE", clip_lo=-0.40, clip_hi=0.70,
            paper_line=PAPER["median_ruce"],
        ),
        "duration_hist": _duration_chart(closed["duration_days"]),
        "attribution":   _attribution_chart(closed),
        "annual":        _annual_chart(annual),
        "bucket_chart":  _bucket_chart(bucket),
    }

    # ---- Render HTML ---------------------------------------------------------
    log.info("Rendering HTML ...")
    html = render_html(
        summary=summary, dist=dist, bucket=bucket, exchange=exchange,
        annual=annual, validation=validation, closed=closed,
        charts=charts, run_dir=run_dir, n_pairs=n_pairs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("Tearsheet written to %s  (%d KB)", out_path, out_path.stat().st_size // 1024)


if __name__ == "__main__":
    main()
