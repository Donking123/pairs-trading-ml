"""Backtest reporting — summary, distributions, and an HTML tearsheet.

Consumes the :class:`~asian_adr.core.RoceRuceResult` list produced by the
calculator and writes the run artefacts the architecture's project structure
expects under a run directory:

    summary.json        — headline medians, counts, abort/force-close rates
    distribution.json   — paper Table 7-B distributions for each metric
    trades.parquet/.csv — per-trade records
    tearsheet.html      — self-contained HTML report with paper-benchmark checks

Self-contained: no quantstats / plotting dependency; the HTML is plain tables so
it renders anywhere. Pandas is used only to write the trade table.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from asian_adr.core import RoceRuceResult

from .roce_ruce_calculator import distribution_stats

logger = logging.getLogger(__name__)

__all__ = ["build_summary", "build_distributions", "write_report"]

# Paper benchmarks (k0=2, kc=0, T=60, H=90) for validation flags (§9.4).
PAPER_BENCHMARKS = {
    "median_roce": (0.028, 0.005),   # target, tolerance
    "median_ruce": (0.053, 0.005),
    "median_duration": (3.0, 1.0),
}


def _f(values: Iterable[Any]) -> list[float]:
    return [float(v) for v in values]


def build_distributions(results: list[RoceRuceResult]) -> dict[str, dict[str, float]]:
    return {
        "roce": distribution_stats(_f(r.roce for r in results)),
        "ruce": distribution_stats(_f(r.ruce for r in results)),
        "roce_net": distribution_stats(_f(r.roce_net for r in results)),
        "ruce_net": distribution_stats(_f(r.ruce_net for r in results)),
        "duration_days": distribution_stats(_f(r.duration_days for r in results)),
        "roll_cost_pct": distribution_stats(_f(r.roll_cost_pct for r in results)),
    }


def build_summary(
    results: list[RoceRuceResult], config: dict | None = None
) -> dict[str, Any]:
    n = len(results)
    closed = [r for r in results if not r.was_aborted]
    aborted = [r for r in results if r.was_aborted]
    force_closed = [r for r in results if r.was_force_closed]
    def med(vals: list[float]) -> float | None:
        return median(vals) if vals else None

    # ADR-leg contribution to total return (paper ≈ 90%).
    adr_abs = sum(abs(float(r.adr_return)) for r in closed)
    loc_abs = sum(abs(float(r.local_return)) for r in closed)
    adr_contribution = adr_abs / (adr_abs + loc_abs) if (adr_abs + loc_abs) else None

    summary: dict[str, Any] = {
        "n_trades": n,
        "n_closed": len(closed),
        "n_overnight_abort": len(aborted),
        "n_force_close": len(force_closed),
        "abort_rate": round(len(aborted) / n, 4) if n else None,
        "median_roce": med(_f(r.roce for r in closed)),
        "median_ruce": med(_f(r.ruce for r in closed)),
        "median_roce_net": med(_f(r.roce_net for r in closed)),
        "median_ruce_net": med(_f(r.ruce_net for r in closed)),
        "median_duration": med(_f(r.duration_days for r in closed)),
        "adr_return_contribution": round(adr_contribution, 4) if adr_contribution else None,
        "n_pairs_traded": len({r.pair_id for r in results}),
        "config": config or {},
    }
    summary["validation"] = _validate(summary)
    return summary


def _validate(summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key, (target, tol) in PAPER_BENCHMARKS.items():
        val = summary.get(key)
        checks[key] = {
            "value": val,
            "target": target,
            "within_tolerance": (val is not None and abs(val - target) <= tol),
        }
    return checks


def _dist_table_html(distributions: dict[str, dict[str, float]]) -> str:
    cols = ["mean", "std", "max", "p90", "p75", "median", "p25", "p10", "min"]
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for metric, stats in distributions.items():
        cells = "".join(f"<td>{stats.get(c, float('nan')):.4f}</td>" for c in cols)
        rows.append(f"<tr><th>{metric}</th>{cells}</tr>")
    return f"<table><tr><th>metric</th>{head}</tr>{''.join(rows)}</table>"


def _summary_table_html(summary: dict[str, Any]) -> str:
    rows = []
    for k, v in summary.items():
        if k in ("config", "validation"):
            continue
        rows.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def _validation_html(validation: dict[str, Any]) -> str:
    rows = []
    for k, c in validation.items():
        ok = "PASS" if c["within_tolerance"] else "FAIL"
        color = "#2e7d32" if c["within_tolerance"] else "#c62828"
        rows.append(
            f"<tr><th>{k}</th><td>{c['value']}</td><td>{c['target']}</td>"
            f"<td style='color:{color};font-weight:bold'>{ok}</td></tr>"
        )
    return (
        "<table><tr><th>check</th><th>value</th><th>paper</th><th>status</th></tr>"
        f"{''.join(rows)}</table>"
    )


def render_html(summary: dict[str, Any], distributions: dict[str, dict[str, float]]) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Asian ADR Backtest Tearsheet</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color:#222; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 1.6rem; }}
 table {{ border-collapse: collapse; margin: 0.5rem 0; }}
 th, td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: right; font-variant-numeric: tabular-nums; }}
 th {{ background: #f5f5f5; text-align: left; }}
 .sub {{ color:#666; font-size:0.85rem; }}
</style></head><body>
<h1>Asian ADR Pairs — Backtest Tearsheet</h1>
<p class="sub">Hong &amp; Susmel (2013) · generated {generated}</p>
<h2>Summary</h2>{_summary_table_html(summary)}
<h2>Paper Validation (k0=2, kc=0, T=60, H=90)</h2>{_validation_html(summary['validation'])}
<h2>Distributions (paper Table 7-B)</h2>{_dist_table_html(distributions)}
</body></html>"""


def write_report(
    results: list[RoceRuceResult],
    out_dir: str | Path,
    config: dict | None = None,
) -> Path:
    """Write all run artefacts; returns the run directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = build_summary(results, config)
    distributions = build_distributions(results)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "distribution.json").write_text(json.dumps(distributions, indent=2, default=str))
    (out / "tearsheet.html").write_text(render_html(summary, distributions))

    if results:
        try:
            import pandas as pd

            df = pd.DataFrame([r.model_dump() for r in results])
            df.to_csv(out / "trades.csv", index=False)
            df.to_parquet(out / "trades.parquet", index=False)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("could not write trade table: %s", exc)

    logger.info("backtest report written to %s", out)
    return out
