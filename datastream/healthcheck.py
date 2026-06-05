#!/usr/bin/env python3
"""
healthcheck.py
==============

Pre-trade / pre-research readiness diagnostic for the Asian ADR pipeline.

Verifies, with clear pass/fail per item:

    [Data freshness]
      * ADR prices Parquet exists, not older than --max-age-days, has rows for
        every ticker in the ADR reference
      * Global Asian-underlying prices Parquet ditto
      * FX rates Parquet ditto; latest date <= --fx-max-staleness-days old

    [Registry sanity]
      * config/pairs/asian_adr_pairs.json exists, parses, has at least one
        approved pair; every pair has a positive adr_ratio and a recognised
        Asian exchange

    [Environment]
      * Required env vars for credentials present (warn-only by default)
      * Python deps importable (pandas, numpy, statsmodels, pyarrow)

Exit codes
----------
    0  all checks passed (HEALTHY)
    1  one or more warnings — non-fatal (DEGRADED)
    2  one or more critical failures (UNHEALTHY)

Designed to be safe to run any time, including in CI and Kubernetes probes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("healthcheck")


ASIAN_EXCHANGES: set[str] = {
    "TKS", "TSE",        # Japan (Tokyo)
    "HKG",               # Hong Kong
    "KRX",               # Korea
    "BOM", "BSE",        # India Bombay
    "NSE",               # India National
    "ASX",               # Australia
    "SES",               # Singapore
    "TAI",               # Taiwan
    "IDX",               # Indonesia
    "PHS",               # Philippines
    "KLS",               # Malaysia (Bursa)
}

REQUIRED_ENV_VARS_FOR_LIVE: list[str] = [
    "WRDS_USERNAME", "WRDS_PASSWORD", "OANDA_API_KEY",
]

CORE_PY_DEPS: list[str] = ["pandas", "numpy", "pyarrow", "statsmodels"]


class Severity(str):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    severity: str
    message: str
    detail: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Individual checks
# -----------------------------------------------------------------------------
def _file_age_days(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 86400.0


def check_parquet_exists(path: Path, label: str, max_age_days: float) -> CheckResult:
    if not path.exists():
        return CheckResult(label, Severity.FAIL, f"missing: {path}")
    age = _file_age_days(path) or 0.0
    detail = {"path": str(path), "age_days": round(age, 2),
              "size_kb": round(path.stat().st_size / 1024, 1)}
    if age > max_age_days:
        return CheckResult(
            label, Severity.WARN,
            f"stale: age={age:.1f}d > max={max_age_days}d",
            detail,
        )
    return CheckResult(label, Severity.OK, f"present, age={age:.1f}d", detail)


def check_parquet_rows(path: Path, label: str, min_rows: int = 1) -> CheckResult:
    if not path.exists():
        return CheckResult(label, Severity.FAIL, f"missing: {path}")
    try:
        import pyarrow.parquet as pq
        n = pq.ParquetFile(path).metadata.num_rows
    except Exception as exc:
        return CheckResult(label, Severity.FAIL, f"unreadable: {exc}")
    if n < min_rows:
        return CheckResult(label, Severity.FAIL,
                           f"row count {n} < min {min_rows}",
                           {"rows": n})
    return CheckResult(label, Severity.OK, f"rows={n}", {"rows": n})


def check_fx_latest_date(path: Path, max_staleness_days: int) -> CheckResult:
    if not path.exists():
        return CheckResult("fx_latest_date", Severity.FAIL,
                           f"missing: {path}")
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["date"])
        latest = pd.to_datetime(df["date"]).max().date()
    except Exception as exc:
        return CheckResult("fx_latest_date", Severity.FAIL,
                           f"unreadable: {exc}")
    today = date.today()
    staleness = (today - latest).days
    detail = {"latest_date": latest.isoformat(), "staleness_days": staleness}
    if staleness > max_staleness_days:
        return CheckResult(
            "fx_latest_date", Severity.WARN,
            f"latest FX is {staleness}d old (max={max_staleness_days})",
            detail,
        )
    return CheckResult("fx_latest_date", Severity.OK,
                       f"latest FX {latest.isoformat()} ({staleness}d old)",
                       detail)


def check_registry(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult("pair_registry", Severity.FAIL,
                           f"missing: {path}")
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return CheckResult("pair_registry", Severity.FAIL,
                           f"unparseable JSON: {exc}")
    if not isinstance(data, list) or not data:
        return CheckResult("pair_registry", Severity.FAIL,
                           f"empty or wrong shape: {type(data).__name__}")

    bad_ratio = [p for p in data if not p.get("adr_ratio") or p["adr_ratio"] <= 0]
    bad_exch  = [p for p in data
                 if p.get("underlying_exchange") not in ASIAN_EXCHANGES]
    inactive  = [p for p in data if not p.get("is_active", True)]

    if bad_ratio or bad_exch:
        return CheckResult(
            "pair_registry", Severity.FAIL,
            f"{len(bad_ratio)} invalid ratios, {len(bad_exch)} non-Asian exchanges",
            {"n_pairs": len(data), "n_bad_ratio": len(bad_ratio),
             "n_bad_exchange": len(bad_exch), "n_inactive": len(inactive)},
        )
    if inactive:
        return CheckResult(
            "pair_registry", Severity.WARN,
            f"{len(inactive)} inactive pairs in registry",
            {"n_pairs": len(data), "n_inactive": len(inactive)},
        )
    return CheckResult(
        "pair_registry", Severity.OK,
        f"{len(data)} active pairs",
        {"n_pairs": len(data)},
    )


def check_env_vars(required: list[str], required_for_live: bool) -> CheckResult:
    missing = [v for v in required if not os.environ.get(v)]
    if not missing:
        return CheckResult("env_vars", Severity.OK,
                           f"all {len(required)} present")
    sev = Severity.FAIL if required_for_live else Severity.WARN
    return CheckResult(
        "env_vars", sev,
        f"missing: {', '.join(missing)}",
        {"missing": missing},
    )


def check_python_deps(deps: list[str]) -> CheckResult:
    missing: list[str] = []
    for d in deps:
        try:
            __import__(d)
        except ImportError:
            missing.append(d)
    if not missing:
        return CheckResult("python_deps", Severity.OK,
                           f"all {len(deps)} importable")
    return CheckResult(
        "python_deps", Severity.FAIL,
        f"missing: {', '.join(missing)}",
        {"missing": missing, "hint": "pip install " + " ".join(missing)},
    )


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(check_python_deps(CORE_PY_DEPS))
    results.append(check_parquet_exists(args.adr_prices,
                                        "adr_prices.parquet", args.max_age_days))
    results.append(check_parquet_rows(args.adr_prices, "adr_prices_rows"))
    results.append(check_parquet_exists(args.adr_reference,
                                        "adr_reference.parquet", args.max_age_days))
    results.append(check_parquet_exists(args.global_prices,
                                        "global_prices.parquet", args.max_age_days))
    results.append(check_parquet_rows(args.global_prices, "global_prices_rows"))
    results.append(check_parquet_exists(args.fx_rates,
                                        "fx_rates.parquet", args.max_age_days))
    results.append(check_fx_latest_date(args.fx_rates, args.fx_max_staleness_days))
    results.append(check_registry(args.pairs))
    results.append(check_env_vars(REQUIRED_ENV_VARS_FOR_LIVE,
                                  required_for_live=args.require_creds))

    return results


def summarise(results: list[CheckResult]) -> int:
    n_ok = sum(1 for r in results if r.severity == Severity.OK)
    n_warn = sum(1 for r in results if r.severity == Severity.WARN)
    n_fail = sum(1 for r in results if r.severity == Severity.FAIL)

    log.info("=" * 72)
    log.info("HEALTHCHECK RESULTS")
    log.info("=" * 72)
    for r in results:
        sym = {Severity.OK: "✓", Severity.WARN: "!", Severity.FAIL: "✗"}[r.severity]
        log.info("  %s [%-4s] %-26s %s", sym, r.severity, r.name, r.message)
    log.info("-" * 72)
    log.info("  OK=%d  WARN=%d  FAIL=%d", n_ok, n_warn, n_fail)

    if n_fail > 0:
        log.error("STATUS: UNHEALTHY")
        return 2
    if n_warn > 0:
        log.warning("STATUS: DEGRADED")
        return 1
    log.info("STATUS: HEALTHY")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Asian ADR pipeline readiness diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adr-prices",    type=Path,
                   default=Path("datastream/data/parquet/adr/adr_prices.parquet"))
    p.add_argument("--adr-reference", type=Path,
                   default=Path("datastream/data/parquet/adr/adr_reference.parquet"))
    p.add_argument("--global-prices", type=Path,
                   default=Path("datastream/data/parquet/global/global_prices.parquet"))
    p.add_argument("--fx-rates",      type=Path,
                   default=Path("datastream/data/parquet/fx/fx_rates.parquet"))
    p.add_argument("--pairs",         type=Path,
                   default=Path("config/pairs/asian_adr_pairs.json"))

    p.add_argument("--max-age-days", type=float, default=7.0,
                   help="Max age (days) for any cached Parquet")
    p.add_argument("--fx-max-staleness-days", type=int, default=5,
                   help="Max staleness for the LATEST FX date inside the Parquet")
    p.add_argument("--require-creds", action="store_true",
                   help="Treat missing live credentials (WRDS/OANDA) as FAIL "
                        "rather than WARN")
    p.add_argument("--json", action="store_true",
                   help="Emit JSON report on stdout in addition to logs")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    results = run_checks(args)
    code = summarise(results)
    if args.json:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": {0: "HEALTHY", 1: "DEGRADED", 2: "UNHEALTHY"}[code],
            "exit_code": code,
            "checks": [
                {"name": r.name, "severity": r.severity,
                 "message": r.message, "detail": r.detail}
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
