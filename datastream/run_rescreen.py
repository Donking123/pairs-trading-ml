#!/usr/bin/env python3
"""
rescreen.py
===========

Incremental re-screening orchestrator for the Asian ADR pairs strategy.

Steps
-----
1. Auto-detect the last date in each Parquet cache (adr_prices, global_prices, fx_rates)
2. Fetch only the missing date range from WRDS (incremental — not a full re-pull)
3. Append new rows to each Parquet, deduplicate on primary keys, sort, write back
4. Back up the current pair registry with a datestamp
5. Re-run the full screening pipeline with the specified --as-of date
6. Write the new registry + a human-readable changelog (added / dropped / changed pairs)

Usage
-----
    cd datastream/

    # Standard weekly run (as-of today, fetch the gap, update registry)
    python rescreen.py

    # Explicit as-of date
    python rescreen.py --as-of 2026-09-30

    # Re-screen on existing Parquet only — no WRDS call
    python rescreen.py --skip-fetch

    # Preview what would change without writing anything
    python rescreen.py --dry-run

    # Also re-fetch the static ADR reference table (new ADR listings, etc.)
    python rescreen.py --refresh-reference
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: make sibling scripts importable regardless of cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fetch_datastream_adr_data    as _fetch_adr   # noqa: E402
import fetch_datastream_global_data as _fetch_global  # noqa: E402
import fetch_fx_history             as _fetch_fx    # noqa: E402
import run_asian_adr_screening      as _screen      # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
)
log = logging.getLogger("rescreen")

# ---------------------------------------------------------------------------
# Default paths  (relative to the datastream/ directory)
# ---------------------------------------------------------------------------
_ADR_DIR    = _HERE / "data" / "parquet" / "adr"
_GLOBAL_DIR = _HERE / "data" / "parquet" / "global"
_FX_DIR     = _HERE / "data" / "parquet" / "fx"
_PAIRS_PATH = _HERE / "config" / "pairs" / "asian_adr_pairs.json"
_DIAG_PATH  = _HERE / "config" / "pairs" / "screening_diagnostics.json"
_BACKUP_DIR = _HERE / "config" / "pairs" / "backups"
_CHANGELOG_DIR = _HERE / "config" / "pairs" / "changelogs"

# Overlap buffer: always re-fetch this many days before the Parquet tail to
# catch late-arriving or holiday-adjusted rows.
_GAP_BUFFER_DAYS = 5


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file, falling back to fastparquet on pyarrow errors."""
    try:
        return pd.read_parquet(path)
    except OSError:
        try:
            import fastparquet as fp  # type: ignore
            return fp.ParquetFile(str(path)).to_pandas()
        except Exception as exc:
            log.error("Cannot read %s: %s", path, exc)
            raise SystemExit(2)


def _get_last_date(path: Path, date_col: str) -> Optional[date]:
    """Return the maximum date in a Parquet file, or None if the file is missing."""
    if not path.exists():
        return None
    df = _read_parquet(path)[[date_col]]
    col = pd.to_datetime(df[date_col], errors="coerce")
    ts = col.max()
    return ts.date() if not pd.isna(ts) else None


def _append_parquet(
    path: Path,
    new_df: pd.DataFrame,
    key_cols: list[str],
    date_col: str,
) -> int:
    """
    Append new_df to an existing Parquet file.

    Deduplicates on key_cols (keeping the new row on conflict), sorts by
    key_cols, and writes back. Returns the number of net-new rows added.
    """
    existing = _read_parquet(path)
    if date_col in existing.columns:
        existing[date_col] = pd.to_datetime(existing[date_col])
    if date_col in new_df.columns:
        new_df = new_df.copy()
        new_df[date_col] = pd.to_datetime(new_df[date_col])

    before = len(existing)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    combined.to_parquet(path, index=False, compression="snappy")

    net_new = len(combined) - before
    log.info("  %s: %d → %d rows (+%d net-new)", path.name, before, len(combined), net_new)
    return net_new


# ---------------------------------------------------------------------------
# Incremental WRDS fetches
# ---------------------------------------------------------------------------

def fetch_incremental_adr(
    gap_start: date,
    gap_end: date,
    adr_dir: Path,
    refresh_reference: bool,
) -> None:
    """Fetch ADR prices for the gap window and append to adr_prices.parquet."""
    log.info("fetching ADR prices: %s → %s", gap_start, gap_end)
    prices, reference = _fetch_adr.fetch_from_wrds(gap_start, gap_end)

    log.info("appending ADR prices...")
    _append_parquet(
        adr_dir / "adr_prices.parquet",
        prices,
        key_cols=["infocode", "marketdate"],
        date_col="marketdate",
    )

    if refresh_reference:
        ref_path = adr_dir / "adr_reference.parquet"
        reference.to_parquet(ref_path, index=False, compression="snappy")
        log.info("refreshed ADR reference: %d rows → %s", len(reference), ref_path)
    else:
        log.info("skipping ADR reference refresh (use --refresh-reference to update)")


def fetch_incremental_global(
    gap_start: date,
    gap_end: date,
    adr_ref_path: Path,
    global_dir: Path,
) -> None:
    """Fetch Asian underlying prices for the gap window and append to global_prices.parquet."""
    log.info("fetching global prices: %s → %s", gap_start, gap_end)

    ref = _read_parquet(adr_ref_path)
    asian_only = _fetch_global.filter_to_asian(ref)
    if asian_only.empty:
        log.warning("no Asian underlyings in ADR reference — skipping global fetch")
        return

    tickers   = asian_only["underlying_ticker"].astype(str).tolist()
    exchanges = sorted(asian_only["underlying_exchange"].unique().tolist())

    new_df = _fetch_global.fetch_from_wrds(gap_start, gap_end, tickers, exchanges)

    log.info("appending global prices...")
    _append_parquet(
        global_dir / "global_prices.parquet",
        new_df,
        key_cols=["infocode", "marketdate"],
        date_col="marketdate",
    )


def fetch_incremental_fx(
    gap_start: date,
    gap_end: date,
    adr_ref_path: Path,
    fx_dir: Path,
) -> None:
    """Fetch FX rates for the gap window and append to fx_rates.parquet."""
    log.info("fetching FX rates: %s → %s", gap_start, gap_end)

    currencies = _fetch_fx.load_currencies_from_reference(adr_ref_path)
    new_df     = _fetch_fx.fetch_from_datastream(currencies, gap_start, gap_end)

    if new_df.empty:
        log.warning("no FX rows returned for the gap — skipping FX append")
        return

    new_df = new_df.copy()
    new_df["date"] = pd.to_datetime(new_df["date"])

    fx_path = fx_dir / "fx_rates.parquet"
    existing = _read_parquet(fx_path)
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")

    before = len(existing)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["base_currency", "date"], keep="last")
    combined = combined.sort_values(["base_currency", "date"]).reset_index(drop=True)
    combined.to_parquet(fx_path, index=False, compression="snappy")

    net_new = len(combined) - before
    log.info("  fx_rates.parquet: %d → %d rows (+%d net-new)",
             before, len(combined), net_new)


# ---------------------------------------------------------------------------
# Registry backup
# ---------------------------------------------------------------------------

def backup_registry(registry_path: Path) -> Optional[Path]:
    """Copy the current registry to backups/<stem>_YYYYMMDD_HHMMSS.json."""
    if not registry_path.exists():
        return None
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = _BACKUP_DIR / f"{registry_path.stem}_{stamp}.json"
    shutil.copy2(registry_path, dest)
    log.info("backed up registry to %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------

_TRACKED_FIELDS = [
    "adr_ratio",
    "underlying_exchange",
    "underlying_currency",
    "zero_return_pct_adr",
    "adf_pvalue",
    "pp_pvalue",
    "roll_spread_adr",
    "roll_spread_local",
    "joint_history_days",
]


def compute_changelog(
    old_pairs: list[dict],
    new_pairs: list[dict],
) -> dict[str, Any]:
    old_map = {p["pair_id"]: p for p in old_pairs}
    new_map = {p["pair_id"]: p for p in new_pairs}

    added   = [new_map[pid] for pid in sorted(set(new_map) - set(old_map))]
    dropped = [old_map[pid] for pid in sorted(set(old_map) - set(new_map))]

    changed = []
    for pid in sorted(set(old_map) & set(new_map)):
        diffs: dict[str, dict] = {}
        for field in _TRACKED_FIELDS:
            ov = old_map[pid].get(field)
            nv = new_map[pid].get(field)
            if _values_differ(ov, nv):
                diffs[field] = {"old": ov, "new": nv}
        if diffs:
            changed.append({"pair_id": pid, "changes": diffs})

    return {
        "as_of":            date.today().isoformat(),
        "old_pair_count":   len(old_pairs),
        "new_pair_count":   len(new_pairs),
        "added_count":      len(added),
        "dropped_count":    len(dropped),
        "changed_count":    len(changed),
        "added":            added,
        "dropped":          dropped,
        "changed":          changed,
    }


def _values_differ(a: Any, b: Any) -> bool:
    """Treat NaN == NaN as equal; use a small tolerance for floats."""
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    try:
        fa, fb = float(a), float(b)
        if np.isnan(fa) and np.isnan(fb):
            return False
        return abs(fa - fb) > 1e-8
    except (TypeError, ValueError):
        return a != b


def _print_changelog_summary(cl: dict[str, Any]) -> None:
    log.info("=" * 60)
    log.info("CHANGELOG SUMMARY")
    log.info("  Pairs before : %d", cl["old_pair_count"])
    log.info("  Pairs after  : %d", cl["new_pair_count"])
    log.info("  Added        : %d", cl["added_count"])
    log.info("  Dropped      : %d", cl["dropped_count"])
    log.info("  Changed      : %d", cl["changed_count"])

    if cl["added"]:
        log.info("  --- Added pairs ---")
        for p in cl["added"]:
            log.info("    + %s  (%s / %s)",
                     p["pair_id"], p.get("underlying_exchange", "?"),
                     p.get("underlying_currency", "?"))

    if cl["dropped"]:
        log.info("  --- Dropped pairs ---")
        for p in cl["dropped"]:
            log.info("    - %s  (%s / %s)",
                     p["pair_id"], p.get("underlying_exchange", "?"),
                     p.get("underlying_currency", "?"))

    if cl["changed"]:
        log.info("  --- Changed pairs (first 10) ---")
        for rec in cl["changed"][:10]:
            for field, diff in rec["changes"].items():
                log.info("    ~ %s  %s: %s → %s",
                         rec["pair_id"], field, diff["old"], diff["new"])

    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incremental re-screening: extend Parquet cache then re-run pair selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--as-of", type=_parse_date, default=date.today(),
        help="Screening 'as of' date — no look-ahead beyond this",
    )
    p.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip the WRDS incremental fetch; re-screen on existing Parquet only",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print the changelog but do not write any files",
    )
    p.add_argument(
        "--refresh-reference", action="store_true",
        help="Re-fetch the static ADR reference table (new ADR listings / delistings)",
    )
    p.add_argument(
        "--gap-buffer-days", type=int, default=_GAP_BUFFER_DAYS,
        help="Re-fetch this many days before the Parquet tail (handles late-arriving data)",
    )
    # Paths (override defaults if needed)
    p.add_argument("--adr-dir",    type=Path, default=_ADR_DIR)
    p.add_argument("--global-dir", type=Path, default=_GLOBAL_DIR)
    p.add_argument("--fx-dir",     type=Path, default=_FX_DIR)
    p.add_argument("--pairs-path", type=Path, default=_PAIRS_PATH)
    p.add_argument("--diag-path",  type=Path, default=_DIAG_PATH)
    # Screening parameters (forwarded to run_pipeline)
    p.add_argument("--estimation-days",         type=int,   default=60)
    p.add_argument("--holding-days",            type=int,   default=90)
    p.add_argument("--k0",                      type=float, default=2.0)
    p.add_argument("--kc",                      type=float, default=0.0)
    p.add_argument("--cointegration-alpha",     type=float, default=0.05)
    p.add_argument("--min-history-days",        type=int,   default=504)
    p.add_argument("--min-non-zero-return-pct", type=float, default=0.50)
    p.add_argument("--max-zero-return-pct-adr", type=float, default=0.50)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    as_of = args.as_of
    log.info("rescreen starting  as_of=%s  skip_fetch=%s  dry_run=%s",
             as_of, args.skip_fetch, args.dry_run)

    adr_prices_path    = args.adr_dir    / "adr_prices.parquet"
    adr_ref_path       = args.adr_dir    / "adr_reference.parquet"
    global_prices_path = args.global_dir / "global_prices.parquet"
    fx_rates_path      = args.fx_dir     / "fx_rates.parquet"

    # ------------------------------------------------------------------ #
    # 1. Detect data gaps                                                  #
    # ------------------------------------------------------------------ #
    log.info("detecting Parquet tail dates...")
    last_adr    = _get_last_date(adr_prices_path,    "marketdate")
    last_global = _get_last_date(global_prices_path, "marketdate")
    last_fx     = _get_last_date(fx_rates_path,      "date")

    log.info("  adr_prices    last: %s", last_adr)
    log.info("  global_prices last: %s", last_global)
    log.info("  fx_rates      last: %s", last_fx)

    if not args.skip_fetch:
        # ------------------------------------------------------------------ #
        # 2. Incremental WRDS fetch                                           #
        # ------------------------------------------------------------------ #
        gap_end = as_of  # inclusive upper bound for the fetch

        # ADR prices
        if last_adr is not None and last_adr < gap_end:
            gap_start = last_adr - timedelta(days=args.gap_buffer_days)
            fetch_incremental_adr(gap_start, gap_end, args.adr_dir, args.refresh_reference)
        elif last_adr is None:
            log.error("adr_prices.parquet not found — run fetch_datastream_adr_data.py first")
            return 2
        else:
            log.info("ADR prices already up to %s — no fetch needed", last_adr)
            if args.refresh_reference:
                log.info("--refresh-reference: fetching reference table")
                _, reference = _fetch_adr.fetch_from_wrds(
                    as_of - timedelta(days=1), as_of
                )
                adr_ref_path.parent.mkdir(parents=True, exist_ok=True)
                reference.to_parquet(adr_ref_path, index=False, compression="snappy")
                log.info("refreshed ADR reference: %d rows", len(reference))

        # Global (Asian underlying) prices
        if last_global is not None and last_global < gap_end:
            gap_start = last_global - timedelta(days=args.gap_buffer_days)
            fetch_incremental_global(gap_start, gap_end, adr_ref_path, args.global_dir)
        elif last_global is None:
            log.error("global_prices.parquet not found — run fetch_datastream_global_data.py first")
            return 2
        else:
            log.info("global prices already up to %s — no fetch needed", last_global)

        # FX rates
        if last_fx is not None and last_fx < gap_end:
            gap_start = last_fx - timedelta(days=args.gap_buffer_days)
            fetch_incremental_fx(gap_start, gap_end, adr_ref_path, args.fx_dir)
        elif last_fx is None:
            log.error("fx_rates.parquet not found — run fetch_fx_history.py first")
            return 2
        else:
            log.info("FX rates already up to %s — no fetch needed", last_fx)

    # ------------------------------------------------------------------ #
    # 3. Load current registry (before overwriting)                        #
    # ------------------------------------------------------------------ #
    old_pairs: list[dict] = []
    if args.pairs_path.exists():
        with open(args.pairs_path) as f:
            old_pairs = json.load(f)
        log.info("loaded current registry: %d pairs", len(old_pairs))
    else:
        log.warning("no existing registry found at %s — treating as empty", args.pairs_path)

    # ------------------------------------------------------------------ #
    # 4. Load Parquet data for screening                                   #
    # ------------------------------------------------------------------ #
    log.info("loading Parquet files for screening...")

    def _load(path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            log.error("%s not found at %s", label, path)
            raise SystemExit(2)
        df = _read_parquet(path)
        if "marketdate" in df.columns:
            df["marketdate"] = pd.to_datetime(df["marketdate"]).dt.normalize()
        log.info("  %s: %d rows", label, len(df))
        return df

    adr_prices    = _load(adr_prices_path,    "adr_prices")
    adr_reference = _load(adr_ref_path,       "adr_reference")
    global_prices = _load(global_prices_path, "global_prices")
    fx_rates      = _load(fx_rates_path,      "fx_rates")

    # ------------------------------------------------------------------ #
    # 5. Re-run screening pipeline                                          #
    # ------------------------------------------------------------------ #
    cfg = {
        "estimation_days":         args.estimation_days,
        "holding_days":            args.holding_days,
        "k0":                      args.k0,
        "kc":                      args.kc,
        "cointegration_alpha":     args.cointegration_alpha,
        "min_history_days":        args.min_history_days,
        "min_non_zero_return_pct": args.min_non_zero_return_pct,
        "max_zero_return_pct_adr": args.max_zero_return_pct_adr,
    }
    log.info("running screening pipeline  as_of=%s  cfg=%s", as_of, cfg)

    approved, diagnostics = _screen.run_pipeline(
        adr_prices, adr_reference, global_prices, fx_rates, as_of, cfg,
    )
    new_pairs = [asdict(p) for p in approved]
    log.info("screening complete: %d approved pairs", len(new_pairs))

    # ------------------------------------------------------------------ #
    # 6. Compute changelog                                                  #
    # ------------------------------------------------------------------ #
    changelog = compute_changelog(old_pairs, new_pairs)
    _print_changelog_summary(changelog)

    # ------------------------------------------------------------------ #
    # 7. Write outputs (unless --dry-run)                                   #
    # ------------------------------------------------------------------ #
    if args.dry_run:
        log.info("--dry-run: no files written")
        return 0

    # Backup old registry
    backup_registry(args.pairs_path)

    # Write new registry
    args.pairs_path.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_path.write_text(json.dumps(new_pairs, indent=2))
    log.info("wrote %d pairs to %s", len(new_pairs), args.pairs_path)

    # Write diagnostics
    args.diag_path.parent.mkdir(parents=True, exist_ok=True)
    args.diag_path.write_text(json.dumps(diagnostics, indent=2, default=str))
    log.info("wrote %d diagnostic records to %s", len(diagnostics), args.diag_path)

    # Write changelog
    _CHANGELOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    changelog_path = _CHANGELOG_DIR / f"changelog_{stamp}.json"
    changelog_path.write_text(json.dumps(changelog, indent=2, default=str))
    log.info("wrote changelog to %s", changelog_path)

    log.info("rescreen complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
