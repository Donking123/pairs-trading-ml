"""
Phase 4d helper — probe WRDS for the most CURRENT CRSP daily table.

The legacy `crsp.dsf` we pulled from stops at 2024-12-31. CRSP's newer CIZ ("v2") tables
are sometimes refreshed sooner, so 2025 data may live there. This script connects to WRDS
and, for each candidate daily table, reports the latest available date — so we can repoint
the pull at whatever is most current.

Run with your WRDS login:
    cd pairs-trading-ml
    python phases/phase4/notebooks/06_probe_crsp_currency.py

Then paste the output back. (Read-only — it only runs MAX(date) queries.)
"""
from __future__ import annotations

import wrds

# (library, table) candidates — legacy + CIZ/v2 daily stock, index, FF for reference.
CANDIDATES = [
    ("crsp", "dsf"),                    # legacy daily stock file (what we used)
    ("crsp", "dsf_v2"),                 # CIZ view, if present
    ("crsp", "stkdlysecuritydata"),     # CIZ daily security data (new format)
    ("crsp_a_stock", "dsf"),            # alt schema
    ("crsp", "dsp500"),                 # S&P 500 index (daily)
    ("crsp", "dsp500list"),             # index constituents
    ("crsp", "dsp500list_v2"),
    ("ff", "fivefactors_daily"),        # reference: should be current
]

# date-like column names to look for, in priority order
DATE_COLS = ["date", "dlycaldt", "caldt", "mthcaldt", "ending", "datadate"]


def main() -> None:
    db = wrds.Connection()
    try:
        print("=== listing crsp tables that look daily ===")
        try:
            tabs = db.list_tables(library="crsp")
            hits = [t for t in tabs if any(k in t.lower() for k in ("dsf", "dly", "daily", "dsp500"))]
            print(" ", ", ".join(sorted(hits)) or "(none matched)")
        except Exception as e:
            print("  list_tables failed:", str(e)[:100])

        print("\n=== latest date per candidate table ===")
        for lib, tbl in CANDIDATES:
            try:
                desc = db.describe_table(library=lib, table=tbl)
                cols = [c.lower() for c in desc["name"].tolist()]
                datecol = next((c for c in DATE_COLS if c in cols), None)
                if datecol is None:
                    print(f"  {lib}.{tbl:24s} — no date-like column; cols: {cols[:8]}")
                    continue
                mx = db.raw_sql(f"SELECT MAX({datecol}) AS mx FROM {lib}.{tbl}")
                print(f"  {lib}.{tbl:24s} — MAX({datecol}) = {mx['mx'].iloc[0]}")
            except Exception as e:
                print(f"  {lib}.{tbl:24s} — n/a ({str(e)[:70]})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
