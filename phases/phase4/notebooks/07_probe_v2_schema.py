"""
Phase 4d helper — dump the CIZ/v2 CRSP table schemas so we can migrate the pull to 2025.

The v2 tables (crsp.dsf_v2, crsp.stkdlysecuritydata) carry 2025 data but use CIZ column
names. This prints the columns (and one sample row) of every table we'd need, so the pull
adapter can be written correctly in one shot.

Run with your WRDS login, then paste the output:
    python phases/phase4/notebooks/07_probe_v2_schema.py
"""
from __future__ import annotations

import wrds

TABLES = [
    ("crsp", "wrds_dsfv2_query"),     # WRDS convenience view (often legacy-compatible)
    ("crsp", "dsf_v2"),               # CIZ daily stock
    ("crsp", "dsp500_v2"),            # CIZ S&P 500 index (daily)
    ("crsp", "dsp500list_v2"),        # CIZ index constituents
    ("crsp", "stkdlysecuritydata"),   # CIZ daily security data (has delisting fields)
    ("crsp", "stksecurityinfohist"),  # CIZ security identifiers/classification (siccd, shrcd)
]


def main() -> None:
    db = wrds.Connection()
    try:
        for lib, tbl in TABLES:
            print("\n" + "=" * 78)
            print(f"{lib}.{tbl}")
            print("=" * 78)
            try:
                desc = db.describe_table(library=lib, table=tbl)
                pairs = list(zip(desc["name"].tolist(), desc["type"].tolist()))
                print("columns:")
                for name, typ in pairs:
                    print(f"  {name:28s} {typ}")
            except Exception as e:
                print("  describe failed:", str(e)[:120])
                continue
            # one sample row to see value formats
            try:
                samp = db.raw_sql(f"SELECT * FROM {lib}.{tbl} LIMIT 1")
                print("sample row:")
                for c in samp.columns:
                    print(f"  {c:28s} = {samp[c].iloc[0]!r}")
            except Exception as e:
                print("  sample failed:", str(e)[:120])

        # what index families / numbers identify the S&P 500 in the v2 constituents?
        print("\n" + "=" * 78)
        print("crsp.dsp500list_v2 — distinct indfam / indno (to identify S&P 500)")
        print("=" * 78)
        try:
            fam = db.raw_sql(
                "SELECT indfam, indno, COUNT(*) AS n FROM crsp.dsp500list_v2 "
                "GROUP BY indfam, indno ORDER BY n DESC")
            print(fam.to_string(index=False))
        except Exception as e:
            print("  query failed:", str(e)[:120])
    finally:
        db.close()


if __name__ == "__main__":
    main()
