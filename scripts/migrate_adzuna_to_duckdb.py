"""Migrate Adzuna parquet store to DuckDB.

Run once: uv run python scripts/migrate_adzuna_to_duckdb.py

Intentionally skips 2025/month_12.parquet so we can test fetch_adzuna
picking it up and dedup_adzuna handling the incremental case.
"""

import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from ai_index.const import adzuna_db_path, adzuna_store_path
from ai_index.utils import build_insert_from_parquet, ensure_ads_table, get_adzuna_conn

SKIP_FILES = {("2025", "month_12.parquet")}


def migrate():
    parquet_dir = adzuna_store_path
    if not parquet_dir.exists():
        print(f"ERROR: Parquet store not found at {parquet_dir}")
        return

    print(f"Source: {parquet_dir}")
    print(f"Target: {adzuna_db_path}")

    if adzuna_db_path.exists():
        print(f"WARNING: DuckDB file already exists at {adzuna_db_path}")
        resp = input("Delete and recreate? [y/N] ")
        if resp.lower() != "y":
            print("Aborting.")
            return
        adzuna_db_path.unlink()

    conn = get_adzuna_conn()
    ensure_ads_table(conn)

    # Discover year directories
    year_dirs = sorted(
        d for d in parquet_dir.iterdir()
        if d.is_dir() and re.match(r"^\d{4}$", d.name)
    )

    total_inserted = 0
    skipped_files = []

    for year_dir in year_dirs:
        year = year_dir.name
        month_files = sorted(year_dir.glob("month_*.parquet"))

        for pf in month_files:
            if (year, pf.name) in SKIP_FILES:
                print(f"  SKIP: {year}/{pf.name} (intentionally skipped for testing)")
                skipped_files.append(f"{year}/{pf.name}")
                continue

            # Extract month number from filename
            m = re.match(r"month_(\d+)\.parquet", pf.name)
            if not m:
                continue
            month = int(m.group(1))

            # Check if already migrated
            existing = conn.execute(
                "SELECT COUNT(*) FROM ads WHERE year = ? AND month = ?",
                [int(year), month],
            ).fetchone()[0]
            if existing > 0:
                print(f"  {year}/{pf.name}: already in DB ({existing} rows), skipping")
                total_inserted += existing
                continue

            # Read source schema for cast decisions
            row_count = pq.read_metadata(str(pf)).num_rows
            insert_sql = build_insert_from_parquet(str(pf), int(year), month)

            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(insert_sql)
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                print(f"  ERROR migrating {year}/{pf.name}: {e}")
                raise

            total_inserted += row_count
            print(f"  {year}/{pf.name}: {row_count} rows inserted")

    # Migrate dedup markers into _meta table
    print("\nMigrating dedup markers...")
    for year_dir in year_dirs:
        year = year_dir.name
        marker_path = year_dir / "_deduplicated.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            meta_key = f"dedup_{year}"

            # Check if any months were skipped for this year
            has_skipped = any(y == year for y, _ in SKIP_FILES)
            if has_skipped:
                print(f"  {year}: skipping dedup marker (has skipped month files)")
                continue

            # Build months fingerprint from the marker's file list
            months = sorted(
                int(re.match(r"month_(\d+)\.parquet", f).group(1))
                for f in marker.get("files", [])
                if re.match(r"month_(\d+)\.parquet", f)
            )
            months_fingerprint = ",".join(str(m) for m in months)

            # Convert row_counts keys from filenames to month numbers
            row_counts = {}
            for fname, count in marker.get("row_counts", {}).items():
                m = re.match(r"month_(\d+)\.parquet", fname)
                if m:
                    row_counts[int(m.group(1))] = count

            marker_data = {
                "months_fingerprint": months_fingerprint,
                "row_counts": row_counts,
                "duplicates_removed": marker.get("duplicates_removed", 0),
            }
            conn.execute(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
                [meta_key, json.dumps(marker_data)],
            )
            print(f"  {year}: dedup marker migrated ({marker.get('duplicates_removed', 0)} dups)")

    # Verify row counts
    print("\nVerification:")
    db_counts = conn.execute("""
        SELECT year, month, COUNT(*) as n
        FROM ads GROUP BY year, month
        ORDER BY year, month
    """).fetchall()

    for year_val, month_val, count in db_counts:
        print(f"  {year_val}/month_{month_val:02d}: {count} rows")

    db_total = conn.execute("SELECT COUNT(*) FROM ads").fetchone()[0]
    print(f"\nTotal rows in DuckDB: {db_total}")
    print(f"Total inserted: {total_inserted}")

    if skipped_files:
        print(f"\nSkipped files (for incremental testing):")
        for sf in skipped_files:
            print(f"  - {sf}")

    conn.close()

    # Rename parquet dir to backup
    backup_dir = parquet_dir.parent / "adzuna_parquet_backup"
    if backup_dir.exists():
        print(f"\nBackup dir already exists at {backup_dir}, not renaming.")
    else:
        parquet_dir.rename(backup_dir)
        print(f"\nRenamed {parquet_dir} -> {backup_dir}")

    print("\nMigration complete!")


if __name__ == "__main__":
    migrate()
