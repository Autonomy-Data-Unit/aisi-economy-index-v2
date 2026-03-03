# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dedup Adzuna
#
# Deduplicate Adzuna job ads by `id` across months within each year.
# Earliest month wins — duplicates in later months are removed.
# Rewrites parquets in-place and writes a `_deduplicated.json` marker.

# %%
#|default_exp nodes.dedup_adzuna
#|export_as_func true

# %%
#|set_func_signature
def main(adzuna_meta, ctx, print) -> {"dedup_meta": dict}:
    """Deduplicate Adzuna job ads by ID across months."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dotenv import load_dotenv; load_dotenv()
from dev_utils import set_node_func_args
set_node_func_args(main)

# %%
#|export
import json
from pathlib import Path

import pandas as pd

from ai_index.const import adzuna_store_path

dedup_meta = {"years": {}}

for year, year_info in adzuna_meta["years"].items():
    year_dir = adzuna_store_path / year
    marker_path = year_dir / "_deduplicated.json"

    # Collect month parquet files that exist
    month_files = sorted(year_dir.glob("month_*.parquet"))
    month_file_names = [f.name for f in month_files]

    # Check marker: skip if already deduplicated with same file set
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get("files") == month_file_names:
            print(f"dedup_adzuna: {year} already deduplicated ({len(month_files)} files match marker), skipping")
            dedup_meta["years"][year] = marker
            continue

    if not month_files:
        print(f"dedup_adzuna: no parquet files for {year}, skipping")
        continue

    # Pass 1: read only 'id' column from each month to build seen_ids and find duplicates
    print(f"dedup_adzuna: {year} — pass 1: scanning IDs across {len(month_files)} months")
    seen_ids: set = set()
    months_with_dups: dict[Path, int] = {}  # path -> num duplicates

    for pf in month_files:
        ids = pd.read_parquet(pf, columns=["id"])["id"]
        dup_mask = ids.isin(seen_ids)
        n_dups = dup_mask.sum()
        if n_dups > 0:
            months_with_dups[pf] = int(n_dups)
        seen_ids.update(ids)
        print(f"  {pf.name}: {len(ids)} rows, {n_dups} duplicates")

    total_dups = sum(months_with_dups.values())

    # Pass 2: rewrite months that have duplicates
    if months_with_dups:
        print(f"dedup_adzuna: {year} — pass 2: rewriting {len(months_with_dups)} month(s) to remove {total_dups} duplicates")
        seen_ids_pass2: set = set()

        for pf in month_files:
            ids = pd.read_parquet(pf, columns=["id"])["id"]
            if pf in months_with_dups:
                df = pd.read_parquet(pf)
                keep_mask = ~df["id"].isin(seen_ids_pass2)
                df_clean = df[keep_mask].reset_index(drop=True)
                df_clean.to_parquet(pf, compression="snappy")
                seen_ids_pass2.update(df_clean["id"])
                print(f"  {pf.name}: {len(df)} -> {len(df_clean)} rows")
                del df, df_clean
            else:
                seen_ids_pass2.update(ids)
    else:
        print(f"dedup_adzuna: {year} — no duplicates found")

    # Collect final row counts
    row_counts = {}
    for pf in month_files:
        n = pd.read_parquet(pf, columns=["id"]).shape[0]
        row_counts[pf.name] = n

    # Write marker
    marker_data = {
        "files": month_file_names,
        "row_counts": row_counts,
        "duplicates_removed": total_dups,
    }
    marker_path.write_text(json.dumps(marker_data, indent=2))
    dedup_meta["years"][year] = marker_data
    print(f"dedup_adzuna: {year} done — {total_dups} duplicates removed, marker written")

print(f"dedup_adzuna: done — {len(dedup_meta['years'])} year(s)")
{"dedup_meta": dedup_meta}  #|func_return_line
