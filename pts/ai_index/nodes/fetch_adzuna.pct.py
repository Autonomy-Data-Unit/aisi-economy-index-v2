# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fetch Adzuna
#
# Download raw Adzuna job ad JSONL files from S3 and convert to monthly parquets.
# Hive-partitioned: `s3://{prefix}/year={Y}/month={M}/day={D}/*.jsonl`

# %%
#|default_exp nodes.fetch_adzuna
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print) -> {"adzuna_meta": dict}:
    """Download raw Adzuna job ads from S3 to monthly parquets."""
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
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from ai_index.const import adzuna_store_path

s3_prefix = ctx.vars.get("adzuna_s3_prefix", "adu-project-data/aisi-economy-index/adzuna2025")
data_year_str = ctx.vars.get("data_year", "2025")
years = [y.strip() for y in data_year_str.split(",")]

adzuna_meta = {"years": {}}

for year in years:
    year_dir = adzuna_store_path / year
    year_dir.mkdir(parents=True, exist_ok=True)
    year_info = {"months": [], "row_counts": {}}

    for month in range(1, 13):
        parquet_path = year_dir / f"month_{month:02d}.parquet"

        # Idempotent: skip if already downloaded
        if parquet_path.exists():
            df_existing = pd.read_parquet(parquet_path)
            year_info["months"].append(month)
            year_info["row_counts"][month] = len(df_existing)
            print(f"fetch_adzuna: {year}/month_{month:02d}.parquet exists ({len(df_existing)} rows), skipping")
            del df_existing
            continue

        # List day partitions for this month
        month_prefix = f"s3://{s3_prefix}/year={year}/month={month}/"
        try:
            ls_result = subprocess.run(
                ["aws", "s3", "ls", month_prefix],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError:
            print(f"fetch_adzuna: no data at {month_prefix}, skipping")
            continue

        # Parse day directories from ls output (format: "PRE day=5/")
        day_dirs = []
        for line in ls_result.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("PRE "):
                day_dir_name = line[4:].rstrip("/")
                day_dirs.append(day_dir_name)

        if not day_dirs:
            print(f"fetch_adzuna: no day partitions for {year}/month={month}, skipping")
            continue

        # Download each day's JSONL and collect DataFrames
        day_dfs = []
        for day_dir_name in sorted(day_dirs):
            day_prefix = f"{month_prefix}{day_dir_name}/"

            # List files in this day partition
            ls_day = subprocess.run(
                ["aws", "s3", "ls", day_prefix],
                capture_output=True, text=True, check=True,
            )
            jsonl_files = []
            for fline in ls_day.stdout.strip().splitlines():
                parts = fline.strip().split()
                if len(parts) >= 4 and parts[-1].endswith(".jsonl"):
                    jsonl_files.append(parts[-1])

            for jsonl_file in jsonl_files:
                s3_path = f"{day_prefix}{jsonl_file}"
                print(f"fetch_adzuna: downloading {s3_path}")

                cp_result = subprocess.run(
                    ["aws", "s3", "cp", s3_path, "-"],
                    capture_output=True, text=True, check=True,
                )
                if cp_result.stdout.strip():
                    df_day = pd.read_json(StringIO(cp_result.stdout), lines=True)
                    day_dfs.append(df_day)
                    print(f"  -> {len(df_day)} rows")

        if not day_dfs:
            print(f"fetch_adzuna: no data found for {year}/month={month}")
            continue

        # Concat all days and write parquet
        df_month = pd.concat(day_dfs, ignore_index=True)
        df_month.to_parquet(parquet_path, compression="snappy")
        year_info["months"].append(month)
        year_info["row_counts"][month] = len(df_month)
        print(f"fetch_adzuna: wrote {parquet_path.name} ({len(df_month)} rows)")
        del day_dfs, df_month

    adzuna_meta["years"][year] = year_info
    print(f"fetch_adzuna: year {year} — {len(year_info['months'])} months, {sum(year_info['row_counts'].values())} total rows")

print(f"fetch_adzuna: done — {len(adzuna_meta['years'])} year(s)")
{"adzuna_meta": adzuna_meta}  #|func_return_line
