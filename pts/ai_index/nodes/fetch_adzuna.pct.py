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
import os
import shutil
import tempfile

import boto3
import pandas as pd
import pyarrow.parquet as pq

from ai_index.const import adzuna_store_path

s3_prefix = ctx.vars.get("adzuna_s3_prefix", "adu-project-data/aisi-economy-index/adzuna2025")
data_year_str = ctx.vars.get("data_year", "2025")
years = [y.strip() for y in data_year_str.split(",")]

# Parse bucket and key prefix from s3_prefix (format: "bucket/key/prefix")
bucket_name, _, key_prefix = s3_prefix.partition("/")
s3 = boto3.client("s3")

adzuna_meta = {"years": {}}

for year in years:
    year_dir = adzuna_store_path / year
    year_dir.mkdir(parents=True, exist_ok=True)
    year_info = {"months": [], "row_counts": {}}

    for month in range(1, 13):
        parquet_path = year_dir / f"month_{month:02d}.parquet"

        # Idempotent: skip if already downloaded
        if parquet_path.exists():
            row_count = pq.read_metadata(parquet_path).num_rows
            year_info["months"].append(month)
            year_info["row_counts"][month] = row_count
            print(f"fetch_adzuna: {year}/month_{month:02d}.parquet exists ({row_count} rows), skipping")
            continue

        # List all JSON/JSONL objects under this month's Hive partition
        month_key = f"{key_prefix}/year={year}/month={month}/"
        paginator = s3.get_paginator("list_objects_v2")
        data_keys = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=month_key):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".json") or obj["Key"].endswith(".jsonl"):
                    data_keys.append(obj["Key"])

        if not data_keys:
            print(f"fetch_adzuna: no data files for {year}/month={month}, skipping")
            continue

        # Download each S3 file to temp, convert to parquet one at a time.
        # Each file is ~2 GB JSONL (~744k rows). We convert each to a temp parquet
        # to free memory, then use pyarrow's dataset reader to merge them.
        tmp_dir = tempfile.mkdtemp()
        tmp_parquets = []
        total_rows = 0
        for i, key in enumerate(sorted(data_keys)):
            print(f"fetch_adzuna: downloading s3://{bucket_name}/{key}")
            tmp_json = f"{tmp_dir}/day_{i}.json"
            s3.download_file(bucket_name, key, tmp_json)
            df_day = pd.read_json(tmp_json, lines=True)
            tmp_pq = f"{tmp_dir}/day_{i}.parquet"
            df_day.to_parquet(tmp_pq, compression="snappy")
            tmp_parquets.append(tmp_pq)
            total_rows += len(df_day)
            print(f"  -> {len(df_day)} rows")
            del df_day
            os.remove(tmp_json)

        if tmp_parquets:
            # Build unified schema from all day parquets (metadata only, no data loading).
            # Different day files may have columns in different order.
            import pyarrow as pa
            all_fields = {}
            for p in tmp_parquets:
                schema = pq.read_schema(p)
                for field in schema:
                    if field.name not in all_fields:
                        all_fields[field.name] = field
            unified_schema = pa.schema(list(all_fields.values()))

            # Write each day's data as a row group, reordered to match the unified schema.
            # Only one day's data is in memory at a time.
            writer = pq.ParquetWriter(str(parquet_path), unified_schema, compression="snappy")
            for p in tmp_parquets:
                table = pq.read_table(p)
                # Add any missing columns as null, then select in unified order
                for field in unified_schema:
                    if field.name not in table.schema.names:
                        table = table.append_column(field, pa.nulls(len(table), type=field.type))
                table = table.select([f.name for f in unified_schema])
                writer.write_table(table)
                del table
            writer.close()

            year_info["months"].append(month)
            year_info["row_counts"][month] = total_rows
            print(f"fetch_adzuna: wrote {parquet_path.name} ({total_rows} rows)")
        else:
            print(f"fetch_adzuna: no data found for {year}/month={month}")

        shutil.rmtree(tmp_dir, ignore_errors=True)

    adzuna_meta["years"][year] = year_info
    print(f"fetch_adzuna: year {year} — {len(year_info['months'])} months, {sum(year_info['row_counts'].values())} total rows")

print(f"fetch_adzuna: done — {len(adzuna_meta['years'])} year(s)")
{"adzuna_meta": adzuna_meta}  #|func_return_line
