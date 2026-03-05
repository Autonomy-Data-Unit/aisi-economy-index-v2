# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
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
from dev_utils import set_node_func_args
set_node_func_args()

# %% [markdown]
# # Function body

# %%
#|export
import os
import shutil
import tempfile

import boto3
import pyarrow as pa
import pyarrow.json as paj
import pyarrow.parquet as pq

from ai_index.const import adzuna_store_path

s3_prefix = ctx.vars["adzuna_s3_prefix"]
years_filter = ctx.vars["years"]

# Parse bucket and key prefix from s3_prefix (format: "bucket/key/prefix")
bucket_name, _, key_prefix = s3_prefix.partition("/")
s3 = boto3.client("s3")

# Discover all year partitions from S3
import re
paginator = s3.get_paginator("list_objects_v2")
years = set()
for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{key_prefix}/year=", Delimiter="/"):
    for prefix_obj in page.get("CommonPrefixes", []):
        m = re.search(r"year=(\d+)", prefix_obj["Prefix"])
        if m:
            years.add(m.group(1))
years = sorted(years)
print(f"fetch_adzuna: discovered {len(years)} year(s) in S3: {years}")

# Filter to requested years (empty string = all)
if years_filter:
    requested = {y.strip() for y in years_filter.split(",")}
    years = [y for y in years if y in requested]
    print(f"fetch_adzuna: filtered to {len(years)} year(s): {years}")

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
        # Uses pyarrow JSON reader (much faster + lower memory than pd.read_json).
        tmp_dir = tempfile.mkdtemp()
        tmp_parquets = []
        total_rows = 0
        n_files = len(data_keys)
        for i, key in enumerate(sorted(data_keys)):
            print(f"fetch_adzuna: [{i+1}/{n_files}] downloading s3://{bucket_name}/{key}")
            tmp_json = f"{tmp_dir}/day_{i}.json"
            s3.download_file(bucket_name, key, tmp_json)
            # Use pyarrow's native JSON reader — columnar, parallelized C++
            read_opts = paj.ReadOptions(block_size=1 << 26)  # 64 MB blocks
            table = paj.read_json(tmp_json, read_options=read_opts)
            tmp_pq = f"{tmp_dir}/day_{i}.parquet"
            pq.write_table(table, tmp_pq, compression="snappy")
            tmp_parquets.append(tmp_pq)
            n_rows = table.num_rows
            total_rows += n_rows
            print(f"  -> {n_rows} rows")
            del table
            os.remove(tmp_json)

        if tmp_parquets:
            # Build unified schema from all day parquets (metadata only, no data loading).
            # Different day files may have columns in different order.
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
