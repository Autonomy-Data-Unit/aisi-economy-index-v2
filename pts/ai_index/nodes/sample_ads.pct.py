# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sample Ads
#
# Deterministically sample a subset of job ads for development.
# If `sample_n == 0`, reference the original deduped parquets directly (full run).
# Otherwise, sample N ads proportionally across months and write subset parquets.

# %%
#|default_exp nodes.sample_ads
#|export_as_func true

# %%
#|set_func_signature
def main(dedup_meta, ctx, print) -> {"ads_manifest": dict}:
    """Sample job ads for processing (or pass through all if sample_n=0)."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args()

# %%
#|export
import numpy as np
import pandas as pd

from ai_index.const import adzuna_store_path, pipeline_store_path

sample_n = int(ctx.vars["sample_n"])
sample_seed = int(ctx.vars["sample_seed"])

# %%
#|export
# Build list of (year, month_file, row_count) from dedup_meta
month_entries = []
for year, year_info in dedup_meta["years"].items():
    row_counts = year_info.get("row_counts", {})
    for filename, count in sorted(row_counts.items()):
        month_entries.append((year, filename, count))

total_ads = sum(c for _, _, c in month_entries)
print(f"sample_ads: {total_ads} total ads across {len(month_entries)} months")

# %%
#|export
if sample_n == 0 or sample_n >= total_ads:
    # Full run: reference original parquets directly
    print(f"sample_ads: using all {total_ads} ads (sample_n={sample_n})")
    manifest = {"months": [], "total_ads": total_ads, "sample_n": 0, "sample_seed": sample_seed}
    for year, filename, count in month_entries:
        manifest["months"].append({
            "year": year,
            "filename": filename,
            "path": str(adzuna_store_path / year / filename),
            "row_count": count,
        })
else:
    # Proportional sampling across months
    rng = np.random.default_rng(sample_seed)

    # Allocate samples proportionally (minimum 1 per month if possible)
    weights = np.array([c for _, _, c in month_entries], dtype=float)
    weights /= weights.sum()
    raw_alloc = weights * sample_n
    alloc = np.floor(raw_alloc).astype(int)
    # Distribute remaining samples by largest remainder
    remainder = sample_n - alloc.sum()
    fractional = raw_alloc - alloc
    top_indices = np.argsort(-fractional)[:remainder]
    alloc[top_indices] += 1

    print(f"sample_ads: sampling {sample_n} ads (seed={sample_seed})")
    store_dir = pipeline_store_path / "sample_ads"
    manifest = {"months": [], "total_ads": int(alloc.sum()), "sample_n": sample_n, "sample_seed": sample_seed}

    for i, (year, filename, count) in enumerate(month_entries):
        n_sample = int(alloc[i])
        if n_sample == 0:
            continue

        out_dir = store_dir / year
        out_path = out_dir / filename

        # Check if sample file already exists with correct count
        if out_path.exists():
            existing_count = pd.read_parquet(out_path, columns=["id"]).shape[0]
            if existing_count == n_sample:
                print(f"  {year}/{filename}: {n_sample} already sampled, skipping")
                manifest["months"].append({
                    "year": year,
                    "filename": filename,
                    "path": str(out_path),
                    "row_count": n_sample,
                })
                continue

        # Read, sample, write
        src_path = adzuna_store_path / year / filename
        df = pd.read_parquet(src_path)
        sampled = df.sample(n=min(n_sample, len(df)), random_state=rng.integers(2**31))
        out_dir.mkdir(parents=True, exist_ok=True)
        sampled.to_parquet(out_path, compression="snappy")
        print(f"  {year}/{filename}: sampled {len(sampled)}/{count}")

        manifest["months"].append({
            "year": year,
            "filename": filename,
            "path": str(out_path),
            "row_count": len(sampled),
        })

    print(f"sample_ads: {manifest['total_ads']} sampled ads across {len(manifest['months'])} months")

ads_manifest = manifest
{"ads_manifest": ads_manifest}  #|func_return_line
