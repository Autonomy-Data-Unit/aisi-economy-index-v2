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
from ai_index import const
from pathlib import Path

# %%
const.store_path

# %%
import pyarrow.dataset as ds

dataset = ds.dataset(
    const.store_path / "inputs/adzuna",
    format="parquet",
)

table = dataset.to_table(columns=["id", "__filename"])

# %%
# Group by filename
grouped = table.group_by("__filename").aggregate([("id", "list")])

def filename_to_datestr(filename):
    p = Path(filename)
    return f"{p.parts[-2]}-{p.parts[-1].split('.')[0].split('_')[1]}"

# Convert to dict
filename_to_ids = {
    filename_to_datestr(row["__filename"]): set(row["id_list"])
    for row in grouped.to_pylist()
}
