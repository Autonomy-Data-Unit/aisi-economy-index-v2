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
def main(adzuna_meta, ctx, print) -> {"ads_manifest": dict}:
    """Sample job ads for processing (or pass through all if sample_n=0)."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
set_node_func_args(run_name='test_local')
show_node_vars(run_name='test_local')

# %% [markdown]
# # Function body

# %%
#|export
from ai_index import const
from pathlib import Path

# %%
from ai_index.utils import get_adzuna_conn

conn = get_adzuna_conn(read_only=True)

# Summary: row counts per year/month
counts = conn.execute("""
    SELECT year, month, COUNT(*) as n
    FROM ads GROUP BY year, month
    ORDER BY year, month
""").fetchdf()
print(counts.to_string(index=False))
conn.close()

# %%
from ai_index.utils import get_ads_by_id

res = get_ads_by_id([2675965976], columns=["title", "category_id"])

# %%
res = get_ads_by_id([2675965976])
res.to_pandas()
