# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.sample_ads
#
# Deterministically sample a subset of job ads for development.
# If `sample_n == 0`, reference the original deduped parquets directly (full run).
# Otherwise, sample N ads proportionally across months and write subset parquets.

# %%
#|default_exp nodes.sample_ads
#|export_as_func true

# %%
#|top_export
import numpy as np

# %%
#|set_func_signature
def main(ctx, print) -> {
    'ad_ids': np.ndarray
}:
    """Sample job ads for processing (or pass through all if sample_n=0)."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('sample_ads', run_name=run_name)
show_node_vars('sample_ads', run_name=run_name)

# %% [markdown]
# # Function body

# %%
#|export
from ai_index import const
from pathlib import Path
from ai_index.utils import get_adzuna_conn

duckdb_memory_limit = ctx.vars["duckdb_memory_limit"]

# %% [markdown]
# Get all ad IDs

# %%
#|export
conn = get_adzuna_conn(read_only=True, memory_limit=duckdb_memory_limit)
res = conn.execute("""
    SELECT id as ad_id FROM ads
""").fetchdf()
conn.close()
ad_ids = res['ad_id']

# %% [markdown]
# Construct a sample. If the `sample_n == -1` then we use all samples.

# %%
#|export
if ctx.vars['sample_n'] == -1:
    sample_ad_ids = ad_ids.to_numpy()
elif ctx.vars['sample_n'] < 0:
    raise ValueError(f"sample_n must be >= 0, got {ctx.vars['sample_n']}")
else:
    np.random.seed(ctx.vars['sample_seed'])
    sample_ad_ids = np.random.choice(ad_ids, size=ctx.vars['sample_n'], replace=False)

# %%
#|export
sample_ad_ids #|func_return_line
