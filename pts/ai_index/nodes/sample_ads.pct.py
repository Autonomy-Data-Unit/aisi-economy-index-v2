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
# If `sample_n == -1`, pass through all ads (full run).
# Otherwise, sample N ads and return their IDs.

# %%
#|default_exp sample_ads
#|export_as_func true

# %%
#|top_export
import numpy as np

# %%
#|set_func_signature
def main(ctx, print) -> {
    'ad_ids': np.ndarray
}:
    """Sample job ads for processing (or pass through all if sample_n=-1)."""
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
    rng = np.random.default_rng(ctx.vars['sample_seed'])
    sample_ad_ids = rng.choice(ad_ids, size=ctx.vars['sample_n'], replace=False)

# %% [markdown]
# Export ad texts (title + description) for sampled IDs to parquet.
# This avoids repeated DuckDB lookups in downstream nodes (e.g. embed_ads).

# %%
#|export
import pyarrow as pa
import pyarrow.parquet as pq

run_name = ctx.vars["run_name"]
ad_texts_dir = const.pipeline_store_path / run_name / "sample_ads"
ad_texts_dir.mkdir(parents=True, exist_ok=True)
ad_texts_path = ad_texts_dir / "ad_texts.parquet"

if ad_texts_path.exists():
    print(f"sample_ads: ad_texts.parquet already exists, skipping export ({const.rel(ad_texts_path)})")
else:
    conn_texts = get_adzuna_conn(read_only=True, memory_limit=duckdb_memory_limit)
    # Register the sampled IDs as an Arrow table so DuckDB can join efficiently.
    # Use a semi-join (WHERE id IN) instead of INNER JOIN + ORDER BY to avoid
    # a full sort of the 30M row table. The parquet is written once (skip-if-exists)
    # so row order doesn't affect content-hash stability across restarts.
    _sample_table = pa.table({"id": pa.array(sample_ad_ids, type=pa.int64())})
    conn_texts.register("_sample", _sample_table)
    conn_texts.execute(f"""
        COPY (
            SELECT a.id, a.title, a.category_name, a.description
            FROM ads a
            WHERE a.id IN (SELECT id FROM _sample)
        ) TO '{ad_texts_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    conn_texts.close()
    del _sample_table
    print(f"sample_ads: wrote {len(sample_ad_ids)} ad texts to {const.rel(ad_texts_path)}")

# %%
#|export
sample_ad_ids #|func_return_line
