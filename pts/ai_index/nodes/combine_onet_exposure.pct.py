# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.combine_onet_exposure
#
# Merge all O\*NET occupation-level score DataFrames into a single combined
# exposure table. Receives a dict of `{name: pd.DataFrame}` from the
# `join_scores` barrier node, joins them on `onet_code`, and saves a single
# `combined_scores.parquet`.
#
# Node variables:
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.combine_onet_exposure
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, score_dfs: dict) -> "pd.DataFrame":
    """Merge all O*NET score DataFrames into a single combined exposure table."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('combine_onet_exposure', run_name=run_name)
show_node_vars('combine_onet_exposure', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Merge score DataFrames

# %%
#|export
import pandas as pd
import functools

from ai_index import const
from ai_index.utils.scoring import OnetScoreSet

# %%
#|export
run_name = ctx.vars["run_name"]
output_dir = const.pipeline_store_path / run_name / "combine_onet_exposure"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
#|export
# Merge all score DataFrames on onet_code
dfs = []
for name, df in sorted(score_dfs.items()):
    assert "onet_code" in df.columns, f"Score '{name}' missing 'onet_code' column"
    dfs.append(df)
    score_cols = [c for c in df.columns if c != "onet_code"]
    print(f"  {name}: {len(df)} rows, columns={score_cols}")

combined = functools.reduce(
    lambda left, right: left.merge(right, on="onet_code", how="outer"),
    dfs,
)
print(f"combine_onet_exposure: {len(combined)} occupations, "
      f"{len(combined.columns) - 1} score columns")

# %% [markdown]
# ## Validate and save

# %%
#|export
score_set = OnetScoreSet(name="combined_exposure", scores=combined)
score_set.save(output_dir)
print(f"combine_onet_exposure: wrote {output_dir / 'scores.parquet'}")

combined #|func_return_line

# %% [markdown]
# ## Sample output

# %%
print(f"\nScore columns: {[c for c in combined.columns if c != 'onet_code']}")
print(f"\nFirst 5 rows:")
print(combined.head())
