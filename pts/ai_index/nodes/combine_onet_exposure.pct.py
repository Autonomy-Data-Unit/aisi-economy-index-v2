# ---
# jupyter:
#   kernelspec:
#     display_name: ai-index (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.combine_onet_exposure
#
# Merge all O\*NET occupation-level score DataFrames into a single combined
# exposure table. Receives a dict of `{name: pd.DataFrame}` from the
# `join_scores` barrier node, joins them on `onet_code`, and saves a single
# `combined_scores.csv`.
#
# Node variables: none

# %%
#|default_exp combine_onet_exposure
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
output_dir = const.onet_exposure_scores_path
output_dir.mkdir(parents=True, exist_ok=True)

# %%
#|export
# Merge all score DataFrames on onet_code
dfs = []
for name, df in sorted(score_dfs.items()):
    if "onet_code" not in df.columns:
        raise ValueError(f"Score '{name}' missing 'onet_code' column")
    dfs.append(df)
    score_cols = [c for c in df.columns if c != "onet_code"]
    print(f"  {name}: {len(df)} rows, columns={score_cols}")

# Validate all score nodes cover the same occupations
onet_sets = {name: set(df["onet_code"]) for name, df in sorted(score_dfs.items())}
all_names = list(onet_sets.keys())
ref_name = all_names[0]
ref_codes = onet_sets[ref_name]
for name in all_names[1:]:
    if onet_sets[name] != ref_codes:
        missing = ref_codes - onet_sets[name]
        extra = onet_sets[name] - ref_codes
        parts = []
        if missing:
            parts.append(f"{len(missing)} codes in {ref_name} but not {name}")
        if extra:
            parts.append(f"{len(extra)} codes in {name} but not {ref_name}")
        raise ValueError(f"Occupation set mismatch between score nodes: {'; '.join(parts)}")

combined = functools.reduce(
    lambda left, right: left.merge(right, on="onet_code", how="inner"),
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
print(f"combine_onet_exposure: wrote {const.rel(output_dir / 'scores.csv')}")

combined; #|func_return_line

# %% [markdown]
# ## Sample output

# %%
print(f"\nScore columns: {[c for c in combined.columns if c != 'onet_code']}")
print(f"\nFirst 5 rows:")
combined.head()
