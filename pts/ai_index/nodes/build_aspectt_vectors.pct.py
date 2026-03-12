# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.build_aspectt_vectors
#
# Build ASPECTT numeric vectors from O\*NET database tables.
#
# ASPECTT = **A**bilities, **S**kills, **K**nowledge, **W**ork Activities:
# four O\*NET categories with Level and Importance scale scores per feature.
#
# 1. Reads filtered occupation codes from `onet_targets.parquet`
#    (produced by `prepare_onet_targets`, already excludes public sector).
# 2. Loads 4 O\*NET tables: Abilities, Skills, Knowledge, Work Activities.
# 3. Pivots each table by Level (LV) and Importance (IM) scale scores.
# 4. Concatenates into a ~157-dimensional vector per occupation.
# 5. Saves as `.npz` in `store/pipeline/{run_name}/aspectt_vectors/`.
#
# Node variables:
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.build_aspectt_vectors
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print) -> bool:
    """Build ASPECTT numeric vectors from O*NET database tables."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('build_aspectt_vectors', run_name=run_name)
show_node_vars('build_aspectt_vectors', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import numpy as np
import pandas as pd

from ai_index import const

# %%
#|export
output_dir = const.aspectt_vectors_path
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load filtered occupations
#
# Use the occupation codes from `onet_targets.parquet` so we automatically
# inherit the public-sector exclusion from `prepare_onet_targets`.

# %%
#|export
onet_targets = pd.read_parquet(const.onet_targets_path)
valid_codes = set(onet_targets["O*NET-SOC Code"].tolist())
code_to_title = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Title"]))
print(f"build_aspectt: {len(valid_codes)} occupations from onet_targets.parquet")

# %% [markdown]
# ## Load O\*NET tables

# %%
#|export
extract_dir = const.onet_store_path / "db_30_0_text"

def _load_onet_table(name):
    return pd.read_csv(extract_dir / f"{name}.txt", sep="\t", header=0, encoding="utf-8", dtype=str)

CATEGORIES = [
    ("Abilities", "Element Name"),
    ("Skills", "Element Name"),
    ("Knowledge", "Element Name"),
    ("Work Activities", "Element Name"),
]

raw_tables = {}
for table_name, _ in CATEGORIES:
    raw_tables[table_name] = _load_onet_table(table_name)
    print(f"  loaded {table_name}: {len(raw_tables[table_name])} rows")

# %% [markdown]
# ## Pivot tables into feature matrices
#
# For each category, pivot on the element name with O\*NET-SOC Code as index.
# Build both Level (LV) and Importance (IM) matrices.

# %%
#|export
def _pivot_category(df, category_name, element_col, scale_id):
    """Pivot one O*NET category table into a feature matrix for a given scale."""
    filtered = df[df["Scale ID"] == scale_id].copy()
    filtered = filtered[filtered["O*NET-SOC Code"].isin(valid_codes)]
    filtered["Data Value"] = pd.to_numeric(filtered["Data Value"], errors="coerce")

    pivoted = filtered.pivot_table(
        index="O*NET-SOC Code",
        columns=element_col,
        values="Data Value",
        aggfunc="sum",
    )
    # Prefix column names with category
    pivoted.columns = [f"{category_name} - {col}" for col in pivoted.columns]
    return pivoted


level_dfs = []
importance_dfs = []

for table_name, element_col in CATEGORIES:
    df = raw_tables[table_name]
    lv = _pivot_category(df, table_name, element_col, "LV")
    im = _pivot_category(df, table_name, element_col, "IM")
    level_dfs.append(lv)
    importance_dfs.append(im)
    print(f"  {table_name}: {lv.shape[1]} features")

level_matrix = pd.concat(level_dfs, axis=1)
importance_matrix = pd.concat(importance_dfs, axis=1)

# Ensure consistent row ordering (use onet_targets order)
ordered_codes = [c for c in onet_targets["O*NET-SOC Code"] if c in level_matrix.index]
level_matrix = level_matrix.loc[ordered_codes]
importance_matrix = importance_matrix.loc[ordered_codes]

print(f"build_aspectt: level matrix {level_matrix.shape}, importance matrix {importance_matrix.shape}")

# %% [markdown]
# ## Save to disk

# %%
#|export
titles = np.array([code_to_title[c] for c in ordered_codes], dtype=str)
codes = np.array(ordered_codes, dtype=str)
columns = np.array(level_matrix.columns.tolist(), dtype=str)

np.savez(
    output_dir / "aspectt_vectors.npz",
    titles=titles,
    codes=codes,
    columns=columns,
    levels=level_matrix.values.astype(np.float32),
    importance=importance_matrix.values.astype(np.float32),
)

print(f"build_aspectt: wrote {const.rel(output_dir / 'aspectt_vectors.npz')}")
print(f"  titles: {titles.shape}")
print(f"  codes: {codes.shape}")
print(f"  columns: {columns.shape}")
print(f"  levels: {level_matrix.values.shape} (float32)")
print(f"  importance: {importance_matrix.values.shape} (float32)")

True #|func_return_line

# %% [markdown]
# ## Sample output

# %%
print(f"\nFeature categories:")
for cat_name, _ in CATEGORIES:
    cat_cols = [c for c in columns if c.startswith(f"{cat_name} - ")]
    print(f"  {cat_name}: {len(cat_cols)} features")

print(f"\nSample occupations (first 3):")
for i in range(min(3, len(titles))):
    vals = level_matrix.values[i]
    print(f"\n  {titles[i]} ({codes[i]})")
    # Show top 5 features by level score
    top_idx = np.argsort(vals)[::-1][:5]
    for j in top_idx:
        print(f"    {columns[j]:<50s}  LV={vals[j]:.2f}  IM={importance_matrix.values[i, j]:.2f}")
