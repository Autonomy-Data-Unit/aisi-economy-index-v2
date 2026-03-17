# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.compute_job_ad_exposure
#
# Map occupation-level exposure scores to individual job ads using filtered
# match weights. Column-agnostic: computes weighted averages for whatever
# score columns exist in the combined exposure table.
#
# For each job ad, takes its filtered O\*NET occupation matches (2-3 per ad),
# normalizes the match scores to weights, and computes a weighted average
# across all exposure score columns.
#
# 1. Receives the combined exposure DataFrame from `combine_onet_exposure`.
# 2. Loads filtered matches from `llm_filter_candidates/filtered_matches.parquet`.
# 3. For each chunk of ads, computes weighted average scores.
# 4. Writes results as `ad_exposure.parquet`.
#
# Node variables:
# - `exposure_chunk_size` (per-node): Number of ads to process per chunk
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.compute_job_ad_exposure
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, ad_ids: list[int], exposure_scores: "pd.DataFrame") -> {
    'ad_ids': list[int]
}:
    """Map occupation-level exposure scores to individual job ads via weighted averaging."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('compute_job_ad_exposure', run_name=run_name)
show_node_vars('compute_job_ad_exposure', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import duckdb
import numpy as np
import pandas as pd

from ai_index import const

# %%
#|export
run_name = ctx.vars["run_name"]
chunk_size = ctx.vars["exposure_chunk_size"]

output_dir = const.pipeline_store_path / run_name / "compute_job_ad_exposure"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Discover score columns
#
# The exposure\_scores DataFrame is small (~861 occupations) and stays in memory.
# Score columns are discovered dynamically, so adding a new score node upstream
# automatically flows through here.

# %%
#|export
score_cols = [c for c in exposure_scores.columns if c != "onet_code"]
print(f"compute_job_ad_exposure: {len(score_cols)} score columns: {score_cols}")

# %% [markdown]
# ## Process in chunks
#
# For each chunk: load filtered matches from parquet via DuckDB, join with
# exposure scores, and compute per-ad weighted averages using vectorized
# pandas operations (no per-ad Python loop). Pre-allocated arrays keep
# memory bounded regardless of total ad count.

# %%
#|export
filtered_path = const.pipeline_store_path / run_name / "rerank_candidates" / "reranked_matches.parquet"
conn = duckdb.connect()
conn.execute(f"CREATE VIEW reranked AS SELECT * FROM read_parquet('{filtered_path}')")

n_ads = len(ad_ids)
n_scores = len(score_cols)
n_chunks = (n_ads + chunk_size - 1) // chunk_size

# Pre-allocate output arrays
ad_id_to_row = {int(aid): i for i, aid in enumerate(ad_ids)}
out_n_matches = np.zeros(n_ads, dtype=np.int32)
out_scores = np.full((n_ads, n_scores), np.nan, dtype=np.float64)

for chunk_idx in range(n_chunks):
    chunk_start = chunk_idx * chunk_size
    chunk_end = min(chunk_start + chunk_size, n_ads)
    chunk_ad_ids = ad_ids[chunk_start:chunk_end]

    # Load filtered matches for this chunk
    id_list = ",".join(str(int(i)) for i in chunk_ad_ids)
    chunk_matches = conn.execute(
        f"SELECT ad_id, onet_code, rerank_score "
        f"FROM reranked "
        f"WHERE ad_id IN ({id_list}) ORDER BY ad_id"
    ).fetchdf()

    if chunk_matches.empty:
        print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads (no matches)")
        continue

    # Inner join with exposure scores (drops unrecognized onet_codes)
    merged = chunk_matches.merge(exposure_scores, on="onet_code", how="inner")

    if merged.empty:
        print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads (no scores)")
        continue

    # Normalize rerank_score to per-ad weights (equal weighting if all scores are 0)
    weight_sums = merged.groupby("ad_id")["rerank_score"].transform("sum")
    counts_per_ad = merged.groupby("ad_id")["rerank_score"].transform("count")
    merged["_weight"] = np.where(weight_sums > 0, merged["rerank_score"] / weight_sums, 1.0 / counts_per_ad)

    # Multiply score columns by weight, then sum per ad
    for col in score_cols:
        merged[col] = merged[col] * merged["_weight"]

    agg = merged.groupby("ad_id")[score_cols].sum()
    counts = merged.groupby("ad_id").size()

    # Write to pre-allocated arrays via index mapping
    row_indices = np.array([ad_id_to_row[int(aid)] for aid in agg.index])
    out_scores[row_indices] = agg.values
    out_n_matches[row_indices] = counts.values

    print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads")

conn.close()

# %% [markdown]
# ## Write output

# %%
#|export
score_dict = {col: out_scores[:, i] for i, col in enumerate(score_cols)}
results_df = pd.DataFrame({"ad_id": ad_ids, "n_matches": out_n_matches, **score_dict})
results_df["error"] = np.where(out_n_matches == 0, "no matched scores", None)

n_ok = int((out_n_matches > 0).sum())
n_err = n_ads - n_ok

output_path = output_dir / "ad_exposure.parquet"
results_df.to_parquet(output_path, index=False)
print(f"compute_job_ad_exposure: done, {n_ok} succeeded, {n_err} failed")
print(f"  output: {const.rel(output_path)}")

ad_ids #|func_return_line

# %% [markdown]
# ## Sample output

# %%
print(f"\nScore columns: {score_cols}")
print(f"\nFirst 5 rows:")
results_df[["ad_id", "n_matches"] + score_cols].head()
