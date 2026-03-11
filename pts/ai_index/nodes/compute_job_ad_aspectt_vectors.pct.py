# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.compute_job_ad_aspectt_vectors
#
# Compute per-ad weighted ASPECTT vectors from filtered occupation matches.
#
# For each job ad, takes its filtered O\*NET occupation matches (2-3 per ad),
# looks up their ASPECTT numeric vectors (Abilities, Skills, Knowledge,
# Work Activities with Level and Importance scores), normalizes the match
# scores to weights, and computes a weighted average to produce a per-ad
# feature vector.
#
# 1. Loads ASPECTT vectors from `build_aspectt_vectors/aspectt_vectors.npz`.
# 2. Loads filtered matches from `llm_filter_candidates/filtered_matches.parquet`.
# 3. For each chunk of ads, computes weighted average ASPECTT vectors.
# 4. Stores results in DuckDB via ResultStore (keyed by ad_id, supports resume).
#
# Node variables:
# - `aspectt_chunk_size` (per-node): Number of ads to process per chunk
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.compute_job_ad_aspectt_vectors
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, ad_ids: list[int], aspectt_done: bool) -> {
    'ad_ids': list[int]
}:
    """Compute per-ad weighted ASPECTT vectors from filtered occupation matches."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('compute_job_ad_aspectt_vectors', run_name=run_name)
show_node_vars('compute_job_ad_aspectt_vectors', run_name=run_name)

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
from ai_index.utils.result_store import ResultStore

# %%
#|export
run_name = ctx.vars["run_name"]
chunk_size = ctx.vars["aspectt_chunk_size"]

output_dir = const.pipeline_store_path / run_name / "compute_job_ad_aspectt_vectors"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load ASPECTT vectors
#
# Small (~861 occupations x ~157 features), stays in memory.

# %%
#|export
aspectt = np.load(const.aspectt_vectors_path / "aspectt_vectors.npz")

aspectt_codes = aspectt["codes"]
aspectt_columns = aspectt["columns"]
aspectt_levels = aspectt["levels"]       # (n_occ, n_features) float32
aspectt_importance = aspectt["importance"]  # (n_occ, n_features) float32

# Build code -> row index mapping
code_to_idx = {code: i for i, code in enumerate(aspectt_codes)}

n_features = aspectt_levels.shape[1]
print(f"compute_ad_aspectt: loaded ASPECTT vectors — {len(aspectt_codes)} occupations, {n_features} features")

# %% [markdown]
# ## Prepare connections and store

# %%
#|export
filtered_path = const.pipeline_store_path / run_name / "llm_filter_candidates" / "filtered_matches.parquet"
_matches_conn = duckdb.connect()  # in-memory, queries parquet directly

db_path = output_dir / "ad_aspectt.duckdb"
store = ResultStore(db_path, {
    "id": "BIGINT NOT NULL",
    "level": "BLOB NOT NULL",
    "importance": "BLOB NOT NULL",
    "n_matches": "INTEGER NOT NULL",
    "error": "VARCHAR",
})

done = store.done_ids()
remaining = [i for i in ad_ids if i not in done]
n_remaining = len(remaining)
print(f"compute_ad_aspectt: {len(done)} already done, {n_remaining} remaining out of {len(ad_ids)}")

# %% [markdown]
# ## Process in chunks
#
# For each chunk: load filtered matches from parquet, compute weighted
# average ASPECTT vectors, write to ResultStore.

# %%
#|export
n_chunks = (n_remaining + chunk_size - 1) // chunk_size

for chunk_idx in range(n_chunks):
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n_remaining)
    chunk_ad_ids = remaining[start:end]

    # Load filtered matches for this chunk
    id_list = ",".join(str(int(i)) for i in chunk_ad_ids)
    chunk_matches = _matches_conn.execute(
        f"SELECT ad_id, onet_code, combined_score "
        f"FROM read_parquet('{filtered_path}') "
        f"WHERE ad_id IN ({id_list}) ORDER BY ad_id"
    ).fetchdf()

    # Group by ad_id
    grouped = chunk_matches.groupby("ad_id")

    records = []
    for ad_id in chunk_ad_ids:
        ad_id_int = int(ad_id)
        if ad_id_int not in grouped.groups:
            records.append({
                "id": ad_id_int,
                "level": np.zeros(n_features, dtype=np.float32).tobytes(),
                "importance": np.zeros(n_features, dtype=np.float32).tobytes(),
                "n_matches": 0,
                "error": "no filtered matches found",
            })
            continue

        group = grouped.get_group(ad_id_int)
        onet_codes = group["onet_code"].tolist()
        scores = group["combined_score"].values.astype(np.float64)

        # Look up ASPECTT indices, skip codes not in ASPECTT
        indices = []
        valid_scores = []
        for code, score in zip(onet_codes, scores):
            if code in code_to_idx:
                indices.append(code_to_idx[code])
                valid_scores.append(score)

        if not indices:
            records.append({
                "id": ad_id_int,
                "level": np.zeros(n_features, dtype=np.float32).tobytes(),
                "importance": np.zeros(n_features, dtype=np.float32).tobytes(),
                "n_matches": 0,
                "error": "no matched codes in ASPECTT vectors",
            })
            continue

        # Normalize scores to weights
        valid_scores = np.array(valid_scores, dtype=np.float64)
        weights = valid_scores / valid_scores.sum()

        # Weighted average
        level_vecs = aspectt_levels[indices]        # (n_matches, n_features)
        importance_vecs = aspectt_importance[indices]  # (n_matches, n_features)
        level_avg = (weights[:, None] * level_vecs).sum(axis=0).astype(np.float32)
        importance_avg = (weights[:, None] * importance_vecs).sum(axis=0).astype(np.float32)

        records.append({
            "id": ad_id_int,
            "level": level_avg.tobytes(),
            "importance": importance_avg.tobytes(),
            "n_matches": len(indices),
            "error": None,
        })

    store.insert(pd.DataFrame(records))
    print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads")

_matches_conn.close()
n_ok, n_err = store.counts()
store.close()
print(f"compute_ad_aspectt: done — {n_ok} succeeded, {n_err} failed")
print(f"  output: {const.rel(db_path)}")

ad_ids #|func_return_line

# %% [markdown]
# ## Sample output

# %%
import duckdb
import numpy as np

conn = duckdb.connect(str(db_path), read_only=True)
sample_rows = conn.execute(
    "SELECT id, level, importance, n_matches FROM results WHERE error IS NULL LIMIT 5"
).fetchall()
conn.close()

for row_id, level_blob, importance_blob, n_matches in sample_rows:
    level_vec = np.frombuffer(level_blob, dtype=np.float32)
    importance_vec = np.frombuffer(importance_blob, dtype=np.float32)
    print(f"\nAd {row_id} ({n_matches} matches):")
    # Show top 5 features by level score
    top_idx = np.argsort(level_vec)[::-1][:5]
    for j in top_idx:
        print(f"  {aspectt_columns[j]:<50s}  LV={level_vec[j]:.2f}  IM={importance_vec[j]:.2f}")
