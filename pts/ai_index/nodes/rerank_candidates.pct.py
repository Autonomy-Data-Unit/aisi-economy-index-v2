# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.rerank_candidates
#
# Score filtered candidates using a reranker model to produce weights for
# downstream exposure computation.
#
# 1. Reads filtered candidates from `llm_filter_candidates/filtered_matches.parquet`.
# 2. For each batch of ads, scores their candidates using `arerank()`.
# 3. Writes scored results to `rerank_candidates/reranked_matches.parquet`.
#
# The rerank scores are used as weights in `compute_job_ad_exposure` for the
# weighted average of occupation-level exposure scores.
#
# Node variables:
# - `rerank_model` (per-node): Model key from rerank_models.toml
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.rerank_candidates
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int]) -> {
    'ad_ids': list[int]
}:
    """Score filtered candidates with a reranker to produce exposure weights."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_sbatch'
set_node_func_args('rerank_candidates', run_name=run_name)
show_node_vars('rerank_candidates', run_name=run_name)

# %% [markdown]
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import json
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ai_index import const
from ai_index.utils import arerank_pairs, get_ads_by_id

# %%
#|export
run_name = ctx.vars["run_name"]
rerank_model = ctx.vars["rerank_model"]
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]
chunk_size = ctx.vars["chunk_size"]

output_dir = const.pipeline_store_path / run_name / "rerank_candidates"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load O*NET metadata and set up connections

# %%
#|export
filtered_path = const.pipeline_store_path / run_name / "llm_filter_candidates" / "filtered_matches.parquet"

onet_targets = pd.read_parquet(const.onet_targets_path)
onet_descs = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Description"]))
onet_titles_lookup = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Title"]))

matches_conn = duckdb.connect()  # in-memory
matches_conn.execute(f"CREATE VIEW filtered AS SELECT * FROM read_parquet('{filtered_path}')")

n_ads = len(ad_ids)
print(f"rerank_candidates: {n_ads} ads")
print(f"  rerank_model: {rerank_model}")
print(f"  reading from: {const.rel(filtered_path)}")

# %% [markdown]
# ## Score in batches
#
# Process ads in chunks. For each chunk, load the filtered candidates,
# build query and document texts, call the reranker, and write results
# incrementally to parquet.

# %%
#|export
CHUNK_SIZE = chunk_size

reranked_schema = pa.schema([
    ("ad_id", pa.int64()),
    ("rank", pa.int32()),
    ("onet_code", pa.string()),
    ("onet_title", pa.string()),
    ("rerank_score", pa.float32()),
])

output_path = output_dir / "reranked_matches.parquet"
writer = pq.ParquetWriter(output_path, reranked_schema)

n_chunks = (n_ads + CHUNK_SIZE - 1) // CHUNK_SIZE
total_scored = 0
slurm_jobs = []
started_at = time.time()

for chunk_idx in range(n_chunks):
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = min(chunk_start + CHUNK_SIZE, n_ads)
    chunk_ad_ids = ad_ids[chunk_start:chunk_end]

    # Load filtered candidates for this chunk
    id_list = ",".join(str(int(i)) for i in chunk_ad_ids)
    chunk_matches = matches_conn.execute(
        f"SELECT ad_id, onet_code, onet_title, cosine_score "
        f"FROM filtered "
        f"WHERE ad_id IN ({id_list}) ORDER BY ad_id, rank"
    ).fetchdf()

    if chunk_matches.empty:
        print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads (no candidates)")
        continue

    # Group candidates by ad
    candidates_by_ad = {}
    for ad_id, group in chunk_matches.groupby("ad_id"):
        candidates_by_ad[int(ad_id)] = group.to_dict("records")

    # Build query texts
    ads_with_candidates = [aid for aid in chunk_ad_ids if aid in candidates_by_ad]
    if not ads_with_candidates:
        continue
    ads_table = get_ads_by_id(ads_with_candidates, columns=["title", "description"])
    ads_df = ads_table.to_pandas().set_index("id")

    # Build (query, documents) items for all ads in the chunk, then score
    # them in a single arerank_pairs call. This sends one sbatch job per
    # chunk (~500 ads) instead of one per ad.
    items = []
    for ad_id in ads_with_candidates:
        candidates = candidates_by_ad[ad_id]
        query = f"{ads_df.loc[ad_id, 'title']}. {str(ads_df.loc[ad_id, 'description'] or '')[:3000]}"
        doc_texts = [f"{onet_titles_lookup[c['onet_code']]}: {onet_descs[c['onet_code']][:300]}" for c in candidates]
        items.append((query, doc_texts))

    _sa = {}
    scores_per_ad = await arerank_pairs(
        items,
        model=rerank_model,
        cache=sbatch_cache, time=sbatch_time,
        slurm_accounting=_sa,
    )
    if _sa:
        slurm_jobs.append(_sa)

    chunk_rows = []
    for i, ad_id in enumerate(ads_with_candidates):
        candidates = candidates_by_ad[ad_id]
        ad_scores = scores_per_ad[i]
        for rank, c in enumerate(candidates):
            chunk_rows.append({
                "ad_id": ad_id,
                "rank": rank,
                "onet_code": c["onet_code"],
                "onet_title": c["onet_title"],
                "rerank_score": float(ad_scores[rank]),
            })

    if chunk_rows:
        writer.write_table(pa.Table.from_pylist(chunk_rows, schema=reranked_schema))
        total_scored += len(chunk_rows)

    print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(ads_with_candidates)} ads, {len(chunk_rows)} scored")

writer.close()
matches_conn.close()

ended_at = time.time()
print(f"rerank_candidates: scored {total_scored} candidate rows for {n_ads} ads")
print(f"  output: {output_path}")

rerank_meta = {
    "n_total": n_ads,
    "n_scored": total_scored,
    "started_at": started_at,
    "ended_at": ended_at,
    "elapsed_seconds": ended_at - started_at,
    "slurm_jobs": slurm_jobs,
    "slurm_total_seconds": sum(j.get("elapsed_seconds", 0) for j in slurm_jobs),
}
meta_path = output_dir / "rerank_meta.json"
with open(meta_path, "w") as f:
    json.dump(rerank_meta, f, indent=2)
print(f"  meta: {const.rel(meta_path)}")

ad_ids #|func_return_line

# %% [markdown]
# ## Sample reranked matches

# %%
from ai_index.utils import get_adzuna_conn

reranked_df = pd.read_parquet(output_path)
sample_ids = ad_ids[:5]
conn = get_adzuna_conn(read_only=True)
conn.execute("CREATE OR REPLACE TEMP TABLE _match_ids (id BIGINT)")
conn.executemany("INSERT INTO _match_ids VALUES (?)", [(int(i),) for i in sample_ids])
raw_ads = {row[0]: {"title": row[1], "category_name": row[2]}
           for row in conn.execute(
    "SELECT a.id, a.title, a.category_name "
    "FROM ads a JOIN _match_ids m ON a.id = m.id"
).fetchall()}
conn.close()

for ad_id in sample_ids:
    ad_id = int(ad_id)
    raw = raw_ads[ad_id]
    ad_matches = reranked_df[reranked_df["ad_id"] == ad_id].head(5)
    print(f"\n{'━'*80}")
    print(f"Ad {ad_id}: {raw['title']} [{raw['category_name']}]")
    print(f"{'─'*80}")
    for _, row in ad_matches.iterrows():
        print(f"  #{row['rank']+1}  {row['onet_code']}  {row['onet_title']:<45s}  "
              f"rerank={row['rerank_score']:.4f}")
