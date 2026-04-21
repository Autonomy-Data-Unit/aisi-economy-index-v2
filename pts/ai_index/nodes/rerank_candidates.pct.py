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
# - `max_concurrent_chunks` (per-node): Max concurrent rerank chunks
# - `run_name` (global): Pipeline run name

# %%
#|default_exp rerank_candidates
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
import asyncio
import json
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ai_index import const
from ai_index.utils import arerank_pairs
from ai_index.utils.batch import _dispatch_wave
from isambard_utils.orchestrate import StagedInput
from isambard_utils.staging import astage_files

# %%
#|export
run_name = ctx.vars["run_name"]
rerank_model = ctx.vars["rerank_model"]
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]
chunk_size = ctx.vars["chunk_size"]
max_concurrent_chunks = ctx.vars["max_concurrent_chunks"]

output_dir = const.pipeline_store_path / run_name / "rerank_candidates"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Prepare and stage data on Isambard
#
# Instead of uploading 500MB of serialized items per sbatch job, stage the
# raw data files on Isambard once. Each job reads its chunk via predicate
# pushdown on the Lustre filesystem.

# %%
#|export
filtered_path = const.pipeline_store_path / run_name / "llm_filter_candidates" / "filtered_matches.parquet"

onet_targets = pd.read_parquet(const.onet_targets_path)
onet_titles_lookup = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Title"]))

# Build rich document text per occupation for reranking
def _build_onet_doc(row):
    parts = [row["Title"]]
    alt = row["Alternate_Titles"]
    if len(alt) > 0:
        parts.append("Also known as: " + ", ".join(alt[:10]))
    parts.append(row["Description"])
    tasks = row["Top_Tasks"]
    if len(tasks) > 0:
        parts.append("Key tasks: " + "; ".join(tasks[:5]))
    return "\n".join(parts)

onet_docs = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets.apply(_build_onet_doc, axis=1)))

n_ads = len(ad_ids)
print(f"rerank_candidates: {n_ads} ads")
print(f"  rerank_model: {rerank_model}")
print(f"  reading from: {const.rel(filtered_path)}")

# Use the ad_texts.parquet already written by sample_ads (deterministic, ZSTD).
# This avoids re-exporting 5M ads from DuckDB (30 min, 30GB RAM) and ensures
# a stable content hash across pipeline restarts for Isambard cache hits.
ad_texts_path = const.pipeline_store_path / run_name / "sample_ads" / "ad_texts.parquet"

# Export onet_docs.json (onet_code -> doc_text)
staging_dir = output_dir / "_staging"
staging_dir.mkdir(parents=True, exist_ok=True)
onet_docs_path = staging_dir / "onet_docs.json"
with open(onet_docs_path, "w") as f:
    json.dump(onet_docs, f, sort_keys=True)

# Stage all three files to Isambard
print("rerank_candidates: staging files to Isambard...")
staged_refs = await astage_files(
    {
        "filtered_matches": filtered_path,
        "ad_texts": ad_texts_path,
        "onet_docs": onet_docs_path,
    },
    print_fn=print,
)
print("rerank_candidates: staging complete")

# %% [markdown]
# ## Score in batches
#
# Process ads in chunks. For each chunk, read the candidate structure from
# filtered_matches.parquet (lightweight, predicate pushdown), then send a
# StagedInput to arerank_pairs. The heavy item building (query + doc texts)
# happens on the Isambard GPU node from the pre-staged files.

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

async def _process_chunk(chunk_idx, chunk_ad_ids):
    # Read candidate structure from parquet (lightweight, predicate pushdown).
    # No DuckDB in-memory view needed: each chunk reads only its rows.
    chunk_matches = pq.read_table(
        filtered_path,
        filters=pc.field("ad_id").isin(chunk_ad_ids),
        columns=["ad_id", "onet_code", "onet_title"],
    )

    if len(chunk_matches) == 0:
        print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads (no candidates)")
        return []

    # Group candidates by ad (lightweight: just onet_code + onet_title)
    candidates_by_ad = {}
    ad_id_col = chunk_matches.column("ad_id").to_pylist()
    onet_code_col = chunk_matches.column("onet_code").to_pylist()
    onet_title_col = chunk_matches.column("onet_title").to_pylist()
    for i in range(len(chunk_matches)):
        aid = ad_id_col[i]
        if aid not in candidates_by_ad:
            candidates_by_ad[aid] = []
        candidates_by_ad[aid].append({
            "onet_code": onet_code_col[i],
            "onet_title": onet_title_col[i],
        })
    del chunk_matches

    ads_with_candidates = [aid for aid in chunk_ad_ids if aid in candidates_by_ad]
    if not ads_with_candidates:
        return []

    # Build a StagedInput: the resolver on Isambard will read the staged
    # parquets/JSON and build the (query, doc_texts) items for this chunk.
    staged_items = StagedInput(
        resolver="rerank_pairs_items",
        sources=staged_refs,
        params={"ad_ids": ads_with_candidates},
    )

    _sa = {}
    scores_per_ad = await arerank_pairs(
        staged_items,
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

    print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(ads_with_candidates)} ads, {len(chunk_rows)} scored")
    return chunk_rows

def _on_chunk_result(chunk_rows):
    nonlocal total_scored
    if chunk_rows:
        writer.write_table(pa.Table.from_pylist(chunk_rows, schema=reranked_schema))
        total_scored += len(chunk_rows)

await _dispatch_wave(
    [(i, ad_ids[i * CHUNK_SIZE : min((i + 1) * CHUNK_SIZE, n_ads)]) for i in range(n_chunks)],
    _process_chunk,
    _on_chunk_result,
    max_concurrent=max_concurrent_chunks,
)

writer.close()

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
