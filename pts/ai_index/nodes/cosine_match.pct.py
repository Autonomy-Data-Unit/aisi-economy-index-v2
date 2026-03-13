# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.cosine_match
#
# Weighted dual cosine similarity matching between job ad embeddings
# and O*NET occupation embeddings.
#
# 1. Loads role and taskskill embeddings from both `embed_ads` and `embed_onet`.
# 2. Computes cosine top-K separately for role and taskskill.
# 3. Merges candidates and combines with weighted scoring:
#    `combined = alpha * role_score + (1 - alpha) * taskskill_score`.
# 4. Saves top-K matches per ad to parquet in `store/pipeline/{run_name}/cosine_match/`.
#
# Node variables:
# - `cosine_mode` (global): Execution mode for cosine similarity ("api", "local", "sbatch")
# - `topk` (global): Number of top matches per ad
# - `cosine_alpha` (per-node): Weight for role score (taskskill weight = 1 - alpha)
# - `cosine_chunk_size` (per-node): Number of ads to process per chunk
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.cosine_match
#|export_as_func true

# %%
#|top_export
import json

import duckdb
import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils import acosine_topk

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int], onet_done: bool) -> {
    'ad_ids': list[int]
}:
    """Weighted dual cosine similarity matching."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('cosine_match', run_name=run_name)
show_node_vars('cosine_match', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
run_name = ctx.vars["run_name"]
cosine_mode = ctx.vars["cosine_mode"]
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]
topk = ctx.vars["topk"]
cosine_alpha = ctx.vars["cosine_alpha"]
chunk_size = ctx.vars["cosine_chunk_size"]

output_dir = const.pipeline_store_path / run_name / "cosine_match"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load O\*NET embeddings
#
# O\*NET embeddings are small (~1000 occupations) and stay in memory.
# Ad embeddings are loaded per-chunk from DuckDB.

# %%
#|export
onet_dir = const.pipeline_store_path / run_name / "embed_onet"

onet_codes = np.load(onet_dir / "onet_codes.npy")
onet_titles = np.load(onet_dir / "onet_titles.npy")
onet_role_embeds = np.load(onet_dir / "role_embeddings.npy")
onet_task_embeds = np.load(onet_dir / "taskskill_embeddings.npy")

embed_db = const.pipeline_store_path / run_name / "embed_ads" / "embeddings.duckdb"
embed_conn = duckdb.connect(str(embed_db), read_only=True)

n_ads = len(ad_ids)
n_onet = len(onet_codes)
n_chunks = (n_ads + chunk_size - 1) // chunk_size
print(f"cosine_match: {n_ads} ads x {n_onet} occupations, topk={topk}, alpha={cosine_alpha}")
print(f"  processing in {n_chunks} chunks of up to {chunk_size}")

# %% [markdown]
# ## Process chunks
#
# For each chunk of ads: load embeddings from DuckDB, compute cosine top-K
# for role and taskskill, merge candidates with weighted scoring, and
# accumulate match rows.

# %%
#|export
def _load_chunk_embeddings(chunk_ad_ids):
    """Load role and taskskill embeddings for a chunk of ad IDs from DuckDB."""
    embed_conn.execute("CREATE OR REPLACE TEMP TABLE _chunk_order (id BIGINT, pos INTEGER)")
    embed_conn.executemany(
        "INSERT INTO _chunk_order VALUES (?, ?)",
        [(int(aid), i) for i, aid in enumerate(chunk_ad_ids)],
    )
    rows = embed_conn.execute(
        "SELECT r.role, r.taskskill FROM results r "
        "JOIN _chunk_order a ON r.id = a.id ORDER BY a.pos"
    ).fetchall()
    role_embeds = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    task_embeds = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return role_embeds, task_embeds


def _weighted_merge(chunk_ad_ids, role_results, task_results):
    """Merge role and task cosine results with weighted scoring for a chunk.

    Each ad has two independent top-K lists: one from role embeddings and one
    from taskskill embeddings. The same O*NET occupation may appear in both
    lists. This function unions the two candidate sets, keeping both scores
    when an occupation appears in both (and 0.0 for the missing channel when
    it appears in only one). The final combined score is:

        combined = alpha * role_score + (1 - alpha) * taskskill_score

    The union may contain up to 2*topk candidates; we take the final top-K
    by combined score.
    """
    role_indices = role_results["indices"]
    role_scores = role_results["scores"]
    task_indices = task_results["indices"]
    task_scores = task_results["scores"]

    rows = []
    for i in range(len(chunk_ad_ids)):
        # Union role and task candidates. When the same O*NET occupation
        # appears in both top-K lists, combine its role and task scores;
        # when it appears in only one, the other score defaults to 0.0.
        candidates = {}
        for j in range(topk):
            idx = int(role_indices[i, j])
            candidates[idx] = [float(role_scores[i, j]), 0.0]
        for j in range(topk):
            idx = int(task_indices[i, j])
            if idx in candidates:
                candidates[idx][1] = float(task_scores[i, j])
            else:
                candidates[idx] = [0.0, float(task_scores[i, j])]

        # Compute combined scores and sort descending
        scored = []
        for idx, (rs, ts) in candidates.items():
            combined = cosine_alpha * rs + (1.0 - cosine_alpha) * ts
            scored.append((idx, rs, ts, combined))
        scored.sort(key=lambda x: -x[3])

        # Take final top-K
        for rank, (idx, rs, ts, combined) in enumerate(scored[:topk]):
            rows.append({
                "ad_id": int(chunk_ad_ids[i]),
                "rank": rank,
                "onet_code": str(onet_codes[idx]),
                "onet_title": str(onet_titles[idx]),
                "role_score": rs,
                "taskskill_score": ts,
                "combined_score": combined,
            })
    return rows

# %%
#|export
all_rows = []
slurm_jobs = []
for chunk_idx in range(n_chunks):
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n_ads)
    chunk_ad_ids = ad_ids[start:end]

    ad_role_embeds, ad_task_embeds = _load_chunk_embeddings(chunk_ad_ids)

    _sa1 = {}
    role_results = await acosine_topk(ad_role_embeds, onet_role_embeds, k=topk, mode=cosine_mode, cache=sbatch_cache, time=sbatch_time, slurm_accounting=_sa1)
    if _sa1: slurm_jobs.append(_sa1)
    _sa2 = {}
    task_results = await acosine_topk(ad_task_embeds, onet_task_embeds, k=topk, mode=cosine_mode, cache=sbatch_cache, time=sbatch_time, slurm_accounting=_sa2)
    if _sa2: slurm_jobs.append(_sa2)

    chunk_rows = _weighted_merge(chunk_ad_ids, role_results, task_results)
    all_rows.extend(chunk_rows)

    print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads, {len(chunk_rows)} match rows")

embed_conn.close()

# %% [markdown]
# ## Save results

# %%
#|export
matches_df = pd.DataFrame(all_rows)
output_path = output_dir / "matches.parquet"
matches_df.to_parquet(output_path, index=False)

print(f"cosine_match: wrote {len(matches_df)} match rows ({n_ads} ads x topk={topk})")
print(f"  output: {output_path}")
print(f"  score range: {matches_df['combined_score'].min():.4f} - {matches_df['combined_score'].max():.4f}")

if slurm_jobs:
    _meta = {"slurm_jobs": slurm_jobs, "slurm_total_seconds": sum(j.get("elapsed_seconds", 0) for j in slurm_jobs)}
    with open(output_dir / "cosine_meta.json", "w") as _f:
        json.dump(_meta, _f, indent=2)
    print(f"cosine_match: wrote {const.rel(output_dir / 'cosine_meta.json')}")

ad_ids #|func_return_line

# %% [markdown]
# ## Sample matches
#
# Show the top O\*NET matches for each job ad, with the ad title for context.

# %%
import duckdb
from ai_index.utils import get_adzuna_conn
from ai_index.nodes.llm_summarise import JobInfoModel

sample_ids = ad_ids[:5]

conn = get_adzuna_conn(read_only=True)
conn.execute("CREATE OR REPLACE TEMP TABLE _match_ids (id BIGINT)")
conn.executemany("INSERT INTO _match_ids VALUES (?)", [(int(i),) for i in sample_ids])
raw_ads = {row[0]: {"title": row[1], "category_name": row[2], "description": row[3]}
           for row in conn.execute(
    "SELECT a.id, a.title, a.category_name, a.description "
    "FROM ads a JOIN _match_ids m ON a.id = m.id"
).fetchall()}
conn.close()

summaries_db = const.pipeline_store_path / run_name / "llm_summarise" / "summaries.duckdb"
sconn = duckdb.connect(str(summaries_db), read_only=True)
sample_set = set(int(i) for i in sample_ids)
summaries = {row[0]: JobInfoModel.model_validate_json(row[1])
             for row in sconn.execute(
    "SELECT id, data FROM results WHERE error IS NULL"
).fetchall() if row[0] in sample_set}
sconn.close()

for ad_id in sample_ids:
    ad_id = int(ad_id)
    raw = raw_ads[ad_id]
    summary = summaries[ad_id]
    ad_matches = matches_df[matches_df["ad_id"] == ad_id].head(5)
    print(f"\n{'━'*80}")
    print(f"Ad {ad_id}: {raw['title']}")
    print(f"  Sector: {raw['category_name']}")
    print(f"  Domain: {summary.domain} | Level: {summary.level}")
    print(f"  Summary: {summary.short_description}")
    print(f"  Tasks: {', '.join(summary.tasks)}")
    print(f"  Skills: {', '.join(summary.skills)}")
    desc = (raw["description"] or "")[:200].strip()
    if desc:
        print(f"  Description: {desc}...")
    print(f"{'─'*80}")
    for _, row in ad_matches.iterrows():
        print(f"  #{row['rank']+1}  {row['onet_code']}  {row['onet_title']:<45s}  "
              f"combined={row['combined_score']:.4f}  (role={row['role_score']:.4f}, task={row['taskskill_score']:.4f})")
