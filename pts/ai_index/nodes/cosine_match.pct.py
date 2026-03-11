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
# - `run_name` (global): Pipeline run name

# %%
from nblite import nbl_export; nbl_export();

# %%
#|default_exp nodes.cosine_match
#|export_as_func true

# %%
#|top_export
import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils import acosine_topk

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int], onet_done: bool):
    """Weighted dual cosine similarity matching."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dotenv import load_dotenv; load_dotenv()
from dev_utils import set_node_func_args
set_node_func_args('cosine_match')

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
run_name = ctx.vars["run_name"]
cosine_mode = ctx.vars["cosine_mode"]
topk = ctx.vars["topk"]
cosine_alpha = ctx.vars["cosine_alpha"]

output_dir = const.pipeline_store_path / run_name / "cosine_match"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load embeddings

# %%
#|export
ads_dir = const.pipeline_store_path / run_name / "embed_ads"
onet_dir = const.pipeline_store_path / run_name / "embed_onet"

ad_role_embeds = np.load(ads_dir / "role_embeddings.npy")
ad_task_embeds = np.load(ads_dir / "taskskill_embeddings.npy")

onet_codes = np.load(onet_dir / "onet_codes.npy")
onet_titles = np.load(onet_dir / "onet_titles.npy")
onet_role_embeds = np.load(onet_dir / "role_embeddings.npy")
onet_task_embeds = np.load(onet_dir / "taskskill_embeddings.npy")

n_ads = len(ad_ids)
n_onet = len(onet_codes)
print(f"cosine_match: {n_ads} ads x {n_onet} occupations, topk={topk}, alpha={cosine_alpha}")

# %% [markdown]
# ## Compute cosine similarity
#
# Run cosine top-K separately for role and taskskill embeddings.
# Candidates appearing in only one top-K list get 0 for the missing score.

# %%
#|export
role_results = await acosine_topk(ad_role_embeds, onet_role_embeds, k=topk, mode=cosine_mode)
print(f"cosine_match: role cosine done — indices {role_results['indices'].shape}")

task_results = await acosine_topk(ad_task_embeds, onet_task_embeds, k=topk, mode=cosine_mode)
print(f"cosine_match: task cosine done — indices {task_results['indices'].shape}")

# %% [markdown]
# ## Weighted merge
#
# For each ad, merge role and task candidates, compute combined score,
# and keep the final top-K.

# %%
#|export
role_indices = role_results["indices"]   # (n_ads, topk)
role_scores = role_results["scores"]     # (n_ads, topk)
task_indices = task_results["indices"]   # (n_ads, topk)
task_scores = task_results["scores"]     # (n_ads, topk)

rows = []
for i in range(n_ads):
    # Build lookup: onet_idx -> [role_score, task_score]
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
            "ad_id": int(ad_ids[i]),
            "rank": rank,
            "onet_code": str(onet_codes[idx]),
            "onet_title": str(onet_titles[idx]),
            "role_score": rs,
            "taskskill_score": ts,
            "combined_score": combined,
        })

# %% [markdown]
# ## Save results

# %%
#|export
matches_df = pd.DataFrame(rows)
output_path = output_dir / "matches.parquet"
matches_df.to_parquet(output_path, index=False)

print(f"cosine_match: wrote {len(matches_df)} match rows ({n_ads} ads x topk={topk})")
print(f"  output: {output_path}")
print(f"  score range: {matches_df['combined_score'].min():.4f} - {matches_df['combined_score'].max():.4f}")

# %% [markdown]
# ## Sample matches
#
# Show the top O\*NET matches for each job ad, with the ad title for context.

# %%
from ai_index.utils import get_adzuna_conn

conn = get_adzuna_conn(read_only=True)
conn.execute("CREATE OR REPLACE TEMP TABLE _match_ids (id BIGINT)")
conn.executemany("INSERT INTO _match_ids VALUES (?)", [(int(i),) for i in ad_ids])
ad_titles = dict(conn.execute(
    "SELECT a.id, a.title FROM ads a JOIN _match_ids m ON a.id = m.id"
).fetchall())
conn.close()

for ad_id in ad_ids[:5]:
    ad_id = int(ad_id)
    title = ad_titles[ad_id]
    ad_matches = matches_df[matches_df["ad_id"] == ad_id].head(5)
    print(f"\n{'─'*80}")
    print(f"Ad {ad_id}: {title}")
    print(f"{'─'*80}")
    for _, row in ad_matches.iterrows():
        print(f"  #{row['rank']+1}  {row['onet_code']}  {row['onet_title']:<45s}  "
              f"combined={row['combined_score']:.4f}  (role={row['role_score']:.4f}, task={row['taskskill_score']:.4f})")
