# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.cosine_candidates
#
# Compute cosine similarity between ad embeddings and O*NET embeddings,
# select top-N candidates per ad for downstream reranking.
#
# 1. Loads ad embeddings from `embed_ads/embeddings.duckdb`.
# 2. Loads O*NET embeddings from `embed_onet/onet_embeddings.npy`.
# 3. Computes cosine similarity (single channel, not dual).
# 4. Saves top-N candidates per ad to `cosine_candidates/candidates.parquet`.
#
# Node variables:
# - `cosine_topk` (per-node): Number of candidates to keep (default 100)
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.cosine_candidates
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int], onet_done: bool) -> {
    'ad_ids': list[int]
}:
    """Cosine similarity top-N candidates for reranking."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('cosine_candidates', run_name=run_name)
show_node_vars('cosine_candidates', run_name=run_name)

# %% [markdown]
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import json

import duckdb
import numpy as np
import pandas as pd

from ai_index import const

# %%
#|export
run_name = ctx.vars["run_name"]
cosine_topk = ctx.vars["cosine_topk"]

output_dir = const.pipeline_store_path / run_name / "cosine_candidates"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load O*NET embeddings

# %%
#|export
onet_dir = const.pipeline_store_path / run_name / "embed_onet"

with open(onet_dir / "onet_codes.json") as f:
    onet_codes = json.load(f)

onet_embeds = np.load(onet_dir / "onet_embeddings.npy")

# Load O*NET titles for output (from the source parquet)
onet_targets = pd.read_parquet(const.onet_targets_path)
onet_titles = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Title"]))

n_onet = len(onet_codes)
print(f"cosine_candidates: loaded {n_onet} O*NET embeddings ({onet_embeds.shape})")

# Normalize O*NET embeddings once
onet_norms = np.linalg.norm(onet_embeds, axis=1, keepdims=True)
onet_norms = np.maximum(onet_norms, 1e-10)
onet_normed = onet_embeds / onet_norms

# %% [markdown]
# ## Load ad embeddings and compute cosine similarity
#
# Process ads in chunks to avoid loading all embeddings at once.

# %%
#|export
embed_db = const.pipeline_store_path / run_name / "embed_ads" / "embeddings.duckdb"
embed_conn = duckdb.connect(str(embed_db), read_only=True)

n_ads = len(ad_ids)
CHUNK_SIZE = 10000
n_chunks = (n_ads + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"cosine_candidates: {n_ads} ads x {n_onet} occupations, topk={cosine_topk}")
print(f"  processing in {n_chunks} chunks")

all_rows = []

for chunk_idx in range(n_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_ads)
    chunk_ad_ids = ad_ids[start:end]

    # Load embeddings for this chunk
    embed_conn.execute("CREATE OR REPLACE TEMP TABLE _chunk_order (id BIGINT, pos INTEGER)")
    embed_conn.executemany(
        "INSERT INTO _chunk_order VALUES (?, ?)",
        [(int(aid), i) for i, aid in enumerate(chunk_ad_ids)],
    )
    rows = embed_conn.execute(
        "SELECT r.embedding FROM results r "
        "JOIN _chunk_order a ON r.id = a.id ORDER BY a.pos"
    ).fetchall()

    ad_embeds = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])

    # Normalize ad embeddings
    ad_norms = np.linalg.norm(ad_embeds, axis=1, keepdims=True)
    ad_norms = np.maximum(ad_norms, 1e-10)
    ad_normed = ad_embeds / ad_norms

    # Cosine similarity: (chunk_size, n_onet)
    sim_matrix = ad_normed @ onet_normed.T

    # Top-K per ad
    topk_clamped = min(cosine_topk, n_onet)
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :topk_clamped]
    top_scores = np.take_along_axis(sim_matrix, top_indices, axis=1)

    for i in range(len(chunk_ad_ids)):
        for rank in range(topk_clamped):
            onet_idx = int(top_indices[i, rank])
            all_rows.append({
                "ad_id": int(chunk_ad_ids[i]),
                "rank": rank,
                "onet_code": onet_codes[onet_idx],
                "onet_title": onet_titles[onet_codes[onet_idx]],
                "cosine_score": float(top_scores[i, rank]),
            })

    print(f"  chunk {chunk_idx + 1}/{n_chunks}: {len(chunk_ad_ids)} ads")

embed_conn.close()

# %% [markdown]
# ## Save results

# %%
#|export
candidates_df = pd.DataFrame(all_rows)
output_path = output_dir / "candidates.parquet"
candidates_df.to_parquet(output_path, index=False)

print(f"cosine_candidates: wrote {len(candidates_df)} candidate rows ({n_ads} ads x topk={cosine_topk})")
print(f"  output: {output_path}")
if len(candidates_df) > 0:
    print(f"  score range: {candidates_df['cosine_score'].min():.4f} - {candidates_df['cosine_score'].max():.4f}")

ad_ids #|func_return_line

# %% [markdown]
# ## Sample candidates

# %%
from ai_index.utils import get_adzuna_conn

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
    ad_matches = candidates_df[candidates_df["ad_id"] == ad_id].head(5)
    print(f"\n{'━'*80}")
    print(f"Ad {ad_id}: {raw['title']} [{raw['category_name']}]")
    print(f"{'─'*80}")
    for _, row in ad_matches.iterrows():
        print(f"  #{row['rank']+1}  {row['onet_code']}  {row['onet_title']:<45s}  "
              f"cosine={row['cosine_score']:.4f}")
