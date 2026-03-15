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
# Rerank cosine candidates using a reranker model (e.g. Qwen3-Reranker on GPU).
#
# 1. Loads cosine candidates from `cosine_candidates/candidates.parquet`.
# 2. Loads raw ad text from Adzuna DuckDB for query construction.
# 3. Builds O*NET document texts for the reranker.
# 4. For each ad, scores its cosine candidates using `arerank()`.
# 5. Saves reranked top-K to `rerank_candidates/reranked_matches.parquet`.
#
# Node variables:
# - `rerank_model` (per-node): Model key from rerank_models.toml
# - `rerank_topk` (per-node): Number of candidates to keep after reranking
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.rerank_candidates
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int]) -> {
    'ad_ids': list[int]
}:
    """Rerank cosine candidates with a cross-encoder/generative reranker."""
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

import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils import arerank, get_ads_by_id

# %%
#|export
run_name = ctx.vars["run_name"]
rerank_model = ctx.vars["rerank_model"]
rerank_topk = ctx.vars["rerank_topk"]
sbatch_time = ctx.vars["sbatch_time"]

output_dir = const.pipeline_store_path / run_name / "rerank_candidates"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load cosine candidates

# %%
#|export
candidates_path = const.pipeline_store_path / run_name / "cosine_candidates" / "candidates.parquet"
candidates_df = pd.read_parquet(candidates_path)

# Load O*NET titles and descriptions for reranker document text
onet_targets = pd.read_parquet(const.onet_targets_path)
onet_descs = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Description"]))
onet_titles = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Title"]))

n_ads = len(ad_ids)
n_candidates_per_ad = candidates_df.groupby("ad_id").size().iloc[0]
print(f"rerank_candidates: {n_ads} ads, {n_candidates_per_ad} candidates each")
print(f"  rerank_model: {rerank_model}")
print(f"  rerank_topk: {rerank_topk}")

# %% [markdown]
# ## Rerank

# %%
#|export
# Build unique O*NET document texts across all candidates
all_candidate_codes = candidates_df["onet_code"].unique().tolist()
doc_texts = [f"{onet_titles[code]}: {onet_descs[code][:300]}" for code in all_candidate_codes]

# Build query texts
print("rerank_candidates: loading ad texts...")
ads_table = get_ads_by_id(ad_ids, columns=["title", "description"])
ads_df = ads_table.to_pandas().set_index("id")

query_texts = []
for ad_id in ad_ids:
    row = ads_df.loc[ad_id]
    query_texts.append(f"{row['title']}. {str(row['description'] or '')[:3000]}")

# Call reranker: score all queries against all candidate documents
print(f"rerank_candidates: scoring {len(query_texts)} queries x {len(doc_texts)} documents...")
rerank_result = await arerank(
    queries=query_texts,
    documents=doc_texts,
    top_k=len(doc_texts),  # get full ranking so we can filter to per-ad candidates
    model=rerank_model,
    time=sbatch_time,
)
rerank_indices = rerank_result["indices"]   # (n_ads, n_docs)
rerank_scores = rerank_result["scores"]     # (n_ads, n_docs)

# For each ad, filter to its cosine candidates and take top-K
reranked_rows = []
for qi, ad_id in enumerate(ad_ids):
    ad_candidates = set(
        candidates_df[candidates_df["ad_id"] == ad_id]["onet_code"].tolist()
    )

    filtered = []
    for j in range(rerank_indices.shape[1]):
        doc_idx = int(rerank_indices[qi, j])
        code = all_candidate_codes[doc_idx]
        if code in ad_candidates:
            filtered.append({
                "ad_id": int(ad_id),
                "onet_code": code,
                "onet_title": onet_titles[code],
                "rerank_score": float(rerank_scores[qi, j]),
            })
        if len(filtered) >= rerank_topk:
            break

    for rank, row in enumerate(filtered):
        row["rank"] = rank
        reranked_rows.append(row)

print(f"rerank_candidates: reranking complete")

# %% [markdown]
# ## Save results

# %%
#|export
reranked_df = pd.DataFrame(reranked_rows)
output_path = output_dir / "reranked_matches.parquet"
reranked_df.to_parquet(output_path, index=False)

print(f"rerank_candidates: wrote {len(reranked_df)} match rows ({n_ads} ads x topk={rerank_topk})")
print(f"  output: {output_path}")
if len(reranked_df) > 0:
    print(f"  score range: {reranked_df['rerank_score'].min():.4f} - {reranked_df['rerank_score'].max():.4f}")

ad_ids #|func_return_line

# %% [markdown]
# ## Sample reranked matches

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
    ad_matches = reranked_df[reranked_df["ad_id"] == ad_id].head(5)
    print(f"\n{'━'*80}")
    print(f"Ad {ad_id}: {raw['title']} [{raw['category_name']}]")
    print(f"{'─'*80}")
    for _, row in ad_matches.iterrows():
        print(f"  #{row['rank']+1}  {row['onet_code']}  {row['onet_title']:<45s}  "
              f"rerank={row['rerank_score']:.4f}")
