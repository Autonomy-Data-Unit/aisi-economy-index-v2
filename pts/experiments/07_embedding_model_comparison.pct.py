# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embedding Model Comparison: API vs Open-Source
#
# Tests the full proposed pipeline with two bi-encoder choices:
# - **text-embedding-3-large** (OpenAI API, 3072 dims)
# - **Qwen3-Embedding-8B** (open-source, 4096 dims, runs on Isambard)
#
# Both pipelines use **Qwen3-Reranker-8B** as the reranker (best from notebook 06).
#
# Pipeline: bi-encoder top-100 -> Qwen3-Reranker-8B top-10 -> GPT-5.4 evaluation
#
# This determines whether the full pipeline can run without API dependencies.

# %% [markdown]
# ## 1. Setup

# %%
import json
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_index.const import onet_targets_path
from ai_index.utils.embed import embed, aembed
from ai_index.utils.rerank import arerank
from ai_index.utils.adzuna_store import get_ads_by_id
from validation.utils import load_matches, matches_to_ranked_lists
from adulib.llm import async_single
from adulib.asynchronous import batch_executor

# %%
API_EMBED_MODEL = "text-embedding-3-large"          # OpenAI API
OSS_EMBED_MODEL = "qwen3-embed-8b-sbatch"           # open-source on Isambard
RERANK_MODEL = "qwen3-reranker-8b-sbatch"            # generative reranker
EVAL_MODEL = "openai/gpt-5.4"
SAMPLE_N = 2000
SAMPLE_SEED = 42
EVAL_SAMPLE_N = 200
EVAL_SEED = 123
REFERENCE_RUN = "val__validation_5k__qwen-32b-sbatch__bge-large-sbatch"
TOP_K = 10
RERANK_CANDIDATES = 100

# %% [markdown]
# ## 2. Load O\*NET and ad data

# %%
onet_df = pd.read_parquet(onet_targets_path)
onet_codes = onet_df["O*NET-SOC Code"].tolist()
onet_titles = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Title"]))
onet_descs = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Description"]))

# O*NET texts for embedding and reranking
onet_embed_texts = [
    f"{row['Title']}\n\n{row['Description']}\n\nKey tasks and skills: {row['Work Activities/Tasks/Skills']}"
    for _, row in onet_df.iterrows()
]
onet_rerank_texts = [f"{onet_titles[code]}: {onet_descs[code][:300]}" for code in onet_codes]

ref_df = load_matches(REFERENCE_RUN, stage="filtered")
ref_ranked = matches_to_ranked_lists(ref_df)
rng = np.random.default_rng(SAMPLE_SEED)
sample_ids = rng.choice(sorted(ref_ranked.keys()), size=min(SAMPLE_N, len(ref_ranked)), replace=False).tolist()

ads_table = get_ads_by_id(sample_ids, columns=["title", "description", "category_name"])
ads_df = ads_table.to_pandas().set_index("id").loc[sample_ids]

# Ad texts for embedding (raw title + description)
ad_embed_texts = [f"{row['title']}. {str(row['description'] or '')}"[:30000] for _, row in ads_df.iterrows()]
# Ad texts for reranking (shorter for cross-encoder context)
ad_rerank_texts = [f"{row['title']}. {str(row['description'] or '')[:3000]}" for _, row in ads_df.iterrows()]

print(f"Data: {len(onet_codes)} occupations, {len(sample_ids)} ads")

# Eval sample
eval_rng = np.random.default_rng(EVAL_SEED)
eval_ids = eval_rng.choice(sample_ids, size=min(EVAL_SAMPLE_N, len(sample_ids)), replace=False).tolist()
eval_id_set = set(eval_ids)
eval_idx_in_sample = [sample_ids.index(ad_id) for ad_id in eval_ids]
print(f"Eval sample: {len(eval_ids)} ads")

# %% [markdown]
# ## 3. Embed with text-embedding-3-large (API, cached)

# %%
print("Embedding O*NET with API model (cached)...")
api_onet_embeds = embed(onet_embed_texts, model=API_EMBED_MODEL)
print("Embedding ads with API model (cached)...")
api_ad_embeds = embed(ad_embed_texts, model=API_EMBED_MODEL)
print(f"API embeddings: O*NET {api_onet_embeds.shape}, ads {api_ad_embeds.shape}")

# %% [markdown]
# ## 4. Embed with Qwen3-Embedding-8B (Isambard)

# %%
print("Embedding O*NET with Qwen3-Embedding-8B on Isambard...")
oss_onet_embeds = await aembed(onet_embed_texts, model=OSS_EMBED_MODEL, time="00:30:00")
print(f"O*NET embeddings: {oss_onet_embeds.shape}")

# %%
print("Embedding ads with Qwen3-Embedding-8B on Isambard...")
oss_ad_embeds = await aembed(ad_embed_texts, model=OSS_EMBED_MODEL, time="01:00:00")
print(f"Ad embeddings: {oss_ad_embeds.shape}")

# %% [markdown]
# ## 5. Bi-encoder top-100 candidates

# %%
def get_candidates(ad_embeds, onet_embeds, top_n):
    """Compute cosine similarity and return top-N candidates per ad."""
    ad_norms = ad_embeds / np.linalg.norm(ad_embeds, axis=1, keepdims=True)
    onet_norms = onet_embeds / np.linalg.norm(onet_embeds, axis=1, keepdims=True)
    sim = ad_norms @ onet_norms.T
    top_idx = np.argsort(-sim, axis=1)[:, :top_n]
    return sim, top_idx

api_sim, api_top_idx = get_candidates(api_ad_embeds, api_onet_embeds, RERANK_CANDIDATES)
oss_sim, oss_top_idx = get_candidates(oss_ad_embeds, oss_onet_embeds, RERANK_CANDIDATES)

# Flat top-10 (no reranking)
api_flat = {sample_ids[i]: [onet_codes[j] for j in api_top_idx[i, :TOP_K]] for i in range(len(sample_ids))}
oss_flat = {sample_ids[i]: [onet_codes[j] for j in oss_top_idx[i, :TOP_K]] for i in range(len(sample_ids))}

# How similar are the bi-encoder candidate sets?
overlap_counts = []
for i, ad_id in enumerate(eval_ids):
    si = sample_ids.index(ad_id)
    api_set = set(onet_codes[j] for j in api_top_idx[si, :RERANK_CANDIDATES])
    oss_set = set(onet_codes[j] for j in oss_top_idx[si, :RERANK_CANDIDATES])
    overlap_counts.append(len(api_set & oss_set))

print(f"\n=== Bi-encoder top-{RERANK_CANDIDATES} overlap (API vs OSS) ===")
print(f"Mean overlap: {np.mean(overlap_counts):.1f} / {RERANK_CANDIDATES}")
print(f"Min: {min(overlap_counts)}, Max: {max(overlap_counts)}")

# %% [markdown]
# ## 6. Qwen3-Reranker-8B on both candidate sets

# %%
# Rerank API embedding candidates
eval_query_texts = [ad_rerank_texts[sample_ids.index(ad_id)] for ad_id in eval_ids]

print(f"Reranking API candidates ({RERANK_CANDIDATES} -> {TOP_K}) with Qwen3-Reranker-8B...")
api_rerank_result = await arerank(
    queries=eval_query_texts,
    documents=onet_rerank_texts,
    top_k=RERANK_CANDIDATES,
    model=RERANK_MODEL,
    time="02:00:00",
)
api_rerank_indices = api_rerank_result["indices"]
print(f"API rerank done: {api_rerank_indices.shape}")

# Build top-10 from reranked results, filtered to API bi-encoder candidates
api_reranked = {}
for i, ad_id in enumerate(eval_ids):
    si = sample_ids.index(ad_id)
    bi_candidates = set(onet_codes[j] for j in api_top_idx[si, :RERANK_CANDIDATES])
    filtered = []
    for j in api_rerank_indices[i]:
        code = onet_codes[j]
        if code in bi_candidates:
            filtered.append(code)
        if len(filtered) >= TOP_K:
            break
    api_reranked[ad_id] = filtered[:TOP_K]

# %%
# Rerank OSS embedding candidates
# The reranker scores are the same (same queries x same documents), but we
# filter to different candidate sets. Since we already have the full rerank
# scores, we can reuse them.
oss_reranked = {}
for i, ad_id in enumerate(eval_ids):
    si = sample_ids.index(ad_id)
    bi_candidates = set(onet_codes[j] for j in oss_top_idx[si, :RERANK_CANDIDATES])
    filtered = []
    for j in api_rerank_indices[i]:  # same rerank order
        code = onet_codes[j]
        if code in bi_candidates:
            filtered.append(code)
        if len(filtered) >= TOP_K:
            break
    oss_reranked[ad_id] = filtered[:TOP_K]

print(f"Built reranked top-{TOP_K} for both pipelines")

# %% [markdown]
# ## 7. LLM evaluation

# %%
EVAL_SYSTEM = f"""You are an expert occupational classification evaluator. You will be given a job advertisement and {TOP_K} candidate O*NET occupation matches. Rate how well EACH candidate matches the job ad.

Rating scale:
5 = Perfect match. The occupation precisely describes this job.
4 = Good match. Captures the core role with minor differences.
3 = Partial match. Related field but notable differences in duties or level.
2 = Weak match. Same broad domain but substantially different role.
1 = Poor match. Unrelated or wrong occupation.

Respond with JSON only:
{{"ratings": [r1, r2, ...], "best_index": 0, "reasoning": "brief explanation"}}

The ratings list must have exactly one rating per candidate match."""

async def run_eval(eval_ids_list, ranked_dict, label):
    prompts = []
    for ad_id in eval_ids_list:
        ad_row = ads_df.loc[ad_id]
        candidates = "\n".join(
            f"{k}. {code} - {onet_titles[code]}: {onet_descs[code][:150]}"
            for k, code in enumerate(ranked_dict[ad_id], 1)
        )
        prompts.append(
            f"## Job Advertisement\n**Title:** {ad_row['title']}\n**Category:** {ad_row['category_name']}\n"
            f"**Description:** {str(ad_row['description'] or '')[:1500]}\n\n## Candidate Matches\n{candidates}\n\n"
            f"Rate each candidate 1-5."
        )

    async def _call(prompt):
        resp, _, _ = await async_single(prompt, model=EVAL_MODEL, system=EVAL_SYSTEM,
                                        max_tokens=400, response_format={"type": "json_object"})
        return resp

    print(f"Evaluating {len(prompts)} ads ({label})...")
    responses = await batch_executor(_call, batch_args=[(p,) for p in prompts], concurrency_limit=20)

    all_ratings = []
    parse_failures = 0
    for resp in responses:
        try:
            ratings = [int(r) for r in json.loads(resp)["ratings"]]
            assert len(ratings) == TOP_K
        except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
            ratings = [None] * TOP_K
            parse_failures += 1
        all_ratings.append(ratings)

    valid = [(i, r) for i, r in enumerate(all_ratings) if r[0] is not None]
    best_in_k = [max(r) for _, r in valid]
    top1_ratings = [r[0] for _, r in valid]

    hit_4plus = sum(1 for b in best_in_k if b >= 4) / len(best_in_k)
    hit_5 = sum(1 for b in best_in_k if b >= 5) / len(best_in_k)

    print(f"\n=== {label} (n={len(valid)}, {parse_failures} failures) ===")
    print(f"Mean top-1:             {np.mean(top1_ratings):.2f}")
    print(f"Mean best-in-{TOP_K}:        {np.mean(best_in_k):.2f}")
    print(f"Top-{TOP_K} has 4+ match:    {hit_4plus:.1%}  <-- PRIMARY")
    print(f"Top-{TOP_K} has 5 match:     {hit_5:.1%}")

    counts = Counter(x for _, r in valid for x in r)
    print(f"\nRating distribution:")
    for rating in sorted(counts):
        print(f"  {rating}: {counts[rating]:4d} ({counts[rating]/sum(counts.values())*100:5.1f}%)")

    return all_ratings, {"label": label, "n": len(valid), "mean_top1": np.mean(top1_ratings),
                         "mean_best": np.mean(best_in_k), "hit_4plus": hit_4plus, "hit_5": hit_5}

# %%
api_flat_ratings, api_flat_stats = await run_eval(eval_ids, api_flat, "API FLAT (text-emb-3-large)")

# %%
oss_flat_ratings, oss_flat_stats = await run_eval(eval_ids, oss_flat, "OSS FLAT (Qwen3-Embed-8B)")

# %%
api_rr_ratings, api_rr_stats = await run_eval(eval_ids, api_reranked, "API + RERANKER")

# %%
oss_rr_ratings, oss_rr_stats = await run_eval(eval_ids, oss_reranked, "OSS + RERANKER")

# %% [markdown]
# ## 8. Results

# %%
summary = pd.DataFrame([api_flat_stats, oss_flat_stats, api_rr_stats, oss_rr_stats])
summary.columns = ["Pipeline", "N", "Mean top-1", f"Mean best-in-{TOP_K}", f"Top-{TOP_K} has 4+", f"Top-{TOP_K} has 5"]
print(f"\n{'='*95}")
print(f"  RESULTS (top-{TOP_K})")
print(f"{'='*95}")
print(summary.to_string(index=False))
print(f"\nPrimary metric: 'Top-{TOP_K} has 4+' = fraction of ads with at least one good match")

# %%
# Head-to-head: API+reranker vs OSS+reranker (the key comparison)
a_wins = b_wins = ties = 0
for i in range(len(eval_ids)):
    ra, rb = api_rr_ratings[i], oss_rr_ratings[i]
    if ra[0] is None or rb[0] is None:
        continue
    ma, mb = max(ra), max(rb)
    if ma > mb: a_wins += 1
    elif mb > ma: b_wins += 1
    else: ties += 1
t = a_wins + b_wins + ties
print(f"\n=== API+Reranker vs OSS+Reranker (best-in-{TOP_K}) ===")
print(f"API wins: {a_wins}/{t} ({a_wins/t:.1%})")
print(f"OSS wins: {b_wins}/{t} ({b_wins/t:.1%})")
print(f"Ties:     {ties}/{t} ({ties/t:.1%})")

# %%
# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = ["API flat", "OSS flat", "API+reranker", "OSS+reranker"]
colors = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f"]

hit_vals = [api_flat_stats["hit_4plus"], oss_flat_stats["hit_4plus"],
            api_rr_stats["hit_4plus"], oss_rr_stats["hit_4plus"]]
axes[0].bar(labels, hit_vals, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_ylabel(f"Fraction with 4+ match in top-{TOP_K}")
axes[0].set_title(f"Recall (top-{TOP_K} has 4+)")
axes[0].set_ylim(0.8, 1.02)
for i, v in enumerate(hit_vals):
    axes[0].text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")

best_vals = [api_flat_stats["mean_best"], oss_flat_stats["mean_best"],
             api_rr_stats["mean_best"], oss_rr_stats["mean_best"]]
axes[1].bar(labels, best_vals, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_ylabel(f"Mean best-in-top-{TOP_K} rating")
axes[1].set_title(f"Best match quality")
axes[1].set_ylim(4.0, 5.1)
for i, v in enumerate(best_vals):
    axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Embedding Model Comparison: API vs Open-Source", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Conclusions
#
# ### Can the full pipeline run without API dependencies?
#
# Compare the two full pipelines:
# - **API pipeline**: text-embedding-3-large + Qwen3-Reranker-8B
# - **OSS pipeline**: Qwen3-Embedding-8B + Qwen3-Reranker-8B (fully on Isambard)
#
# If the OSS pipeline matches or exceeds the API pipeline, the entire matching
# stage can run on Isambard with no API costs for embedding.
#
# ### Cost implications at 30M ads
#
# - **API embedding**: ~30M x 2K tokens x $0.13/M tokens = ~$7,800
# - **OSS embedding on Isambard**: GPU-hours only (~0.25 NHR/GPU-hour)
# - **Reranking**: same cost either way (Qwen3-Reranker-8B on Isambard)
