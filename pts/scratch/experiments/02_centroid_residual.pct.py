# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Centroid-Residual Matching
#
# **Idea**: Subtract the group centroid from both ad and occupation embeddings
# before computing similarity. This "zooms in" on within-group differences,
# removing the dominant "this is healthcare" / "this is construction" signal
# and focusing on what distinguishes occupations within a group.
#
# Compare top-5 quality against flat cosine similarity baseline.

# %% [markdown]
# ## 1. Setup and data loading

# %%
import json
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_index.const import onet_targets_path, onet_store_path
from ai_index.utils.embed import embed
from ai_index.utils.adzuna_store import get_ads_by_id
from validation.utils import load_matches, matches_to_ranked_lists
from adulib.llm import async_single
from adulib.asynchronous import batch_executor

# %%
EMBEDDING_MODEL = "text-embedding-3-large"
EVAL_MODEL = "openai/gpt-5.4"
SAMPLE_N = 2000
SAMPLE_SEED = 42
EVAL_SAMPLE_N = 200
EVAL_SEED = 123
REFERENCE_RUN = "val__validation_5k__qwen-32b-sbatch__bge-large-sbatch"
TOP_K = 5

# %% [markdown]
# ## 2. Load O\*NET data, build hierarchy, embed

# %%
onet_df = pd.read_parquet(onet_targets_path)
onet_df["major_group"] = onet_df["O*NET-SOC Code"].str[:2]
onet_codes = onet_df["O*NET-SOC Code"].tolist()
onet_titles = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Title"]))
onet_descs = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Description"]))
major_group_codes = sorted(onet_df["major_group"].unique())
group_to_idx = {mg: onet_df[onet_df["major_group"] == mg].index.tolist() for mg in major_group_codes}

print(f"O*NET: {len(onet_df)} occupations, {len(major_group_codes)} major groups")

# %%
# Embed O*NET occupations (cached from notebook 01)
onet_texts = [
    f"{row['Title']}\n\n{row['Description']}\n\nKey tasks and skills: {row['Work Activities/Tasks/Skills']}"
    for _, row in onet_df.iterrows()
]
print("Embedding O*NET occupations (cached)...")
onet_embeds = embed(onet_texts, model=EMBEDDING_MODEL)
onet_norms = onet_embeds / np.linalg.norm(onet_embeds, axis=1, keepdims=True)

# Compute group centroids
centroids = np.zeros((len(major_group_codes), onet_embeds.shape[1]))
for i, mg in enumerate(major_group_codes):
    centroids[i] = onet_embeds[group_to_idx[mg]].mean(axis=0)
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
print(f"Centroids: {centroids.shape}")

# %% [markdown]
# ## 3. Load and embed ads

# %%
ref_df = load_matches(REFERENCE_RUN, stage="filtered")
ref_ranked = matches_to_ranked_lists(ref_df)
ref_ad_ids = sorted(ref_ranked.keys())
rng = np.random.default_rng(SAMPLE_SEED)
sample_ids = rng.choice(ref_ad_ids, size=min(SAMPLE_N, len(ref_ad_ids)), replace=False).tolist()

ads_table = get_ads_by_id(sample_ids, columns=["title", "description", "category_name"])
ads_df = ads_table.to_pandas().set_index("id").loc[sample_ids]

ad_texts = [f"{row['title']}. {str(row['description'] or '')}"[:30000] for _, row in ads_df.iterrows()]
print(f"Embedding {len(ad_texts)} ads (cached)...")
ad_embeds = embed(ad_texts, model=EMBEDDING_MODEL)
ad_norms = ad_embeds / np.linalg.norm(ad_embeds, axis=1, keepdims=True)
print(f"Ads: {ad_embeds.shape}, O*NET: {onet_embeds.shape}")

# %% [markdown]
# ## 4. Flat matching baseline (top-5)

# %%
sim_matrix = ad_norms @ onet_norms.T
flat_top_k_idx = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]

flat_ranked = {}
for i, ad_id in enumerate(sample_ids):
    flat_ranked[ad_id] = [onet_codes[j] for j in flat_top_k_idx[i]]

print(f"Flat matching: {len(flat_ranked)} ads, top-{TOP_K} per ad")

# %% [markdown]
# ## 5. Centroid-residual matching
#
# For each (ad, occupation) pair, we subtract the occupation's group centroid
# from both vectors, then compute cosine similarity on the residuals.
#
# Geometrically: this removes the "group direction" and measures similarity
# in the subspace of within-group variation.

# %%
residual_sim_matrix = np.zeros_like(sim_matrix)
for gi, mg in enumerate(major_group_codes):
    group_indices = group_to_idx[mg]
    if not group_indices:
        continue
    c = centroids[gi]
    # Ad residuals w.r.t. this group's centroid
    ad_res = ad_embeds - c
    ad_res_norms = np.maximum(np.linalg.norm(ad_res, axis=1, keepdims=True), 1e-10)
    ad_res_normed = ad_res / ad_res_norms
    # Occupation residuals
    occ_res = onet_embeds[group_indices] - c
    occ_res_norms = np.maximum(np.linalg.norm(occ_res, axis=1, keepdims=True), 1e-10)
    occ_res_normed = occ_res / occ_res_norms
    residual_sim_matrix[:, group_indices] = ad_res_normed @ occ_res_normed.T

residual_top_k_idx = np.argsort(-residual_sim_matrix, axis=1)[:, :TOP_K]
residual_ranked = {}
for i, ad_id in enumerate(sample_ids):
    residual_ranked[ad_id] = [onet_codes[j] for j in residual_top_k_idx[i]]

print(f"Centroid-residual matching: {len(residual_ranked)} ads, top-{TOP_K} per ad")

# %%
# How often do the two approaches agree?
top1_agree = sum(1 for ad_id in sample_ids if flat_ranked[ad_id][0] == residual_ranked[ad_id][0])
top5_overlap = []
for ad_id in sample_ids:
    overlap = len(set(flat_ranked[ad_id]) & set(residual_ranked[ad_id]))
    top5_overlap.append(overlap)

print(f"\n=== Flat vs Residual Agreement ===")
print(f"Top-1 agreement: {top1_agree / len(sample_ids):.4f}")
print(f"Mean top-5 overlap: {np.mean(top5_overlap):.2f} / 5")
print(f"Top-5 identical: {sum(1 for o in top5_overlap if o == 5) / len(sample_ids):.4f}")

# %% [markdown]
# ## 6. LLM evaluation of top-5

# %%
EVAL_SYSTEM = """You are an expert occupational classification evaluator. You will be given a job advertisement and 5 candidate O*NET occupation matches. Rate how well EACH candidate matches the job ad.

Rating scale:
5 = Perfect match. The occupation precisely describes this job.
4 = Good match. Captures the core role with minor differences.
3 = Partial match. Related field but notable differences in duties or level.
2 = Weak match. Same broad domain but substantially different role.
1 = Poor match. Unrelated or wrong occupation.

Respond with JSON only:
{"ratings": [r1, r2, r3, r4, r5], "best_index": 0, "reasoning": "brief explanation"}"""

def build_eval_prompt(ad_row, top5_codes):
    candidates = ""
    for k, code in enumerate(top5_codes, 1):
        candidates += f"{k}. {code} - {onet_titles[code]}: {onet_descs[code][:150]}\n"
    return f"""## Job Advertisement
**Title:** {ad_row['title']}
**Category:** {ad_row['category_name']}
**Description:** {str(ad_row['description'] or '')[:1500]}

## Candidate Matches
{candidates}
Rate each candidate 1-5."""

async def run_eval(eval_ids, ranked_dict, label):
    prompts = []
    for ad_id in eval_ids:
        ad_row = ads_df.loc[ad_id]
        prompts.append(build_eval_prompt(ad_row, ranked_dict[ad_id]))

    async def _call(prompt):
        resp, _, _ = await async_single(prompt, model=EVAL_MODEL, system=EVAL_SYSTEM,
                                        max_tokens=300, response_format={"type": "json_object"})
        return resp

    print(f"Evaluating {len(prompts)} ads ({label}) with {EVAL_MODEL}...")
    responses = await batch_executor(_call, batch_args=[(p,) for p in prompts], concurrency_limit=20)

    all_ratings = []
    parse_failures = 0
    for resp in responses:
        try:
            parsed = json.loads(resp)
            ratings = [int(r) for r in parsed["ratings"]]
            assert len(ratings) == 5
        except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
            ratings = [None] * 5
            parse_failures += 1
        all_ratings.append(ratings)

    # Compute metrics
    valid = [(i, r) for i, r in enumerate(all_ratings) if r[0] is not None]
    best_in_5 = [max(r) for _, r in valid]
    top1_ratings = [r[0] for _, r in valid]
    all_individual = [x for _, r in valid for x in r]

    print(f"\n=== {label} (n={len(valid)}, {parse_failures} parse failures) ===")
    print(f"Mean top-1 rating:      {np.mean(top1_ratings):.2f}")
    print(f"Mean best-in-top-5:     {np.mean(best_in_5):.2f}")
    print(f"Top-5 has a 4+ match:   {sum(1 for b in best_in_5 if b >= 4) / len(best_in_5):.1%}")
    print(f"Top-5 has a 5 match:    {sum(1 for b in best_in_5 if b >= 5) / len(best_in_5):.1%}")
    print(f"\nAll ratings distribution:")
    counts = Counter(all_individual)
    for r in sorted(counts):
        pct = counts[r] / len(all_individual) * 100
        print(f"  {r}: {counts[r]:4d} ({pct:5.1f}%)")

    return all_ratings

# %%
# Select evaluation sample (same across notebooks for fair comparison)
eval_rng = np.random.default_rng(EVAL_SEED)
eval_ids = eval_rng.choice(sample_ids, size=min(EVAL_SAMPLE_N, len(sample_ids)), replace=False).tolist()
print(f"Evaluation sample: {len(eval_ids)} ads")

# %%
flat_ratings = await run_eval(eval_ids, flat_ranked, "FLAT EMBEDDING")

# %%
residual_ratings = await run_eval(eval_ids, residual_ranked, "CENTROID-RESIDUAL")

# %%
# Head-to-head comparison: for each ad, which approach got a better best-in-top-5?
n_flat_wins = 0
n_residual_wins = 0
n_ties = 0
for i in range(len(eval_ids)):
    fr = flat_ratings[i]
    rr = residual_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    flat_best = max(fr)
    res_best = max(rr)
    if flat_best > res_best:
        n_flat_wins += 1
    elif res_best > flat_best:
        n_residual_wins += 1
    else:
        n_ties += 1

total = n_flat_wins + n_residual_wins + n_ties
print(f"\n=== Head-to-Head (best-in-top-5 rating) ===")
print(f"Flat wins:     {n_flat_wins:4d} ({n_flat_wins/total:.1%})")
print(f"Residual wins: {n_residual_wins:4d} ({n_residual_wins/total:.1%})")
print(f"Ties:          {n_ties:4d} ({n_ties/total:.1%})")

# %% [markdown]
# ## 7. Sample inspection

# %%
def print_match_comparison(ad_id, flat_codes, flat_rats, resid_codes, resid_rats):
    ad_row = ads_df.loc[ad_id]
    desc = textwrap.shorten(str(ad_row["description"] or "")[:300], 200)
    print(f"{'='*90}")
    print(f"Ad: {ad_row['title']} [{ad_row['category_name']}]")
    print(f"    {desc}")
    print(f"\n  {'FLAT':40s} {'RESIDUAL':40s}")
    print(f"  {'-'*40} {'-'*40}")
    for k in range(TOP_K):
        fc, fr_val = flat_codes[k], flat_rats[k] if flat_rats[k] is not None else "?"
        rc, rr_val = resid_codes[k], resid_rats[k] if resid_rats[k] is not None else "?"
        ft = onet_titles[fc][:35]
        rt = onet_titles[rc][:35]
        print(f"  [{fr_val}] {ft:35s}   [{rr_val}] {rt:35s}")
    print()

# Show cases where residual outperforms flat
print("=== CASES WHERE RESIDUAL OUTPERFORMS FLAT ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, rr = flat_ratings[i], residual_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    if max(rr) > max(fr):
        print_match_comparison(ad_id, flat_ranked[ad_id], fr, residual_ranked[ad_id], rr)
        shown += 1
        if shown >= 5:
            break

print("\n=== CASES WHERE FLAT OUTPERFORMS RESIDUAL ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, rr = flat_ratings[i], residual_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    if max(fr) > max(rr):
        print_match_comparison(ad_id, flat_ranked[ad_id], fr, residual_ranked[ad_id], rr)
        shown += 1
        if shown >= 5:
            break

# %% [markdown]
# ## 8. Summary
#
# Compare the centroid-residual approach against flat cosine similarity:
# - Does "zooming in" on within-group differences improve top-5 quality?
# - How often does it change the results at all?
# - Is the improvement consistent or limited to certain groups?
