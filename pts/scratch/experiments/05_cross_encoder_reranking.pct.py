# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-Encoder Reranking Experiment
#
# Repeats the reranking comparison from notebook 04 using the new cross-encoder
# infrastructure (`ai_index.utils.rerank`). Tests the full pipeline:
# bi-encoder top-100 -> cross-encoder top-10 -> LLM evaluation.
#
# Uses Voyage rerank-2.5 API for this experiment. For production, switch to
# `bge-reranker-v2-m3-sbatch` which runs on Isambard GPU.
#
# ## Pipeline comparison
#
# | Approach | Source | Candidate pool | Final top-10 |
# |----------|--------|---------------|-------------|
# | Flat embedding | bi-encoder | 861 | top-10 by cosine |
# | Cross-encoder | bi-encoder + CE | top-100 -> CE rerank | top-10 by CE score |
# | LLM reranking | bi-encoder + LLM | top-20 -> LLM rerank | top-10 by LLM |

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
from ai_index.utils.embed import embed
from ai_index.utils.adzuna_store import get_ads_by_id
from validation.utils import load_matches, matches_to_ranked_lists
from ai_index.utils.rerank import arerank
from adulib.llm import async_single
from adulib.asynchronous import batch_executor
import asyncio

# %%
EMBEDDING_MODEL = "text-embedding-3-large"
CE_RERANK_MODEL = "bge-reranker-v2-m3-sbatch"  # cross-encoder on Isambard GPU
LLM_RERANK_MODEL = "openai/gpt-5.4"
EVAL_MODEL = "openai/gpt-5.4"
SAMPLE_N = 2000
SAMPLE_SEED = 42
EVAL_SAMPLE_N = 200
EVAL_SEED = 123
REFERENCE_RUN = "val__validation_5k__qwen-32b-sbatch__bge-large-sbatch"
TOP_K = 10
CE_CANDIDATES = 100
LLM_CANDIDATES = 20

# %% [markdown]
# ## 2. Load data and embed (all cached)

# %%
onet_df = pd.read_parquet(onet_targets_path)
onet_codes = onet_df["O*NET-SOC Code"].tolist()
onet_titles = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Title"]))
onet_descs = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Description"]))

onet_texts = [
    f"{row['Title']}\n\n{row['Description']}\n\nKey tasks and skills: {row['Work Activities/Tasks/Skills']}"
    for _, row in onet_df.iterrows()
]
print("Embedding O*NET (cached)...")
onet_embeds = embed(onet_texts, model=EMBEDDING_MODEL)
onet_norms = onet_embeds / np.linalg.norm(onet_embeds, axis=1, keepdims=True)

ref_df = load_matches(REFERENCE_RUN, stage="filtered")
ref_ranked = matches_to_ranked_lists(ref_df)
rng = np.random.default_rng(SAMPLE_SEED)
sample_ids = rng.choice(sorted(ref_ranked.keys()), size=min(SAMPLE_N, len(ref_ranked)), replace=False).tolist()

ads_table = get_ads_by_id(sample_ids, columns=["title", "description", "category_name"])
ads_df = ads_table.to_pandas().set_index("id").loc[sample_ids]

ad_texts = [f"{row['title']}. {str(row['description'] or '')}"[:30000] for _, row in ads_df.iterrows()]
print(f"Embedding {len(ad_texts)} ads (cached)...")
ad_embeds = embed(ad_texts, model=EMBEDDING_MODEL)
ad_norms = ad_embeds / np.linalg.norm(ad_embeds, axis=1, keepdims=True)
print(f"Data loaded: {len(onet_codes)} occupations, {len(sample_ids)} ads")

# %% [markdown]
# ## 3. Bi-encoder candidates

# %%
sim_matrix = ad_norms @ onet_norms.T

flat_top_idx = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]
flat_ranked = {sample_ids[i]: [onet_codes[j] for j in flat_top_idx[i]] for i in range(len(sample_ids))}

ce_top_idx = np.argsort(-sim_matrix, axis=1)[:, :CE_CANDIDATES]
ce_candidates = {sample_ids[i]: [onet_codes[j] for j in ce_top_idx[i]] for i in range(len(sample_ids))}

llm_top_idx = np.argsort(-sim_matrix, axis=1)[:, :LLM_CANDIDATES]
llm_candidates = {sample_ids[i]: [onet_codes[j] for j in llm_top_idx[i]] for i in range(len(sample_ids))}

eval_rng = np.random.default_rng(EVAL_SEED)
eval_ids = eval_rng.choice(sample_ids, size=min(EVAL_SAMPLE_N, len(sample_ids)), replace=False).tolist()
print(f"Eval sample: {len(eval_ids)} ads")

# %% [markdown]
# ## 4. Cross-encoder reranking (top-100 -> top-10)

# %%
# Build O*NET document texts for the reranker (title + description)
# Use all 861 occupations as the shared document list. After reranking, we'll
# filter to each ad's bi-encoder top-100 candidates.
onet_doc_texts_list = [
    f"{onet_titles[code]}: {onet_descs[code][:300]}"
    for code in onet_codes
]

# Build query texts for the eval sample
eval_ad_texts = []
for ad_id in eval_ids:
    ad_row = ads_df.loc[ad_id]
    eval_ad_texts.append(f"{ad_row['title']}. {str(ad_row['description'] or '')[:3000]}")

print(f"{len(eval_ad_texts)} queries x {len(onet_doc_texts_list)} documents")

# %%
# Run cross-encoder on Isambard: score all 200 ads against all 861 occupations
# in a single sbatch job. This is 200 * 861 = 172,200 pairs.
print(f"Cross-encoder reranking {len(eval_ids)} ads x {len(onet_codes)} occupations on Isambard...")

# We request top-K from the full occupation list, but we'll also build a
# candidate-filtered version after
ce_result = await arerank(
    queries=eval_ad_texts,
    documents=onet_doc_texts_list,
    top_k=CE_CANDIDATES,  # get top-100 from CE to compare with bi-encoder top-100
    model=CE_RERANK_MODEL,
    time="00:30:00",
)

ce_indices = ce_result["indices"]   # (200, 100)
ce_scores = ce_result["scores"]     # (200, 100)
print(f"CE reranking complete: {ce_indices.shape}")

# %%
# Build ranked lists: take top-10 from the CE-ranked results
# Option A: pure CE ranking (ignoring bi-encoder candidates entirely)
ce_ranked = {}
for i, ad_id in enumerate(eval_ids):
    ce_ranked[ad_id] = [onet_codes[j] for j in ce_indices[i, :TOP_K]]

# Option B: CE ranking filtered to bi-encoder top-100 candidates
# This is the proposed pipeline: bi-encoder recall -> CE rerank
ce_filtered_ranked = {}
for i, ad_id in enumerate(eval_ids):
    bi_candidates = set(ce_candidates[ad_id])  # bi-encoder top-100
    filtered = []
    for j in ce_indices[i]:
        code = onet_codes[j]
        if code in bi_candidates:
            filtered.append(code)
        if len(filtered) >= TOP_K:
            break
    # Pad if needed (shouldn't happen with 100 candidates)
    if len(filtered) < TOP_K:
        for code in ce_candidates[ad_id]:
            if code not in filtered:
                filtered.append(code)
            if len(filtered) >= TOP_K:
                break
    ce_filtered_ranked[ad_id] = filtered

print(f"CE pure top-{TOP_K}: {len(ce_ranked)} ads")
print(f"CE filtered (bi-encoder top-{CE_CANDIDATES} -> CE top-{TOP_K}): {len(ce_filtered_ranked)} ads")

# %% [markdown]
# ## 5. LLM reranking (top-20 -> top-10)

# %%
RERANK_SYSTEM = f"""You are an expert occupational classifier. Given a job advertisement and {LLM_CANDIDATES} candidate O*NET occupations, select the {TOP_K} best matches and rank them.

Consider: job duties, required skills, professional level, specialization, and industry context.

Respond with JSON only:
{{"ranked_codes": ["code1", "code2", ...], "reasoning": "brief explanation"}}

Return exactly {TOP_K} codes."""

# %%
print(f"LLM reranking {len(eval_ids)} ads...")

async def _llm_call(prompt):
    resp, _, _ = await async_single(prompt, model=LLM_RERANK_MODEL, system=RERANK_SYSTEM,
                                    max_tokens=400, response_format={"type": "json_object"})
    return resp

rerank_prompts = []
for ad_id in eval_ids:
    ad_row = ads_df.loc[ad_id]
    candidates = "\n".join(
        f"{k}. {code} - {onet_titles[code]}: {onet_descs[code][:200]}"
        for k, code in enumerate(llm_candidates[ad_id], 1)
    )
    rerank_prompts.append(
        f"## Job Advertisement\n**Title:** {ad_row['title']}\n**Category:** {ad_row['category_name']}\n"
        f"**Description:** {str(ad_row['description'] or '')[:2000]}\n\n## Candidates\n{candidates}\n\n"
        f"Select the {TOP_K} best matches."
    )

llm_responses = await batch_executor(_llm_call, batch_args=[(p,) for p in rerank_prompts], concurrency_limit=20)

llm_ranked = {}
for i, (ad_id, resp) in enumerate(zip(eval_ids, llm_responses)):
    cand_set = set(llm_candidates[ad_id])
    try:
        codes = json.loads(resp)["ranked_codes"]
        valid = [c for c in codes if c in cand_set]
        if len(valid) < TOP_K:
            for c in llm_candidates[ad_id]:
                if c not in valid:
                    valid.append(c)
                if len(valid) >= TOP_K:
                    break
        llm_ranked[ad_id] = valid[:TOP_K]
    except (json.JSONDecodeError, KeyError):
        llm_ranked[ad_id] = llm_candidates[ad_id][:TOP_K]

print(f"LLM reranking complete: {len(llm_ranked)} ads")

# %% [markdown]
# ## 6. LLM evaluation of all three approaches

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
flat_ratings, flat_stats = await run_eval(eval_ids, flat_ranked, "FLAT EMBEDDING")

# %%
ce_ratings, ce_stats = await run_eval(eval_ids, ce_ranked, "CE PURE (861->10)")

# %%
cef_ratings, cef_stats = await run_eval(eval_ids, ce_filtered_ranked, "CE FILTERED (100->10)")

# %%
llm_ratings, llm_stats = await run_eval(eval_ids, llm_ranked, "LLM RERANKED (20->10)")

# %% [markdown]
# ## 7. Results

# %%
summary = pd.DataFrame([flat_stats, ce_stats, cef_stats, llm_stats])
summary.columns = ["Approach", "N", "Mean top-1", f"Mean best-in-{TOP_K}", f"Top-{TOP_K} has 4+", f"Top-{TOP_K} has 5"]
print(f"\n{'='*90}")
print(f"  RESULTS (top-{TOP_K})")
print(f"{'='*90}")
print(summary.to_string(index=False))

# %%
# Head-to-head
comparisons = [
    ("CE pure vs Flat", ce_ratings, flat_ratings),
    ("CE filtered vs Flat", cef_ratings, flat_ratings),
    ("LLM vs Flat", llm_ratings, flat_ratings),
    ("CE filtered vs LLM", cef_ratings, llm_ratings),
]
for comp_label, rats_a, rats_b in comparisons:
    a_wins = b_wins = ties = 0
    for i in range(len(eval_ids)):
        ra, rb = rats_a[i], rats_b[i]
        if ra[0] is None or rb[0] is None:
            continue
        ma, mb = max(ra), max(rb)
        if ma > mb: a_wins += 1
        elif mb > ma: b_wins += 1
        else: ties += 1
    t = a_wins + b_wins + ties
    names = comp_label.split(" vs ")
    print(f"\n{comp_label} (best-in-{TOP_K}): {names[0]} wins {a_wins}/{t} ({a_wins/t:.1%}), "
          f"{names[1]} wins {b_wins}/{t} ({b_wins/t:.1%}), ties {ties}/{t} ({ties/t:.1%})")

# %%
# Sample: cases where CE found good matches that flat missed
print("\n=== CE FOUND GOOD MATCH NOT IN FLAT TOP-10 ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, cr = flat_ratings[i], ce_ratings[i]
    if fr[0] is None or cr[0] is None:
        continue
    if max(cr) >= 4 and max(fr) <= 2:
        ad_row = ads_df.loc[ad_id]
        print(f"Ad: {ad_row['title']} [{ad_row['category_name']}]")
        print(f"  Flat best rating: {max(fr)}, CE best rating: {max(cr)}")
        best_ce_idx = cr.index(max(cr))
        best_ce_code = ce_ranked[ad_id][best_ce_idx]
        print(f"  CE best match: {best_ce_code} - {onet_titles[best_ce_code]}")
        print()
        shown += 1
        if shown >= 5:
            break

if shown == 0:
    print("  (No cases where CE found 4+ and flat had only <=2)")

# %% [markdown]
# ## 8. Conclusions
#
# ### Experiment setup
#
# This notebook tested the proposed three-stage pipeline using the new
# cross-encoder infrastructure. The cross-encoder (Voyage rerank-2.5 API)
# rescored the bi-encoder's top-100 candidates and selected the top-10.
# For comparison, LLM reranking (GPT-5.4) selected top-10 from top-20.
#
# ### For production on Isambard
#
# The infrastructure is now in place to run cross-encoder reranking on
# Isambard via sbatch:
#
# ```python
# from ai_index.utils.rerank import arerank
#
# result = await arerank(
#     queries=ad_texts,
#     documents=onet_doc_texts,
#     top_k=10,
#     model="bge-reranker-v2-m3-sbatch",
#     time="01:00:00",
# )
# # result["indices"]: (n_ads, 10) top-10 O*NET indices per ad
# # result["scores"]: (n_ads, 10) corresponding scores
# ```
#
# This integrates with the existing isambard_utils orchestration: content-addressed
# caching, SBATCH submission, automatic model download, accounting.
#
# ### Next steps
#
# 1. Cache the O\*NET cross-encoder model on Isambard
#    (`isambard_utils.models.cache_model("BAAI/bge-reranker-v2-m3")`)
# 2. Run a calibration job to measure GPU-hours per 1000 ads
# 3. Integrate as a pipeline node between `cosine_match` and `llm_filter_candidates`
