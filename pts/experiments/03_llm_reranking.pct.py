# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM Reranking of Embedding Candidates
#
# **Idea**: Use embeddings for cheap recall (top-10 candidates), then use a
# high-quality LLM to rerank those candidates into a better top-5.
#
# The LLM can understand semantic nuances that embeddings miss: job level,
# specialization within a field, domain-specific terminology.
#
# Compare the reranked top-5 against the flat embedding top-5.

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

from ai_index.const import onet_targets_path
from ai_index.utils.embed import embed
from ai_index.utils.adzuna_store import get_ads_by_id
from validation.utils import load_matches, matches_to_ranked_lists
from adulib.llm import async_single
from adulib.asynchronous import batch_executor

# %%
EMBEDDING_MODEL = "text-embedding-3-large"
RERANK_MODEL = "openai/gpt-5.4"
EVAL_MODEL = "openai/gpt-5.4"
SAMPLE_N = 2000
SAMPLE_SEED = 42
EVAL_SAMPLE_N = 200
EVAL_SEED = 123
REFERENCE_RUN = "val__validation_5k__qwen-32b-sbatch__bge-large-sbatch"
TOP_K = 20
RERANK_CANDIDATES = 40  # embed top-N candidates to rerank from

# %% [markdown]
# ## 2. Load O\*NET data and embed

# %%
onet_df = pd.read_parquet(onet_targets_path)
onet_codes = onet_df["O*NET-SOC Code"].tolist()
onet_titles = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Title"]))
onet_descs = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Description"]))

onet_texts = [
    f"{row['Title']}\n\n{row['Description']}\n\nKey tasks and skills: {row['Work Activities/Tasks/Skills']}"
    for _, row in onet_df.iterrows()
]
print("Embedding O*NET occupations (cached)...")
onet_embeds = embed(onet_texts, model=EMBEDDING_MODEL)
onet_norms = onet_embeds / np.linalg.norm(onet_embeds, axis=1, keepdims=True)
print(f"O*NET: {onet_embeds.shape}")

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
print(f"Ads: {ad_embeds.shape}")

# %% [markdown]
# ## 4. Flat matching baseline (top-5)

# %%
sim_matrix = ad_norms @ onet_norms.T
flat_top_k_idx = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]

flat_ranked = {}
for i, ad_id in enumerate(sample_ids):
    flat_ranked[ad_id] = [onet_codes[j] for j in flat_top_k_idx[i]]

# Also get top-RERANK_CANDIDATES for the reranking step
rerank_top_idx = np.argsort(-sim_matrix, axis=1)[:, :RERANK_CANDIDATES]
rerank_candidates = {}
for i, ad_id in enumerate(sample_ids):
    rerank_candidates[ad_id] = [onet_codes[j] for j in rerank_top_idx[i]]

print(f"Flat top-{TOP_K} and top-{RERANK_CANDIDATES} candidates ready for {len(sample_ids)} ads")

# %% [markdown]
# ## 5. LLM reranking
#
# Give GPT-5.4 the job ad and top-10 embedding candidates. Ask it to select
# and rank the best 5.

# %%
RERANK_SYSTEM = f"""You are an expert occupational classifier. Given a job advertisement and {RERANK_CANDIDATES} candidate O*NET occupations (ranked by embedding similarity), select the {TOP_K} best matches and rank them from best to worst.

Consider: job duties, required skills, professional level, specialization, and industry context.

Respond with JSON only:
{{"ranked_codes": ["code1", "code2", ...], "reasoning": "brief explanation"}}

Return exactly {TOP_K} codes."""

def build_rerank_prompt(ad_row, candidate_codes):
    candidates = ""
    for k, code in enumerate(candidate_codes, 1):
        candidates += f"{k}. {code} - {onet_titles[code]}: {onet_descs[code][:200]}\n"
    return f"""## Job Advertisement
**Title:** {ad_row['title']}
**Category:** {ad_row['category_name']}
**Description:** {str(ad_row['description'] or '')[:2000]}

## Candidate Occupations (from embedding similarity)
{candidates}
Select the {TOP_K} best matches and rank them."""

# %%
# Only rerank the evaluation sample (200 ads) to keep costs manageable
eval_rng = np.random.default_rng(EVAL_SEED)
eval_ids = eval_rng.choice(sample_ids, size=min(EVAL_SAMPLE_N, len(sample_ids)), replace=False).tolist()
print(f"Reranking {len(eval_ids)} ads with {RERANK_MODEL}...")

rerank_prompts = []
for ad_id in eval_ids:
    ad_row = ads_df.loc[ad_id]
    rerank_prompts.append(build_rerank_prompt(ad_row, rerank_candidates[ad_id]))

async def _rerank_call(prompt):
    resp, _, _ = await async_single(prompt, model=RERANK_MODEL, system=RERANK_SYSTEM,
                                    max_tokens=300, response_format={"type": "json_object"})
    return resp

rerank_responses = await batch_executor(
    _rerank_call, batch_args=[(p,) for p in rerank_prompts], concurrency_limit=20
)
print(f"Got {len(rerank_responses)} reranking responses")

# %%
# Parse reranking responses
reranked = {}
rerank_failures = 0
for i, (ad_id, resp) in enumerate(zip(eval_ids, rerank_responses)):
    candidates_set = set(rerank_candidates[ad_id])
    try:
        parsed = json.loads(resp)
        codes = parsed["ranked_codes"]
        # Validate: codes must be from the candidate set
        valid_codes = [c for c in codes if c in candidates_set]
        if len(valid_codes) < TOP_K:
            # Pad with remaining candidates in original order
            for c in rerank_candidates[ad_id]:
                if c not in valid_codes:
                    valid_codes.append(c)
                if len(valid_codes) >= TOP_K:
                    break
        reranked[ad_id] = valid_codes[:TOP_K]
    except (json.JSONDecodeError, KeyError):
        # Fall back to flat ranking
        reranked[ad_id] = flat_ranked[ad_id]
        rerank_failures += 1

print(f"Reranking complete. Failures (fell back to flat): {rerank_failures}")

# %%
# How much did reranking change the results?
top1_changed = sum(1 for ad_id in eval_ids if reranked[ad_id][0] != flat_ranked[ad_id][0])
top5_overlap = []
for ad_id in eval_ids:
    overlap = len(set(reranked[ad_id]) & set(flat_ranked[ad_id]))
    top5_overlap.append(overlap)

print(f"\n=== Reranking vs Flat ===")
print(f"Top-1 changed:      {top1_changed}/{len(eval_ids)} ({top1_changed/len(eval_ids):.1%})")
print(f"Mean top-5 overlap:  {np.mean(top5_overlap):.2f} / 5")
print(f"Top-5 identical:     {sum(1 for o in top5_overlap if o == 5) / len(eval_ids):.1%}")

# Note: since reranking picks from top-10, the top-5 set can only contain
# codes from the embedding top-10. It can promote candidates from positions
# 6-10 into the top-5.
new_in_top5 = []
for ad_id in eval_ids:
    new_codes = set(reranked[ad_id]) - set(flat_ranked[ad_id])
    new_in_top5.append(len(new_codes))
print(f"Mean new codes in top-5 (from positions 6-10): {np.mean(new_in_top5):.2f}")

# %% [markdown]
# ## 6. LLM evaluation of top-5
#
# **Caveat**: We use GPT-5.4 for both reranking and evaluation, which could
# bias results in favor of the reranked approach. The evaluation still provides
# an absolute quality signal for both approaches.

# %%
EVAL_SYSTEM = """You are an expert occupational classification evaluator. You will be given a job advertisement and 5 candidate O*NET occupation matches. Rate how well EACH candidate matches the job ad.

Rating scale:
5 = Perfect match. The occupation precisely describes this job.
4 = Good match. Captures the core role with minor differences.
3 = Partial match. Related field but notable differences in duties or level.
2 = Weak match. Same broad domain but substantially different role.
1 = Poor match. Unrelated or wrong occupation.

Respond with JSON only:
{"ratings": [r1, r2, ...], "best_index": 0, "reasoning": "brief explanation"}

The ratings list must have exactly one rating per candidate match."""

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

async def run_eval(eval_ids_list, ranked_dict, label):
    prompts = []
    for ad_id in eval_ids_list:
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
            assert len(ratings) == TOP_K
        except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
            ratings = [None] * TOP_K
            parse_failures += 1
        all_ratings.append(ratings)

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
flat_ratings = await run_eval(eval_ids, flat_ranked, "FLAT EMBEDDING")

# %%
reranked_ratings = await run_eval(eval_ids, reranked, "LLM RERANKED")

# %%
# Head-to-head comparison
n_flat_wins = 0
n_reranked_wins = 0
n_ties = 0
for i in range(len(eval_ids)):
    fr, rr = flat_ratings[i], reranked_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    flat_best = max(fr)
    rerank_best = max(rr)
    if flat_best > rerank_best:
        n_flat_wins += 1
    elif rerank_best > flat_best:
        n_reranked_wins += 1
    else:
        n_ties += 1

total = n_flat_wins + n_reranked_wins + n_ties
print(f"\n=== Head-to-Head (best-in-top-5 rating) ===")
print(f"Flat wins:     {n_flat_wins:4d} ({n_flat_wins/total:.1%})")
print(f"Reranked wins: {n_reranked_wins:4d} ({n_reranked_wins/total:.1%})")
print(f"Ties:          {n_ties:4d} ({n_ties/total:.1%})")

# Did reranking improve the top-1 specifically?
top1_flat_better = 0
top1_reranked_better = 0
top1_same = 0
for i in range(len(eval_ids)):
    fr, rr = flat_ratings[i], reranked_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    if fr[0] > rr[0]:
        top1_flat_better += 1
    elif rr[0] > fr[0]:
        top1_reranked_better += 1
    else:
        top1_same += 1

t2 = top1_flat_better + top1_reranked_better + top1_same
print(f"\n=== Head-to-Head (top-1 rating) ===")
print(f"Flat better:     {top1_flat_better:4d} ({top1_flat_better/t2:.1%})")
print(f"Reranked better: {top1_reranked_better:4d} ({top1_reranked_better/t2:.1%})")
print(f"Same:            {top1_same:4d} ({top1_same/t2:.1%})")

# %% [markdown]
# ## 7. Sample inspection

# %%
def print_match_comparison(ad_id, flat_codes, flat_rats, rerank_codes, rerank_rats):
    ad_row = ads_df.loc[ad_id]
    desc = textwrap.shorten(str(ad_row["description"] or "")[:300], 200)
    print(f"{'='*90}")
    print(f"Ad: {ad_row['title']} [{ad_row['category_name']}]")
    print(f"    {desc}")
    print(f"\n  {'FLAT EMBEDDING':40s} {'LLM RERANKED':40s}")
    print(f"  {'-'*40} {'-'*40}")
    for k in range(TOP_K):
        fc = flat_codes[k]
        rc = rerank_codes[k]
        fr_val = flat_rats[k] if flat_rats[k] is not None else "?"
        rr_val = rerank_rats[k] if rerank_rats[k] is not None else "?"
        ft = onet_titles[fc][:35]
        rt = onet_titles[rc][:35]
        print(f"  [{fr_val}] {ft:35s}   [{rr_val}] {rt:35s}")
    print()

# Show cases where reranking helped
print("=== CASES WHERE LLM RERANKING IMPROVED TOP-5 ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, rr = flat_ratings[i], reranked_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    if max(rr) > max(fr):
        print_match_comparison(ad_id, flat_ranked[ad_id], fr, reranked[ad_id], rr)
        shown += 1
        if shown >= 5:
            break

print("\n=== CASES WHERE LLM RERANKING HURT TOP-5 ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, rr = flat_ratings[i], reranked_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    if max(fr) > max(rr):
        print_match_comparison(ad_id, flat_ranked[ad_id], fr, reranked[ad_id], rr)
        shown += 1
        if shown >= 5:
            break

print("\n=== CASES WHERE RERANKING CHANGED TOP-1 (both approaches rated) ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, rr = flat_ratings[i], reranked_ratings[i]
    if fr[0] is None or rr[0] is None:
        continue
    if flat_ranked[ad_id][0] != reranked[ad_id][0]:
        print_match_comparison(ad_id, flat_ranked[ad_id], fr, reranked[ad_id], rr)
        shown += 1
        if shown >= 5:
            break

# %% [markdown]
# ## 8. Summary
#
# Key questions answered:
# - Does LLM reranking of embedding top-10 produce better top-5 than flat
#   embedding alone?
# - How often does reranking change the top-1?
# - Is the improvement worth the cost of an LLM call per ad?
#
# **Caveat**: GPT-5.4 is used for both reranking and evaluation, so the
# evaluation may be biased in favor of the reranked results.
