# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Cross-Encoder vs LLM Reranking: Systematic Comparison
#
# ## Background
#
# Previous experiments established:
# - **Bi-encoder embedding** (text-embedding-3-large on raw ad text) achieves 98%
#   recall at top-20: i.e. 98% of ads have a 4+ rated match in their top-20.
# - **LLM reranking** dramatically improves top-1 quality (3.67 -> 4.59) but adds
#   little to recall (the good candidates are already there).
# - **Centroid-residual matching** hurts quality. Subtracting group centroids
#   destroys useful signal rather than enhancing within-group discrimination.
#
# ## Proposed pipeline
#
# The emerging pipeline design is:
# 1. **Bi-encoder** for cheap recall: embed raw ad text, cosine match against all
#    861 O\*NET occupations, take top-N candidates (N=100)
# 2. **Reranker** to refine: score each (ad, candidate) pair, produce a top-10
# 3. **LLM filter** for final selection: one LLM call per ad on the top-10
#
# The key question for step 2: should the reranker be a **cross-encoder** (cheap,
# fast, purpose-built) or an **LLM** (expensive but more capable)?
#
# ## This experiment
#
# Systematic head-to-head comparison:
# - **(a) Cross-encoder reranking**: top-100 from bi-encoder, cross-encoder scores
#   each pair, take top-10
# - **(b) LLM reranking**: top-20 from bi-encoder, LLM selects and ranks top-10
#
# Primary metric: **fraction of ads where at least one top-10 match is rated 4+**
# by an independent LLM evaluator (GPT-5.4).

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
from adulib.llm import async_single
from adulib.asynchronous import batch_executor

# %%
EMBEDDING_MODEL = "text-embedding-3-large"
RERANK_MODEL_CE = "voyage/rerank-2.5"         # cross-encoder via API
RERANK_MODEL_LLM = "openai/gpt-5.4"           # LLM reranker
EVAL_MODEL = "openai/gpt-5.4"                 # evaluator (same for both)
SAMPLE_N = 2000
SAMPLE_SEED = 42
EVAL_SAMPLE_N = 200
EVAL_SEED = 123
REFERENCE_RUN = "val__validation_5k__qwen-32b-sbatch__bge-large-sbatch"
TOP_K = 10                                     # final candidate set size
CE_CANDIDATES = 100                            # bi-encoder top-N for cross-encoder
LLM_CANDIDATES = 20                            # bi-encoder top-N for LLM reranker

# %% [markdown]
# ## 2. Load data and embed

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
# ## 3. Bi-encoder baseline

# %%
sim_matrix = ad_norms @ onet_norms.T

# Flat top-10 (baseline)
flat_top_k_idx = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]
flat_ranked = {}
for i, ad_id in enumerate(sample_ids):
    flat_ranked[ad_id] = [onet_codes[j] for j in flat_top_k_idx[i]]

# Top-100 for cross-encoder, top-20 for LLM reranker
ce_top_idx = np.argsort(-sim_matrix, axis=1)[:, :CE_CANDIDATES]
ce_candidates = {}
for i, ad_id in enumerate(sample_ids):
    ce_candidates[ad_id] = [onet_codes[j] for j in ce_top_idx[i]]

llm_top_idx = np.argsort(-sim_matrix, axis=1)[:, :LLM_CANDIDATES]
llm_candidates = {}
for i, ad_id in enumerate(sample_ids):
    llm_candidates[ad_id] = [onet_codes[j] for j in llm_top_idx[i]]

print(f"Flat top-{TOP_K}: {len(flat_ranked)} ads")
print(f"CE candidates: top-{CE_CANDIDATES} per ad")
print(f"LLM candidates: top-{LLM_CANDIDATES} per ad")

# %% [markdown]
# ## 4. Select evaluation sample
#
# Both reranking methods are only run on the eval sample (200 ads) to keep
# costs manageable. This is the same sample used in previous experiments.

# %%
eval_rng = np.random.default_rng(EVAL_SEED)
eval_ids = eval_rng.choice(sample_ids, size=min(EVAL_SAMPLE_N, len(sample_ids)), replace=False).tolist()
print(f"Evaluation sample: {len(eval_ids)} ads")

# %% [markdown]
# ## 5. Cross-encoder reranking (top-100 -> top-10)
#
# Use Voyage rerank-2.5 API to score each (ad, occupation) pair. The query is
# the ad text; the documents are O\*NET occupation descriptions.

# %%
import litellm

async def cross_encoder_rerank(ad_id):
    """Rerank top-100 candidates for one ad using cross-encoder API."""
    ad_row = ads_df.loc[ad_id]
    query = f"{ad_row['title']}. {str(ad_row['description'] or '')[:3000]}"
    candidates = ce_candidates[ad_id]

    # Build document texts for the reranker
    documents = [
        f"{onet_titles[code]}: {onet_descs[code][:300]}"
        for code in candidates
    ]

    result = await litellm.arerank(
        model=RERANK_MODEL_CE,
        query=query,
        documents=documents,
        top_n=TOP_K,
    )

    # Extract top-K codes in ranked order
    ranked_codes = []
    for r in result.results:
        ranked_codes.append(candidates[r["index"]])
    return ranked_codes

# %%
import asyncio
import time as _time

print(f"Cross-encoder reranking {len(eval_ids)} ads ({CE_CANDIDATES} -> {TOP_K})...")

# Process one at a time with retry to respect Voyage 2M TPM rate limit.
# Each call sends ~100 documents (~30K tokens), so we can do ~60/min.
ce_ranked = {}
for idx, ad_id in enumerate(eval_ids):
    for attempt in range(5):
        try:
            ce_ranked[ad_id] = await cross_encoder_rerank(ad_id)
            break
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    if (idx + 1) % 20 == 0:
        print(f"  {idx + 1}/{len(eval_ids)} done")
    await asyncio.sleep(0.5)  # ~120 calls/min baseline pacing

print(f"Cross-encoder reranking complete: {len(ce_ranked)} ads")

# %% [markdown]
# ## 6. LLM reranking (top-20 -> top-10)

# %%
RERANK_SYSTEM = f"""You are an expert occupational classifier. Given a job advertisement and {LLM_CANDIDATES} candidate O*NET occupations (ranked by embedding similarity), select the {TOP_K} best matches and rank them from best to worst.

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
print(f"LLM reranking {len(eval_ids)} ads ({LLM_CANDIDATES} -> {TOP_K})...")

rerank_prompts = []
for ad_id in eval_ids:
    rerank_prompts.append(build_rerank_prompt(ads_df.loc[ad_id], llm_candidates[ad_id]))

async def _llm_rerank_call(prompt):
    resp, _, _ = await async_single(prompt, model=RERANK_MODEL_LLM, system=RERANK_SYSTEM,
                                    max_tokens=400, response_format={"type": "json_object"})
    return resp

llm_rerank_responses = await batch_executor(
    _llm_rerank_call, batch_args=[(p,) for p in rerank_prompts], concurrency_limit=20
)

# Parse responses
llm_ranked = {}
llm_rerank_failures = 0
for i, (ad_id, resp) in enumerate(zip(eval_ids, llm_rerank_responses)):
    candidates_set = set(llm_candidates[ad_id])
    try:
        parsed = json.loads(resp)
        codes = parsed["ranked_codes"]
        valid_codes = [c for c in codes if c in candidates_set]
        if len(valid_codes) < TOP_K:
            for c in llm_candidates[ad_id]:
                if c not in valid_codes:
                    valid_codes.append(c)
                if len(valid_codes) >= TOP_K:
                    break
        llm_ranked[ad_id] = valid_codes[:TOP_K]
    except (json.JSONDecodeError, KeyError):
        llm_ranked[ad_id] = llm_candidates[ad_id][:TOP_K]
        llm_rerank_failures += 1

print(f"LLM reranking complete. Failures: {llm_rerank_failures}")

# %% [markdown]
# ## 7. Compare reranking approaches (before LLM evaluation)

# %%
# How much do the reranked top-10s overlap with flat and with each other?
flat_ce_overlap = []
flat_llm_overlap = []
ce_llm_overlap = []

for ad_id in eval_ids:
    flat_set = set(flat_ranked[ad_id])
    ce_set = set(ce_ranked[ad_id])
    llm_set = set(llm_ranked[ad_id])
    flat_ce_overlap.append(len(flat_set & ce_set))
    flat_llm_overlap.append(len(flat_set & llm_set))
    ce_llm_overlap.append(len(ce_set & llm_set))

print(f"=== Top-{TOP_K} Overlap ===")
print(f"Flat vs CE:   {np.mean(flat_ce_overlap):.1f} / {TOP_K} mean overlap")
print(f"Flat vs LLM:  {np.mean(flat_llm_overlap):.1f} / {TOP_K} mean overlap")
print(f"CE vs LLM:    {np.mean(ce_llm_overlap):.1f} / {TOP_K} mean overlap")

# How many new codes (not in flat top-10) did each reranker surface?
ce_new = [len(set(ce_ranked[ad_id]) - set(flat_ranked[ad_id])) for ad_id in eval_ids]
llm_new = [len(set(llm_ranked[ad_id]) - set(flat_ranked[ad_id])) for ad_id in eval_ids]
print(f"\nMean new codes vs flat top-{TOP_K}:")
print(f"  CE:  {np.mean(ce_new):.1f} new (from positions {TOP_K+1}-{CE_CANDIDATES})")
print(f"  LLM: {np.mean(llm_new):.1f} new (from positions {TOP_K+1}-{LLM_CANDIDATES})")

# %% [markdown]
# ## 8. LLM evaluation of all three approaches

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

def build_eval_prompt(ad_row, top_codes):
    candidates = ""
    for k, code in enumerate(top_codes, 1):
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
        prompts.append(build_eval_prompt(ads_df.loc[ad_id], ranked_dict[ad_id]))

    async def _call(prompt):
        resp, _, _ = await async_single(prompt, model=EVAL_MODEL, system=EVAL_SYSTEM,
                                        max_tokens=400, response_format={"type": "json_object"})
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
    best_in_k = [max(r) for _, r in valid]
    top1_ratings = [r[0] for _, r in valid]
    all_individual = [x for _, r in valid for x in r]

    hit_4plus = sum(1 for b in best_in_k if b >= 4) / len(best_in_k)
    hit_5 = sum(1 for b in best_in_k if b >= 5) / len(best_in_k)

    print(f"\n=== {label} (n={len(valid)}, {parse_failures} parse failures) ===")
    print(f"Mean top-1 rating:          {np.mean(top1_ratings):.2f}")
    print(f"Mean best-in-top-{TOP_K}:        {np.mean(best_in_k):.2f}")
    print(f"Top-{TOP_K} has a 4+ match:      {hit_4plus:.1%}  <-- primary metric")
    print(f"Top-{TOP_K} has a 5 match:       {hit_5:.1%}")
    print(f"\nAll ratings distribution:")
    counts = Counter(all_individual)
    for r in sorted(counts):
        pct = counts[r] / len(all_individual) * 100
        print(f"  {r}: {counts[r]:4d} ({pct:5.1f}%)")

    return all_ratings, {"label": label, "n": len(valid), "mean_top1": np.mean(top1_ratings),
                         "mean_best": np.mean(best_in_k), "hit_4plus": hit_4plus, "hit_5": hit_5}

# %%
flat_ratings, flat_stats = await run_eval(eval_ids, flat_ranked, "FLAT EMBEDDING (baseline)")

# %%
ce_ratings, ce_stats = await run_eval(eval_ids, ce_ranked, "CROSS-ENCODER RERANKED")

# %%
llm_ratings, llm_stats = await run_eval(eval_ids, llm_ranked, "LLM RERANKED")

# %% [markdown]
# ## 9. Results comparison

# %%
# Summary table
summary = pd.DataFrame([flat_stats, ce_stats, llm_stats])
summary = summary[["label", "n", "mean_top1", "mean_best", "hit_4plus", "hit_5"]]
summary.columns = ["Approach", "N", "Mean top-1", f"Mean best-in-{TOP_K}", f"Top-{TOP_K} has 4+", f"Top-{TOP_K} has 5"]
print(f"\n{'='*90}")
print(f"  RESULTS SUMMARY (top-{TOP_K})")
print(f"{'='*90}")
print(summary.to_string(index=False))
print(f"\nPrimary metric: 'Top-{TOP_K} has 4+' = fraction of ads with at least one good match")

# %%
# Head-to-head: CE vs LLM reranking
n_ce_wins = 0
n_llm_wins = 0
n_ties = 0
for i in range(len(eval_ids)):
    cr, lr = ce_ratings[i], llm_ratings[i]
    if cr[0] is None or lr[0] is None:
        continue
    ce_best = max(cr)
    llm_best = max(lr)
    if ce_best > llm_best:
        n_ce_wins += 1
    elif llm_best > ce_best:
        n_llm_wins += 1
    else:
        n_ties += 1

total = n_ce_wins + n_llm_wins + n_ties
print(f"\n=== CE vs LLM Head-to-Head (best-in-top-{TOP_K}) ===")
print(f"CE wins:   {n_ce_wins:4d} ({n_ce_wins/total:.1%})")
print(f"LLM wins:  {n_llm_wins:4d} ({n_llm_wins/total:.1%})")
print(f"Ties:      {n_ties:4d} ({n_ties/total:.1%})")

# %%
# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = ["Flat\n(baseline)", "Cross-encoder\n(top-100→10)", "LLM reranked\n(top-20→10)"]
colors = ["#4878d0", "#ee854a", "#6acc64"]

# Left: primary metric (hit rate)
hit_vals = [flat_stats["hit_4plus"], ce_stats["hit_4plus"], llm_stats["hit_4plus"]]
axes[0].bar(labels, hit_vals, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_ylabel(f"Fraction of ads with 4+ match in top-{TOP_K}")
axes[0].set_title(f"Primary Metric: Top-{TOP_K} Recall")
axes[0].set_ylim(0.7, 1.02)
for i, v in enumerate(hit_vals):
    axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

# Right: mean best-in-top-K
best_vals = [flat_stats["mean_best"], ce_stats["mean_best"], llm_stats["mean_best"]]
axes[1].bar(labels, best_vals, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_ylabel(f"Mean best-in-top-{TOP_K} rating")
axes[1].set_title(f"Mean Best Match Quality")
axes[1].set_ylim(3.5, 5.1)
for i, v in enumerate(best_vals):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

fig.suptitle(f"Reranking Comparison: Cross-Encoder vs LLM (top-{TOP_K})", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Sample inspection

# %%
def print_comparison(ad_id, flat_r, ce_codes, ce_r, llm_codes, llm_r):
    ad_row = ads_df.loc[ad_id]
    desc = textwrap.shorten(str(ad_row["description"] or "")[:300], 200)
    print(f"{'='*95}")
    print(f"Ad: {ad_row['title']} [{ad_row['category_name']}]")
    print(f"    {desc}")
    print(f"\n  {'FLAT':30s} {'CROSS-ENCODER':30s} {'LLM RERANKED':30s}")
    print(f"  {'-'*30} {'-'*30} {'-'*30}")
    for k in range(min(5, TOP_K)):  # show first 5 of each
        fc = flat_ranked[ad_id][k] if k < len(flat_ranked[ad_id]) else "---"
        cc = ce_codes[k] if k < len(ce_codes) else "---"
        lc = llm_codes[k] if k < len(llm_codes) else "---"
        fr = flat_r[k] if flat_r[k] is not None else "?"
        cr = ce_r[k] if ce_r[k] is not None else "?"
        lr = llm_r[k] if llm_r[k] is not None else "?"
        ft = onet_titles.get(fc, "---")[:26]
        ct = onet_titles.get(cc, "---")[:26]
        lt = onet_titles.get(lc, "---")[:26]
        print(f"  [{fr}] {ft:26s}   [{cr}] {ct:26s}   [{lr}] {lt:26s}")
    print()

# Cases where CE and LLM disagree most
print("=== CASES WHERE CE OUTPERFORMS LLM ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    cr, lr = ce_ratings[i], llm_ratings[i]
    if cr[0] is None or lr[0] is None:
        continue
    if max(cr) > max(lr):
        print_comparison(ad_id, flat_ratings[i], ce_ranked[ad_id], cr, llm_ranked[ad_id], lr)
        shown += 1
        if shown >= 3:
            break

print("\n=== CASES WHERE LLM OUTPERFORMS CE ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    cr, lr = ce_ratings[i], llm_ratings[i]
    if cr[0] is None or lr[0] is None:
        continue
    if max(lr) > max(cr):
        print_comparison(ad_id, flat_ratings[i], ce_ranked[ad_id], cr, llm_ranked[ad_id], lr)
        shown += 1
        if shown >= 3:
            break

print("\n=== CASES WHERE BOTH IMPROVE OVER FLAT ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, cr, lr = flat_ratings[i], ce_ratings[i], llm_ratings[i]
    if fr[0] is None or cr[0] is None or lr[0] is None:
        continue
    if max(cr) > max(fr) and max(lr) > max(fr):
        print_comparison(ad_id, fr, ce_ranked[ad_id], cr, llm_ranked[ad_id], lr)
        shown += 1
        if shown >= 3:
            break

# %% [markdown]
# ## 11. Conclusions
#
# ### Results summary
#
# | Approach | Mean top-1 | Mean best-in-10 | **Top-10 has 4+** | Top-10 has 5 |
# |----------|-----------|----------------|-------------------|-------------|
# | Flat embedding (baseline) | 3.67 | 4.75 | 95.5% | 81.0% |
# | Cross-encoder (100->10) | 4.28 | 4.84 | **99.0%** | **86.0%** |
# | LLM reranked (20->10) | 4.58 | 4.75 | 95.5% | 80.0% |
#
# ### What we learned
#
# **Cross-encoder reranking is the clear winner for candidate recall.** It
# achieves 99% of ads with a good match (4+) in the top-10, compared to 95.5%
# for both flat embedding and LLM reranking. The key advantage is that it draws
# from a much larger pool (top-100 from bi-encoder) and can surface good
# candidates that the bi-encoder ranked at positions 11-100.
#
# **LLM reranking is better at picking the single best match** (top-1 rating
# 4.58 vs 4.28 for cross-encoder), but it does not improve recall at all. It
# ties flat embedding at 95.5%. This is because it only selects from the
# bi-encoder's top-20, which limits how much it can improve the candidate set.
#
# **For our use case, recall matters more than top-1 precision.** The reranker's
# job is to surface good candidates for a downstream LLM filter. The LLM filter
# will make the final selection, so we need the right answer to be *somewhere*
# in the top-10, not necessarily at position 1.
#
# ### Recommended pipeline
#
# 1. **Bi-encoder** (e.g. text-embedding-3-large): embed raw ad text, cosine
#    match against all 861 O\*NET occupations, take top-100. This is cheap and
#    parallelizable. No LLM summarisation needed.
#
# 2. **Cross-encoder reranker**: score each (ad, candidate) pair, produce a
#    top-10. For production, run a local model on GPU (e.g. `BAAI/bge-reranker-v2-m3`
#    or `cross-encoder/ms-marco-MiniLM-L-12-v2`) on Isambard rather than the
#    Voyage API used in this experiment. Local cross-encoders process thousands
#    of pairs per second on GPU, making this step fast even at 30M ads x 100
#    candidates.
#
# 3. **LLM filter**: one LLM call per ad on the top-10 candidates. Same role
#    as the current `llm_filter_candidates` node.
#
# ### Cost comparison vs current pipeline
#
# Current: LLM summarise (30M calls) -> embed summaries -> cosine match ->
# LLM filter (30M calls) = **60M LLM calls**
#
# Proposed: embed raw text -> cosine top-100 -> cross-encoder rerank (GPU,
# no LLM) -> LLM filter (30M calls) = **30M LLM calls** + GPU reranking
#
# The proposed pipeline halves the LLM cost by dropping the summarisation step
# entirely, and replaces it with a much cheaper cross-encoder reranking step.
#
# ### Caveats
#
# - The cross-encoder used here (Voyage rerank-2.5) is an API-based reranker,
#   not a local model. Production would use a local cross-encoder on Isambard.
# - GPT-5.4 was used for both LLM reranking and evaluation, which may bias
#   the evaluation in favor of the LLM-reranked approach. Despite this bias,
#   the cross-encoder still won on recall.
# - The evaluation sample is 200 ads. Larger-scale validation on the full 5K
#   sample would increase confidence.
# - The bi-encoder used (text-embedding-3-large, 3072 dims) is an API model.
#   For production, we may want to test whether the existing local models
#   (bge-large, 1024 dims) achieve similar recall at top-100.
