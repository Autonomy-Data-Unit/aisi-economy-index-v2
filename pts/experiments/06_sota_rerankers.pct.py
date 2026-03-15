# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SOTA Reranker Comparison
#
# Previous experiment (05) showed bge-reranker-v2-m3 (278M) performed poorly
# on our occupation matching task. This notebook tests two SOTA models:
#
# - **gte-reranker-modernbert-base** (149M, Alibaba, classic cross-encoder):
#   Modern BERT architecture, scored 83% Hit@1 on AI Multiple benchmark.
#   Uses sentence-transformers CrossEncoder.
#
# - **Qwen3-Reranker-8B** (8B, Qwen, generative reranker): Highest MTEB-R
#   scores among open-source rerankers. Instruction-following. Uses vLLM backend
#   with yes/no logprob extraction.
#
# Both run on Isambard GPU. Compared against flat bi-encoder embedding baseline.

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
from ai_index.utils.rerank import arerank
from ai_index.utils.adzuna_store import get_ads_by_id
from validation.utils import load_matches, matches_to_ranked_lists
from adulib.llm import async_single
from adulib.asynchronous import batch_executor

# %%
EMBEDDING_MODEL = "text-embedding-3-large"
CE_MODEL = "gte-reranker-modernbert-sbatch"        # classic cross-encoder
GEN_MODEL = "qwen3-reranker-8b-sbatch"             # generative reranker
EVAL_MODEL = "openai/gpt-5.4"
SAMPLE_N = 2000
SAMPLE_SEED = 42
EVAL_SAMPLE_N = 200
EVAL_SEED = 123
REFERENCE_RUN = "val__validation_5k__qwen-32b-sbatch__bge-large-sbatch"
TOP_K = 10
CE_CANDIDATES = 100  # bi-encoder top-N for cross-encoder reranking

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
print(f"Data: {len(onet_codes)} occupations, {len(sample_ids)} ads")

# %% [markdown]
# ## 3. Bi-encoder candidates and eval sample

# %%
sim_matrix = ad_norms @ onet_norms.T

flat_top_idx = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]
flat_ranked = {sample_ids[i]: [onet_codes[j] for j in flat_top_idx[i]] for i in range(len(sample_ids))}

eval_rng = np.random.default_rng(EVAL_SEED)
eval_ids = eval_rng.choice(sample_ids, size=min(EVAL_SAMPLE_N, len(sample_ids)), replace=False).tolist()
print(f"Eval sample: {len(eval_ids)} ads")

# Build query and document texts for reranking
onet_doc_texts = [f"{onet_titles[code]}: {onet_descs[code][:300]}" for code in onet_codes]

eval_query_texts = []
for ad_id in eval_ids:
    ad_row = ads_df.loc[ad_id]
    eval_query_texts.append(f"{ad_row['title']}. {str(ad_row['description'] or '')[:3000]}")

print(f"Reranking: {len(eval_query_texts)} queries x {len(onet_doc_texts)} documents")

# %% [markdown]
# ## 4. Cross-encoder reranking (gte-reranker-modernbert-base)

# %%
print(f"Running {CE_MODEL} on Isambard...")
ce_result = await arerank(
    queries=eval_query_texts,
    documents=onet_doc_texts,
    top_k=CE_CANDIDATES,
    model=CE_MODEL,
    time="00:30:00",
)
ce_indices = ce_result["indices"]
ce_scores = ce_result["scores"]
print(f"CE result shape: {ce_indices.shape}")

# Build top-10 ranked lists
ce_ranked = {}
for i, ad_id in enumerate(eval_ids):
    ce_ranked[ad_id] = [onet_codes[j] for j in ce_indices[i, :TOP_K]]

print(f"CE top-{TOP_K}: {len(ce_ranked)} ads")

# %% [markdown]
# ## 5. Generative reranker (Qwen3-Reranker-8B)

# %%
print(f"Running {GEN_MODEL} on Isambard...")
gen_result = await arerank(
    queries=eval_query_texts,
    documents=onet_doc_texts,
    top_k=CE_CANDIDATES,
    model=GEN_MODEL,
    time="02:00:00",
)
gen_indices = gen_result["indices"]
gen_scores = gen_result["scores"]
print(f"Generative result shape: {gen_indices.shape}")

gen_ranked = {}
for i, ad_id in enumerate(eval_ids):
    gen_ranked[ad_id] = [onet_codes[j] for j in gen_indices[i, :TOP_K]]

print(f"Generative top-{TOP_K}: {len(gen_ranked)} ads")

# %% [markdown]
# ## 6. LLM evaluation

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
ce_ratings, ce_stats = await run_eval(eval_ids, ce_ranked, "GTE-MODERNBERT (149M)")

# %%
gen_ratings, gen_stats = await run_eval(eval_ids, gen_ranked, "QWEN3-RERANKER-8B")

# %% [markdown]
# ## 7. Results

# %%
summary = pd.DataFrame([flat_stats, ce_stats, gen_stats])
summary.columns = ["Approach", "N", "Mean top-1", f"Mean best-in-{TOP_K}", f"Top-{TOP_K} has 4+", f"Top-{TOP_K} has 5"]
print(f"\n{'='*90}")
print(f"  RESULTS (top-{TOP_K})")
print(f"{'='*90}")
print(summary.to_string(index=False))
print(f"\nPrimary metric: 'Top-{TOP_K} has 4+' = fraction of ads with at least one good match")

# %%
# Head-to-head comparisons
comparisons = [
    ("GTE vs Flat", ce_ratings, flat_ratings),
    ("Qwen3 vs Flat", gen_ratings, flat_ratings),
    ("Qwen3 vs GTE", gen_ratings, ce_ratings),
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
    print(f"{comp_label} (best-in-{TOP_K}): {names[0]} wins {a_wins}/{t} ({a_wins/t:.1%}), "
          f"{names[1]} wins {b_wins}/{t} ({b_wins/t:.1%}), ties {ties}/{t} ({ties/t:.1%})")

# %%
# Sample inspection: cases where rerankers find good matches flat missed
print("\n=== QWEN3 FOUND GOOD MATCH NOT IN FLAT TOP-10 ===\n")
shown = 0
for i, ad_id in enumerate(eval_ids):
    fr, gr = flat_ratings[i], gen_ratings[i]
    if fr[0] is None or gr[0] is None:
        continue
    if max(gr) > max(fr) and max(gr) >= 4:
        ad_row = ads_df.loc[ad_id]
        print(f"Ad: {ad_row['title']} [{ad_row['category_name']}]")
        print(f"  Flat best: {max(fr)}, Qwen3 best: {max(gr)}")
        best_idx = gr.index(max(gr))
        best_code = gen_ranked[ad_id][best_idx]
        print(f"  Qwen3 best match: {best_code} - {onet_titles[best_code]}")
        flat_best_idx = fr.index(max(fr))
        flat_best_code = flat_ranked[ad_id][flat_best_idx]
        print(f"  Flat best match:  {flat_best_code} - {onet_titles[flat_best_code]}")
        print()
        shown += 1
        if shown >= 5:
            break

if shown == 0:
    print("  (No cases found)")

# %% [markdown]
# ## 8. Conclusions
#
# Compare the SOTA rerankers against the flat embedding baseline and against
# each other. Key questions:
# - Does a better cross-encoder model fix the poor results from bge-reranker-v2-m3?
# - Does the generative reranker (Qwen3-8B) outperform the classic cross-encoder?
# - Is either reranker worth adding to the pipeline?
