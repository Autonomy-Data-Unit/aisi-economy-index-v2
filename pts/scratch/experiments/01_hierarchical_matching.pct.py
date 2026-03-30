# ---
# jupyter:
#   kernelspec:
#     display_name: ai-index (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hierarchical O*NET Matching Experiment
#
# **Question**: If we classify ads to the top-N major groups first, what fraction
# of the correct detailed O\*NET codes fall within those groups?
#
# This tests whether a two-stage approach (major group -> detailed occupation) is
# viable for reducing the candidate set without losing accuracy.

# %% [markdown]
# ## 1. Setup and config

# %%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ai_index.const import inputs_path, onet_targets_path, onet_store_path, pipeline_store_path
from ai_index.utils.embed import embed
from ai_index.utils.adzuna_store import get_ads_by_id, get_all_ad_ids
from validation.utils import load_matches, matches_to_ranked_lists, top1_agreement, topk_jaccard

# %%
EMBEDDING_MODEL = "text-embedding-3-large"  # API mode, 8K tokens, no GPU needed
SAMPLE_N = 2000
SAMPLE_SEED = 42
REFERENCE_RUN = "val__validation_5k__qwen-32b-sbatch__bge-large-sbatch"

# %% [markdown]
# ## 2. Load O\*NET data and build hierarchy

# %%
onet_df = pd.read_parquet(onet_targets_path)
print(f"O*NET occupations: {len(onet_df)}")
print(f"Columns: {onet_df.columns.tolist()}")

# %%
# Parse hierarchy from SOC codes (XX-XXXX.YY)
# Major group = first 2 digits
onet_df["major_group"] = onet_df["O*NET-SOC Code"].str[:2]

major_groups = onet_df.groupby("major_group").agg(
    n_occupations=("Title", "count"),
    example_title=("Title", "first"),
).reset_index()

print(f"\nMajor groups: {len(major_groups)}")
print(major_groups.to_string(index=False))

# %%
# Build major group name lookup from the first part of the SOC code descriptions
# Use the most common prefix pattern in titles within each group
major_group_names = {}
for mg, group in onet_df.groupby("major_group"):
    # The major group name comes from the SOC system; we'll use the most general title
    titles = group["Title"].tolist()
    major_group_names[mg] = f"Group {mg} ({len(titles)} occupations)"

print(f"\nMajor group sizes:")
for mg in sorted(major_group_names):
    n = (onet_df["major_group"] == mg).sum()
    example = onet_df[onet_df["major_group"] == mg]["Title"].iloc[0]
    print(f"  {mg}: {n:3d} occupations (e.g. {example})")

# %%
# Load alternate titles
alt_titles_path = onet_store_path / "db_30_0_text" / "Alternate Titles.txt"
alt_titles_df = pd.read_csv(alt_titles_path, sep="\t")
print(f"\nAlternate titles: {len(alt_titles_df)} rows")
print(f"Columns: {alt_titles_df.columns.tolist()}")

# Build lookup: O*NET code -> list of alternate titles
alt_titles_by_code = alt_titles_df.groupby("O*NET-SOC Code")["Alternate Title"].apply(list).to_dict()
print(f"Codes with alternate titles: {len(alt_titles_by_code)}")

# %% [markdown]
# ## 3. Build O\*NET text representations

# %%
# Rich document per occupation: title + description + tasks/skills
onet_texts = []
for _, row in onet_df.iterrows():
    text = f"{row['Title']}\n\n{row['Description']}\n\nKey tasks and skills: {row['Work Activities/Tasks/Skills']}"
    onet_texts.append(text)

print(f"Built {len(onet_texts)} O*NET text documents")
print(f"Mean length: {np.mean([len(t) for t in onet_texts]):.0f} chars")
print(f"Max length: {max(len(t) for t in onet_texts)} chars")

# %%
# Alternate-title-enriched variant
onet_texts_with_alts = []
for _, row in onet_df.iterrows():
    code = row["O*NET-SOC Code"]
    alts = alt_titles_by_code.get(code, [])
    alt_str = ", ".join(alts[:10]) if alts else ""  # cap at 10 to avoid huge texts

    if alt_str:
        text = f"{row['Title']} (also known as: {alt_str})\n\n{row['Description']}\n\nKey tasks and skills: {row['Work Activities/Tasks/Skills']}"
    else:
        text = f"{row['Title']}\n\n{row['Description']}\n\nKey tasks and skills: {row['Work Activities/Tasks/Skills']}"
    onet_texts_with_alts.append(text)

print(f"Built {len(onet_texts_with_alts)} enriched O*NET text documents")
print(f"Mean length: {np.mean([len(t) for t in onet_texts_with_alts]):.0f} chars")

# %% [markdown]
# ## 4. Sample ads and load reference matches

# %%
# Load reference matches from the validation run, then subsample from those
ref_df = load_matches(REFERENCE_RUN, stage="filtered")
ref_ranked = matches_to_ranked_lists(ref_df)
print(f"Reference run has matches for {len(ref_ranked)} ads")

# Sample from the reference run's ad IDs so we have ground truth for all
ref_ad_ids = sorted(ref_ranked.keys())
rng = np.random.default_rng(SAMPLE_SEED)
sample_ids = rng.choice(ref_ad_ids, size=min(SAMPLE_N, len(ref_ad_ids)), replace=False).tolist()
print(f"Sampled {len(sample_ids)} ads from reference run")

# %%
# Load raw ad text
ads_table = get_ads_by_id(sample_ids, columns=["title", "description", "category_name"])
ads_df = ads_table.to_pandas()
ads_df = ads_df.set_index("id")
# Reorder to match sample_ids
ads_df = ads_df.loc[sample_ids]
print(f"Loaded {len(ads_df)} ads")
print(f"Columns: {ads_df.columns.tolist()}")
print(f"\nSample ad:")
print(f"  Title: {ads_df.iloc[0]['title']}")
print(f"  Category: {ads_df.iloc[0]['category_name']}")
print(f"  Description length: {len(str(ads_df.iloc[0]['description']))} chars")

# %% [markdown]
# ## 5. Embed everything

# %%
# Embed O*NET occupations (861 texts)
print("Embedding O*NET occupations...")
onet_embeds = embed(onet_texts, model=EMBEDDING_MODEL)
print(f"O*NET embeddings shape: {onet_embeds.shape}")

# %%
# Also embed the alternate-title-enriched variant
print("Embedding enriched O*NET occupations...")
onet_embeds_with_alts = embed(onet_texts_with_alts, model=EMBEDDING_MODEL)
print(f"Enriched O*NET embeddings shape: {onet_embeds_with_alts.shape}")

# %%
# Compute major group centroids by averaging occupation embeddings per group
major_group_codes = sorted(onet_df["major_group"].unique())
group_to_idx = {mg: onet_df[onet_df["major_group"] == mg].index.tolist() for mg in major_group_codes}

centroids = np.zeros((len(major_group_codes), onet_embeds.shape[1]))
for i, mg in enumerate(major_group_codes):
    idxs = group_to_idx[mg]
    centroids[i] = onet_embeds[idxs].mean(axis=0)

# Normalize centroids
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
print(f"Major group centroids shape: {centroids.shape}")

# %%
# Embed job ads (raw title + description)
ad_texts = []
for _, row in ads_df.iterrows():
    title = str(row["title"] or "")
    desc = str(row["description"] or "")
    text = f"{title}. {desc}"[:30000]  # truncate to ~30K chars
    ad_texts.append(text)

print(f"Embedding {len(ad_texts)} job ads...")
ad_embeds = embed(ad_texts, model=EMBEDDING_MODEL)
print(f"Ad embeddings shape: {ad_embeds.shape}")

# %% [markdown]
# ## 6. Flat matching (baseline)

# %%
# Full cosine similarity matrix (SAMPLE_N x 861)
# Normalize embeddings for cosine similarity
ad_norms = ad_embeds / np.linalg.norm(ad_embeds, axis=1, keepdims=True)
onet_norms = onet_embeds / np.linalg.norm(onet_embeds, axis=1, keepdims=True)

sim_matrix = ad_norms @ onet_norms.T
print(f"Similarity matrix shape: {sim_matrix.shape}")

# %%
# Get top-K matches per ad from flat matching
TOP_K = 10
flat_top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :TOP_K]
flat_top_k_scores = np.take_along_axis(sim_matrix, flat_top_k_indices, axis=1)

# Build ranked lists for flat matching
onet_codes = onet_df["O*NET-SOC Code"].tolist()
flat_ranked = {}
for i, ad_id in enumerate(sample_ids):
    flat_ranked[ad_id] = [onet_codes[j] for j in flat_top_k_indices[i]]

# %%
# Compare flat embedding matching against the LLM-based reference
flat_top1 = top1_agreement(flat_ranked, ref_ranked)
flat_jaccard = topk_jaccard(flat_ranked, ref_ranked)

print("=== Flat Embedding Matching vs LLM-based Reference ===")
print(f"Top-1 agreement: {flat_top1:.4f}")
print(f"Top-K Jaccard:   {flat_jaccard:.4f}")

# Major-group-level agreement
flat_mg_agree = 0
for ad_id in sample_ids:
    flat_mg = flat_ranked[ad_id][0][:2]
    ref_mg = ref_ranked[ad_id][0][:2]
    if flat_mg == ref_mg:
        flat_mg_agree += 1
flat_mg_agreement = flat_mg_agree / len(sample_ids)
print(f"Major group agreement: {flat_mg_agreement:.4f}")

# %% [markdown]
# ## 7. Hierarchical matching (core experiment)

# %%
# Stage 1: Cosine similarity of each ad against major group centroids
ad_vs_centroids = ad_norms @ centroids.T  # (SAMPLE_N, 23)
print(f"Ad-centroid similarity matrix shape: {ad_vs_centroids.shape}")

# %%
# For each N (top groups to consider), measure coverage and candidate reduction
results = []
for n_groups in [1, 2, 3, 5, 7, 10]:
    top_groups_idx = np.argsort(-ad_vs_centroids, axis=1)[:, :n_groups]

    coverage_count = 0
    flat_agree_count = 0
    total_candidates = 0

    for i, ad_id in enumerate(sample_ids):
        # Get the selected major groups for this ad
        selected_groups = {major_group_codes[g] for g in top_groups_idx[i]}

        # Count candidates in selected groups
        n_candidates = sum(len(group_to_idx[mg]) for mg in selected_groups)
        total_candidates += n_candidates

        # Check coverage: is the reference top-1 in one of the selected groups?
        ref_top1_code = ref_ranked[ad_id][0]
        ref_mg = ref_top1_code[:2]
        if ref_mg in selected_groups:
            coverage_count += 1

        # Check agreement with flat matching
        # Get top-1 from hierarchical (mask out non-selected occupations)
        mask = np.zeros(len(onet_codes), dtype=bool)
        for mg in selected_groups:
            for idx in group_to_idx[mg]:
                mask[idx] = True
        masked_sims = sim_matrix[i].copy()
        masked_sims[~mask] = -1
        hier_top1_idx = np.argmax(masked_sims)
        hier_top1_code = onet_codes[hier_top1_idx]

        if hier_top1_code == flat_ranked[ad_id][0]:
            flat_agree_count += 1

    coverage = coverage_count / len(sample_ids)
    flat_agreement = flat_agree_count / len(sample_ids)
    mean_candidates = total_candidates / len(sample_ids)
    reduction = 1 - mean_candidates / len(onet_codes)

    results.append({
        "n_groups": n_groups,
        "mean_candidates": mean_candidates,
        "reduction": reduction,
        "coverage_of_ref_top1": coverage,
        "agreement_with_flat": flat_agreement,
    })

    print(f"N={n_groups:2d} groups | "
          f"Mean candidates: {mean_candidates:6.1f} | "
          f"Reduction: {reduction:.1%} | "
          f"Coverage of ref top-1: {coverage:.4f} | "
          f"Agreement w/ flat: {flat_agreement:.4f}")

# %%
results_df = pd.DataFrame(results)
print("\n=== Hierarchical Matching Results ===")
print(results_df.to_string(index=False))

# %%
# Plot coverage vs N groups
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(results_df["n_groups"], results_df["coverage_of_ref_top1"],
         "b-o", linewidth=2, markersize=8, label="Coverage of ref top-1")
ax1.plot(results_df["n_groups"], results_df["agreement_with_flat"],
         "g-s", linewidth=2, markersize=8, label="Agreement with flat")
ax1.set_xlabel("Number of top groups (N)")
ax1.set_ylabel("Fraction")
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(results_df["n_groups"], results_df["reduction"],
         "r--^", linewidth=2, markersize=8, label="Candidate reduction")
ax2.set_ylabel("Candidate reduction", color="r")
ax2.tick_params(axis="y", labelcolor="r")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="center right")

ax1.set_title("Hierarchical Matching: Coverage vs Candidate Reduction")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Group classification accuracy

# %%
# Using reference matches as pseudo-ground-truth
# True major group = first 2 digits of reference top-1
true_groups = [ref_ranked[ad_id][0][:2] for ad_id in sample_ids]
predicted_group_ranks = np.argsort(-ad_vs_centroids, axis=1)  # full ranking

# Top-1/3/5 accuracy
for k in [1, 3, 5]:
    correct = 0
    for i, ad_id in enumerate(sample_ids):
        top_k_groups = {major_group_codes[g] for g in predicted_group_ranks[i, :k]}
        if true_groups[i] in top_k_groups:
            correct += 1
    acc = correct / len(sample_ids)
    print(f"Top-{k} group classification accuracy: {acc:.4f}")

# %%
# Per-group accuracy breakdown
from collections import Counter

group_correct = Counter()
group_total = Counter()

for i, ad_id in enumerate(sample_ids):
    true_mg = true_groups[i]
    pred_mg = major_group_codes[predicted_group_ranks[i, 0]]
    group_total[true_mg] += 1
    if pred_mg == true_mg:
        group_correct[true_mg] += 1

print("\n=== Per-group classification accuracy (top-1) ===")
print(f"{'Group':>6} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
print("-" * 35)
for mg in sorted(group_total.keys()):
    acc = group_correct[mg] / group_total[mg] if group_total[mg] > 0 else 0
    print(f"{mg:>6} {group_correct[mg]:>8} {group_total[mg]:>6} {acc:>9.4f}")

# %%
# Confusion matrix heatmap
n_groups_total = len(major_group_codes)
confusion = np.zeros((n_groups_total, n_groups_total), dtype=int)
mg_to_idx = {mg: i for i, mg in enumerate(major_group_codes)}

for i, ad_id in enumerate(sample_ids):
    true_mg = true_groups[i]
    pred_mg = major_group_codes[predicted_group_ranks[i, 0]]
    confusion[mg_to_idx[true_mg], mg_to_idx[pred_mg]] += 1

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(confusion, cmap="Blues")
ax.set_xticks(range(n_groups_total))
ax.set_yticks(range(n_groups_total))
ax.set_xticklabels(major_group_codes, rotation=90, fontsize=8)
ax.set_yticklabels(major_group_codes, fontsize=8)
ax.set_xlabel("Predicted major group")
ax.set_ylabel("True major group (from reference)")
ax.set_title("Major Group Classification Confusion Matrix")
fig.colorbar(im)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Alternative text representations

# %%
# Test different ad text constructions
text_variants = {}

# Variant 1: Title only
text_variants["title_only"] = [str(row["title"] or "") for _, row in ads_df.iterrows()]

# Variant 2: Title + first 500 chars of description
text_variants["title_desc_500"] = [
    f"{row['title']}. {str(row['description'] or '')[:500]}"
    for _, row in ads_df.iterrows()
]

# Variant 3: Title + full description (default, already embedded)
text_variants["title_desc_full"] = ad_texts

# Variant 4: Title + category + full description
text_variants["title_cat_desc"] = [
    f"{row['title']} [{row['category_name']}]. {str(row['description'] or '')}"[:30000]
    for _, row in ads_df.iterrows()
]

for name, texts in text_variants.items():
    mean_len = np.mean([len(t) for t in texts])
    print(f"{name:20s}: mean length {mean_len:.0f} chars")

# %%
# Embed each variant and compute flat matching agreement with reference
variant_results = []
for name, texts in text_variants.items():
    if name == "title_desc_full":
        # Already embedded
        v_embeds = ad_embeds
    else:
        print(f"Embedding variant: {name}...")
        v_embeds = embed(texts, model=EMBEDDING_MODEL)

    # Normalize and compute similarities
    v_norms = v_embeds / np.linalg.norm(v_embeds, axis=1, keepdims=True)
    v_sims = v_norms @ onet_norms.T

    # Build ranked lists
    v_top_k_indices = np.argsort(-v_sims, axis=1)[:, :TOP_K]
    v_ranked = {}
    for i, ad_id in enumerate(sample_ids):
        v_ranked[ad_id] = [onet_codes[j] for j in v_top_k_indices[i]]

    v_top1 = top1_agreement(v_ranked, ref_ranked)
    v_jaccard = topk_jaccard(v_ranked, ref_ranked)

    # Major group agreement
    v_mg_agree = sum(
        1 for ad_id in sample_ids
        if v_ranked[ad_id][0][:2] == ref_ranked[ad_id][0][:2]
    ) / len(sample_ids)

    variant_results.append({
        "variant": name,
        "top1_vs_ref": v_top1,
        "jaccard_vs_ref": v_jaccard,
        "major_group_agreement": v_mg_agree,
    })

variant_df = pd.DataFrame(variant_results)
print("\n=== Ad Text Variant Comparison ===")
print(variant_df.to_string(index=False))

# %%
# Also compare with the enriched O*NET text variant (alternate titles)
onet_norms_alts = onet_embeds_with_alts / np.linalg.norm(onet_embeds_with_alts, axis=1, keepdims=True)
sims_alts = ad_norms @ onet_norms_alts.T

alt_top_k_indices = np.argsort(-sims_alts, axis=1)[:, :TOP_K]
alt_ranked = {}
for i, ad_id in enumerate(sample_ids):
    alt_ranked[ad_id] = [onet_codes[j] for j in alt_top_k_indices[i]]

alt_top1 = top1_agreement(alt_ranked, ref_ranked)
alt_jaccard = topk_jaccard(alt_ranked, ref_ranked)
alt_mg = sum(
    1 for ad_id in sample_ids
    if alt_ranked[ad_id][0][:2] == ref_ranked[ad_id][0][:2]
) / len(sample_ids)

print(f"\n=== O*NET Text with Alternate Titles ===")
print(f"Top-1 vs reference: {alt_top1:.4f}")
print(f"Jaccard vs reference: {alt_jaccard:.4f}")
print(f"Major group agreement: {alt_mg:.4f}")

# Comparison
print(f"\nBaseline (no alt titles): top1={flat_top1:.4f}, jaccard={flat_jaccard:.4f}")
print(f"With alt titles:         top1={alt_top1:.4f}, jaccard={alt_jaccard:.4f}")

# %% [markdown]
# ## 10. LLM evaluation of match quality
#
# Use GPT-5.4 to judge whether each top-1 match is a good occupation match for
# the ad. This gives us an absolute quality signal rather than comparing against
# another pipeline's output.

# %%
import json
from adulib.llm import async_single
from adulib.asynchronous import batch_executor

EVAL_MODEL = "openai/gpt-5.4"
EVAL_SAMPLE_N = 200  # evaluate a subsample to keep costs reasonable
EVAL_SEED = 123

# %%
# Build evaluation set: for each sampled ad, pair with its flat-matching top-1
eval_rng = np.random.default_rng(EVAL_SEED)
eval_ids = eval_rng.choice(sample_ids, size=min(EVAL_SAMPLE_N, len(sample_ids)), replace=False).tolist()

onet_title_lookup = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Title"]))
onet_desc_lookup = dict(zip(onet_df["O*NET-SOC Code"], onet_df["Description"]))

eval_records = []
for ad_id in eval_ids:
    ad_row = ads_df.loc[ad_id]
    top1_code = flat_ranked[ad_id][0]
    top1_title = onet_title_lookup[top1_code]
    top1_desc = onet_desc_lookup[top1_code]
    eval_records.append({
        "ad_id": ad_id,
        "ad_title": str(ad_row["title"] or ""),
        "ad_desc": str(ad_row["description"] or "")[:1500],
        "ad_category": str(ad_row["category_name"] or ""),
        "onet_code": top1_code,
        "onet_title": top1_title,
        "onet_desc": top1_desc,
    })

print(f"Built {len(eval_records)} evaluation pairs")

# %%
EVAL_SYSTEM = """You are an expert occupational classification evaluator. You will be given a job advertisement and a candidate O*NET occupation match. Judge how well the occupation matches the job ad.

Respond with JSON only:
{
  "rating": <1-5>,
  "reasoning": "<brief explanation>"
}

Rating scale:
5 = Perfect match. The occupation precisely describes this job.
4 = Good match. The occupation captures the core role, minor differences in specifics.
3 = Partial match. Related field but notable differences in duties or level.
2 = Weak match. Same broad domain but substantially different role.
1 = Poor match. Unrelated or wrong occupation."""

eval_prompts = []
for r in eval_records:
    prompt = f"""## Job Advertisement
**Title:** {r['ad_title']}
**Category:** {r['ad_category']}
**Description:** {r['ad_desc']}

## Candidate O*NET Occupation
**Code:** {r['onet_code']}
**Title:** {r['onet_title']}
**Description:** {r['onet_desc']}

Rate this match (1-5) and explain briefly."""
    eval_prompts.append(prompt)

print(f"Built {len(eval_prompts)} evaluation prompts")

# %%
# Run LLM evaluation
async def _eval_call(prompt):
    response_text, _cache_hit, _call_log = await async_single(
        prompt,
        model=EVAL_MODEL,
        system=EVAL_SYSTEM,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    return response_text

print(f"Evaluating {len(eval_prompts)} matches with {EVAL_MODEL}...")
eval_responses = await batch_executor(
    _eval_call,
    batch_args=[(p,) for p in eval_prompts],
    concurrency_limit=20,
)
print(f"Got {len(eval_responses)} responses")

# %%
# Parse responses
eval_ratings = []
eval_reasonings = []
parse_failures = 0
for i, resp in enumerate(eval_responses):
    try:
        parsed = json.loads(resp)
        rating = int(parsed["rating"])
        reasoning = parsed["reasoning"]
    except (json.JSONDecodeError, KeyError, ValueError):
        rating = None
        reasoning = f"PARSE FAILURE: {resp[:200]}"
        parse_failures += 1
    eval_ratings.append(rating)
    eval_reasonings.append(reasoning)

valid_ratings = [r for r in eval_ratings if r is not None]
print(f"\n=== LLM Evaluation Results ({EVAL_MODEL}) ===")
print(f"Evaluated: {len(eval_ratings)}, Parse failures: {parse_failures}")
print(f"Mean rating: {np.mean(valid_ratings):.2f} (1=poor, 5=perfect)")
print(f"Median rating: {np.median(valid_ratings):.1f}")
print(f"\nRating distribution:")
from collections import Counter
rating_counts = Counter(valid_ratings)
for r in sorted(rating_counts):
    pct = rating_counts[r] / len(valid_ratings) * 100
    bar = "#" * int(pct / 2)
    print(f"  {r}: {rating_counts[r]:4d} ({pct:5.1f}%) {bar}")

good_match_pct = sum(1 for r in valid_ratings if r >= 4) / len(valid_ratings) * 100
print(f"\nGood matches (rating >= 4): {good_match_pct:.1f}%")

# %% [markdown]
# ## 11. Sample matches for manual inspection

# %%
# Print a stratified sample: some good, some bad
eval_df = pd.DataFrame({
    "ad_id": eval_ids,
    "ad_title": [r["ad_title"] for r in eval_records],
    "ad_category": [r["ad_category"] for r in eval_records],
    "onet_code": [r["onet_code"] for r in eval_records],
    "onet_title": [r["onet_title"] for r in eval_records],
    "rating": eval_ratings,
    "reasoning": eval_reasonings,
})

import textwrap

def print_match(row, ad_desc_lookup):
    print(f"{'='*80}")
    print(f"RATING: {row['rating']}/5")
    print(f"Ad title:    {row['ad_title']}")
    print(f"Ad category: {row['ad_category']}")
    desc = ad_desc_lookup.get(row['ad_id'], '')[:300]
    print(f"Ad desc:     {textwrap.shorten(desc, 200)}")
    print(f"O*NET match: {row['onet_code']} - {row['onet_title']}")
    print(f"Reasoning:   {row['reasoning']}")
    print()

# Build ad description lookup from ads_df
ad_desc_lookup = {ad_id: str(row["description"] or "") for ad_id, row in ads_df.iterrows()}

# Show 5 best and 5 worst matches
valid_eval_df = eval_df[eval_df["rating"].notna()].copy()

print("=== TOP 5 BEST MATCHES ===\n")
best = valid_eval_df.nlargest(5, "rating")
for _, row in best.iterrows():
    print_match(row, ad_desc_lookup)

print("\n=== TOP 5 WORST MATCHES ===\n")
worst = valid_eval_df.nsmallest(5, "rating")
for _, row in worst.iterrows():
    print_match(row, ad_desc_lookup)

print("\n=== 5 RANDOM MIDDLE MATCHES (rating=3) ===\n")
middle = valid_eval_df[valid_eval_df["rating"] == 3]
if len(middle) > 5:
    middle = middle.sample(5, random_state=42)
for _, row in middle.iterrows():
    print_match(row, ad_desc_lookup)

# %% [markdown]
# ## 12. Summary
#
# ### Key findings
#
# **Match quality (LLM-evaluated):**
# - Check the mean rating and rating distribution above
# - The fraction of matches rated 4+ tells us how often raw-text embedding
#   finds a genuinely good occupation match
#
# **Hierarchical classification viability:**
# - Check the results table for coverage at N=3 and N=5 groups
# - If coverage at N=3 exceeds ~90%, hierarchical matching is viable
# - The candidate set reduction tells us the computational savings
#
# **Text representation:**
# - The variant comparison shows which ad text construction works best
# - Title-only vs full text quantifies the value of the description
#
# **Recommendations for pipeline redesign:**
# - If hierarchical N=3 gives >90% coverage with >50% candidate reduction,
#   adopt the two-stage approach
# - If LLM evaluation shows >60% of matches rated 4+, raw-text embedding
#   may be viable without LLM summarisation
# - Use whichever text variant scores highest for the final pipeline
