# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Validation: Model Agreement Analysis
#
# Compares the 14 completed Arm 1 validation runs (vary LLM, fix embedding to
# `bge-large-sbatch`, fix reranker to `qwen3-reranker-8b-sbatch`).
#
# **Questions:**
# 1. Cosine candidates: are they identical across runs? (Same embedding = should be)
# 2. LLM filter: how much do different LLMs agree on which candidates to keep?
# 3. Rerank: how much do the reranked orderings agree across runs?

# %%
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

from ai_index import const

# %% [markdown]
# ## Discover completed validation runs

# %%
pipeline_dir = const.pipeline_store_path
val_dirs = sorted([
    d for d in pipeline_dir.iterdir()
    if d.is_dir() and d.name.startswith("val__")
])

# Check which runs have all three stages complete
completed_runs = {}
for d in val_dirs:
    has_cosine = (d / "cosine_candidates" / "candidates.parquet").exists()
    has_filter = (d / "llm_filter_candidates" / "filtered_matches.parquet").exists()
    has_rerank = (d / "rerank_candidates" / "reranked_matches.parquet").exists()
    if has_cosine and has_filter and has_rerank:
        # Extract LLM model name from run name: val__validation_5k__{llm}__bge-large-sbatch
        parts = d.name.split("__")
        llm_model = parts[2] if len(parts) >= 3 else d.name
        completed_runs[llm_model] = d

print(f"Found {len(completed_runs)} completed validation runs:")
for name, path in sorted(completed_runs.items()):
    print(f"  {name}")

# %% [markdown]
# ## 1. Cosine Candidates: Verify Identity
#
# All runs use the same embedding model (`bge-large-sbatch`) and same sample
# (`sample_n=5000, sample_seed=42`), so the cosine candidates should be
# identical. Let's verify.

# %%
cosine_dfs = {}
for llm, run_dir in sorted(completed_runs.items()):
    df = pd.read_parquet(run_dir / "cosine_candidates" / "candidates.parquet")
    cosine_dfs[llm] = df

# Compare all runs against the first one
ref_name = sorted(cosine_dfs.keys())[0]
ref_df = cosine_dfs[ref_name]

print(f"Reference run: {ref_name} ({len(ref_df)} rows)")
all_identical = True
for llm, df in sorted(cosine_dfs.items()):
    if llm == ref_name:
        continue
    # Compare ad_id + onet_code pairs (ignore score floating point noise)
    ref_pairs = set(zip(ref_df["ad_id"], ref_df["onet_code"]))
    other_pairs = set(zip(df["ad_id"], df["onet_code"]))
    if ref_pairs == other_pairs:
        print(f"  {llm}: IDENTICAL candidate sets")
    else:
        all_identical = False
        only_ref = ref_pairs - other_pairs
        only_other = other_pairs - ref_pairs
        print(f"  {llm}: DIFFERENT! {len(only_ref)} only in ref, {len(only_other)} only in {llm}")

if all_identical:
    print("\nAll cosine candidate sets are identical (as expected with same embedding model).")

# %% [markdown]
# ### Cosine overlap at different top-K thresholds
#
# Even though the full top-20 sets are identical, let's verify at different
# cutoffs. This also sets up the methodology for comparing with other embeddings
# in future Arm 2 runs.

# %%
ref_cosine = cosine_dfs[ref_name]

# Build per-ad ranked candidate lists from the reference
cosine_by_ad = ref_cosine.groupby("ad_id").apply(
    lambda g: g.sort_values("rank")["onet_code"].tolist(),
    include_groups=False,
)

n_ads = len(cosine_by_ad)
print(f"Cosine candidates: {n_ads} ads, top-{ref_cosine['rank'].max() + 1} each")
print(f"\n(Since all runs are identical, overlap is 100% at all thresholds.)")
print(f"This section will become meaningful when Arm 2 (vary embedding) runs are added.")

# %% [markdown]
# ## 2. LLM Filter Agreement
#
# For each job ad, each LLM model selects a subset of the 20 cosine candidates
# to keep. We compute pairwise Jaccard similarity of the kept sets.

# %%
# Load filtered matches for each LLM
filter_dfs = {}
for llm, run_dir in sorted(completed_runs.items()):
    df = pd.read_parquet(run_dir / "llm_filter_candidates" / "filtered_matches.parquet")
    filter_dfs[llm] = df

# Summary stats
print("LLM Filter summary:")
print(f"{'Model':<25s} {'Ads':>6s} {'Matches':>8s} {'Mean/ad':>8s} {'Median':>7s}")
print("-" * 60)
for llm in sorted(filter_dfs.keys()):
    df = filter_dfs[llm]
    counts = df.groupby("ad_id").size()
    print(f"{llm:<25s} {len(counts):>6d} {len(df):>8d} {counts.mean():>8.1f} {counts.median():>7.0f}")

# %%
# Build per-ad kept sets for each LLM
filter_sets = {}
for llm, df in filter_dfs.items():
    by_ad = df.groupby("ad_id")["onet_code"].apply(set).to_dict()
    filter_sets[llm] = by_ad

# Find ads that appear in ALL runs (intersection of successful ads)
all_ad_sets = [set(s.keys()) for s in filter_sets.values()]
common_ads = sorted(set.intersection(*all_ad_sets))
print(f"\nAds present in all {len(filter_sets)} runs: {len(common_ads)}")

# %% [markdown]
# ### Pairwise Jaccard similarity (LLM filter)

# %%
llm_names = sorted(filter_sets.keys())
n_llms = len(llm_names)

# Compute pairwise mean Jaccard over common ads
jaccard_matrix = np.zeros((n_llms, n_llms))
for i, llm_a in enumerate(llm_names):
    for j, llm_b in enumerate(llm_names):
        if i == j:
            jaccard_matrix[i, j] = 1.0
            continue
        jaccards = []
        for ad_id in common_ads:
            set_a = filter_sets[llm_a].get(ad_id, set())
            set_b = filter_sets[llm_b].get(ad_id, set())
            if not set_a and not set_b:
                continue
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccards.append(intersection / union if union > 0 else 0.0)
        jaccard_matrix[i, j] = np.mean(jaccards) if jaccards else 0.0

jaccard_df = pd.DataFrame(jaccard_matrix, index=llm_names, columns=llm_names)

print("Pairwise mean Jaccard similarity (LLM filter kept sets):")
print(jaccard_df.round(3).to_string())

# %%
# Overall statistics
upper_tri = jaccard_matrix[np.triu_indices(n_llms, k=1)]
print(f"\nOverall pairwise Jaccard statistics:")
print(f"  Mean:   {upper_tri.mean():.3f}")
print(f"  Median: {np.median(upper_tri):.3f}")
print(f"  Min:    {upper_tri.min():.3f}")
print(f"  Max:    {upper_tri.max():.3f}")
print(f"  Std:    {upper_tri.std():.3f}")

# %% [markdown]
# ### Top-1 agreement (LLM filter)
#
# The LLM filter does not rank candidates. It only selects which to keep.
# The `rank` column preserves the original cosine ordering among kept candidates.
# So "top-1 agreement" means: do both LLMs keep the same highest-cosine-ranked
# candidate?

# %%
# Top-1 agreement: do LLMs agree on the first-ranked kept candidate?
filter_top1 = {}
for llm, df in filter_dfs.items():
    top1 = df.sort_values(["ad_id", "rank"]).groupby("ad_id").first()["onet_code"].to_dict()
    filter_top1[llm] = top1

top1_agreement = np.zeros((n_llms, n_llms))
for i, llm_a in enumerate(llm_names):
    for j, llm_b in enumerate(llm_names):
        if i == j:
            top1_agreement[i, j] = 1.0
            continue
        agrees = 0
        total = 0
        for ad_id in common_ads:
            a = filter_top1[llm_a].get(ad_id)
            b = filter_top1[llm_b].get(ad_id)
            if a is not None and b is not None:
                total += 1
                if a == b:
                    agrees += 1
        top1_agreement[i, j] = agrees / total if total > 0 else 0.0

top1_df = pd.DataFrame(top1_agreement, index=llm_names, columns=llm_names)
print("Pairwise top-1 agreement (LLM filter, fraction of ads where top-ranked match is same):")
print(top1_df.round(3).to_string())

# %%
upper_tri_top1 = top1_agreement[np.triu_indices(n_llms, k=1)]
print(f"\nOverall top-1 agreement statistics:")
print(f"  Mean:   {upper_tri_top1.mean():.3f}")
print(f"  Median: {np.median(upper_tri_top1):.3f}")
print(f"  Min:    {upper_tri_top1.min():.3f}")
print(f"  Max:    {upper_tri_top1.max():.3f}")

# %% [markdown]
# ### "At least N overlap" (LLM filter)
#
# For each pair of LLMs, what fraction of ads have at least N candidates in
# common? The threshold is clamped to the smaller set size: if both models keep
# only 2 candidates and share both, that passes "at least 3" since they agree
# on everything available.

# %%
def compute_at_least_n_overlap(filter_sets, llm_names, common_ads, n_threshold):
    """Fraction of ads where two models share at least min(n, min(|A|, |B|)) candidates."""
    n_llms = len(llm_names)
    matrix = np.zeros((n_llms, n_llms))
    for i, llm_a in enumerate(llm_names):
        for j, llm_b in enumerate(llm_names):
            if i == j:
                matrix[i, j] = 1.0
                continue
            passes = 0
            total = 0
            for ad_id in common_ads:
                set_a = filter_sets[llm_a].get(ad_id, set())
                set_b = filter_sets[llm_b].get(ad_id, set())
                if set_a and set_b:
                    total += 1
                    effective_threshold = min(n_threshold, len(set_a), len(set_b))
                    if len(set_a & set_b) >= effective_threshold:
                        passes += 1
            matrix[i, j] = passes / total if total > 0 else 0.0
    return matrix

for n_thresh in [1, 2, 3]:
    matrix = compute_at_least_n_overlap(filter_sets, llm_names, common_ads, n_thresh)
    upper = matrix[np.triu_indices(n_llms, k=1)]
    print(f"'At least {n_thresh} overlap' (clamped to smaller set):")
    print(f"  Mean:   {upper.mean():.3f}    Median: {np.median(upper):.3f}    Min: {upper.min():.3f}")
    if n_thresh == 1:
        upper_tri_overlap = upper
    print()

# Show full matrix for at-least-1 (most interpretable)
overlap_1 = compute_at_least_n_overlap(filter_sets, llm_names, common_ads, 1)
overlap_1_df = pd.DataFrame(overlap_1, index=llm_names, columns=llm_names)
print("Pairwise 'at least 1 overlap' matrix:")
print(overlap_1_df.round(3).to_string())

# Show full matrix for at-least-3
print()
overlap_3 = compute_at_least_n_overlap(filter_sets, llm_names, common_ads, 3)
overlap_3_df = pd.DataFrame(overlap_3, index=llm_names, columns=llm_names)
print("Pairwise 'at least 3 overlap' matrix:")
print(overlap_3_df.round(3).to_string())

# %% [markdown]
# ### Per-model agreement with majority vote
#
# For each ad, compute the "majority" set of candidates (kept by more than half
# of models). Then measure each model's F1 against this consensus.

# %%
n_models = len(llm_names)
majority_threshold = n_models / 2

# Count how many models keep each (ad_id, onet_code) pair
candidate_votes = {}
for llm in llm_names:
    for ad_id in common_ads:
        for code in filter_sets[llm].get(ad_id, set()):
            key = (ad_id, code)
            candidate_votes[key] = candidate_votes.get(key, 0) + 1

# Build majority set per ad
majority_sets = {}
for (ad_id, code), count in candidate_votes.items():
    if count > majority_threshold:
        majority_sets.setdefault(ad_id, set()).add(code)

ads_with_majority = [ad_id for ad_id in common_ads if ad_id in majority_sets]
print(f"Ads with at least one majority candidate: {len(ads_with_majority)} / {len(common_ads)}")

# Compute per-model F1 against majority
print(f"\n{'Model':<25s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s}")
print("-" * 55)
for llm in llm_names:
    tp, fp, fn = 0, 0, 0
    for ad_id in ads_with_majority:
        model_set = filter_sets[llm].get(ad_id, set())
        maj_set = majority_sets[ad_id]
        tp += len(model_set & maj_set)
        fp += len(model_set - maj_set)
        fn += len(maj_set - model_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"{llm:<25s} {precision:>10.3f} {recall:>8.3f} {f1:>6.3f}")

# %% [markdown]
# ### Best model subsets by Jaccard
#
# Find the subset of size k with the highest mean pairwise Jaccard. With 14
# models, exhaustive search over all C(14,k) subsets is feasible.

# %%
from itertools import combinations

def mean_pairwise_jaccard(subset_indices, jaccard_matrix):
    """Mean of upper-triangle Jaccard values for a subset of model indices."""
    pairs = list(combinations(subset_indices, 2))
    if not pairs:
        return 1.0
    return np.mean([jaccard_matrix[i, j] for i, j in pairs])

print(f"{'k':>3s}  {'Mean Jaccard':>12s}  Best subset")
print("-" * 80)
for k in range(len(llm_names), 1, -1):
    best_score = -1
    best_subset = None
    for subset in combinations(range(n_llms), k):
        score = mean_pairwise_jaccard(subset, jaccard_matrix)
        if score > best_score:
            best_score = score
            best_subset = subset
    names = [llm_names[i] for i in best_subset]
    # Truncate display for large subsets
    if len(names) > 6:
        removed = sorted(set(llm_names) - set(names))
        display = f"all except: {', '.join(removed)}" if removed else "all"
    else:
        display = ", ".join(names)
    print(f"{k:>3d}  {best_score:>12.3f}  {display}")

# %% [markdown]
# ## 3. Rerank Agreement
#
# All runs use the same reranker (`qwen3-reranker-8b-sbatch`), but they receive
# different candidate sets (from different LLM filters). We compare:
# 1. The rerank orderings on the intersection of candidates kept by each pair
# 2. The top-1 reranked candidate per ad

# %%
# Load reranked matches (skip corrupted parquets)
rerank_dfs = {}
for llm, run_dir in sorted(completed_runs.items()):
    try:
        df = pd.read_parquet(run_dir / "rerank_candidates" / "reranked_matches.parquet")
        rerank_dfs[llm] = df
    except Exception as e:
        print(f"  SKIPPING {llm}: {e}")

print(f"Loaded rerank data for {len(rerank_dfs)} / {len(completed_runs)} runs")

# Build per-ad ranked lists (ordered by rerank_score descending)
rerank_ranked = {}
for llm, df in rerank_dfs.items():
    by_ad = {}
    for ad_id, group in df.groupby("ad_id"):
        ranked = group.sort_values("rerank_score", ascending=False)["onet_code"].tolist()
        by_ad[int(ad_id)] = ranked
    rerank_ranked[llm] = by_ad

rerank_llm_names = sorted(rerank_dfs.keys())
n_rerank_llms = len(rerank_llm_names)

# Common ads across rerank runs
rerank_ad_sets = [set(rerank_ranked[llm].keys()) for llm in rerank_llm_names]
rerank_common_ads = sorted(set.intersection(*rerank_ad_sets))
print(f"Ads present in all {n_rerank_llms} rerank runs: {len(rerank_common_ads)}")

# %% [markdown]
# ### Rerank-weighted Jaccard
#
# Weighted Jaccard (Ruzicka similarity) uses rerank scores as weights instead
# of binary set membership. For each ad:
# - Weighted intersection = sum(min(score_A[c], score_B[c])) for shared candidates
# - Weighted union = sum(max(score_A[c], score_B[c])) for all candidates
#
# High-scoring candidates that both models agree on contribute a lot. Low-scoring
# candidates in only one model barely affect the metric.

# %%
# Build per-ad score dicts (needed here and later for Spearman)
rerank_scores = {}
for llm, df in rerank_dfs.items():
    by_ad = {}
    for ad_id, group in df.groupby("ad_id"):
        by_ad[int(ad_id)] = dict(zip(group["onet_code"], group["rerank_score"]))
    rerank_scores[llm] = by_ad

def weighted_jaccard_ad(scores_a, scores_b):
    """Weighted Jaccard (Ruzicka) for one ad's candidate scores."""
    all_codes = set(scores_a.keys()) | set(scores_b.keys())
    if not all_codes:
        return np.nan
    w_intersection = sum(min(scores_a.get(c, 0.0), scores_b.get(c, 0.0)) for c in all_codes)
    w_union = sum(max(scores_a.get(c, 0.0), scores_b.get(c, 0.0)) for c in all_codes)
    return w_intersection / w_union if w_union > 0 else 0.0

wj_matrix = np.zeros((n_rerank_llms, n_rerank_llms))
for i, llm_a in enumerate(rerank_llm_names):
    for j, llm_b in enumerate(rerank_llm_names):
        if i == j:
            wj_matrix[i, j] = 1.0
            continue
        wjs = []
        for ad_id in rerank_common_ads:
            sa = rerank_scores[llm_a].get(ad_id, {})
            sb = rerank_scores[llm_b].get(ad_id, {})
            wj = weighted_jaccard_ad(sa, sb)
            if not np.isnan(wj):
                wjs.append(wj)
        wj_matrix[i, j] = np.mean(wjs) if wjs else 0.0

wj_df = pd.DataFrame(wj_matrix, index=rerank_llm_names, columns=rerank_llm_names)
print("Pairwise rerank-weighted Jaccard (Ruzicka similarity):")
print(wj_df.round(3).to_string())

upper_tri_wj = wj_matrix[np.triu_indices(n_rerank_llms, k=1)]
print(f"\nOverall weighted Jaccard statistics:")
print(f"  Mean:   {upper_tri_wj.mean():.3f}")
print(f"  Median: {np.median(upper_tri_wj):.3f}")
print(f"  Min:    {upper_tri_wj.min():.3f}")
print(f"  Max:    {upper_tri_wj.max():.3f}")

# %% [markdown]
# ### Best model subsets by weighted Jaccard

# %%
def mean_pairwise_from_matrix(subset_indices, matrix):
    pairs = list(combinations(subset_indices, 2))
    if not pairs:
        return 1.0
    return np.mean([matrix[i, j] for i, j in pairs])

print(f"{'k':>3s}  {'Weighted J':>11s}  Best subset")
print("-" * 80)
for k in range(n_rerank_llms, 1, -1):
    best_score = -1
    best_subset = None
    for subset in combinations(range(n_rerank_llms), k):
        score = mean_pairwise_from_matrix(subset, wj_matrix)
        if score > best_score:
            best_score = score
            best_subset = subset
    names = [rerank_llm_names[i] for i in best_subset]
    if len(names) > 6:
        removed = sorted(set(rerank_llm_names) - set(names))
        display = f"all except: {', '.join(removed)}" if removed else "all"
    else:
        display = ", ".join(names)
    print(f"{k:>3d}  {best_score:>11.3f}  {display}")

# %% [markdown]
# ### Top-1 agreement (rerank)
#
# After reranking, do different runs agree on the best-scoring occupation?

# %%
rerank_top1 = {}
for llm, ranked in rerank_ranked.items():
    rerank_top1[llm] = {ad_id: codes[0] for ad_id, codes in ranked.items() if codes}

top1_rerank_agreement = np.zeros((n_rerank_llms, n_rerank_llms))
for i, llm_a in enumerate(rerank_llm_names):
    for j, llm_b in enumerate(rerank_llm_names):
        if i == j:
            top1_rerank_agreement[i, j] = 1.0
            continue
        agrees = 0
        total = 0
        for ad_id in rerank_common_ads:
            a = rerank_top1[llm_a].get(ad_id)
            b = rerank_top1[llm_b].get(ad_id)
            if a is not None and b is not None:
                total += 1
                if a == b:
                    agrees += 1
        top1_rerank_agreement[i, j] = agrees / total if total > 0 else 0.0

top1_rerank_df = pd.DataFrame(top1_rerank_agreement, index=rerank_llm_names, columns=rerank_llm_names)
print("Pairwise top-1 agreement (after reranking):")
print(top1_rerank_df.round(3).to_string())

upper_tri_rerank = top1_rerank_agreement[np.triu_indices(n_rerank_llms, k=1)]
print(f"\nOverall top-1 rerank agreement:")
print(f"  Mean:   {upper_tri_rerank.mean():.3f}")
print(f"  Median: {np.median(upper_tri_rerank):.3f}")
print(f"  Min:    {upper_tri_rerank.min():.3f}")
print(f"  Max:    {upper_tri_rerank.max():.3f}")

# %% [markdown]
# ### Rerank score-weighted overlap
#
# For candidates kept by both runs, compute the Spearman rank correlation
# of their rerank scores.

# %%
from scipy.stats import spearmanr

# rerank_scores already built above in the weighted Jaccard section

spearman_matrix = np.zeros((n_rerank_llms, n_rerank_llms))
for i, llm_a in enumerate(rerank_llm_names):
    for j, llm_b in enumerate(rerank_llm_names):
        if i == j:
            spearman_matrix[i, j] = 1.0
            continue
        correlations = []
        for ad_id in rerank_common_ads:
            scores_a = rerank_scores[llm_a].get(ad_id, {})
            scores_b = rerank_scores[llm_b].get(ad_id, {})
            common_codes = set(scores_a.keys()) & set(scores_b.keys())
            if len(common_codes) >= 3:  # need at least 3 points for meaningful correlation
                vals_a = [scores_a[c] for c in common_codes]
                vals_b = [scores_b[c] for c in common_codes]
                rho, _ = spearmanr(vals_a, vals_b)
                if not np.isnan(rho):
                    correlations.append(rho)
        spearman_matrix[i, j] = np.mean(correlations) if correlations else np.nan

spearman_df = pd.DataFrame(spearman_matrix, index=rerank_llm_names, columns=rerank_llm_names)
print("Pairwise mean Spearman correlation of rerank scores (on shared candidates):")
print(spearman_df.round(3).to_string())

upper_tri_spearman = spearman_matrix[np.triu_indices(n_rerank_llms, k=1)]
valid = upper_tri_spearman[~np.isnan(upper_tri_spearman)]
print(f"\nOverall Spearman statistics (on shared candidates):")
print(f"  Mean:   {valid.mean():.3f}")
print(f"  Median: {np.median(valid):.3f}")
print(f"  Min:    {valid.min():.3f}")
print(f"  Max:    {valid.max():.3f}")
print(f"  N pairs with data: {len(valid)} / {len(upper_tri_spearman)}")

# %% [markdown]
# ### Rerank: fraction of shared candidates
#
# Since different LLM filters produce different candidate sets, how much
# overlap is there after filtering (which is what the reranker sees)?

# %%
shared_frac_matrix = np.zeros((n_rerank_llms, n_rerank_llms))
for i, llm_a in enumerate(rerank_llm_names):
    for j, llm_b in enumerate(rerank_llm_names):
        if i == j:
            shared_frac_matrix[i, j] = 1.0
            continue
        fracs = []
        for ad_id in rerank_common_ads:
            codes_a = set(rerank_scores[llm_a].get(ad_id, {}).keys())
            codes_b = set(rerank_scores[llm_b].get(ad_id, {}).keys())
            union = codes_a | codes_b
            if union:
                fracs.append(len(codes_a & codes_b) / len(union))
        shared_frac_matrix[i, j] = np.mean(fracs) if fracs else 0.0

shared_df = pd.DataFrame(shared_frac_matrix, index=rerank_llm_names, columns=rerank_llm_names)
print("Pairwise mean Jaccard of reranked candidate sets:")
print(shared_df.round(3).to_string())

# %% [markdown]
# ## 4. Exposure Score Agreement
#
# The final output: per-ad exposure scores. Since O\*NET occupation-level scores
# are identical across runs (same scoring nodes), all variation comes from which
# candidates the LLM filter kept and how the reranker weighted them.

# %%
# Load exposure scores from each completed run
exposure_dfs = {}
for llm, run_dir in sorted(completed_runs.items()):
    path = run_dir / "compute_job_ad_exposure" / "ad_exposure.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        exposure_dfs[llm] = df

print(f"Loaded exposure data for {len(exposure_dfs)} runs")

# Identify score columns (exclude ad_id, n_matches, error)
sample_df = next(iter(exposure_dfs.values()))
score_cols = [c for c in sample_df.columns if c not in ("ad_id", "n_matches", "error")]
print(f"Score columns: {score_cols}")

# Find common ads across all runs (exclude NaN rows)
exposure_by_llm = {}
for llm, df in exposure_dfs.items():
    clean = df.dropna(subset=score_cols).set_index("ad_id")
    exposure_by_llm[llm] = clean

exposure_ad_sets = [set(df.index) for df in exposure_by_llm.values()]
exposure_common_ads = sorted(set.intersection(*exposure_ad_sets))
print(f"Common ads with valid scores across all runs: {len(exposure_common_ads)}")

# %%
exposure_llm_names = sorted(exposure_by_llm.keys())
n_exp_llms = len(exposure_llm_names)

# %% [markdown]
# ### Pairwise Pearson correlation per score column

# %%
from scipy.stats import pearsonr, spearmanr as spearmanr_fn

for col in score_cols:
    pearson_matrix = np.zeros((n_exp_llms, n_exp_llms))
    for i, llm_a in enumerate(exposure_llm_names):
        for j, llm_b in enumerate(exposure_llm_names):
            if i == j:
                pearson_matrix[i, j] = 1.0
                continue
            vals_a = exposure_by_llm[llm_a].loc[exposure_common_ads, col].values
            vals_b = exposure_by_llm[llm_b].loc[exposure_common_ads, col].values
            r, _ = pearsonr(vals_a, vals_b)
            pearson_matrix[i, j] = r

    upper = pearson_matrix[np.triu_indices(n_exp_llms, k=1)]
    print(f"{col}:")
    print(f"  Pearson  mean={upper.mean():.4f}  median={np.median(upper):.4f}  min={upper.min():.4f}  max={upper.max():.4f}")

# %% [markdown]
# ### Pairwise Spearman rank correlation per score column

# %%
for col in score_cols:
    spearman_matrix = np.zeros((n_exp_llms, n_exp_llms))
    for i, llm_a in enumerate(exposure_llm_names):
        for j, llm_b in enumerate(exposure_llm_names):
            if i == j:
                spearman_matrix[i, j] = 1.0
                continue
            vals_a = exposure_by_llm[llm_a].loc[exposure_common_ads, col].values
            vals_b = exposure_by_llm[llm_b].loc[exposure_common_ads, col].values
            rho, _ = spearmanr_fn(vals_a, vals_b)
            spearman_matrix[i, j] = rho

    upper = spearman_matrix[np.triu_indices(n_exp_llms, k=1)]
    print(f"{col}:")
    print(f"  Spearman mean={upper.mean():.4f}  median={np.median(upper):.4f}  min={upper.min():.4f}  max={upper.max():.4f}")

# %% [markdown]
# ### Mean absolute difference per score column

# %%
for col in score_cols:
    mad_matrix = np.zeros((n_exp_llms, n_exp_llms))
    for i, llm_a in enumerate(exposure_llm_names):
        for j, llm_b in enumerate(exposure_llm_names):
            if i == j:
                continue
            vals_a = exposure_by_llm[llm_a].loc[exposure_common_ads, col].values
            vals_b = exposure_by_llm[llm_b].loc[exposure_common_ads, col].values
            mad_matrix[i, j] = np.mean(np.abs(vals_a - vals_b))

    upper = mad_matrix[np.triu_indices(n_exp_llms, k=1)]
    # Also show the overall range of this score for context
    all_vals = np.concatenate([
        exposure_by_llm[llm].loc[exposure_common_ads, col].values
        for llm in exposure_llm_names
    ])
    score_range = all_vals.max() - all_vals.min()
    score_mean = all_vals.mean()
    print(f"{col}:")
    print(f"  MAD  mean={upper.mean():.4f}  median={np.median(upper):.4f}  max={upper.max():.4f}")
    print(f"  Score range: [{all_vals.min():.3f}, {all_vals.max():.3f}]  mean={score_mean:.3f}  "
          f"MAD as % of range: {upper.mean()/score_range*100:.1f}%")

# %% [markdown]
# ### Full Pearson matrix for key scores
#
# Show the full pairwise matrix for the two most important scores:
# `felten_score` and `task_exposure_importance_weighted`.

# %%
for col in ["felten_score", "task_exposure_importance_weighted"]:
    pearson_matrix = np.zeros((n_exp_llms, n_exp_llms))
    for i, llm_a in enumerate(exposure_llm_names):
        for j, llm_b in enumerate(exposure_llm_names):
            if i == j:
                pearson_matrix[i, j] = 1.0
                continue
            vals_a = exposure_by_llm[llm_a].loc[exposure_common_ads, col].values
            vals_b = exposure_by_llm[llm_b].loc[exposure_common_ads, col].values
            r, _ = pearsonr(vals_a, vals_b)
            pearson_matrix[i, j] = r

    df = pd.DataFrame(pearson_matrix, index=exposure_llm_names, columns=exposure_llm_names)
    print(f"Pearson correlation matrix: {col}")
    print(df.round(4).to_string())
    print()

# %% [markdown]
# ### Best model subsets by exposure Pearson correlation
#
# Find model subsets with highest mean pairwise Pearson on
# `task_exposure_importance_weighted` (the primary outcome score).

# %%
# Build Pearson matrix for the primary score
primary_col = "task_exposure_importance_weighted"
exp_pearson = np.zeros((n_exp_llms, n_exp_llms))
for i, llm_a in enumerate(exposure_llm_names):
    for j, llm_b in enumerate(exposure_llm_names):
        if i == j:
            exp_pearson[i, j] = 1.0
            continue
        vals_a = exposure_by_llm[llm_a].loc[exposure_common_ads, primary_col].values
        vals_b = exposure_by_llm[llm_b].loc[exposure_common_ads, primary_col].values
        r, _ = pearsonr(vals_a, vals_b)
        exp_pearson[i, j] = r

print(f"Best subsets by mean pairwise Pearson on {primary_col}:")
print(f"{'k':>3s}  {'Pearson':>8s}  Best subset")
print("-" * 80)
for k in range(n_exp_llms, 1, -1):
    best_score = -1
    best_subset = None
    for subset in combinations(range(n_exp_llms), k):
        pairs = list(combinations(subset, 2))
        score = np.mean([exp_pearson[i, j] for i, j in pairs])
        if score > best_score:
            best_score = score
            best_subset = subset
    names = [exposure_llm_names[i] for i in best_subset]
    if len(names) > 6:
        removed = sorted(set(exposure_llm_names) - set(names))
        display = f"all except: {', '.join(removed)}" if removed else "all"
    else:
        display = ", ".join(names)
    print(f"{k:>3d}  {best_score:>8.4f}  {display}")

# %% [markdown]
# ## 5. Summary: Agreement Across Pipeline Stages

# %%
print("=" * 70)
print("SUMMARY: Mean pairwise agreement across pipeline stages")
print("=" * 70)
print()

# LLM Filter
print("LLM Filter (Jaccard of kept sets):")
print(f"  Mean pairwise Jaccard:       {upper_tri.mean():.3f}")
print(f"  Mean top-1 agreement:        {upper_tri_top1.mean():.3f}")
print(f"  Mean 'any overlap' fraction: {upper_tri_overlap.mean():.3f}")
print()

# Rerank
print("After Reranking:")
upper_shared = shared_frac_matrix[np.triu_indices(n_rerank_llms, k=1)]
print(f"  Mean pairwise Jaccard:       {upper_shared.mean():.3f} (unweighted, same as filter)")
print(f"  Mean weighted Jaccard:       {upper_tri_wj.mean():.3f} (rerank-score weighted)")
print(f"  Mean top-1 agreement:        {upper_tri_rerank.mean():.3f}")
print(f"  Mean Spearman on shared:     {valid.mean():.3f}")
print()

# Interpretation
print("Interpretation:")
print("  - Cosine candidates are identical (same embedding model across all runs).")
print("  - LLM filter is where model disagreement enters the pipeline.")
print("  - Reranking (same reranker) can either amplify or dampen LLM disagreement")
print("    depending on whether it reorders the shared candidates consistently.")
