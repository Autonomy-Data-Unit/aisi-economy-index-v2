# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Job Ads Pipeline Validation: Arm 1 (LLM Sensitivity)
#
# Fix the embedding model (node: `embed_ads`) and reranker (node: `rerank_candidates`),
# vary the LLM used in the filter step (node: `llm_filter_candidates`).
#
# The cosine candidates (node: `cosine_candidates`) are identical across all runs
# with the same embedding, so all disagreement enters at the LLM filter stage.
# We measure agreement at the filter stage, then track how it propagates through
# reranking and into the final exposure scores.

# %% [markdown]
# ## Setup

# %%
import tomllib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown as IPyMarkdown

from ai_index.const import validation_config_path, pipeline_store_path
from validation.utils import (
    discover_completed_runs,
    load_parquet,
    build_model_name_lookup,
    build_model_info_table,
    pairwise_jaccard,
    pairwise_top1,
    pairwise_weighted_jaccard,
    pairwise_spearman,
    build_pairwise_matrix,
    pairwise_correlation_matrix,
    upper_tri_stats,
    best_subsets,
)

import os
run_def = os.environ.get("VALIDATION_RUN_DEF", "validation_5k")

with open(validation_config_path, "rb") as f:
    config = tomllib.load(f)

completed = discover_completed_runs(run_def)
fixed_embeddings = config["fixed_embeddings"]
fixed_rerankers = config["fixed_rerankers"]

# Model name lookup: config key -> short display name (e.g. "qwen-7b-sbatch" -> "Qwen2.5-7B-Instruct")
mn = build_model_name_lookup()

print(f"Run definition: {run_def}")
print(f"Completed runs: {len(completed)}")

# %% [markdown]
# ### Models used

# %%
all_llms = sorted(config["llm_models"])

print("LLMs (varied):")
display(build_model_info_table(all_llms, mn))
print("\nEmbeddings (fixed):")
display(build_model_info_table(fixed_embeddings, mn))
print("\nRerankers (fixed):")
display(build_model_info_table(fixed_rerankers, mn))

# %% [markdown]
# ## Load data
#
# For each (embedding, reranker) pair, collect all LLM runs and load their
# filter outputs: the set of kept candidates per ad, match counts, and top-1.

# %%
arm1_results = {}

for fixed_embed in fixed_embeddings:
    for fixed_rerank in fixed_rerankers:
        # Collect runs for this arm: same embedding + reranker, vary LLM
        arm_runs = [
            (rn, llm) for rn, rd, llm, embed, rerank in completed
            if embed == fixed_embed and rerank == fixed_rerank
        ]

        if len(arm_runs) < 2:
            continue

        llm_names = [mn.get(llm, llm) for _, llm in arm_runs]

        # Load filtered matches: per-ad kept sets and per-ad match counts
        filter_sets = {}
        filter_counts = {}
        for rn, llm in arm_runs:
            name = mn.get(llm, llm)
            df = load_parquet(rn, "llm_filter_candidates", "filtered_matches.parquet")
            filter_sets[name] = df.groupby("ad_id")["onet_code"].apply(set).to_dict()
            filter_counts[name] = df.groupby("ad_id").size()

        # Common ads across all runs in this arm
        common_ads = sorted(set.intersection(*[set(s.keys()) for s in filter_sets.values()]))

        # Top-1 kept candidate per ad (highest cosine-ranked that survived the filter)
        filter_top1 = {}
        for rn, llm in arm_runs:
            name = mn.get(llm, llm)
            df = load_parquet(rn, "llm_filter_candidates", "filtered_matches.parquet")
            filter_top1[name] = df.sort_values(["ad_id", "rank"]).groupby("ad_id").first()["onet_code"].to_dict()

        key = (fixed_embed, fixed_rerank)
        arm1_results[key] = {
            "llm_names": llm_names,
            "common_ads": common_ads,
            "filter_sets": filter_sets,
            "filter_counts": filter_counts,
            "filter_top1": filter_top1,
        }

print(f"Arm 1: {len(arm1_results)} (embedding, reranker) groups loaded")
for (embed, rerank), r in arm1_results.items():
    print(f"  {mn.get(embed, embed)} + {mn.get(rerank, rerank)}: {len(r['llm_names'])} LLMs, {len(r['common_ads'])} common ads")

# %% [markdown]
# ## Filter stage
#
# ### Filter selectivity
#
# Each LLM receives 20 cosine candidates per ad and decides which to keep.
# This table shows how many candidates each LLM retains on average. A selective
# model (mean ~3) keeps only the strongest matches; a permissive model (mean ~8)
# keeps most candidates. Selectivity directly affects Jaccard: if one model keeps
# 3 candidates and another keeps 8, even with full overlap on the 3 the Jaccard
# is only 3/8 = 0.375.

# %%
selectivity_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    for llm in r["llm_names"]:
        c = r["filter_counts"][llm]
        selectivity_rows.append({
            "embedding": mn.get(fixed_embed, fixed_embed), "reranker": mn.get(fixed_rerank, fixed_rerank), "llm": llm,
            "mean_candidates": round(c.mean(), 1),
            "median_candidates": int(c.median()),
        })
arm1_selectivity = pd.DataFrame(selectivity_rows)

# %%
arm1_selectivity["group"] = arm1_selectivity["embedding"] + " + " + arm1_selectivity["reranker"]
arm1_selectivity = arm1_selectivity.pivot(index="llm", columns="group", values=["mean_candidates", "median_candidates"])
arm1_selectivity = arm1_selectivity.swaplevel(axis=1).sort_index(axis=1)
arm1_selectivity

# %%
groups = sorted(arm1_selectivity.columns.get_level_values(0).unique())
n_groups = len(groups)

fig, ax = plt.subplots(figsize=(max(10, n_groups * 3), 5))
llm_order = arm1_selectivity.index.tolist()
x = np.arange(len(llm_order))
width = 0.8 / n_groups

for i, group in enumerate(groups):
    vals = arm1_selectivity[(group, "mean_candidates")].values
    ax.bar(x + i * width - (n_groups - 1) * width / 2, vals, width, label=group)

ax.set_xticks(x)
ax.set_xticklabels(llm_order, rotation=45, ha="right")
ax.set_ylabel("Mean candidates per ad")
ax.set_title("Arm 1: LLM filter selectivity by (embedding, reranker)")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ### Pairwise Jaccard (LLM filter kept sets)
#
# For each ad, two LLMs each produce a set of kept O\*NET candidates. The
# Jaccard similarity for that ad is $|A \cap B| \,/\, |A \cup B|$. We average
# this across all common ads to get the pairwise Jaccard between two LLMs.
#
# A Jaccard of 0.5 means that on a typical ad, the two models share half their
# candidates. A Jaccard of 1.0 means they keep exactly the same set.
# Note that Jaccard is sensitive to set size differences (see selectivity above).

# %%
arm1_jaccard = {}

for (fixed_embed, fixed_rerank), r in arm1_results.items():
    arm1_jaccard[(fixed_embed, fixed_rerank)] = build_pairwise_matrix(
        r["llm_names"], r["filter_sets"], r["common_ads"], pairwise_jaccard,
    )

jaccard_summary_rows = []
for (fixed_embed, fixed_rerank), matrix in arm1_jaccard.items():
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    jaccard_summary_rows.append(stats)
arm1_jaccard_summary = pd.DataFrame(jaccard_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_jaccard_summary

# %%
for (fixed_embed, fixed_rerank), matrix in arm1_jaccard.items():
    display(IPyMarkdown(f"**Jaccard matrix ({mn.get(fixed_embed, fixed_embed)} + {mn.get(fixed_rerank, fixed_rerank)})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ### Best subsets by Jaccard
#
# For each subset size k (from all models down to 2), find the subset of
# k LLMs with the highest mean pairwise Jaccard. This searches all possible
# combinations exhaustively.
#
# Reading the table from top to bottom shows which models get removed first
# as k shrinks. If a large subset (say k=8 or k=9) already achieves high
# agreement, that consensus is evidence of a robust matching. The models
# removed earliest are the biggest outliers: their filter behaviour diverges
# most from the group.

# %%
subset_rows = []
for (fixed_embed, fixed_rerank), matrix in arm1_jaccard.items():
    llm_names = list(matrix.index)
    for k, score, names in best_subsets(llm_names, matrix.values):
        if len(names) > 6:
            removed = sorted(set(llm_names) - set(names))
            label = f"all except: {', '.join(removed)}" if removed else "all"
        else:
            label = ", ".join(names)
        subset_rows.append({"embedding": mn.get(fixed_embed, fixed_embed), "reranker": mn.get(fixed_rerank, fixed_rerank), "k": k, "jaccard": round(score, 3), "subset": label})
arm1_best_subsets = pd.DataFrame(subset_rows)

# %%
arm1_best_subsets

# %% [markdown]
# ### Top-1 agreement (filter stage)
#
# For each ad, we take the highest-cosine-ranked candidate that survived the
# LLM filter (i.e. the candidate with rank 0 in the filtered output). Top-1
# agreement is the fraction of ads where two LLMs agree on this best candidate.
#
# This is stricter than Jaccard: two models can have high Jaccard (overlapping
# sets) but still disagree on which candidate ranks first. Top-1 agreement
# directly measures whether the models agree on the most likely O\*NET match.

# %%
top1_summary_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    matrix = build_pairwise_matrix(
        r["llm_names"], r["filter_top1"], r["common_ads"], pairwise_top1,
    )
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    top1_summary_rows.append(stats)
arm1_top1_summary = pd.DataFrame(top1_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_top1_summary

# %% [markdown]
# ## Rerank stage
#
# Each (embedding, reranker) group uses the same reranker model, but it receives
# different candidate sets from the different LLM filters. The reranker assigns
# a cross-encoder score to each (ad, candidate) pair. We measure agreement on
# the reranker's output using three metrics.

# %%
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    rerank_scores = {}
    rerank_top1 = {}
    for rn, llm in [(rn, llm) for rn, rd, llm, embed, rerank in completed
                     if embed == fixed_embed and rerank == fixed_rerank]:
        name = mn.get(llm, llm)
        df = load_parquet(rn, "rerank_candidates", "reranked_matches.parquet")
        by_ad_scores = {}
        by_ad_top1 = {}
        for ad_id, group in df.groupby("ad_id"):
            by_ad_scores[ad_id] = dict(zip(group["onet_code"], group["rerank_score"]))
            by_ad_top1[ad_id] = group.sort_values("rerank_score", ascending=False).iloc[0]["onet_code"]
        rerank_scores[name] = by_ad_scores
        rerank_top1[name] = by_ad_top1

    r["rerank_scores"] = rerank_scores
    r["rerank_top1"] = rerank_top1

print("Rerank data loaded.")

# %% [markdown]
# ### Weighted Jaccard (Ruzicka similarity)
#
# Standard Jaccard treats all candidates equally: a low-scoring candidate that
# only one model kept counts the same as a high-scoring one both models agree on.
# Ruzicka similarity (weighted Jaccard) uses rerank scores as weights:
#
# For each ad with candidate set $C = C_A \cup C_B$:
#
# $$\text{Ruzicka} = \frac{\sum_{c \in C} \min(s_A(c),\, s_B(c))}{\sum_{c \in C} \max(s_A(c),\, s_B(c))}$$
#
# where $s_A(c)$ is the rerank score for candidate $c$ in run A (0 if absent).
# Candidates with high rerank scores in both runs contribute heavily; candidates
# with low scores or present in only one run contribute little.
#
# A large gap between unweighted Jaccard and Ruzicka similarity means most of the
# disagreement between LLM filters is on low-scoring candidates that the reranker
# considers poor matches.

# %%
arm1_wj = {}

for (fixed_embed, fixed_rerank), r in arm1_results.items():
    arm1_wj[(fixed_embed, fixed_rerank)] = build_pairwise_matrix(
        r["llm_names"], r["rerank_scores"], r["common_ads"], pairwise_weighted_jaccard,
    )

wj_summary_rows = []
for (fixed_embed, fixed_rerank), matrix in arm1_wj.items():
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    wj_summary_rows.append(stats)
arm1_wj_summary = pd.DataFrame(wj_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_wj_summary

# %%
for (fixed_embed, fixed_rerank), matrix in arm1_wj.items():
    display(IPyMarkdown(f"**Weighted Jaccard matrix ({mn.get(fixed_embed, fixed_embed)} + {mn.get(fixed_rerank, fixed_rerank)})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ### Top-1 agreement after reranking
#
# Same concept as filter-stage top-1, but now using the reranker's best-scoring
# candidate (highest `rerank_score`) rather than the cosine-ranked order.
#
# If top-1 agreement is higher here than at the filter stage, the reranker is
# acting as a consensus mechanism: even when two LLMs pass different candidate
# sets, the reranker often picks the same best match from whatever it receives.

# %%
rerank_top1_summary_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    matrix = build_pairwise_matrix(
        r["llm_names"], r["rerank_top1"], r["common_ads"], pairwise_top1,
    )
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    rerank_top1_summary_rows.append(stats)
arm1_rerank_top1_summary = pd.DataFrame(rerank_top1_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_rerank_top1_summary

# %% [markdown]
# ### Spearman rank correlation on shared candidates
#
# For candidates that both LLM runs kept, does the reranker assign them the
# same relative ordering? Spearman rank correlation measures whether the ranks
# (not the raw scores) agree. Only computed for ads where at least 3 candidates
# are shared (needed for a meaningful correlation).
#
# A Spearman of 1.0 means the reranker produces a fully deterministic ordering
# on its inputs, regardless of which other candidates are in the set. In that
# case, all remaining disagreement in top-1 comes purely from candidates that
# one LLM kept and the other dropped.

# %%
spearman_summary_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    matrix = build_pairwise_matrix(
        r["llm_names"], r["rerank_scores"], r["common_ads"], pairwise_spearman,
    )
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    spearman_summary_rows.append(stats)
arm1_spearman_summary = pd.DataFrame(spearman_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_spearman_summary

# %% [markdown]
# ## Exposure scores
#
# The final pipeline output: per-ad exposure scores (node: `compute_job_ad_exposure`).
# Each ad's exposure is a rerank-score-weighted average of the O\*NET occupation-level
# scores for its matched candidates. Since the occupation-level scores are identical
# across all runs (they come from the scoring nodes which don't depend on LLM choice),
# all variation comes from which candidates the LLM filter kept and how the reranker
# weighted them.
#
# This is the bottom line: even if filter-stage Jaccard is moderate, do the final
# exposure scores still agree?

# %%
score_cols = [
    "felten_score", "presence_physical", "presence_emotional",
    "presence_creative", "presence_composite",
    "task_exposure_mean", "task_exposure_importance_weighted",
]

for (fixed_embed, fixed_rerank), r in arm1_results.items():
    exposure_by_llm = {}
    for rn, llm in [(rn, llm) for rn, rd, llm, embed, rerank in completed
                     if embed == fixed_embed and rerank == fixed_rerank]:
        name = mn.get(llm, llm)
        df = load_parquet(rn, "compute_job_ad_exposure", "ad_exposure.parquet")
        exposure_by_llm[name] = df.dropna(subset=score_cols).set_index("ad_id")

    # Common ads with valid scores across all LLMs
    common_ads = sorted(set.intersection(*[set(df.index) for df in exposure_by_llm.values()]))
    r["exposure_by_llm"] = exposure_by_llm
    r["exposure_common_ads"] = common_ads

print("Exposure data loaded.")
for (embed, rerank), r in arm1_results.items():
    print(f"  {embed} + {rerank}: {len(r['exposure_common_ads'])} common ads with valid scores")

# %% [markdown]
# ### Pearson correlation per score column
#
# Pearson correlation measures linear agreement between two vectors of per-ad
# scores. For each LLM pair, we align their score vectors on common ads and
# compute Pearson $r$. The table shows the mean across all LLM pairs.
#
# $r = 1.0$ means perfect agreement; $r = 0.95$ means 95% of the variance
# is shared. Values above 0.95 indicate that the LLM choice has minimal effect
# on the final exposure scores.

# %%
pearson_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    common = r["exposure_common_ads"]
    for col in score_cols:
        vectors = {llm: r["exposure_by_llm"][llm].loc[common, col].values for llm in r["llm_names"]}
        matrix = pairwise_correlation_matrix(r["llm_names"], vectors, method="pearson")
        stats = upper_tri_stats(matrix.values)
        stats["embedding"] = fixed_embed
        stats["reranker"] = fixed_rerank
        stats["score"] = col
        pearson_rows.append(stats)

arm1_pearson = pd.DataFrame(pearson_rows)
arm1_pearson_pivot = arm1_pearson.pivot(index="score", columns=["embedding", "reranker"], values="mean")

# %%
arm1_pearson_pivot

# %% [markdown]
# ### Mean absolute difference (MAD) per score column
#
# While Pearson measures correlation (do the scores move together?), MAD measures
# the actual magnitude of disagreement. For each LLM pair, we compute
# $\text{MAD} = \frac{1}{N}\sum_i |s_A(i) - s_B(i)|$ across common ads,
# then average across all pairs.
#
# MAD as a percentage of the score range gives an intuitive sense of scale: 2%
# means the typical per-ad disagreement between two LLM runs is 2% of the full
# range of that score.

# %%
mad_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    common = r["exposure_common_ads"]
    for col in score_cols:
        all_vals = np.concatenate([r["exposure_by_llm"][llm].loc[common, col].values for llm in r["llm_names"]])
        score_range = all_vals.max() - all_vals.min()

        # Mean pairwise MAD
        mads = []
        for i, llm_a in enumerate(r["llm_names"]):
            for j, llm_b in enumerate(r["llm_names"]):
                if i >= j:
                    continue
                va = r["exposure_by_llm"][llm_a].loc[common, col].values
                vb = r["exposure_by_llm"][llm_b].loc[common, col].values
                mads.append(np.mean(np.abs(va - vb)))

        mad_rows.append({
            "embedding": mn.get(fixed_embed, fixed_embed), "reranker": mn.get(fixed_rerank, fixed_rerank), "score": col,
            "mad_mean": np.mean(mads),
            "mad_max": np.max(mads),
            "score_range": score_range,
            "mad_pct_range": np.mean(mads) / score_range * 100 if score_range > 0 else 0,
        })

arm1_mad = pd.DataFrame(mad_rows)

# %%
arm1_mad.pivot(index="score", columns=["embedding", "reranker"], values="mad_pct_range").style.format("{:.1f}%")

# %% [markdown]
# ### Pearson matrix for primary score
#
# Full pairwise Pearson correlation matrix for `task_exposure_importance_weighted`,
# the primary outcome score. Each cell shows how well two LLMs agree on the
# final per-ad exposure values. The colour scale starts at 0.9 to highlight
# differences within the high-correlation range.

# %%
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    common = r["exposure_common_ads"]
    vectors = {llm: r["exposure_by_llm"][llm].loc[common, "task_exposure_importance_weighted"].values
               for llm in r["llm_names"]}
    matrix = pairwise_correlation_matrix(r["llm_names"], vectors, method="pearson")
    display(IPyMarkdown(f"**Pearson: task_exposure_importance_weighted ({mn.get(fixed_embed, fixed_embed)} + {mn.get(fixed_rerank, fixed_rerank)})**"))
    display(matrix.style.format("{:.4f}").background_gradient(cmap="YlOrRd", vmin=0.9, vmax=1))

# %% [markdown]
# ## Summary
#
# The analysis tracks disagreement through three pipeline stages:
#
# 1. **Filter stage**: Jaccard ~0.45, top-1 ~0.73. LLMs disagree on about half
#    the candidates for a typical ad, but agree on the top candidate ~73% of
#    the time.
#
# 2. **Rerank stage**: Weighted Jaccard ~0.75, top-1 ~0.83, Spearman ~1.0.
#    The reranker dampens disagreement: most of what the LLMs disagree on turns
#    out to be low-scoring candidates. The reranker orders shared candidates
#    identically (Spearman ~1.0), so remaining top-1 disagreement comes entirely
#    from candidates one LLM kept and the other dropped.
#
# 3. **Exposure scores**: Pearson ~0.96, MAD ~2% of range. The final per-ad
#    exposure scores are highly robust to LLM choice. Even the worst LLM pair
#    achieves Pearson >0.90.
#
# The key insight is that disagreement is concentrated on low-importance
# candidates that barely affect the final weighted scores.
