# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Arm 1: LLM Sensitivity
#
# Fix the embedding model (node: `embed_ads`) and reranker (node: `rerank_candidates`),
# vary the LLM used in the filter step (node: `llm_filter_candidates`).
#
# The cosine candidates (node: `cosine_candidates`) are identical across all runs
# with the same embedding, so all disagreement enters at the LLM filter stage.
# We measure agreement at the filter stage, then track how it propagates through
# reranking.

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
    pairwise_jaccard,
    pairwise_top1,
    pairwise_weighted_jaccard,
    pairwise_spearman,
    build_pairwise_matrix,
    upper_tri_stats,
    best_subsets,
)

run_def = "validation_5k"

with open(validation_config_path, "rb") as f:
    config = tomllib.load(f)

completed = discover_completed_runs(run_def)
fixed_embeddings = config["fixed_embeddings"]
fixed_rerankers = config["fixed_rerankers"]

print(f"Run definition: {run_def}")
print(f"Completed runs: {len(completed)}")

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

        llm_names = [llm for _, llm in arm_runs]

        # Load filtered matches: per-ad kept sets and per-ad match counts
        filter_sets = {}
        filter_counts = {}
        for rn, llm in arm_runs:
            df = load_parquet(rn, "llm_filter_candidates", "filtered_matches.parquet")
            filter_sets[llm] = df.groupby("ad_id")["onet_code"].apply(set).to_dict()
            filter_counts[llm] = df.groupby("ad_id").size()

        # Common ads across all runs in this arm
        common_ads = sorted(set.intersection(*[set(s.keys()) for s in filter_sets.values()]))

        # Top-1 kept candidate per ad (highest cosine-ranked that survived the filter)
        filter_top1 = {}
        for rn, llm in arm_runs:
            df = load_parquet(rn, "llm_filter_candidates", "filtered_matches.parquet")
            filter_top1[llm] = df.sort_values(["ad_id", "rank"]).groupby("ad_id").first()["onet_code"].to_dict()

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
    print(f"  {embed} + {rerank}: {len(r['llm_names'])} LLMs, {len(r['common_ads'])} common ads")

# %% [markdown]
# ## Filter stage
#
# ### Filter selectivity
#
# How many candidates does each LLM keep per ad? More selective models (fewer
# candidates) tend to have lower Jaccard with permissive models, but may still
# agree on the top-ranked candidate.

# %%
selectivity_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    for llm in r["llm_names"]:
        c = r["filter_counts"][llm]
        selectivity_rows.append({
            "embedding": fixed_embed, "reranker": fixed_rerank, "llm": llm,
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
# For each pair of LLMs, the Jaccard similarity of the candidate sets they
# keep, averaged across all common ads. A Jaccard of 0.5 means that on a
# typical ad, the two models agree on half the candidates.

# %%
arm1_jaccard = {}

for (fixed_embed, fixed_rerank), r in arm1_results.items():
    arm1_jaccard[(fixed_embed, fixed_rerank)] = build_pairwise_matrix(
        r["llm_names"], r["filter_sets"], r["common_ads"], pairwise_jaccard,
    )

jaccard_summary_rows = []
for (fixed_embed, fixed_rerank), matrix in arm1_jaccard.items():
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = fixed_embed
    stats["reranker"] = fixed_rerank
    jaccard_summary_rows.append(stats)
arm1_jaccard_summary = pd.DataFrame(jaccard_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_jaccard_summary

# %%
for (fixed_embed, fixed_rerank), matrix in arm1_jaccard.items():
    display(IPyMarkdown(f"**Jaccard matrix ({fixed_embed} + {fixed_rerank})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ### Best subsets by Jaccard
#
# For each subset size k (from all 12 models down to 2), find the subset of
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
        subset_rows.append({"embedding": fixed_embed, "reranker": fixed_rerank, "k": k, "jaccard": round(score, 3), "subset": label})
arm1_best_subsets = pd.DataFrame(subset_rows)

# %%
arm1_best_subsets

# %% [markdown]
# ### Top-1 agreement (filter stage)
#
# For each pair of LLMs, the fraction of ads where both models keep the same
# highest-cosine-ranked candidate. This is a stricter measure than Jaccard:
# two models can have high Jaccard (overlapping sets) but still disagree on
# which candidate is best.

# %%
top1_summary_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    matrix = build_pairwise_matrix(
        r["llm_names"], r["filter_top1"], r["common_ads"], pairwise_top1,
    )
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = fixed_embed
    stats["reranker"] = fixed_rerank
    top1_summary_rows.append(stats)
arm1_top1_summary = pd.DataFrame(top1_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_top1_summary

# %% [markdown]
# ## Rerank stage
#
# Each (embedding, reranker) group uses the same reranker model, but it receives
# different candidate sets from the different LLM filters. We measure agreement
# on the reranker's output:
#
# - **Weighted Jaccard (Ruzicka similarity)**: like Jaccard, but weights each
#   candidate by its rerank score. High-scoring candidates that both models
#   agree on contribute more than low-scoring candidates only in one model.
# - **Top-1 agreement after reranking**: do both runs pick the same
#   best-scoring occupation after reranking?
# - **Spearman rank correlation**: for candidates shared by both runs, does
#   the reranker order them the same way?

# %%
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    rerank_scores = {}
    rerank_top1 = {}
    for rn, llm in [(rn, llm) for rn, rd, llm, embed, rerank in completed
                     if embed == fixed_embed and rerank == fixed_rerank]:
        df = load_parquet(rn, "rerank_candidates", "reranked_matches.parquet")
        by_ad_scores = {}
        by_ad_top1 = {}
        for ad_id, group in df.groupby("ad_id"):
            by_ad_scores[ad_id] = dict(zip(group["onet_code"], group["rerank_score"]))
            by_ad_top1[ad_id] = group.sort_values("rerank_score", ascending=False).iloc[0]["onet_code"]
        rerank_scores[llm] = by_ad_scores
        rerank_top1[llm] = by_ad_top1

    r["rerank_scores"] = rerank_scores
    r["rerank_top1"] = rerank_top1

print("Rerank data loaded.")

# %% [markdown]
# ### Weighted Jaccard (Ruzicka similarity)
#
# Standard Jaccard treats all candidates equally. Ruzicka similarity weights
# each candidate by its rerank score: for shared candidates, take the min of
# the two scores; for the union, take the max. This tells us whether models
# agree on the *important* candidates, even if they disagree on low-scoring ones.

# %%
arm1_wj = {}

for (fixed_embed, fixed_rerank), r in arm1_results.items():
    arm1_wj[(fixed_embed, fixed_rerank)] = build_pairwise_matrix(
        r["llm_names"], r["rerank_scores"], r["common_ads"], pairwise_weighted_jaccard,
    )

wj_summary_rows = []
for (fixed_embed, fixed_rerank), matrix in arm1_wj.items():
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = fixed_embed
    stats["reranker"] = fixed_rerank
    wj_summary_rows.append(stats)
arm1_wj_summary = pd.DataFrame(wj_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_wj_summary

# %%
for (fixed_embed, fixed_rerank), matrix in arm1_wj.items():
    display(IPyMarkdown(f"**Weighted Jaccard matrix ({fixed_embed} + {fixed_rerank})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ### Top-1 agreement after reranking
#
# After reranking, do different LLM runs agree on the best-scoring occupation?
# This should be higher than the filter-stage top-1 if the reranker is acting
# as a consensus mechanism.

# %%
rerank_top1_summary_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    matrix = build_pairwise_matrix(
        r["llm_names"], r["rerank_top1"], r["common_ads"], pairwise_top1,
    )
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = fixed_embed
    stats["reranker"] = fixed_rerank
    rerank_top1_summary_rows.append(stats)
arm1_rerank_top1_summary = pd.DataFrame(rerank_top1_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_rerank_top1_summary

# %% [markdown]
# ### Spearman rank correlation on shared candidates
#
# For candidates that both runs kept, does the reranker assign them the same
# relative ordering? A Spearman of 1.0 means the reranker is fully deterministic
# on its inputs: the only source of disagreement is which candidates made it
# through the filter.

# %%
spearman_summary_rows = []
for (fixed_embed, fixed_rerank), r in arm1_results.items():
    matrix = build_pairwise_matrix(
        r["llm_names"], r["rerank_scores"], r["common_ads"], pairwise_spearman,
    )
    stats = upper_tri_stats(matrix.values)
    stats["embedding"] = fixed_embed
    stats["reranker"] = fixed_rerank
    spearman_summary_rows.append(stats)
arm1_spearman_summary = pd.DataFrame(spearman_summary_rows).set_index(["embedding", "reranker"])

# %%
arm1_spearman_summary

# %% [markdown]
# ## Summary
#
# Comparing filter-stage and rerank-stage agreement shows how the reranker
# dampens or amplifies LLM disagreement. If weighted Jaccard is much higher
# than unweighted Jaccard, most disagreement is on low-scoring candidates.
# If top-1 agreement improves after reranking, the reranker is acting as a
# consensus mechanism.
