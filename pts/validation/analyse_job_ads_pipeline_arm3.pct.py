# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Job Ads Pipeline Validation: Arm 3 (Reranker Sensitivity)
#
# Fix the embedding model (node: `embed_ads`) and LLM (node: `llm_filter_candidates`),
# vary the reranker model (node: `rerank_candidates`).
#
# The cosine candidates and LLM filter output are identical across all runs in each
# group (same embedding + same LLM). The reranker is the only source of variation:
# it receives the same filtered candidate set but assigns different scores, potentially
# changing which occupation is ranked first and how the final exposure scores are
# weighted.

# %% [markdown]
# ## Setup

# %%
import os
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

run_def = os.environ.get("VALIDATION_RUN_DEF", "validation_5k")

with open(validation_config_path, "rb") as f:
    config = tomllib.load(f)

completed = discover_completed_runs(run_def)
fixed_llms = config["fixed_llms"]
fixed_embeddings = config["fixed_embeddings"]
all_rerankers = sorted(config["rerank_models"])

mn = build_model_name_lookup()

print(f"Run definition: {run_def}")
print(f"Completed runs: {len(completed)}")

# %% [markdown]
# ### Models used

# %%
print("Rerankers (varied):")
display(build_model_info_table(all_rerankers, mn))
print("\nLLMs (fixed):")
display(build_model_info_table(fixed_llms, mn))
print("\nEmbeddings (fixed):")
display(build_model_info_table(fixed_embeddings, mn))

# %% [markdown]
# ## Load data
#
# For each (LLM, embedding) pair, collect all reranker runs. The cosine candidates
# and LLM filter output are shared across all rerankers in each group, so we only
# need to load the rerank and exposure outputs.

# %%
arm3_results = {}

for fixed_llm in fixed_llms:
    for fixed_embed in fixed_embeddings:
        arm_runs = [
            (rn, rerank) for rn, rd, llm, embed, rerank in completed
            if llm == fixed_llm and embed == fixed_embed
        ]

        if len(arm_runs) < 2:
            continue

        rerank_names = [mn.get(rerank, rerank) for _, rerank in arm_runs]

        # Rerank outputs: per-ad score dicts and top-1
        rerank_scores = {}
        rerank_top1 = {}
        for rn, rerank in arm_runs:
            name = mn.get(rerank, rerank)
            df = load_parquet(rn, "rerank_candidates", "reranked_matches.parquet")
            by_ad_scores = {}
            by_ad_top1 = {}
            for ad_id, group in df.groupby("ad_id"):
                by_ad_scores[ad_id] = dict(zip(group["onet_code"], group["rerank_score"]))
                by_ad_top1[ad_id] = group.sort_values("rerank_score", ascending=False).iloc[0]["onet_code"]
            rerank_scores[name] = by_ad_scores
            rerank_top1[name] = by_ad_top1

        # Common ads
        common_ads = sorted(set.intersection(*[set(s.keys()) for s in rerank_scores.values()]))

        key = (fixed_llm, fixed_embed)
        arm3_results[key] = {
            "rerank_names": rerank_names,
            "arm_runs": arm_runs,
            "common_ads": common_ads,
            "rerank_scores": rerank_scores,
            "rerank_top1": rerank_top1,
        }

print(f"Arm 3: {len(arm3_results)} (LLM, embedding) groups loaded")
for (llm, embed), r in arm3_results.items():
    print(f"  {mn.get(llm, llm)} + {mn.get(embed, embed)}: {len(r['rerank_names'])} rerankers, {len(r['common_ads'])} common ads")

# %% [markdown]
# ## Rerank stage
#
# Since the candidate sets are identical (same embedding + same LLM filter), the
# only question is how different rerankers *score* those candidates. Two rerankers
# might assign very different raw scores but still agree on which candidate is best.
#
# ### Top-1 agreement
#
# The fraction of ads where two rerankers assign the highest score to the same
# O\*NET occupation. This is the most direct measure of reranker agreement.

# %%
top1_rows = []
for (fixed_llm, fixed_embed), r in arm3_results.items():
    matrix = build_pairwise_matrix(
        r["rerank_names"], r["rerank_top1"], r["common_ads"], pairwise_top1,
    )
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    top1_rows.append(stats)

arm3_top1_summary = pd.DataFrame(top1_rows).set_index(["llm", "embedding"])

# %%
arm3_top1_summary

# %%
for (fixed_llm, fixed_embed), r in arm3_results.items():
    matrix = build_pairwise_matrix(
        r["rerank_names"], r["rerank_top1"], r["common_ads"], pairwise_top1,
    )
    display(IPyMarkdown(f"**Top-1 agreement ({mn.get(fixed_llm, fixed_llm)} + {mn.get(fixed_embed, fixed_embed)})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ### Spearman rank correlation
#
# For each ad, do two rerankers rank the candidates in the same order? Since all
# rerankers see the same candidate set (unlike Arms 1 and 2 where the sets differ),
# Spearman is computed over the full candidate list, not just shared candidates.

# %%
spearman_rows = []
for (fixed_llm, fixed_embed), r in arm3_results.items():
    matrix = build_pairwise_matrix(
        r["rerank_names"], r["rerank_scores"], r["common_ads"], pairwise_spearman,
    )
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    spearman_rows.append(stats)

arm3_spearman_summary = pd.DataFrame(spearman_rows).set_index(["llm", "embedding"])

# %%
arm3_spearman_summary

# %% [markdown]
# ### Weighted Jaccard (Ruzicka similarity)
#
# Even though the candidate sets are identical, rerankers assign different scores.
# Ruzicka similarity measures how much the score distributions agree: for each
# candidate, take min/max of the two scores. High Ruzicka means the rerankers
# not only agree on rankings but also on the magnitude of scores.

# %%
wj_rows = []
for (fixed_llm, fixed_embed), r in arm3_results.items():
    matrix = build_pairwise_matrix(
        r["rerank_names"], r["rerank_scores"], r["common_ads"], pairwise_weighted_jaccard,
    )
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["embedding"] = mn.get(fixed_embed, fixed_embed)
    wj_rows.append(stats)

arm3_wj_summary = pd.DataFrame(wj_rows).set_index(["llm", "embedding"])

# %%
arm3_wj_summary

# %%
for (fixed_llm, fixed_embed), r in arm3_results.items():
    matrix = build_pairwise_matrix(
        r["rerank_names"], r["rerank_scores"], r["common_ads"], pairwise_weighted_jaccard,
    )
    display(IPyMarkdown(f"**Weighted Jaccard ({mn.get(fixed_llm, fixed_llm)} + {mn.get(fixed_embed, fixed_embed)})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ## Exposure scores
#
# The final per-ad exposure scores (node: `compute_job_ad_exposure`). Since the
# candidate sets are identical, all variation comes from how the reranker weights
# the candidates. Rerankers that agree on top-1 but disagree on score magnitudes
# may still produce different weighted exposure averages.

# %%
score_cols = [
    "felten_score", "presence_physical", "presence_emotional",
    "presence_creative", "presence_composite",
    "task_exposure_mean", "task_exposure_importance_weighted",
]

for (fixed_llm, fixed_embed), r in arm3_results.items():
    exposure_by_rerank = {}
    for rn, rerank in r["arm_runs"]:
        name = mn.get(rerank, rerank)
        df = load_parquet(rn, "compute_job_ad_exposure", "ad_exposure.parquet")
        exposure_by_rerank[name] = df.dropna(subset=score_cols).set_index("ad_id")

    common_ads = sorted(set.intersection(*[set(df.index) for df in exposure_by_rerank.values()]))
    r["exposure_by_rerank"] = exposure_by_rerank
    r["exposure_common_ads"] = common_ads

print("Exposure data loaded.")
for (llm, embed), r in arm3_results.items():
    print(f"  {mn.get(llm, llm)} + {mn.get(embed, embed)}: {len(r['exposure_common_ads'])} common ads")

# %% [markdown]
# ### Pearson correlation per score column

# %%
pearson_rows = []
for (fixed_llm, fixed_embed), r in arm3_results.items():
    common = r["exposure_common_ads"]
    for col in score_cols:
        vectors = {rr: r["exposure_by_rerank"][rr].loc[common, col].values for rr in r["rerank_names"]}
        matrix = pairwise_correlation_matrix(r["rerank_names"], vectors, method="pearson")
        stats = upper_tri_stats(matrix.values)
        stats["llm"] = mn.get(fixed_llm, fixed_llm)
        stats["embedding"] = mn.get(fixed_embed, fixed_embed)
        stats["score"] = col
        pearson_rows.append(stats)

arm3_pearson = pd.DataFrame(pearson_rows)
arm3_pearson_pivot = arm3_pearson.pivot(index="score", columns=["llm", "embedding"], values="mean")

# %%
arm3_pearson_pivot

# %% [markdown]
# ### MAD as % of range

# %%
mad_rows = []
for (fixed_llm, fixed_embed), r in arm3_results.items():
    common = r["exposure_common_ads"]
    for col in score_cols:
        all_vals = np.concatenate([r["exposure_by_rerank"][rr].loc[common, col].values for rr in r["rerank_names"]])
        score_range = all_vals.max() - all_vals.min()

        mads = []
        for i, rr_a in enumerate(r["rerank_names"]):
            for j, rr_b in enumerate(r["rerank_names"]):
                if i >= j:
                    continue
                va = r["exposure_by_rerank"][rr_a].loc[common, col].values
                vb = r["exposure_by_rerank"][rr_b].loc[common, col].values
                mads.append(np.mean(np.abs(va - vb)))

        mad_rows.append({
            "llm": mn.get(fixed_llm, fixed_llm), "embedding": mn.get(fixed_embed, fixed_embed), "score": col,
            "mad_pct_range": np.mean(mads) / score_range * 100 if score_range > 0 else 0,
        })

arm3_mad = pd.DataFrame(mad_rows)

# %%
arm3_mad.pivot(index="score", columns=["llm", "embedding"], values="mad_pct_range").style.format("{:.1f}%")

# %% [markdown]
# ### Pearson matrix for primary score

# %%
for (fixed_llm, fixed_embed), r in arm3_results.items():
    common = r["exposure_common_ads"]
    vectors = {rr: r["exposure_by_rerank"][rr].loc[common, "task_exposure_importance_weighted"].values
               for rr in r["rerank_names"]}
    matrix = pairwise_correlation_matrix(r["rerank_names"], vectors, method="pearson")
    display(IPyMarkdown(f"**Pearson: task_exposure_importance_weighted ({mn.get(fixed_llm, fixed_llm)} + {mn.get(fixed_embed, fixed_embed)})**"))
    display(matrix.style.format("{:.4f}").background_gradient(cmap="YlOrRd", vmin=0.9, vmax=1))

# %% [markdown]
# ## Summary
#
# Reranker sensitivity is qualitatively different from LLM and embedding sensitivity
# because the candidate set is held constant. All variation comes from how the
# reranker scores and orders the same candidates. Key questions:
#
# 1. **Do rerankers agree on which candidate is best?** (top-1 agreement)
#    If yes, the choice of reranker mainly affects score magnitudes, not the
#    primary occupation assignment.
#
# 2. **Do they rank candidates in the same order?** (Spearman)
#    High Spearman means the rerankers have a consistent quality signal,
#    even if their raw score scales differ.
#
# 3. **How much does reranker choice affect final exposure?** (Pearson, MAD)
#    Since exposure is a rerank-score-weighted average, score magnitude
#    differences can propagate even when rankings agree.
