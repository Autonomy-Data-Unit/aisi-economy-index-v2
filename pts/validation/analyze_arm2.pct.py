# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Arm 2: Embedding Sensitivity
#
# Fix the LLM (node: `llm_filter_candidates`) and reranker (node: `rerank_candidates`),
# vary the embedding model (node: `embed_ads`, `embed_onet`).
#
# Unlike Arm 1, the cosine candidates (node: `cosine_candidates`) are now *different*
# across runs because different embeddings produce different similarity rankings.
# Disagreement enters at the very first retrieval stage, before the LLM filter
# even sees the candidates. We track how this propagates through the full pipeline.

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
fixed_rerankers = config["fixed_rerankers"]

mn = build_model_name_lookup()

print(f"Run definition: {run_def}")
print(f"Completed runs: {len(completed)}")

# %% [markdown]
# ### Models used

# %%
all_embeds = sorted(config["embed_models"])

print("Embeddings (varied):")
display(build_model_info_table(all_embeds, mn))
print("\nLLMs (fixed):")
display(build_model_info_table(fixed_llms, mn))
print("\nRerankers (fixed):")
display(build_model_info_table(fixed_rerankers, mn))

# %% [markdown]
# ## Load data
#
# For each (LLM, reranker) pair, collect all embedding runs and load their
# cosine candidates, filter outputs, rerank results, and exposure scores.

# %%
arm2_results = {}

for fixed_llm in fixed_llms:
    for fixed_rerank in fixed_rerankers:
        arm_runs = [
            (rn, embed) for rn, rd, llm, embed, rerank in completed
            if llm == fixed_llm and rerank == fixed_rerank
        ]

        if len(arm_runs) < 2:
            continue

        embed_names = [mn.get(embed, embed) for _, embed in arm_runs]

        # Cosine candidates: per-ad candidate sets (full top-20 and top-5)
        cosine_sets = {}
        cosine_sets_top5 = {}
        for rn, embed in arm_runs:
            name = mn.get(embed, embed)
            df = load_parquet(rn, "cosine_candidates", "candidates.parquet")
            cosine_sets[name] = df.groupby("ad_id")["onet_code"].apply(set).to_dict()
            top5 = df[df["rank"] < 5]
            cosine_sets_top5[name] = top5.groupby("ad_id")["onet_code"].apply(set).to_dict()

        # Filter outputs
        filter_sets = {}
        filter_counts = {}
        filter_top1 = {}
        for rn, embed in arm_runs:
            name = mn.get(embed, embed)
            df = load_parquet(rn, "llm_filter_candidates", "filtered_matches.parquet")
            filter_sets[name] = df.groupby("ad_id")["onet_code"].apply(set).to_dict()
            filter_counts[name] = df.groupby("ad_id").size()
            filter_top1[name] = df.sort_values(["ad_id", "rank"]).groupby("ad_id").first()["onet_code"].to_dict()

        # Common ads
        common_ads = sorted(set.intersection(*[set(s.keys()) for s in filter_sets.values()]))

        key = (fixed_llm, fixed_rerank)
        arm2_results[key] = {
            "embed_names": embed_names,
            "arm_runs": arm_runs,
            "common_ads": common_ads,
            "cosine_sets": cosine_sets,
            "cosine_sets_top5": cosine_sets_top5,
            "filter_sets": filter_sets,
            "filter_counts": filter_counts,
            "filter_top1": filter_top1,
        }

print(f"Arm 2: {len(arm2_results)} (LLM, reranker) groups loaded")
for (llm, rerank), r in arm2_results.items():
    print(f"  {mn.get(llm, llm)} + {mn.get(rerank, rerank)}: {len(r['embed_names'])} embeddings, {len(r['common_ads'])} common ads")

# %% [markdown]
# ## Cosine stage
#
# Different embedding models retrieve different candidate sets from the O\*NET
# occupation space. For each ad, we compare the candidate sets from two
# embeddings: $J = |A \cap B| \,/\, |A \cup B|$.
#
# This is the earliest point of divergence in the pipeline. We measure Jaccard
# at two depths: the full top-20 (all candidates passed to the LLM filter) and
# the top-5 (the highest-confidence matches). If top-5 Jaccard is higher than
# top-20, the embeddings agree more on the best candidates and diverge mainly
# on the tail.
#
# ### Pairwise Jaccard (cosine candidates, top-20)

# %%
arm2_cosine_jaccard = {}

for (fixed_llm, fixed_rerank), r in arm2_results.items():
    arm2_cosine_jaccard[(fixed_llm, fixed_rerank)] = build_pairwise_matrix(
        r["embed_names"], r["cosine_sets"], r["common_ads"], pairwise_jaccard,
    )

cosine_jaccard_rows = []
for (fixed_llm, fixed_rerank), matrix in arm2_cosine_jaccard.items():
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    cosine_jaccard_rows.append(stats)
arm2_cosine_jaccard_summary = pd.DataFrame(cosine_jaccard_rows).set_index(["llm", "reranker"])

# %%
arm2_cosine_jaccard_summary

# %%
for (fixed_llm, fixed_rerank), matrix in arm2_cosine_jaccard.items():
    display(IPyMarkdown(f"**Cosine Jaccard, top-20 ({mn.get(fixed_llm, fixed_llm)} + {mn.get(fixed_rerank, fixed_rerank)})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ### Pairwise Jaccard (cosine candidates, top-5)
#
# Restricting to only the 5 highest-ranked cosine candidates per ad. If two
# embeddings agree on the top-5 but disagree on ranks 6-20, that tail
# disagreement is less likely to affect the final output.

# %%
arm2_cosine_jaccard_top5 = {}

for (fixed_llm, fixed_rerank), r in arm2_results.items():
    arm2_cosine_jaccard_top5[(fixed_llm, fixed_rerank)] = build_pairwise_matrix(
        r["embed_names"], r["cosine_sets_top5"], r["common_ads"], pairwise_jaccard,
    )

cosine_top5_rows = []
for (fixed_llm, fixed_rerank), matrix in arm2_cosine_jaccard_top5.items():
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    cosine_top5_rows.append(stats)
arm2_cosine_top5_summary = pd.DataFrame(cosine_top5_rows).set_index(["llm", "reranker"])

# %%
arm2_cosine_top5_summary

# %%
for (fixed_llm, fixed_rerank), matrix in arm2_cosine_jaccard_top5.items():
    display(IPyMarkdown(f"**Cosine Jaccard, top-5 ({mn.get(fixed_llm, fixed_llm)} + {mn.get(fixed_rerank, fixed_rerank)})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ## Filter stage
#
# ### Pairwise Jaccard (LLM filter kept sets)
#
# After the LLM filter, different embeddings produce different kept sets for two
# reasons: (1) they started with different cosine candidates, and (2) the LLM may
# respond differently to different candidate presentations. Comparing this Jaccard
# with the cosine-stage Jaccard shows whether the LLM filter amplifies or dampens
# embedding disagreement.

# %%
arm2_filter_jaccard = {}

for (fixed_llm, fixed_rerank), r in arm2_results.items():
    arm2_filter_jaccard[(fixed_llm, fixed_rerank)] = build_pairwise_matrix(
        r["embed_names"], r["filter_sets"], r["common_ads"], pairwise_jaccard,
    )

filter_jaccard_rows = []
for (fixed_llm, fixed_rerank), matrix in arm2_filter_jaccard.items():
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    filter_jaccard_rows.append(stats)
arm2_filter_jaccard_summary = pd.DataFrame(filter_jaccard_rows).set_index(["llm", "reranker"])

# %%
arm2_filter_jaccard_summary

# %%
for (fixed_llm, fixed_rerank), matrix in arm2_filter_jaccard.items():
    display(IPyMarkdown(f"**Filter Jaccard ({mn.get(fixed_llm, fixed_llm)} + {mn.get(fixed_rerank, fixed_rerank)})**"))
    display(matrix.style.format("{:.3f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=1))

# %% [markdown]
# ### Top-1 agreement (filter stage)

# %%
filter_top1_rows = []
for (fixed_llm, fixed_rerank), r in arm2_results.items():
    matrix = build_pairwise_matrix(
        r["embed_names"], r["filter_top1"], r["common_ads"], pairwise_top1,
    )
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    filter_top1_rows.append(stats)
arm2_filter_top1_summary = pd.DataFrame(filter_top1_rows).set_index(["llm", "reranker"])

# %%
arm2_filter_top1_summary

# %% [markdown]
# ## Rerank stage

# %%
for (fixed_llm, fixed_rerank), r in arm2_results.items():
    rerank_scores = {}
    rerank_top1 = {}
    for rn, embed in r["arm_runs"]:
        name = mn.get(embed, embed)
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

# %%
arm2_wj = {}

for (fixed_llm, fixed_rerank), r in arm2_results.items():
    arm2_wj[(fixed_llm, fixed_rerank)] = build_pairwise_matrix(
        r["embed_names"], r["rerank_scores"], r["common_ads"], pairwise_weighted_jaccard,
    )

wj_rows = []
for (fixed_llm, fixed_rerank), matrix in arm2_wj.items():
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    wj_rows.append(stats)
arm2_wj_summary = pd.DataFrame(wj_rows).set_index(["llm", "reranker"])

# %%
arm2_wj_summary

# %% [markdown]
# ### Top-1 agreement after reranking

# %%
rerank_top1_rows = []
for (fixed_llm, fixed_rerank), r in arm2_results.items():
    matrix = build_pairwise_matrix(
        r["embed_names"], r["rerank_top1"], r["common_ads"], pairwise_top1,
    )
    stats = upper_tri_stats(matrix.values)
    stats["llm"] = mn.get(fixed_llm, fixed_llm)
    stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
    rerank_top1_rows.append(stats)
arm2_rerank_top1_summary = pd.DataFrame(rerank_top1_rows).set_index(["llm", "reranker"])

# %%
arm2_rerank_top1_summary

# %% [markdown]
# ## Exposure scores
#
# The final per-ad exposure scores (node: `compute_job_ad_exposure`). Since
# different embeddings produce entirely different candidate pipelines, embedding
# sensitivity is expected to be larger than LLM sensitivity (where the cosine
# candidates were identical).

# %%
score_cols = [
    "felten_score", "presence_physical", "presence_emotional",
    "presence_creative", "presence_composite",
    "task_exposure_mean", "task_exposure_importance_weighted",
]

for (fixed_llm, fixed_rerank), r in arm2_results.items():
    exposure_by_embed = {}
    for rn, embed in r["arm_runs"]:
        name = mn.get(embed, embed)
        df = load_parquet(rn, "compute_job_ad_exposure", "ad_exposure.parquet")
        exposure_by_embed[name] = df.dropna(subset=score_cols).set_index("ad_id")

    common_ads = sorted(set.intersection(*[set(df.index) for df in exposure_by_embed.values()]))
    r["exposure_by_embed"] = exposure_by_embed
    r["exposure_common_ads"] = common_ads

print("Exposure data loaded.")
for (llm, rerank), r in arm2_results.items():
    print(f"  {mn.get(llm, llm)} + {mn.get(rerank, rerank)}: {len(r['exposure_common_ads'])} common ads")

# %% [markdown]
# ### Pearson correlation per score column

# %%
pearson_rows = []
for (fixed_llm, fixed_rerank), r in arm2_results.items():
    common = r["exposure_common_ads"]
    for col in score_cols:
        vectors = {emb: r["exposure_by_embed"][emb].loc[common, col].values for emb in r["embed_names"]}
        matrix = pairwise_correlation_matrix(r["embed_names"], vectors, method="pearson")
        stats = upper_tri_stats(matrix.values)
        stats["llm"] = mn.get(fixed_llm, fixed_llm)
        stats["reranker"] = mn.get(fixed_rerank, fixed_rerank)
        stats["score"] = col
        pearson_rows.append(stats)

arm2_pearson = pd.DataFrame(pearson_rows)
arm2_pearson_pivot = arm2_pearson.pivot(index="score", columns=["llm", "reranker"], values="mean")

# %%
arm2_pearson_pivot

# %% [markdown]
# ### MAD as % of range

# %%
mad_rows = []
for (fixed_llm, fixed_rerank), r in arm2_results.items():
    common = r["exposure_common_ads"]
    for col in score_cols:
        all_vals = np.concatenate([r["exposure_by_embed"][emb].loc[common, col].values for emb in r["embed_names"]])
        score_range = all_vals.max() - all_vals.min()

        mads = []
        for i, emb_a in enumerate(r["embed_names"]):
            for j, emb_b in enumerate(r["embed_names"]):
                if i >= j:
                    continue
                va = r["exposure_by_embed"][emb_a].loc[common, col].values
                vb = r["exposure_by_embed"][emb_b].loc[common, col].values
                mads.append(np.mean(np.abs(va - vb)))

        mad_rows.append({
            "llm": mn.get(fixed_llm, fixed_llm), "reranker": mn.get(fixed_rerank, fixed_rerank), "score": col,
            "mad_pct_range": np.mean(mads) / score_range * 100 if score_range > 0 else 0,
        })

arm2_mad = pd.DataFrame(mad_rows)

# %%
arm2_mad.pivot(index="score", columns=["llm", "reranker"], values="mad_pct_range").style.format("{:.1f}%")

# %% [markdown]
# ### Pearson matrix for primary score

# %%
for (fixed_llm, fixed_rerank), r in arm2_results.items():
    common = r["exposure_common_ads"]
    vectors = {emb: r["exposure_by_embed"][emb].loc[common, "task_exposure_importance_weighted"].values
               for emb in r["embed_names"]}
    matrix = pairwise_correlation_matrix(r["embed_names"], vectors, method="pearson")
    display(IPyMarkdown(f"**Pearson: task_exposure_importance_weighted ({mn.get(fixed_llm, fixed_llm)} + {mn.get(fixed_rerank, fixed_rerank)})**"))
    display(matrix.style.format("{:.4f}").background_gradient(cmap="YlOrRd", vmin=0.9, vmax=1))

# %% [markdown]
# ## Summary
#
# Embedding sensitivity differs from LLM sensitivity because disagreement enters
# at the cosine retrieval stage, before any LLM processing. The key questions:
#
# 1. **How much do cosine candidate sets overlap?** Low overlap means the
#    embeddings retrieve fundamentally different occupations for the same ad.
#
# 2. **Does the LLM filter amplify or dampen this?** If filter Jaccard is lower
#    than cosine Jaccard, the LLM is making the disagreement worse. If higher,
#    the LLM is finding common ground among different candidate pools.
#
# 3. **How much does embedding choice affect final scores?** Compare the exposure
#    Pearson here with Arm 1. If embedding sensitivity is much larger than LLM
#    sensitivity, embedding choice is the more critical design decision.
