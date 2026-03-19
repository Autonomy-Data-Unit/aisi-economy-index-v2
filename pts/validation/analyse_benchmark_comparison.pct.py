# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Job Ads Pipeline Validation: Benchmark Comparison
#
# Compare each validation run against the API benchmark run (`benchmark_5k`),
# which uses frontier models throughout: GPT-5.4 (node: `llm_filter_candidates`),
# text-embedding-3-large (node: `embed_ads`), and voyage-rerank-2.5
# (node: `rerank_candidates`).
#
# For each validation run, we measure agreement with the benchmark at three
# pipeline stages: LLM filter, rerank, and final exposure scores. This gives
# a reference-based quality measure rather than just pairwise model agreement.

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
    pairwise_correlation_matrix,
    upper_tri_stats,
)

run_def = os.environ.get("VALIDATION_RUN_DEF", "validation_5k")
benchmark_run = f"benchmark_{run_def.split('_')[-1]}"  # e.g. "benchmark_5k"

with open(validation_config_path, "rb") as f:
    config = tomllib.load(f)

mn = build_model_name_lookup()

completed = discover_completed_runs(run_def)
print(f"Run definition: {run_def}")
print(f"Benchmark run: {benchmark_run}")
print(f"Validation runs: {len(completed)}")

# %% [markdown]
# ### Benchmark model

# %%
benchmark_dir = pipeline_store_path / benchmark_run
if not (benchmark_dir / "compute_job_ad_exposure" / "exposure_meta.json").exists():
    raise ValueError(f"Benchmark run '{benchmark_run}' not complete. Run: uv run run-pipeline {benchmark_run}")

print("Benchmark pipeline:")
print(f"  LLM filter: GPT-5.4 (API)")
print(f"  Embedding:  text-embedding-3-large (API)")
print(f"  Reranker:   voyage-rerank-2.5 (API)")

# %% [markdown]
# ## Load benchmark data

# %%
# Benchmark filter output
bench_filter_df = load_parquet(benchmark_run, "llm_filter_candidates", "filtered_matches.parquet")
bench_filter_sets = bench_filter_df.groupby("ad_id")["onet_code"].apply(set).to_dict()
bench_filter_top1 = bench_filter_df.sort_values(["ad_id", "rank"]).groupby("ad_id").first()["onet_code"].to_dict()

bench_filter_counts = bench_filter_df.groupby("ad_id").size()
print(f"Benchmark filter: {len(bench_filter_sets)} ads, mean {bench_filter_counts.mean():.1f} candidates/ad (median {bench_filter_counts.median():.0f})")

# Benchmark rerank output
bench_rerank_df = load_parquet(benchmark_run, "rerank_candidates", "reranked_matches.parquet")
bench_rerank_scores = {}
bench_rerank_top1 = {}
for ad_id, group in bench_rerank_df.groupby("ad_id"):
    bench_rerank_scores[ad_id] = dict(zip(group["onet_code"], group["rerank_score"]))
    bench_rerank_top1[ad_id] = group.sort_values("rerank_score", ascending=False).iloc[0]["onet_code"]

# Benchmark exposure
score_cols = [
    "felten_score", "presence_physical", "presence_emotional",
    "presence_creative", "presence_composite",
    "task_exposure_mean", "task_exposure_importance_weighted",
]
bench_exposure = load_parquet(benchmark_run, "compute_job_ad_exposure", "ad_exposure.parquet")
bench_exposure = bench_exposure.dropna(subset=score_cols).set_index("ad_id")
print(f"Benchmark exposure: {len(bench_exposure)} ads with valid scores")

# %% [markdown]
# ## Compare each validation run against the benchmark
#
# For each completed validation run, compute:
# - **Filter Jaccard**: overlap of kept candidate sets with benchmark
# - **Filter top-1**: fraction of ads where the top cosine-ranked kept candidate matches
# - **Rerank top-1**: fraction of ads where the best-scoring candidate matches
# - **Exposure Pearson**: correlation of per-ad exposure scores with benchmark
# - **Exposure MAD%**: mean absolute difference as percentage of score range

# %%
rows = []

for rn, rd, llm, embed, rerank in completed:
    row = {
        "run": rn,
        "llm": mn.get(llm, llm),
        "embedding": mn.get(embed, embed),
        "reranker": mn.get(rerank, rerank),
    }

    # Filter comparison
    val_filter_df = load_parquet(rn, "llm_filter_candidates", "filtered_matches.parquet")
    val_filter_sets = val_filter_df.groupby("ad_id")["onet_code"].apply(set).to_dict()
    val_filter_top1 = val_filter_df.sort_values(["ad_id", "rank"]).groupby("ad_id").first()["onet_code"].to_dict()

    common_filter = sorted(set(val_filter_sets.keys()) & set(bench_filter_sets.keys()))
    if common_filter:
        jaccards = []
        top1_agrees = 0
        for ad_id in common_filter:
            a = val_filter_sets[ad_id]
            b = bench_filter_sets.get(ad_id, set())
            if a or b:
                jaccards.append(len(a & b) / len(a | b))
            va = val_filter_top1.get(ad_id)
            vb = bench_filter_top1.get(ad_id)
            if va and vb and va == vb:
                top1_agrees += 1
        row["filter_jaccard"] = np.mean(jaccards)
        row["filter_top1"] = top1_agrees / len(common_filter)
        row["filter_n_common"] = len(common_filter)
        row["filter_mean_candidates"] = val_filter_df.groupby("ad_id").size().mean()

    # Rerank comparison
    val_rerank_df = load_parquet(rn, "rerank_candidates", "reranked_matches.parquet")
    val_rerank_top1 = {}
    for ad_id, group in val_rerank_df.groupby("ad_id"):
        val_rerank_top1[ad_id] = group.sort_values("rerank_score", ascending=False).iloc[0]["onet_code"]

    common_rerank = sorted(set(val_rerank_top1.keys()) & set(bench_rerank_top1.keys()))
    if common_rerank:
        agrees = sum(1 for ad_id in common_rerank if val_rerank_top1[ad_id] == bench_rerank_top1[ad_id])
        row["rerank_top1"] = agrees / len(common_rerank)

    # Exposure comparison
    val_exposure = load_parquet(rn, "compute_job_ad_exposure", "ad_exposure.parquet")
    val_exposure = val_exposure.dropna(subset=score_cols).set_index("ad_id")
    common_exp = sorted(val_exposure.index.intersection(bench_exposure.index))

    if common_exp:
        from scipy.stats import pearsonr
        col = "task_exposure_importance_weighted"
        va = val_exposure.loc[common_exp, col].values
        vb = bench_exposure.loc[common_exp, col].values
        r, _ = pearsonr(va, vb)
        mad = np.mean(np.abs(va - vb))
        score_range = max(va.max(), vb.max()) - min(va.min(), vb.min())
        row["exposure_pearson"] = r
        row["exposure_mad_pct"] = mad / score_range * 100 if score_range > 0 else 0

    rows.append(row)

benchmark_comparison = pd.DataFrame(rows)
print(f"Compared {len(benchmark_comparison)} validation runs against benchmark")

# %% [markdown]
# ## Filter stage agreement with benchmark
#
# How well does each validation run's LLM filter agree with GPT-5.4's filter?
# Note that runs using different embeddings will have different cosine candidate
# sets feeding into the filter, so lower agreement is expected for those.

# %%
filter_cols = ["llm", "embedding", "reranker", "filter_mean_candidates", "filter_jaccard", "filter_top1"]
filter_df = benchmark_comparison[filter_cols].copy()
filter_df = filter_df.sort_values("filter_jaccard", ascending=False)
filter_df.style.format({
    "filter_jaccard": "{:.3f}",
    "filter_top1": "{:.3f}",
    "filter_mean_candidates": "{:.1f}",
}).background_gradient(subset=["filter_jaccard", "filter_top1"], cmap="YlOrRd", vmin=0, vmax=1)

# %% [markdown]
# ## Rerank top-1 agreement with benchmark
#
# After reranking, does each run agree with the benchmark on the best-scoring
# occupation? This combines the effects of embedding, LLM filter, and reranker
# differences.

# %%
rerank_cols = ["llm", "embedding", "reranker", "rerank_top1"]
rerank_df = benchmark_comparison[rerank_cols].copy()
rerank_df = rerank_df.sort_values("rerank_top1", ascending=False)
rerank_df.style.format({"rerank_top1": "{:.3f}"}).background_gradient(
    subset=["rerank_top1"], cmap="YlOrRd", vmin=0, vmax=1)

# %% [markdown]
# ## Exposure score agreement with benchmark
#
# Pearson correlation and MAD (as % of range) for `task_exposure_importance_weighted`.
# This is the bottom line: how much does each model configuration's final output
# differ from the frontier API benchmark?

# %%
exp_cols = ["llm", "embedding", "reranker", "exposure_pearson", "exposure_mad_pct"]
exp_df = benchmark_comparison[exp_cols].copy()
exp_df = exp_df.sort_values("exposure_pearson", ascending=False)
exp_df.style.format({
    "exposure_pearson": "{:.4f}",
    "exposure_mad_pct": "{:.1f}%",
}).background_gradient(subset=["exposure_pearson"], cmap="YlOrRd", vmin=0.8, vmax=1)

# %% [markdown]
# ## Summary by model dimension
#
# Average agreement with benchmark, grouped by each model dimension. Shows
# which choice (LLM, embedding, or reranker) has the largest impact on
# deviation from the benchmark.

# %%
for dim in ["llm", "embedding", "reranker"]:
    grouped = benchmark_comparison.groupby(dim).agg({
        "filter_jaccard": "mean",
        "filter_top1": "mean",
        "rerank_top1": "mean",
        "exposure_pearson": "mean",
        "exposure_mad_pct": "mean",
    }).sort_values("exposure_pearson", ascending=False)

    display(IPyMarkdown(f"**Mean agreement with benchmark by {dim}**"))
    display(grouped.style.format({
        "filter_jaccard": "{:.3f}",
        "filter_top1": "{:.3f}",
        "rerank_top1": "{:.3f}",
        "exposure_pearson": "{:.4f}",
        "exposure_mad_pct": "{:.1f}%",
    }))

# %% [markdown]
# ## Summary
#
# This notebook measures how far each validation run deviates from a
# high-quality API benchmark. Unlike the arm notebooks (which measure
# pairwise agreement within a model dimension), this gives a reference-based
# quality assessment.
#
# Key questions:
# 1. **Which LLMs agree most with GPT-5.4?** Higher filter Jaccard and top-1
#    suggest the model makes similar matching decisions to the frontier model.
# 2. **Does embedding choice matter more than LLM choice?** Compare the spread
#    of Pearson values within the "by LLM" table vs the "by embedding" table.
# 3. **Are any configurations clearly worse?** Low exposure Pearson (<0.9)
#    suggests a model is producing meaningfully different results from the
#    benchmark.
