# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experiment: Benchmark candidate pruning
#
# The benchmark (GPT-5.4) keeps far fewer candidates per ad than the sbatch
# LLMs. This experiment tests whether agreement with the benchmark improves
# if we prune the lowest-ranked candidates from each validation run's filter
# output before comparing.
#
# `MAX_CANDIDATES` controls the maximum number of candidates kept per ad.
# Candidates are ranked by cosine score (from the filter stage). If an ad
# has more candidates than `MAX_CANDIDATES`, the lowest-ranked ones are
# dropped. Set to `None` to disable pruning.

# %% [markdown]
# ## Configuration

# %%
# Maximum candidates per ad. Set to None to disable pruning.
MAX_CANDIDATES = 3

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
print(f"MAX_CANDIDATES: {MAX_CANDIDATES}")

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
# ## Candidate pruning helper

# %%
def prune_candidates(df, max_candidates):
    """Keep only the top `max_candidates` per ad by rank (lowest rank = best).

    Returns the pruned DataFrame. If max_candidates is None, returns df unchanged.
    """
    if max_candidates is None:
        return df
    return df[df["rank"] < max_candidates]

# %% [markdown]
# ## Load benchmark data

# %%
# Benchmark filter output
bench_filter_df = load_parquet(benchmark_run, "llm_filter_candidates", "filtered_matches.parquet")
bench_filter_df = prune_candidates(bench_filter_df, MAX_CANDIDATES)

bench_filter_sets = bench_filter_df.groupby("ad_id")["onet_code"].apply(set).to_dict()
bench_filter_top1 = bench_filter_df.sort_values(["ad_id", "rank"]).groupby("ad_id").first()["onet_code"].to_dict()

bench_filter_counts = bench_filter_df.groupby("ad_id").size()
print(f"Benchmark filter: {len(bench_filter_sets)} ads, mean {bench_filter_counts.mean():.1f} candidates/ad (median {bench_filter_counts.median():.0f})")

# Benchmark rerank output
bench_rerank_df = load_parquet(benchmark_run, "rerank_candidates", "reranked_matches.parquet")
bench_rerank_df = prune_candidates(bench_rerank_df, MAX_CANDIDATES)

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

# %%
rows = []
skipped = []

for rn, rd, llm, embed, rerank in completed:
    row = {
        "run": rn,
        "llm": mn.get(llm, llm),
        "embedding": mn.get(embed, embed),
        "reranker": mn.get(rerank, rerank),
    }

    # Filter comparison
    try:
        val_filter_df = load_parquet(rn, "llm_filter_candidates", "filtered_matches.parquet")
    except Exception as e:
        print(f"WARNING: skipping {rn} (corrupt filter parquet: {e})")
        skipped.append(rn)
        continue

    val_filter_df = prune_candidates(val_filter_df, MAX_CANDIDATES)

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
    try:
        val_rerank_df = load_parquet(rn, "rerank_candidates", "reranked_matches.parquet")
    except Exception as e:
        print(f"WARNING: skipping rerank/exposure for {rn} (corrupt rerank parquet: {e})")
        rows.append(row)
        continue

    val_rerank_df = prune_candidates(val_rerank_df, MAX_CANDIDATES)

    val_rerank_top1 = {}
    val_rerank_scores = {}
    for ad_id, group in val_rerank_df.groupby("ad_id"):
        val_rerank_top1[ad_id] = group.sort_values("rerank_score", ascending=False).iloc[0]["onet_code"]
        val_rerank_scores[ad_id] = dict(zip(group["onet_code"], group["rerank_score"]))

    common_rerank = sorted(set(val_rerank_top1.keys()) & set(bench_rerank_top1.keys()))
    if common_rerank:
        agrees = sum(1 for ad_id in common_rerank if val_rerank_top1[ad_id] == bench_rerank_top1[ad_id])
        row["rerank_top1"] = agrees / len(common_rerank)
        row["rerank_ruzicka"] = pairwise_weighted_jaccard(val_rerank_scores, bench_rerank_scores, common_rerank)

    # Exposure comparison (not pruned, uses saved pipeline output)
    try:
        val_exposure = load_parquet(rn, "compute_job_ad_exposure", "ad_exposure.parquet")
    except Exception as e:
        print(f"WARNING: skipping exposure for {rn} (corrupt exposure parquet: {e})")
        rows.append(row)
        continue

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
if skipped:
    print(f"Skipped {len(skipped)} runs with corrupt data: {skipped}")

# %% [markdown]
# ## Filter stage agreement with benchmark
#
# How well does each validation run's LLM filter agree with GPT-5.4's filter?
# Candidates are pruned to the top `MAX_CANDIDATES` by cosine rank before
# comparison.

# %%
print(f"Benchmark ({benchmark_run}): mean {bench_filter_counts.mean():.1f} candidates/ad (median {bench_filter_counts.median():.0f})")
print(f"MAX_CANDIDATES = {MAX_CANDIDATES}")

filter_cols = ["llm", "embedding", "reranker", "filter_mean_candidates", "filter_jaccard"]
filter_df = benchmark_comparison[filter_cols].copy()
filter_df = filter_df.sort_values("filter_jaccard", ascending=False)
filter_df.style.format({
    "filter_jaccard": "{:.3f}",
    "filter_mean_candidates": "{:.1f}",
}).background_gradient(subset=["filter_jaccard"], cmap="YlOrRd", vmin=0, vmax=1)

# %% [markdown]
# ## Rerank stage agreement with benchmark
#
# After reranking, how similar are each run's scores to the benchmark?
# Candidates are pruned to the top `MAX_CANDIDATES` by cosine rank before
# comparison.
#
# - **Top-1 agreement**: fraction of ads where the highest-scoring occupation matches
#   the benchmark. Combines the effects of embedding, LLM filter, and reranker.
# - **Ruzicka similarity**: weighted Jaccard over rerank score vectors per ad,
#   $R = \frac{1}{N}\sum_i \frac{\sum_c \min(s^v_{ic},\, s^b_{ic})}{\sum_c \max(s^v_{ic},\, s^b_{ic})}$,
#   capturing agreement across the full score distribution, not just the top pick.

# %%
rerank_cols = ["llm", "embedding", "reranker", "rerank_top1", "rerank_ruzicka"]
rerank_df = benchmark_comparison[rerank_cols].copy()
rerank_df = rerank_df.sort_values("rerank_ruzicka", ascending=False)
rerank_df.style.format({"rerank_top1": "{:.3f}", "rerank_ruzicka": "{:.3f}"}).background_gradient(
    subset=["rerank_top1", "rerank_ruzicka"], cmap="YlOrRd", vmin=0, vmax=1)

# %% [markdown]
# ## Exposure score agreement with benchmark
#
# Pearson correlation and MAD (as % of range) for `task_exposure_importance_weighted`.
# Note: exposure scores use the saved pipeline output (not pruned), since
# the exposure computation already happened with the full candidate set.

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

# %%
for dim in ["llm", "embedding", "reranker"]:
    grouped = benchmark_comparison.groupby(dim).agg({
        "filter_jaccard": "mean",
        "rerank_top1": "mean",
        "rerank_ruzicka": "mean",
        "exposure_pearson": "mean",
        "exposure_mad_pct": "mean",
    }).sort_values("exposure_pearson", ascending=False)

    display(IPyMarkdown(f"**Mean agreement with benchmark by {dim}**"))
    display(grouped.style.format({
        "filter_jaccard": "{:.3f}",
        "rerank_top1": "{:.3f}",
        "rerank_ruzicka": "{:.3f}",
        "exposure_pearson": "{:.4f}",
        "exposure_mad_pct": "{:.1f}%",
    }))
