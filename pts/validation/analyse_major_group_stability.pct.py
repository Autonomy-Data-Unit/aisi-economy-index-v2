# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Major Group Stability Analysis
#
# Does pipeline stability hold within individual O\*NET major groups, or is the
# high overall Pearson correlation an artifact of averaging across diverse
# occupational domains?
#
# For each validation run, ads are assigned to an O\*NET major group based on
# the top-1 reranked occupation code (first two digits of the SOC code). Within
# each major group, we compute the same pairwise Pearson correlations on
# exposure scores that the arm notebooks compute on the full sample.
#
# This analysis reuses existing validation run data; no new pipeline runs are
# needed.

# %% [markdown]
# ## Setup

# %%
import os
import tomllib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from IPython.display import display, Markdown as IPyMarkdown

from ai_index.const import validation_config_path
from validation.utils import (
    discover_completed_runs,
    load_parquet,
    build_model_name_lookup,
    pairwise_correlation_matrix,
    upper_tri_stats,
    best_subsets,
    collect_arm_groups,
    SOC_MAJOR_GROUPS,
)

run_def = os.environ.get("VALIDATION_RUN_DEF", "validation_5k")

with open(validation_config_path, "rb") as f:
    config = tomllib.load(f)

completed = discover_completed_runs(run_def)
mn = build_model_name_lookup()

PRIMARY_SCORE = "task_exposure_importance_weighted"
MIN_GROUP_SIZE = 30

print(f"Run definition: {run_def}")
print(f"Completed runs: {len(completed)}")

# %% [markdown]
# ## Collect arm groups
#
# We collect all arm groups across the three arms of the sensitivity study.
# Each arm group has one fixed model combination and varies one dimension
# (LLM, embedding, or reranker).

# %%
arm_groups = collect_arm_groups(completed, config, mn)

print(f"Total arm groups: {len(arm_groups)}")
for ag in arm_groups:
    print(f"  Arm {ag['arm']} ({ag['varied']}): {ag['fixed']} -- {len(ag['runs'])} runs")

# %% [markdown]
# ## Assign ads to major groups and compute per-group Pearson
#
# For each arm group, we:
# 1. Assign each ad to a major group based on the first run's top-1 reranked
#    O\*NET code (first 2 digits of the SOC code).
# 2. Load exposure scores for all runs.
# 3. Compute the overall (all-ad) mean pairwise Pearson as a baseline.
# 4. For each major group with at least 30 ads, compute the mean pairwise
#    Pearson on the primary exposure score.

# %%
all_group_results = []
overall_results = []

for ag in arm_groups:
    # Assign major groups from first run's reranked top-1
    ref_run = ag["runs"][0][0]
    rerank_df = load_parquet(ref_run, "rerank_candidates", "reranked_matches.parquet")
    ad_top1 = (
        rerank_df.sort_values("rerank_score", ascending=False)
        .groupby("ad_id")
        .first()
    )
    ad_major = ad_top1["onet_code"].str[:2].to_dict()

    # Load exposure scores
    exposure = {}
    for rn, key in ag["runs"]:
        name = mn.get(key, key)
        df = load_parquet(rn, "compute_job_ad_exposure", "ad_exposure.parquet")
        exposure[name] = df.dropna(subset=[PRIMARY_SCORE]).set_index("ad_id")

    common_ads = sorted(set.intersection(*[set(df.index) for df in exposure.values()]))
    names = ag["names"]

    # Overall Pearson (all ads)
    pearsons_all = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            va = exposure[names[i]].loc[common_ads, PRIMARY_SCORE].values
            vb = exposure[names[j]].loc[common_ads, PRIMARY_SCORE].values
            r, _ = pearsonr(va, vb)
            pearsons_all.append(r)

    overall_results.append({
        "arm": ag["arm"],
        "varied": ag["varied"],
        "fixed": ag["fixed"],
        "n_ads": len(common_ads),
        "mean_pearson": np.mean(pearsons_all),
    })

    # Group ads by major group
    groups = {}
    for ad_id in common_ads:
        mg = ad_major.get(ad_id)
        if mg is not None:
            groups.setdefault(mg, []).append(ad_id)

    # Per-group Pearson
    for mg, ads in sorted(groups.items()):
        if len(ads) < MIN_GROUP_SIZE:
            continue
        pearsons = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                va = exposure[names[i]].loc[ads, PRIMARY_SCORE].values
                vb = exposure[names[j]].loc[ads, PRIMARY_SCORE].values
                r, _ = pearsonr(va, vb)
                pearsons.append(r)

        all_group_results.append({
            "arm": ag["arm"],
            "varied": ag["varied"],
            "fixed": ag["fixed"],
            "major_group": mg,
            "group_name": SOC_MAJOR_GROUPS.get(mg, mg),
            "n_ads": len(ads),
            "mean_pearson": np.mean(pearsons),
            "min_pearson": np.min(pearsons),
            "max_pearson": np.max(pearsons),
            "n_pairs": len(pearsons),
        })

results_df = pd.DataFrame(all_group_results)
overall_df = pd.DataFrame(overall_results)
overall_pearson_mean = overall_df["mean_pearson"].mean()

print(f"Per-group results: {len(results_df)} rows "
      f"({results_df['major_group'].nunique()} major groups, "
      f"{len(arm_groups)} arm groups)")
print(f"Overall mean Pearson (reference): {overall_pearson_mean:.4f}")

# %% [markdown]
# ## Major group distribution
#
# How many ads fall into each major group? Groups with few ads produce noisier
# Pearson estimates. Groups below the minimum threshold (N < 30) are excluded
# from the stability analysis.

# %%
ref_ag = arm_groups[0]
ref_run = ref_ag["runs"][0][0]
rerank_df = load_parquet(ref_run, "rerank_candidates", "reranked_matches.parquet")
ad_top1 = rerank_df.sort_values("rerank_score", ascending=False).groupby("ad_id").first()
all_ad_majors = ad_top1["onet_code"].str[:2]

dist = all_ad_majors.value_counts().sort_index()
dist_df = pd.DataFrame({
    "major_group": dist.index,
    "group_name": [SOC_MAJOR_GROUPS.get(mg, mg) for mg in dist.index],
    "n_ads": dist.values,
    "pct": (dist.values / dist.values.sum() * 100).round(1),
    "included": ["yes" if n >= MIN_GROUP_SIZE else "no" for n in dist.values],
})
dist_df

# %%
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#4c72b0" if n >= MIN_GROUP_SIZE else "#cccccc" for n in dist_df["n_ads"]]
ax.bar(range(len(dist_df)), dist_df["n_ads"], color=colors)
ax.set_xticks(range(len(dist_df)))
ax.set_xticklabels(
    [f"{r['major_group']}\n{r['group_name']}" for _, r in dist_df.iterrows()],
    rotation=45, ha="right", fontsize=7,
)
ax.axhline(MIN_GROUP_SIZE, color="red", linestyle="--", linewidth=0.8, label=f"min N = {MIN_GROUP_SIZE}")
ax.set_ylabel("Number of ads")
ax.set_title("Ad count per O*NET major group (grey = excluded from analysis)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## Per-group Pearson: summary across all arms
#
# For each major group, the table shows the mean and minimum pairwise Pearson
# correlation on `task_exposure_importance_weighted`, averaged across all arm
# groups. A value close to the overall Pearson (typically $r \geq 0.95$) means
# that stability holds within that occupational domain.
#
# The `delta_from_overall` column shows how much the per-group Pearson deviates
# from the overall (all-ad) mean. Negative values indicate groups where model
# choice has a larger-than-average effect on per-ad exposure scores.

# %%
summary = (
    results_df.groupby(["major_group", "group_name"])
    .agg(
        n_ads=("n_ads", "first"),
        mean_pearson=("mean_pearson", "mean"),
        min_pearson=("min_pearson", "min"),
        max_pearson=("max_pearson", "max"),
        n_arm_groups=("arm", "count"),
    )
    .sort_values("mean_pearson")
    .reset_index()
)
summary["delta_from_overall"] = summary["mean_pearson"] - overall_pearson_mean

# %%
display(
    summary.style
    .format({
        "mean_pearson": "{:.4f}", "min_pearson": "{:.4f}",
        "max_pearson": "{:.4f}", "delta_from_overall": "{:+.4f}",
    })
    .background_gradient(subset=["mean_pearson"], cmap="YlOrRd", vmin=0.8, vmax=1)
)

# %% [markdown]
# ## Per-group Pearson: breakdown by arm
#
# Does stability within major groups depend on which pipeline dimension is
# varied? Arm 1 (LLM) and Arm 3 (reranker) typically show higher overall
# Pearson than Arm 2 (embedding). The same pattern may hold within groups.

# %%
by_arm = (
    results_df.groupby(["major_group", "group_name", "arm", "varied"])
    .agg(mean_pearson=("mean_pearson", "mean"), n_ads=("n_ads", "first"))
    .reset_index()
)
by_arm_pivot = by_arm.pivot(
    index=["major_group", "group_name", "n_ads"],
    columns="varied",
    values="mean_pearson",
)

# %%
display(
    by_arm_pivot.style
    .format("{:.4f}", na_rep="--")
    .background_gradient(cmap="YlOrRd", vmin=0.8, vmax=1)
)

# %% [markdown]
# ## Scatter: group size vs. stability
#
# Is low per-group Pearson driven by small sample sizes? If so, we would expect
# a positive relationship between group size and Pearson. Groups that fall
# below the trend line are genuinely less stable (not just noisy).
#
# The dashed line shows the overall (all-ad) mean Pearson for reference.

# %%
fig, ax = plt.subplots(figsize=(8, 6))

for arm in sorted(results_df["arm"].unique()):
    subset = results_df[results_df["arm"] == arm]
    arm_summary = subset.groupby("major_group").agg(
        n_ads=("n_ads", "first"),
        mean_pearson=("mean_pearson", "mean"),
    )
    ax.scatter(
        arm_summary["n_ads"], arm_summary["mean_pearson"],
        label=f"Arm {arm} ({subset.iloc[0]['varied']})",
        alpha=0.7, s=40,
    )

ax.axhline(overall_pearson_mean, color="grey", linestyle="--", linewidth=0.8,
           label=f"Overall r = {overall_pearson_mean:.3f}")

# Label the lowest-Pearson groups
for _, row in summary.iterrows():
    if row["mean_pearson"] < summary["mean_pearson"].quantile(0.25):
        ax.annotate(
            f"{row['major_group']} {row['group_name']}",
            (row["n_ads"], row["mean_pearson"]),
            fontsize=7, alpha=0.8,
            xytext=(5, -5), textcoords="offset points",
        )

ax.set_xlabel("Number of ads in group")
ax.set_ylabel(f"Mean pairwise Pearson ({PRIMARY_SCORE})")
ax.set_title("Per-group stability vs. group size")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## Pearson matrices for lowest-stability groups
#
# Full pairwise Pearson matrices within the 5 major groups with the lowest
# mean Pearson, shown for the arm group with the most model runs. This
# highlights whether particular model pairs drive low per-group stability.

# %%
largest_ag = max(arm_groups, key=lambda ag: len(ag["runs"]))
ref_run = largest_ag["runs"][0][0]
rerank_df = load_parquet(ref_run, "rerank_candidates", "reranked_matches.parquet")
ad_top1 = rerank_df.sort_values("rerank_score", ascending=False).groupby("ad_id").first()
ad_major = ad_top1["onet_code"].str[:2].to_dict()

exposure = {}
for rn, key in largest_ag["runs"]:
    name = mn.get(key, key)
    df = load_parquet(rn, "compute_job_ad_exposure", "ad_exposure.parquet")
    exposure[name] = df.dropna(subset=[PRIMARY_SCORE]).set_index("ad_id")

common_ads = sorted(set.intersection(*[set(df.index) for df in exposure.values()]))

groups = {}
for ad_id in common_ads:
    mg = ad_major.get(ad_id)
    if mg is not None:
        groups.setdefault(mg, []).append(ad_id)

display(IPyMarkdown(
    f"**Arm {largest_ag['arm']} ({largest_ag['varied']}): {largest_ag['fixed']}** "
    f"-- {len(largest_ag['runs'])} models"
))

# Show the 5 groups with lowest mean Pearson
bottom_groups = summary.head(5)["major_group"].tolist()

for mg in bottom_groups:
    if mg not in groups or len(groups[mg]) < MIN_GROUP_SIZE:
        continue
    ads = groups[mg]
    names = largest_ag["names"]
    vectors = {name: exposure[name].loc[ads, PRIMARY_SCORE].values for name in names}
    matrix = pairwise_correlation_matrix(names, vectors, method="pearson")
    stats = upper_tri_stats(matrix.values)
    display(IPyMarkdown(
        f"**{mg} {SOC_MAJOR_GROUPS.get(mg, mg)}** (N={len(ads)}, "
        f"mean r={stats['mean']:.4f}, min r={stats['min']:.4f})"
    ))
    display(matrix.style.format("{:.4f}").background_gradient(cmap="YlOrRd", vmin=0.8, vmax=1))

# %% [markdown]
# ## Best model subsets for lowest-stability groups
#
# Are a few outlier models dragging down the within-group Pearson? For each of
# the bottom-5 groups, we find the best model subset at each size $k$ (from all
# models down to 2) by exhaustive search over pairwise Pearson. Reading top to
# bottom shows which models get dropped first.
#
# If dropping 1--2 models dramatically improves the per-group Pearson, the low
# stability is concentrated in those outliers rather than being a broad issue.

# %%
for mg in bottom_groups:
    if mg not in groups or len(groups[mg]) < MIN_GROUP_SIZE:
        continue
    ads = groups[mg]
    names = largest_ag["names"]
    vectors = {name: exposure[name].loc[ads, PRIMARY_SCORE].values for name in names}
    matrix = pairwise_correlation_matrix(names, vectors, method="pearson")

    subset_rows = []
    for k, score, subset_names in best_subsets(names, matrix.values):
        if len(subset_names) > 6:
            removed = sorted(set(names) - set(subset_names))
            label = f"all except: {', '.join(removed)}" if removed else "all"
        else:
            label = ", ".join(subset_names)
        subset_rows.append({"k": k, "mean_pearson": round(score, 4), "subset": label})

    display(IPyMarkdown(
        f"**{mg} {SOC_MAJOR_GROUPS.get(mg, mg)}** (N={len(ads)})"
    ))
    display(pd.DataFrame(subset_rows))

# %% [markdown]
# ## Per-group Pearson after dropping weakest models
#
# To separate "outlier model" effects from genuine within-group instability,
# we recompute per-group Pearson using only the best-k subset for each group.
# We use the largest arm group and drop models until 2/3 of the original set
# remains (i.e. best $\lceil 2N/3 \rceil$ models).

# %%
import math

n_models = len(largest_ag["names"])
k_target = max(2, math.ceil(2 * n_models / 3))

pruned_rows = []
for mg in sorted(groups.keys()):
    ads = groups[mg]
    if len(ads) < MIN_GROUP_SIZE:
        continue
    names = largest_ag["names"]
    vectors = {name: exposure[name].loc[ads, PRIMARY_SCORE].values for name in names}
    matrix = pairwise_correlation_matrix(names, vectors, method="pearson")

    all_stats = upper_tri_stats(matrix.values)

    # Find best subset of size k_target
    for k, score, subset_names in best_subsets(names, matrix.values):
        if k == k_target:
            removed = sorted(set(names) - set(subset_names))
            pruned_rows.append({
                "major_group": mg,
                "group_name": SOC_MAJOR_GROUPS.get(mg, mg),
                "n_ads": len(ads),
                "all_models_pearson": all_stats["mean"],
                f"best_{k_target}_pearson": score,
                "improvement": score - all_stats["mean"],
                "dropped": ", ".join(removed) if removed else "(none)",
            })
            break

pruned_df = pd.DataFrame(pruned_rows).sort_values("all_models_pearson")

# %%
display(
    pruned_df.style
    .format({
        "all_models_pearson": "{:.4f}",
        f"best_{k_target}_pearson": "{:.4f}",
        "improvement": "{:+.4f}",
    })
    .background_gradient(
        subset=["all_models_pearson", f"best_{k_target}_pearson"],
        cmap="YlOrRd", vmin=0.6, vmax=1,
    )
)

# %% [markdown]
# ## Summary
#
# This analysis decomposes the overall pipeline stability into per-major-group
# contributions. The key questions it answers:
#
# 1. **Does stability hold within every group?** If all groups have Pearson
#    close to the overall mean, stability is not an artifact of cross-group
#    averaging.
#
# 2. **Are certain occupational domains less stable?** Groups with low Pearson
#    and adequate sample size represent domains where model choice matters more.
#    These may warrant targeted validation or a domain-specific model selection.
#
# 3. **Is there a size effect?** If only small groups show low Pearson, the
#    instability is likely sampling noise rather than a genuine pipeline weakness.
#    Larger validation runs (e.g. 50K ads) would resolve this.
