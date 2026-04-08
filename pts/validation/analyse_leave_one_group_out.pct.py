# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Leave-One-Group-Out Stability Analysis
#
# Does any single O\*NET major group disproportionately drive the overall
# pipeline stability? For each major group, we remove its ads and recompute
# the pairwise Pearson correlation on the remaining ads. If removing a group
# substantially changes the Pearson, that group is either a source of
# instability (removal increases Pearson) or a stability anchor (removal
# decreases Pearson).
#
# This is the complement of the stratified major-group analysis: instead of
# asking "is each group stable?" it asks "is overall stability robust to
# the removal of any group?"

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
    collect_arm_groups,
    SOC_MAJOR_GROUPS,
)

run_def = os.environ.get("VALIDATION_RUN_DEF", "validation_5k")

with open(validation_config_path, "rb") as f:
    config = tomllib.load(f)

completed = discover_completed_runs(run_def)
mn = build_model_name_lookup()

PRIMARY_SCORE = "task_exposure_importance_weighted"

print(f"Run definition: {run_def}")
print(f"Completed runs: {len(completed)}")

# %% [markdown]
# ## Collect arm groups

# %%
arm_groups = collect_arm_groups(completed, config, mn)

print(f"Total arm groups: {len(arm_groups)}")
for ag in arm_groups:
    print(f"  Arm {ag['arm']} ({ag['varied']}): {ag['fixed']} -- {len(ag['runs'])} runs")

# %% [markdown]
# ## Compute leave-one-group-out Pearson
#
# For each arm group:
# 1. Compute the **baseline** mean pairwise Pearson on all common ads.
# 2. For each major group, remove its ads and recompute the mean pairwise Pearson.
# 3. Record the delta: $\Delta r = r_{\text{baseline}} - r_{\text{LOO}}$.
#
# Positive delta means the removed group was contributing to stability (its
# removal hurts Pearson). Negative delta means the group was a source of
# noise (its removal improves Pearson).

# %%
def mean_pairwise_pearson(names, exposure, ads, score_col):
    """Mean pairwise Pearson for a set of ads across all model pairs."""
    pearsons = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            va = exposure[names[i]].loc[ads, score_col].values
            vb = exposure[names[j]].loc[ads, score_col].values
            r, _ = pearsonr(va, vb)
            pearsons.append(r)
    return np.mean(pearsons)

# %%
loo_results = []

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

    # Group ads by major group
    groups = {}
    for ad_id in common_ads:
        mg = ad_major.get(ad_id)
        if mg is not None:
            groups.setdefault(mg, []).append(ad_id)

    # Baseline Pearson (all ads)
    baseline_r = mean_pairwise_pearson(names, exposure, common_ads, PRIMARY_SCORE)

    # Leave-one-out for each group
    for mg, mg_ads in sorted(groups.items()):
        mg_ad_set = set(mg_ads)
        remaining = [a for a in common_ads if a not in mg_ad_set]
        if len(remaining) < 50:
            continue
        loo_r = mean_pairwise_pearson(names, exposure, remaining, PRIMARY_SCORE)
        loo_results.append({
            "arm": ag["arm"],
            "varied": ag["varied"],
            "fixed": ag["fixed"],
            "major_group": mg,
            "group_name": SOC_MAJOR_GROUPS.get(mg, mg),
            "n_removed": len(mg_ads),
            "n_remaining": len(remaining),
            "baseline_pearson": baseline_r,
            "loo_pearson": loo_r,
            "delta": baseline_r - loo_r,
        })

loo_df = pd.DataFrame(loo_results)
print(f"LOO results: {len(loo_df)} rows "
      f"({loo_df['major_group'].nunique()} major groups, "
      f"{len(arm_groups)} arm groups)")

# %% [markdown]
# ## Baseline Pearson per arm group
#
# The starting point: overall mean pairwise Pearson on the full sample,
# against which we measure the effect of removing each group.

# %%
baselines = (
    loo_df.groupby(["arm", "varied", "fixed"])
    ["baseline_pearson"]
    .first()
    .reset_index()
)
baselines

# %% [markdown]
# ## Delta Pearson by major group
#
# For each major group, the mean delta across all arm groups. Positive delta
# means removing the group decreases overall Pearson (the group contributes
# to stability). Negative delta means removing the group increases Pearson
# (the group is a source of instability).
#
# Deltas on the order of 0.001 are negligible. Only deltas larger than ~0.01
# represent meaningful effects on the overall stability assessment.

# %%
delta_summary = (
    loo_df.groupby(["major_group", "group_name"])
    .agg(
        n_removed=("n_removed", "first"),
        mean_delta=("delta", "mean"),
        min_delta=("delta", "min"),
        max_delta=("delta", "max"),
        n_arm_groups=("arm", "count"),
    )
    .sort_values("mean_delta")
    .reset_index()
)

# %%
display(
    delta_summary.style
    .format({"mean_delta": "{:+.5f}", "min_delta": "{:+.5f}", "max_delta": "{:+.5f}"})
    .background_gradient(subset=["mean_delta"], cmap="RdYlGn", vmin=-0.01, vmax=0.01)
)

# %% [markdown]
# ## Delta Pearson: breakdown by arm
#
# The same delta analysis split by which dimension is varied. Some groups
# might be stable when varying the LLM but unstable when varying the embedding.

# %%
delta_by_arm = (
    loo_df.groupby(["major_group", "group_name", "arm", "varied"])
    .agg(mean_delta=("delta", "mean"), n_removed=("n_removed", "first"))
    .reset_index()
)
delta_pivot = delta_by_arm.pivot(
    index=["major_group", "group_name", "n_removed"],
    columns="varied",
    values="mean_delta",
)

# %%
display(
    delta_pivot.style
    .format("{:+.5f}", na_rep="--")
    .background_gradient(cmap="RdYlGn", vmin=-0.01, vmax=0.01)
)

# %% [markdown]
# ## Visualization: delta Pearson per group
#
# Horizontal bar chart showing the mean delta for each major group across all
# arm groups. Red bars (negative delta) indicate groups whose removal improves
# overall Pearson; green bars (positive delta) indicate groups that contribute
# to stability.

# %%
fig, ax = plt.subplots(figsize=(10, 7))

plot_df = delta_summary.sort_values("mean_delta")
colors = ["#d62728" if d < 0 else "#2ca02c" for d in plot_df["mean_delta"]]

ax.barh(range(len(plot_df)), plot_df["mean_delta"], color=colors)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(
    [f"{r['major_group']} {r['group_name']} (N={r['n_removed']})"
     for _, r in plot_df.iterrows()],
    fontsize=8,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(f"$\\Delta r$ (baseline $-$ LOO Pearson on {PRIMARY_SCORE})")
ax.set_title(
    "Leave-one-group-out: effect of removing each major group\n"
    "Red = removal improves Pearson (group adds noise)   "
    "Green = removal hurts Pearson (group adds stability)"
)
fig.tight_layout()

# %% [markdown]
# ## Scatter: group size vs. delta
#
# Larger groups have more influence on the overall Pearson, so we might expect
# their removal to produce larger deltas (in either direction). If all groups
# cluster near zero regardless of size, the stability result is robust.

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(delta_summary["n_removed"], delta_summary["mean_delta"], alpha=0.7, s=50)

for _, row in delta_summary.iterrows():
    ax.annotate(
        f"{row['major_group']}",
        (row["n_removed"], row["mean_delta"]),
        fontsize=7, alpha=0.8,
        xytext=(4, 4), textcoords="offset points",
    )

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Number of ads in removed group")
ax.set_ylabel(f"$\\Delta r$ (baseline $-$ LOO)")
ax.set_title("Group size vs. stability contribution")
fig.tight_layout()

# %% [markdown]
# ## Per-arm LOO detail
#
# For the arm groups with the most model variation, show the full per-group
# LOO results so we can see whether any single group causes a large delta
# in any specific arm.

# %%
for arm in sorted(loo_df["arm"].unique()):
    arm_data = loo_df[loo_df["arm"] == arm]
    arm_pivot = arm_data.pivot_table(
        index=["major_group", "group_name", "n_removed"],
        columns="fixed",
        values="delta",
    )
    display(IPyMarkdown(f"**Arm {arm} ({arm_data.iloc[0]['varied']})**"))
    display(
        arm_pivot.style
        .format("{:+.5f}", na_rep="--")
        .background_gradient(cmap="RdYlGn", vmin=-0.01, vmax=0.01)
    )

# %% [markdown]
# ## Summary
#
# The leave-one-group-out analysis tests whether overall pipeline stability
# depends on any single occupational domain.
#
# **Interpreting the results:**
#
# - If all deltas are near zero ($|\Delta r| < 0.005$), the stability result
#   is robust: no single group is driving the high Pearson correlation observed
#   in the arm-level analyses.
#
# - Groups with large negative deltas (where removal improves Pearson) are
#   candidates for closer examination: the pipeline may be less reliable for
#   those occupational domains, or the O\*NET descriptions may not discriminate
#   well within those domains.
#
# - Groups with large positive deltas are stability anchors, likely because
#   they contain ads that are easy for all models to match consistently.
#
# - The magnitude of the deltas matters more than the sign. With ~20 major
#   groups each containing a few percent of the total ads, individual group
#   removal typically shifts Pearson by less than 0.01.
