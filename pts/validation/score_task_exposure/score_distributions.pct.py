# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Score Task Exposure: Distribution Analysis
#
# Investigate the distribution of AI exposure scores produced by the
# `score_task_exposure` node across all O\*NET occupations. We have results from
# two LLM scorers (GPT-5.2 and Qwen-7B-sbatch) and want to check for
# pathological patterns or unexpected clustering.
#
# The node classifies each O\*NET task into three exposure levels:
# - **Level 0** (no change): physical, in-person, sensory tasks
# - **Level 1** (human + LLM collaboration): >=30% productivity gain, human judgment essential
# - **Level 2** (LLM independent): end-to-end with minimal human oversight
#
# Occupation-level scores are the mean of per-task levels, normalized from [0,2] to [0,1].

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai_index.const import onet_exposure_scores_path, inputs_path

score_dir = onet_exposure_scores_path / "score_task_exposure"
models = sorted([p.name for p in score_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
print(f"Available models: {models}")

# Load O*NET occupation titles for readable output
onet_titles = pd.read_csv(
    inputs_path / "onet" / "db_30_0_text" / "Occupation Data.txt",
    sep="\t",
    usecols=["O*NET-SOC Code", "Title"],
).rename(columns={"O*NET-SOC Code": "onet_code", "Title": "title"})

def add_titles(df):
    """Merge O*NET titles onto a dataframe with onet_code column."""
    return df.merge(onet_titles, on="onet_code", how="left")

# %% [markdown]
# ## Load scores and task details for all models

# %%
scores = {}
task_details = {}
task_results = {}

for model in models:
    model_dir = score_dir / model
    scores[model] = pd.read_csv(model_dir / "scores.csv")
    task_details[model] = pd.read_parquet(model_dir / "task_details.parquet")
    tr_path = model_dir / "task_results.parquet"
    if tr_path.exists():
        task_results[model] = pd.read_parquet(tr_path)
    n = len(scores[model])
    has_tr = "task_results" if model in task_results else "no task_results"
    print(f"{model}: {n} occupations, score range [{scores[model]['task_exposure_mean'].min():.3f}, {scores[model]['task_exposure_mean'].max():.3f}] ({has_tr})")

# %% [markdown]
# ## Headline statistics

# %%
for model in models:
    print(f"\n=== {model} ===")
    display(scores[model][["task_exposure_mean", "task_exposure_importance_weighted"]].describe().round(4))

# %% [markdown]
# **Observation:** Both models produce scores with mean ~0.29-0.32 and max ~0.57-0.88. The
# entire upper half of the [0, 1] scale is essentially unused. The importance-weighted
# scores track the unweighted mean very closely (std differs by <0.001), suggesting task
# importance weights don't meaningfully reshape the distribution.

# %% [markdown]
# ## Distribution of occupation-level exposure scores

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    s = scores[model]["task_exposure_mean"]
    ax.hist(s, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(s.mean(), color="red", linestyle="--", label=f"mean={s.mean():.3f}")
    ax.axvline(s.median(), color="orange", linestyle="--", label=f"median={s.median():.3f}")
    ax.set_xlabel("task_exposure_mean")
    ax.set_ylabel("Number of occupations")
    ax.set_title(model)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

fig.suptitle("Distribution of Task Exposure Scores (occupation level)", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** Both models produce a clearly **bimodal** distribution:
#
# 1. A cluster of occupations near 0 (mostly physical/manual jobs where nearly all
#    tasks are level 0)
# 2. A tall spike just below 0.5 (occupations where nearly all tasks are level 1,
#    i.e. "collaboration" mode)
#
# GPT-5.2 has a sharper spike at ~0.5 (~190 occupations in the tallest bin). Qwen-7B
# has a similar spike (~160 occupations) but spreads mass more evenly in the 0-0.3 range.
#
# The right half of the scale (0.5-1.0) is almost entirely empty. This is a direct
# consequence of level 2 (LLM independent) almost never being assigned.

# %% [markdown]
# ## Per-task level classification breakdown
#
# The occupation score is derived from per-task classifications (levels 0/1/2).
# If one level completely dominates, the occupation scores will be compressed
# into a narrow band.

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    td = task_details[model]
    total_tasks = td["n_tasks"].sum()
    level_counts = [td["n_level_0"].sum(), td["n_level_1"].sum(), td["n_level_2"].sum()]
    level_pcts = [c / total_tasks * 100 for c in level_counts]

    bars = ax.bar(
        ["Level 0\n(no change)", "Level 1\n(collaboration)", "Level 2\n(LLM independent)"],
        level_pcts,
        color=["#2ca02c", "#ff7f0e", "#d62728"],
        edgecolor="black",
    )
    for bar, pct, cnt in zip(bars, level_pcts, level_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n({cnt:,})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of all task classifications")
    ax.set_title(model)
    ax.set_ylim(0, 100)

fig.suptitle("Task-Level Classification Distribution (all occupations pooled)", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** The root cause of the compressed score range is clear: **level 2 is
# almost never used.** GPT-5.2 assigns level 2 to only 0.5% of tasks (86 out of ~17.5k),
# and Qwen-7B to just 0.4% (44 tasks). The classification is effectively binary (level 0
# vs level 1), which maps to a [0, 0.5] effective score range.
#
# GPT-5.2 assigns level 1 more liberally (64.8%) vs Qwen-7B (53.8%), which explains why
# GPT's mean score (0.32) is higher than Qwen's (0.29).

# %% [markdown]
# ## How many occupations have ANY level 2 tasks?
#
# Level 2 drives the upper half of the score range. If almost no occupations
# have level 2 tasks, the effective score ceiling is ~0.5.

# %%
for model in models:
    td = task_details[model]
    has_l2 = (td["n_level_2"] > 0).sum()
    print(f"{model}: {has_l2}/{len(td)} occupations ({has_l2/len(td)*100:.1f}%) have at least one level 2 task")
    if has_l2 > 0:
        l2_occupations = add_titles(td[td["n_level_2"] > 0]).sort_values("n_level_2", ascending=False)
        print(f"  Top level-2 occupations:")
        for _, row in l2_occupations.head(10).iterrows():
            print(f"    {row['onet_code']} ({row['title']}): "
                  f"{row['n_level_2']} level-2 tasks out of {row['n_tasks']} ({row['pct_level_2']:.1f}%)")
    print()

# %% [markdown]
# **Observation:** Only 5.8% (GPT) / 4.1% (Qwen) of occupations have any level-2 tasks
# at all. The occupations that do get level 2 are concentrated in clerical/administrative
# roles (SOC 43-xxxx): telephone operators, word processors, data entry keyers, mail
# clerks, etc. This makes intuitive sense as these are the most "automatable" roles.
#
# One GPT outlier stands out: 41-9041.00 (Telemarketers) with 9/12 tasks at level 2 and
# a score of 0.875. This is the single occupation that pushes the GPT max well above 0.5.

# %% [markdown]
# ## Occupation scores as a function of level-1 fraction
#
# Since level 2 is so rare, the occupation score should be almost perfectly
# determined by the fraction of tasks classified as level 1. We verify this below.

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    td = task_details[model]
    merged = td.merge(scores[model], on="onet_code")
    ax.scatter(merged["pct_level_1"], merged["task_exposure_mean"],
               alpha=0.4, s=10, color="steelblue")
    ax.set_xlabel("% tasks classified as Level 1 (collaboration)")
    ax.set_ylabel("task_exposure_mean")
    ax.set_title(model)

    # Perfect relationship line: if no level 2 tasks, score = pct_level_1/100 * 0.5
    x = np.linspace(0, 100, 100)
    ax.plot(x, x / 100 * 0.5, color="red", linestyle="--", alpha=0.7, label="theoretical (no L2)")
    ax.legend(fontsize=9)

fig.suptitle("Occupation Score vs Level-1 Task Fraction", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** The points fall almost perfectly on the theoretical red line
# (score = pct_level_1 / 100 * 0.5). This confirms that the score is, in practice,
# just a linear rescaling of the level-1 task fraction. The few points above the line
# are occupations with level-2 tasks (they get a bonus above the 0.5 ceiling).
#
# In other words, the 3-level classification is effectively functioning as a 2-level
# classification. Whether this is acceptable depends on whether the prompts are
# correctly calibrated, or whether the LLMs are simply reluctant to assign level 2.

# %% [markdown]
# ## Score comparison across models
#
# How correlated are the two models' scores? High correlation suggests the
# classification is robust to model choice.

# %%
if len(models) >= 2:
    merged = scores[models[0]].merge(
        scores[models[1]], on="onet_code", suffixes=(f"_{models[0]}", f"_{models[1]}")
    )

    col_a = f"task_exposure_mean_{models[0]}"
    col_b = f"task_exposure_mean_{models[1]}"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged[col_a], merged[col_b], alpha=0.3, s=10, color="steelblue")
    lims = [0, max(merged[col_a].max(), merged[col_b].max()) + 0.02]
    ax.plot(lims, lims, "r--", alpha=0.5, label="y=x")
    ax.set_xlabel(f"task_exposure_mean ({models[0]})")
    ax.set_ylabel(f"task_exposure_mean ({models[1]})")
    ax.set_title("Cross-model Score Comparison")
    ax.set_aspect("equal")
    ax.legend()

    corr = merged[col_a].corr(merged[col_b])
    rank_corr = merged[col_a].corr(merged[col_b], method="spearman")
    mae = (merged[col_a] - merged[col_b]).abs().mean()
    print(f"Pearson r:  {corr:.4f}")
    print(f"Spearman r: {rank_corr:.4f}")
    print(f"MAE:        {mae:.4f}")

    plt.tight_layout()
    plt.show()
else:
    print("Only one model available, skipping cross-model comparison.")

# %% [markdown]
# **Observation:** The two models are strongly correlated (Pearson r=0.93, Spearman
# r=0.92) with low MAE (0.055). The scatter shows:
#
# - Strong agreement in the low-score region (physical occupations both models rate near 0)
# - A visible "grid" pattern near 0.5, where both models assign most tasks to level 1
#   but disagree on the exact count, producing discrete score steps
# - GPT-5.2 tends to score slightly higher (points below the y=x line), consistent with
#   its more liberal level-1 assignment
# - The single GPT outlier at 0.875 (Telemarketers) has no Qwen equivalent (Qwen scores it ~0.5)
#
# Overall the ranking is robust to model choice, though absolute scores differ slightly.

# %% [markdown]
# ## Confidence scores
#
# Mean LLM confidence per occupation. Low or uniform confidence might indicate
# the model is uncertain about its classifications.

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    td = task_details[model]
    ax.hist(td["mean_confidence"], bins=40, edgecolor="black", alpha=0.7, color="mediumpurple")
    ax.axvline(td["mean_confidence"].mean(), color="red", linestyle="--",
               label=f"mean={td['mean_confidence'].mean():.3f}")
    ax.set_xlabel("Mean LLM confidence")
    ax.set_ylabel("Number of occupations")
    ax.set_title(model)
    ax.legend(fontsize=9)

fig.suptitle("Distribution of Mean LLM Confidence per Occupation", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** Both models report mean confidence around 0.82, but with very different
# shapes:
#
# - **GPT-5.2**: Narrow, unimodal distribution in the 0.74-0.94 range. Most occupations
#   cluster around 0.78-0.80. This is reassuringly consistent.
# - **Qwen-7B**: Much wider range (0.4-1.0) with a bimodal shape, one peak near 0.70 and
#   another near 0.95-1.0. The low-confidence tail (0.4-0.6) is concerning and may indicate
#   occupations where Qwen struggled with the classification.
#
# The high-confidence spike at 1.0 in Qwen likely reflects occupations with clear-cut
# physical tasks where the model is very certain of level 0.

# %% [markdown]
# ## Extreme occupations
#
# Which occupations have the highest and lowest exposure scores? Sanity-check
# against intuition: physical/manual jobs should score low, cognitive/information
# jobs should score higher.

# %%
for model in models:
    s = add_titles(scores[model]).sort_values("task_exposure_mean")
    print(f"\n=== {model}: 15 LOWEST exposure scores ===")
    display(s[["onet_code", "title", "task_exposure_mean"]].head(15))
    print(f"\n=== {model}: 15 HIGHEST exposure scores ===")
    display(s[["onet_code", "title", "task_exposure_mean"]].tail(15))

# %% [markdown]
# **Observation:** The extreme occupations pass the intuition test:
#
# **Lowest scores (near 0):** Roofers, rock splitters, dishwashers, welders, hoist
# operators, rail-track layers. These are unambiguously physical/manual roles.
#
# **Highest scores (0.5-0.88):** Telemarketers, order clerks, file clerks, customer
# service representatives, billing clerks, and (for Qwen) software developers and
# data scientists. These are information-processing or administrative roles.
#
# The ordering makes good face-validity sense. The one puzzle is whether Telemarketers
# (GPT: 0.875) should really be the single highest-scoring occupation, since the job
# involves persuasion and human interaction that might resist full automation.

# %% [markdown]
# ## Score density: KDE overlay
#
# Overlay kernel density estimates for both models to compare their distributional
# shapes directly.

# %%
fig, ax = plt.subplots(figsize=(8, 5))

for model, color in zip(models, ["steelblue", "coral"]):
    s = scores[model]["task_exposure_mean"]
    s.plot.kde(ax=ax, label=model, color=color, linewidth=2)

ax.set_xlabel("task_exposure_mean")
ax.set_ylabel("Density")
ax.set_title("Score Density by Model (KDE)")
ax.set_xlim(0, 1)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** The KDE clearly shows the bimodal structure in both models. GPT-5.2
# has a sharper, taller peak at ~0.48 and a smaller left peak at ~0.08. Qwen-7B has
# a broader right peak at ~0.50 and a higher left peak, meaning it classifies more
# occupations as predominantly physical. Both models agree on zero density above 0.6.

# %% [markdown]
# ## Per-level percentage distributions across occupations

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    td = task_details[model]
    data = [td["pct_level_0"], td["pct_level_1"], td["pct_level_2"]]
    bp = ax.boxplot(data, labels=["Level 0", "Level 1", "Level 2"],
                    patch_artist=True)
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("% of tasks in occupation")
    ax.set_title(model)

fig.suptitle("Distribution of Task-Level Percentages Across Occupations", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** The box plots confirm the binary classification pattern. Level 0 and
# level 1 percentages are roughly complementary (wide IQR from ~5% to ~95% for both),
# while level 2 is pinned at 0 with rare outliers. The median level-1 percentage is
# ~73% (GPT) / ~65% (Qwen), reflecting the tendency to classify most tasks as
# "collaboration" mode.

# %% [markdown]
# ## All occupations with level-2 tasks
#
# Full list of every occupation that received at least one level-2 classification,
# sorted by number of level-2 tasks descending.

# %%
for model in models:
    td = task_details[model]
    l2 = add_titles(td[td["n_level_2"] > 0]).sort_values("n_level_2", ascending=False)
    l2_display = l2[["onet_code", "title", "n_level_2", "n_tasks", "pct_level_2"]].copy()
    l2_display.columns = ["O*NET Code", "Title", "L2 Tasks", "Total Tasks", "% L2"]
    print(f"\n=== {model}: {len(l2)} occupations with level-2 tasks ===")
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
        display(l2_display.reset_index(drop=True))

# %% [markdown]
# ## Score distribution of level-2 occupations only
#
# Restrict to just the occupations that have at least one level-2 task and plot
# their score distributions. These are the occupations that push above the 0.5
# ceiling.

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    td = task_details[model]
    l2_codes = set(td.loc[td["n_level_2"] > 0, "onet_code"])
    s = scores[model]
    s_l2 = s[s["onet_code"].isin(l2_codes)]["task_exposure_mean"]

    ax.hist(s_l2, bins=20, edgecolor="black", alpha=0.7, color="indianred")
    ax.axvline(s_l2.mean(), color="red", linestyle="--", label=f"mean={s_l2.mean():.3f}")
    ax.axvline(s_l2.median(), color="orange", linestyle="--", label=f"median={s_l2.median():.3f}")
    ax.set_xlabel("task_exposure_mean")
    ax.set_ylabel("Number of occupations")
    ax.set_title(f"{model} (n={len(s_l2)})")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

fig.suptitle("Score Distribution: Occupations with at Least One Level-2 Task", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** Even among the occupations that *do* have level-2 tasks, most still
# score in the 0.4-0.55 range. The level-2 tasks are too few per occupation to push the
# mean much above 0.5. GPT-5.2's distribution (n=50) peaks around 0.50 with a long left
# tail; the Telemarketer outlier at 0.875 is the only occupation clearly separated from
# the pack. Qwen-7B's subset (n=35) is even more tightly clustered around 0.45-0.55.
#
# This means that even the "most automatable" occupations barely crack the 0.5 mark under
# the current scoring scheme.

# %% [markdown]
# ## Binary exposure: collapsing level 1 and level 2
#
# Since level 2 is so rare, consider a simplified binary scheme where any
# non-zero exposure (level 1 or 2) counts as "exposed" (=1) and level 0 is
# "not exposed" (=0). The occupation score then becomes the fraction of tasks
# that are exposed at all (i.e. `(n_level_1 + n_level_2) / n_tasks`).

# %%
binary_scores = {}
for model in models:
    td = task_details[model].copy()
    td["n_exposed"] = td["n_level_1"] + td["n_level_2"]
    td["binary_exposure"] = td["n_exposed"] / td["n_tasks"]
    binary_scores[model] = td[["onet_code", "binary_exposure", "n_exposed", "n_tasks"]]

    print(f"\n=== {model}: binary exposure stats ===")
    display(binary_scores[model]["binary_exposure"].describe().round(4))

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    s = binary_scores[model]["binary_exposure"]
    ax.hist(s, bins=40, edgecolor="black", alpha=0.7, color="teal")
    ax.axvline(s.mean(), color="red", linestyle="--", label=f"mean={s.mean():.3f}")
    ax.axvline(s.median(), color="orange", linestyle="--", label=f"median={s.median():.3f}")
    ax.set_xlabel("Binary exposure (fraction of tasks with level >= 1)")
    ax.set_ylabel("Number of occupations")
    ax.set_title(model)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

fig.suptitle("Binary Exposure Score Distribution (L1 + L2 collapsed to 'exposed')", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** Collapsing L1 and L2 into a single "exposed" category dramatically
# changes the distribution. The binary score now uses the **full [0, 1] range**, with
# mean 0.64 (GPT) / 0.57 (Qwen) and median 0.75 / 0.64. The distribution is still
# bimodal but now the dominant peak is near 1.0 (occupations where nearly all tasks are
# "exposed"), with a secondary peak near 0.0-0.2.
#
# This is exactly 2x the original 3-level score (since binary_exposure = 2 * task_exposure_mean
# when there are no L2 tasks). The shape is the same, just rescaled to fill the range.

# %% [markdown]
# ### Comparison: 3-level score vs binary score

# %%
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
if len(models) == 1:
    axes = [axes]

for ax, model in zip(axes, models):
    merged = scores[model].merge(binary_scores[model], on="onet_code")
    ax.scatter(merged["binary_exposure"], merged["task_exposure_mean"],
               alpha=0.4, s=10, color="teal")
    ax.set_xlabel("Binary exposure (fraction exposed)")
    ax.set_ylabel("task_exposure_mean (3-level)")
    ax.set_title(model)

    # Theoretical line: if no L2 tasks, 3-level score = binary_exposure * 0.5
    x = np.linspace(0, 1, 100)
    ax.plot(x, x * 0.5, color="red", linestyle="--", alpha=0.7, label="theoretical (no L2)")
    ax.legend(fontsize=9)

fig.suptitle("3-Level Score vs Binary Exposure Score", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Observation:** The 3-level score is almost perfectly a linear function of the binary
# score: `task_exposure_mean = binary_exposure * 0.5`. Points fall tightly on the red
# theoretical line, with only the few L2 occupations sitting slightly above it. This
# confirms that the 3-level and binary scores carry essentially the same information,
# just at different scales.

# %% [markdown]
# ### Binary exposure: KDE overlay

# %%
fig, ax = plt.subplots(figsize=(8, 5))

for model, color in zip(models, ["steelblue", "coral"]):
    s = binary_scores[model]["binary_exposure"]
    s.plot.kde(ax=ax, label=model, color=color, linewidth=2)

ax.set_xlabel("Binary exposure (fraction of tasks with level >= 1)")
ax.set_ylabel("Density")
ax.set_title("Binary Exposure Density by Model (KDE)")
ax.set_xlim(0, 1)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Binary exposure: extreme occupations

# %%
for model in models:
    bs = add_titles(binary_scores[model]).sort_values("binary_exposure")
    print(f"\n=== {model}: 10 LOWEST binary exposure ===")
    display(bs[["onet_code", "title", "binary_exposure"]].head(10))
    print(f"\n=== {model}: 10 HIGHEST binary exposure ===")
    display(bs[["onet_code", "title", "binary_exposure"]].tail(10))

# %% [markdown]
# ## Largest cross-model disagreements
#
# Identify occupations where the two models disagree the most, to understand where
# model sensitivity is concentrated.

# %%
if len(models) >= 2:
    merged = scores[models[0]].merge(
        scores[models[1]], on="onet_code", suffixes=(f"_{models[0]}", f"_{models[1]}")
    )
    col_a = f"task_exposure_mean_{models[0]}"
    col_b = f"task_exposure_mean_{models[1]}"
    merged["abs_diff"] = (merged[col_a] - merged[col_b]).abs()
    merged = add_titles(merged)

    display_cols = ["onet_code", "title", col_a, col_b, "abs_diff"]
    print("Top 20 occupations by absolute score difference:")
    display(merged.sort_values("abs_diff", ascending=False)[display_cols].head(20))

# %% [markdown]
# ## Per-task reasoning analysis
#
# For models with granular task-level results (`task_results.parquet`), we can
# inspect the LLM's reasoning for individual classifications. This section
# samples tasks from each exposure level and shows the reasoning.

# %%
for model in task_results:
    tr = task_results[model]
    print(f"=== {model}: {len(tr)} task classifications ===")
    print(f"Columns: {list(tr.columns)}")
    print(f"Exposure distribution: {tr['exposure'].value_counts().sort_index().to_dict()}")
    print(f"Mean reasoning length: {tr['reasoning'].str.len().mean():.0f} chars")
    print()

# %% [markdown]
# ### Sample level-2 task classifications with reasoning
#
# These are the rarest classifications. Inspect the reasoning to understand
# what makes the LLM decide a task is fully automatable.

# %%
for model in task_results:
    tr = task_results[model]
    l2 = tr[tr["exposure"] == 2].copy()
    print(f"=== {model}: all {len(l2)} level-2 task classifications ===\n")
    for _, row in l2.iterrows():
        print(f"  [{row['onet_code']}] {row['occupation_title']}")
        print(f"  Task: {row['task_text']}")
        print(f"  Reasoning: {row['reasoning']}")
        print(f"  Confidence: {row['confidence']:.2f}")
        print()

# %% [markdown]
# ### Sample level-0 reasoning (highest confidence)
#
# Check that level-0 reasoning cites physical/in-person factors as expected.

# %%
for model in task_results:
    tr = task_results[model]
    l0_top = tr[tr["exposure"] == 0].nlargest(10, "confidence")
    print(f"=== {model}: 10 highest-confidence level-0 classifications ===\n")
    for _, row in l0_top.iterrows():
        print(f"  [{row['onet_code']}] {row['occupation_title']}")
        print(f"  Task: {row['task_text']}")
        print(f"  Reasoning: {row['reasoning']}")
        print(f"  Confidence: {row['confidence']:.2f}")
        print()

# %% [markdown]
# ### Confidence vs exposure level

# %%
for model in task_results:
    tr = task_results[model]
    fig, ax = plt.subplots(figsize=(8, 5))
    for level, color, label in [(0, "#2ca02c", "Level 0"), (1, "#ff7f0e", "Level 1"), (2, "#d62728", "Level 2")]:
        subset = tr[tr["exposure"] == level]["confidence"]
        if len(subset) > 0:
            subset.plot.kde(ax=ax, color=color, label=f"{label} (n={len(subset)})", linewidth=2)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"{model}: Confidence Distribution by Exposure Level")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ### Reasoning length vs exposure level
#
# Does the model write longer reasoning for harder classifications?

# %%
for model in task_results:
    tr = task_results[model].copy()
    tr["reasoning_len"] = tr["reasoning"].str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [tr.loc[tr["exposure"] == lvl, "reasoning_len"] for lvl in [0, 1, 2]]
    bp = ax.boxplot(data, labels=["Level 0", "Level 1", "Level 2"], patch_artist=True)
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Reasoning length (characters)")
    ax.set_title(f"{model}: Reasoning Length by Exposure Level")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary of findings
#
# 1. **Compressed score range**: Both models produce scores in [0, ~0.55], with the
#    upper half of the [0, 1] scale essentially unused. This is because level 2
#    (LLM independent) is assigned to <0.5% of tasks.
#
# 2. **Bimodal distribution**: Scores cluster into two groups: near-zero (physical/manual
#    occupations) and near-0.5 (cognitive/administrative occupations). The gap between
#    ~0.2 and ~0.35 has relatively fewer occupations.
#
# 3. **Effectively binary classification**: The 3-level scheme (0/1/2) is functioning as
#    a 2-level scheme (0/1). The occupation score is almost perfectly predicted by the
#    fraction of tasks classified as level 1.
#
# 4. **Level-2 occupations**: Only 50 (GPT) / 35 (Qwen) occupations have any L2 tasks,
#    concentrated in clerical/administrative roles (SOC 43-xxxx). Even these rarely score
#    above 0.55. One GPT outlier: Telemarketers at 0.875.
#
# 5. **Binary exposure recovers full range**: Collapsing L1+L2 into "exposed" gives a
#    score in the full [0, 1] range (mean 0.64 GPT / 0.57 Qwen). It carries the same
#    ranking information as the 3-level score (just 2x rescaled), but avoids the
#    misleading compression into [0, 0.5].
#
# 6. **Strong cross-model agreement**: Pearson r=0.93, Spearman r=0.92. Rankings are
#    robust to model choice, though GPT-5.2 assigns level 1 more liberally.
#
# 7. **Face validity is good**: Lowest-scoring occupations are physical/manual; highest
#    are clerical/administrative/IT. The ordering passes intuition checks.
#
# 8. **Open question**: Is the near-absence of level 2 a genuine reflection of current
#    AI capabilities (very few tasks are truly automatable end-to-end), or is the prompt
#    calibration too conservative? This deserves investigation by examining the prompt
#    wording and a sample of task-level classifications.
