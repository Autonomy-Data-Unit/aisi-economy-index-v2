# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bradley-Terry Pairwise Exposure Scores: Analysis
#
# Analyse the AI exposure scores produced by pairwise comparison + Bradley-Terry
# model fitting (`score_task_exposure_bt` node). Compare against the previous
# absolute 3-level classification (`score_task_exposure`).
#
# The BT approach asks "which of these two tasks is more affected by AI?" across
# ~123k pairwise comparisons, then fits a Bradley-Terry model to recover continuous
# latent exposure scores. This should produce a better-calibrated, continuous
# distribution compared to the degenerate [0, 0.5] range of the absolute method.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai_index.const import onet_exposure_scores_path, inputs_path

# Load BT results
bt_dir = onet_exposure_scores_path / "score_task_exposure_bt" / "gpt-4.1-mini"
bt_scores = pd.read_csv(bt_dir / "scores.csv")
bt_tasks = pd.read_parquet(bt_dir / "task_bt_scores.parquet")
bt_comps = pd.read_parquet(bt_dir / "comparisons.parquet")

# Load old absolute results
abs_dir = onet_exposure_scores_path / "score_task_exposure" / "gpt-5.2"
abs_scores = pd.read_csv(abs_dir / "scores.csv")
abs_tasks = pd.read_parquet(abs_dir / "task_results.parquet")

# O*NET titles
onet_titles = pd.read_csv(
    inputs_path / "onet" / "db_30_0_text" / "Occupation Data.txt",
    sep="\t", usecols=["O*NET-SOC Code", "Title"],
).rename(columns={"O*NET-SOC Code": "onet_code", "Title": "title"})

def add_titles(df):
    return df.merge(onet_titles, on="onet_code", how="left")

print(f"BT: {len(bt_scores)} occupations, {len(bt_tasks)} tasks, {len(bt_comps)} comparisons")
print(f"Absolute: {len(abs_scores)} occupations, {len(abs_tasks)} tasks")

# %% [markdown]
# ## Headline statistics

# %%
print("=== BT scores (occupation level) ===")
display(bt_scores[["task_exposure_bt_mean", "task_exposure_bt_importance_weighted"]].describe().round(4))

print("\n=== Absolute scores (occupation level) ===")
display(abs_scores[["task_exposure_mean", "task_exposure_importance_weighted"]].describe().round(4))

# %% [markdown]
# ## Score distributions: BT vs absolute
#
# The absolute method compresses into [0, 0.5]. Does BT use the full range?

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(bt_scores["task_exposure_bt_mean"], bins=50, edgecolor="black", alpha=0.7,
        color="steelblue", label="BT (GPT-4.1-mini)")
ax.axvline(bt_scores["task_exposure_bt_mean"].mean(), color="red", linestyle="--",
           label=f"mean={bt_scores['task_exposure_bt_mean'].mean():.3f}")
ax.set_xlabel("Exposure score")
ax.set_ylabel("Number of occupations")
ax.set_title("BT Pairwise Scores")
ax.set_xlim(0, 1)
ax.legend()

ax = axes[1]
ax.hist(abs_scores["task_exposure_mean"], bins=50, edgecolor="black", alpha=0.7,
        color="coral", label="Absolute (GPT-5.2)")
ax.axvline(abs_scores["task_exposure_mean"].mean(), color="red", linestyle="--",
           label=f"mean={abs_scores['task_exposure_mean'].mean():.3f}")
ax.set_xlabel("Exposure score")
ax.set_ylabel("Number of occupations")
ax.set_title("Absolute 3-Level Scores")
ax.set_xlim(0, 1)
ax.legend()

fig.suptitle("Occupation-Level Score Distributions", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## KDE overlay

# %%
fig, ax = plt.subplots(figsize=(10, 5))
bt_scores["task_exposure_bt_mean"].plot.kde(ax=ax, color="steelblue", linewidth=2, label="BT")
abs_scores["task_exposure_mean"].plot.kde(ax=ax, color="coral", linewidth=2, label="Absolute")
ax.set_xlabel("Exposure score")
ax.set_ylabel("Density")
ax.set_title("Score Density: BT vs Absolute")
ax.set_xlim(0, 1)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Cross-method comparison
#
# How correlated are the BT and absolute scores? High rank correlation means
# the methods agree on ordering, even if the absolute values differ.

# %%
merged = bt_scores.merge(abs_scores, on="onet_code")

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(merged["task_exposure_mean"], merged["task_exposure_bt_mean"],
           alpha=0.3, s=12, color="steelblue")
ax.set_xlabel("Absolute score (GPT-5.2)")
ax.set_ylabel("BT score (GPT-4.1-mini)")
ax.set_title("BT vs Absolute Scores")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot([0, 1], [0, 1], "r--", alpha=0.3)

pearson = merged["task_exposure_mean"].corr(merged["task_exposure_bt_mean"])
spearman = merged["task_exposure_mean"].corr(merged["task_exposure_bt_mean"], method="spearman")
print(f"Pearson:  {pearson:.4f}")
print(f"Spearman: {spearman:.4f}")

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Comparison outcome distribution
#
# How often does the LLM pick A, B, or tie?

# %%
outcome_counts = bt_comps["outcome"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(outcome_counts.index, outcome_counts.values,
              color=["steelblue", "coral", "gray"], edgecolor="black")
for bar, count in zip(bars, outcome_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{count:,}\n({count/len(bt_comps)*100:.1f}%)", ha="center", fontsize=10)
ax.set_ylabel("Number of comparisons")
ax.set_title("Pairwise Comparison Outcomes")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Comparisons per item
#
# Each item (occupation-task pair) should have participated in ~12-15
# comparisons. Check uniformity.

# %%
item_counts = np.zeros(len(bt_tasks), dtype=int)
np.add.at(item_counts, bt_comps["item_a_idx"].values, 1)
np.add.at(item_counts, bt_comps["item_b_idx"].values, 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(item_counts, bins=range(item_counts.min(), item_counts.max() + 2),
        edgecolor="black", alpha=0.7, color="mediumpurple")
ax.axvline(item_counts.mean(), color="red", linestyle="--",
           label=f"mean={item_counts.mean():.1f}")
ax.set_xlabel("Number of comparisons per item")
ax.set_ylabel("Number of items")
ax.set_title("Comparison Count Distribution")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Task-level score distribution

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(bt_tasks["bt_score"], bins=60, edgecolor="black", alpha=0.7, color="teal")
ax.axvline(bt_tasks["bt_score"].mean(), color="red", linestyle="--",
           label=f"mean={bt_tasks['bt_score'].mean():.3f}")
ax.set_xlabel("BT score (task level)")
ax.set_ylabel("Number of tasks")
ax.set_title("Task-Level BT Score Distribution")
ax.set_xlim(0, 1)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Extreme occupations

# %%
s = add_titles(bt_scores).sort_values("task_exposure_bt_mean")
print("=== 15 LOWEST BT exposure scores ===")
display(s[["onet_code", "title", "task_exposure_bt_mean"]].head(15))
print("\n=== 15 HIGHEST BT exposure scores ===")
display(s[["onet_code", "title", "task_exposure_bt_mean"]].tail(15))

# %% [markdown]
# ## Extreme tasks
#
# Which individual tasks score highest and lowest?

# %%
t = bt_tasks.sort_values("bt_score")
print("=== 15 LOWEST scoring tasks ===")
for _, row in t.head(15).iterrows():
    print(f"  {row['bt_score']:.3f}  [{row['occupation_title']}] {row['task_text'][:80]}")

print("\n=== 15 HIGHEST scoring tasks ===")
for _, row in t.tail(15).iterrows():
    print(f"  {row['bt_score']:.3f}  [{row['occupation_title']}] {row['task_text'][:80]}")

# %% [markdown]
# ## Spotlight: Petroleum Engineers
#
# Under the absolute method, petroleum engineers had 22/23 tasks at level 1,
# scoring 0.500. The BT score should be more nuanced.

# %%
petro_tasks = bt_tasks[bt_tasks["onet_code"] == "17-2171.00"].sort_values("bt_score", ascending=False)
petro_occ = bt_scores[bt_scores["onet_code"] == "17-2171.00"]
print(f"Petroleum Engineers occupation score: {petro_occ['task_exposure_bt_mean'].values[0]:.3f}")
print(f"  (absolute method was: {abs_scores[abs_scores['onet_code'] == '17-2171.00']['task_exposure_mean'].values[0]:.3f})")
print(f"\nTask-level scores ({len(petro_tasks)} tasks):")
for _, row in petro_tasks.iterrows():
    print(f"  {row['bt_score']:.3f}  {row['task_text'][:90]}")

# %% [markdown]
# ## Within-occupation score spread
#
# The BT model gives each task its own score. How much variation is there
# within occupations? Wide spread means the method discriminates between
# tasks within a single occupation.

# %%
occ_spread = bt_tasks.groupby("onet_code").agg(
    mean=("bt_score", "mean"),
    std=("bt_score", "std"),
    min=("bt_score", "min"),
    max=("bt_score", "max"),
    range=("bt_score", lambda x: x.max() - x.min()),
    n_tasks=("bt_score", "count"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(occ_spread["range"], bins=40, edgecolor="black", alpha=0.7, color="steelblue")
ax.set_xlabel("Within-occupation score range (max - min)")
ax.set_ylabel("Number of occupations")
ax.set_title("Within-Occupation Task Score Spread")

ax = axes[1]
ax.scatter(occ_spread["mean"], occ_spread["std"], alpha=0.4, s=10, color="steelblue")
ax.set_xlabel("Occupation mean BT score")
ax.set_ylabel("Occupation std of BT scores")
ax.set_title("Score Variability vs Mean")
fig.tight_layout()
plt.show()

print(f"Median within-occupation range: {occ_spread['range'].median():.3f}")
print(f"Median within-occupation std: {occ_spread['std'].median():.3f}")

# %% [markdown]
# ## Largest movers: BT vs absolute
#
# Which occupations changed rank the most between methods?

# %%
merged_ranked = add_titles(merged).copy()
merged_ranked["rank_bt"] = merged_ranked["task_exposure_bt_mean"].rank(ascending=False)
merged_ranked["rank_abs"] = merged_ranked["task_exposure_mean"].rank(ascending=False)
merged_ranked["rank_change"] = (merged_ranked["rank_abs"] - merged_ranked["rank_bt"]).abs()

print("Top 20 occupations by rank change (BT vs absolute):")
display(merged_ranked.nlargest(20, "rank_change")[
    ["onet_code", "title", "task_exposure_mean", "task_exposure_bt_mean",
     "rank_abs", "rank_bt", "rank_change"]
])

# %% [markdown]
# ## BT score by SOC major group
#
# Group occupations by their 2-digit SOC code to see exposure patterns
# across broad occupational categories.

# %%
bt_with_titles = add_titles(bt_scores).copy()
bt_with_titles["soc_major"] = bt_with_titles["onet_code"].str[:2]

soc_labels = {
    "11": "Management", "13": "Business/Financial", "15": "Computer/Math",
    "17": "Architecture/Engineering", "19": "Life/Physical/Social Science",
    "21": "Community/Social Service", "23": "Legal", "25": "Education",
    "27": "Arts/Media", "29": "Healthcare Practitioners", "31": "Healthcare Support",
    "33": "Protective Service", "35": "Food Preparation", "37": "Cleaning/Maintenance",
    "39": "Personal Care", "41": "Sales", "43": "Office/Administrative",
    "45": "Farming/Fishing", "47": "Construction", "49": "Installation/Repair",
    "51": "Production", "53": "Transportation",
}

group_stats = bt_with_titles.groupby("soc_major")["task_exposure_bt_mean"].agg(["mean", "median", "count"])
group_stats["label"] = group_stats.index.map(soc_labels)
group_stats = group_stats.sort_values("mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = range(len(group_stats))
ax.barh(y_pos, group_stats["mean"], color="steelblue", edgecolor="black", alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{idx} {row['label']} (n={row['count']:.0f})"
                     for idx, row in group_stats.iterrows()], fontsize=9)
ax.set_xlabel("Mean BT Exposure Score")
ax.set_title("Mean AI Exposure by SOC Major Group")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Duplicate pair analysis
#
# Check how many pairs were compared more than once (across rounds).

# %%
pair_key = bt_comps.apply(lambda r: (min(r["item_a_idx"], r["item_b_idx"]),
                                      max(r["item_a_idx"], r["item_b_idx"])), axis=1)
pair_counts = pair_key.value_counts()
n_dup = (pair_counts > 1).sum()
print(f"Unique pairs: {len(pair_counts):,}")
print(f"Pairs compared >1 time: {n_dup} ({n_dup/len(pair_counts)*100:.2f}%)")

# %% [markdown]
# ## Convergence analysis
#
# Did the scores stabilize across rounds? We refit the BT model using
# cumulative comparisons after each round and measure how much the scores
# change between successive fits. Convergence means later rounds produce
# negligible changes to the ranking.

# %%
from scipy.stats import spearmanr
from ai_index.utils.bradley_terry import fit_bradley_terry, normalize_scores

# Identify round boundaries from the comparisons file
# Known round sizes from the run logs
known_round_sizes = [43867, 45155, 34473, 34423, 34230]
n_total = len(bt_comps)

# Build cumulative boundaries, stopping when we exceed total comparisons
round_boundaries = []
cumulative = 0
for size in known_round_sizes:
    cumulative += size
    if cumulative <= n_total:
        round_boundaries.append(cumulative)
    else:
        break

# If there are comparisons beyond the known rounds, add the total as the final boundary
if round_boundaries[-1] < n_total:
    round_boundaries.append(n_total)

print(f"Total comparisons: {n_total:,}")
print(f"Round boundaries: {round_boundaries}")

# %%
# Fit BT model at each round boundary and track convergence
round_thetas = []
round_scores = []
round_labels = []

for idx, end in enumerate(round_boundaries):
    comp_list = [
        (int(r["item_a_idx"]), int(r["item_b_idx"]), r["outcome"])
        for _, r in bt_comps.iloc[:end].iterrows()
    ]
    theta = fit_bradley_terry(comp_list, len(bt_tasks))
    score = normalize_scores(theta, floor_cutoff=0.1)
    round_thetas.append(theta)
    round_scores.append(score)
    round_labels.append(f"After {end:,} comps")
    print(f"  {round_labels[-1]}: theta std={theta.std():.3f}, score mean={score.mean():.3f}")

# %%
# Pairwise convergence metrics between successive rounds
print("\n=== Convergence between successive rounds ===\n")
print(f"{'Comparison':<45s} {'Spearman':>10s} {'Pearson':>10s} {'MAE':>8s}")
print("-" * 75)

for i in range(1, len(round_scores)):
    sp, _ = spearmanr(round_scores[i - 1], round_scores[i])
    pe = np.corrcoef(round_scores[i - 1], round_scores[i])[0, 1]
    mae = np.abs(round_scores[i - 1] - round_scores[i]).mean()
    label = f"{round_labels[i-1]} vs {round_labels[i]}"
    print(f"{label:<45s} {sp:>10.6f} {pe:>10.6f} {mae:>8.4f}")

# Also compare each round to final
print(f"\n{'Comparison to final':<45s} {'Spearman':>10s} {'Pearson':>10s} {'MAE':>8s}")
print("-" * 75)
final_scores = round_scores[-1]
for i in range(len(round_scores) - 1):
    sp, _ = spearmanr(round_scores[i], final_scores)
    pe = np.corrcoef(round_scores[i], final_scores)[0, 1]
    mae = np.abs(round_scores[i] - final_scores).mean()
    print(f"{round_labels[i]:<45s} {sp:>10.6f} {pe:>10.6f} {mae:>8.4f}")

# %%
# Plot convergence: Spearman between successive rounds
successive_spearman = []
for i in range(1, len(round_scores)):
    sp, _ = spearmanr(round_scores[i - 1], round_scores[i])
    successive_spearman.append(sp)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(range(2, len(successive_spearman) + 2), successive_spearman, "o-", color="steelblue", linewidth=2)
ax.set_xlabel("Round")
ax.set_ylabel("Spearman correlation with previous round")
ax.set_title("Score Convergence (successive rounds)")
ax.set_ylim(0.95, 1.0)
ax.axhline(0.995, color="red", linestyle="--", alpha=0.5, label="0.995 threshold")
ax.legend()

# Score change distribution for last two rounds
ax = axes[1]
if len(round_scores) >= 2:
    delta = round_scores[-1] - round_scores[-2]
    ax.hist(delta, bins=50, edgecolor="black", alpha=0.7, color="mediumpurple")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Score change (last round)")
    ax.set_ylabel("Number of tasks")
    ax.set_title(f"Score Change Distribution (MAE={np.abs(delta).mean():.4f})")

fig.suptitle("Bradley-Terry Convergence Diagnostics", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Physical-task spike analysis
#
# The task-level distribution shows a sharp spike at 0.10-0.15 (before cutoff).
# This is a cluster of purely physical tasks that the LLM cannot distinguish
# from each other, producing very high tie rates in within-cluster comparisons.

# %%
# Compute scores WITHOUT the cutoff to show the spike
scores_no_cutoff = normalize_scores(bt_tasks["bt_theta"].values)

# Within-spike tie rate
spike_idx_set = set(bt_tasks[scores_no_cutoff <= 0.1].index)
both_spike = bt_comps[
    bt_comps["item_a_idx"].isin(spike_idx_set) & bt_comps["item_b_idx"].isin(spike_idx_set)
]
tie_rate = (both_spike["outcome"] == "tie").mean() if len(both_spike) > 0 else 0

print(f"Tasks in spike (score <= 0.1 before cutoff): {len(spike_idx_set)} ({len(spike_idx_set)/len(bt_tasks)*100:.1f}%)")
print(f"Within-spike comparisons: {len(both_spike)}")
print(f"Within-spike tie rate: {tie_rate*100:.1f}%")

# SOC groups in spike
spike_tasks = bt_tasks.loc[list(spike_idx_set)].copy()
spike_tasks["soc_major"] = spike_tasks["onet_code"].str[:2]
soc_labels = {
    "11": "Management", "13": "Business/Financial", "15": "Computer/Math",
    "17": "Arch/Engineering", "19": "Science", "21": "Community/Social",
    "23": "Legal", "25": "Education", "27": "Arts/Media",
    "29": "Healthcare Practitioners", "31": "Healthcare Support",
    "33": "Protective Service", "35": "Food Prep", "37": "Cleaning",
    "39": "Personal Care", "41": "Sales", "43": "Office/Admin",
    "45": "Farming", "47": "Construction", "49": "Install/Repair",
    "51": "Production", "53": "Transportation",
}
print("\nSOC groups in spike:")
for code, count in spike_tasks["soc_major"].value_counts().head(8).items():
    print(f"  {soc_labels.get(code, code)}: {count}")

# %% [markdown]
# ### Before and after floor cutoff
#
# We apply a floor cutoff at 0.1 (on the min-max scale): all tasks at or
# below this threshold are set to 0, and the remaining tasks are rescaled to
# [0, 1]. This collapses the physical-task tie cluster to zero.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Task-level before
ax = axes[0, 0]
ax.hist(scores_no_cutoff, bins=60, edgecolor="black", alpha=0.7, color="teal")
ax.axvline(scores_no_cutoff.mean(), color="red", linestyle="--", label=f"mean={scores_no_cutoff.mean():.3f}")
ax.axvline(0.1, color="black", linestyle=":", linewidth=2, label="cutoff=0.1")
ax.set_xlabel("BT score")
ax.set_ylabel("Number of tasks")
ax.set_title("Task-Level (before cutoff)")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)

# Task-level after
ax = axes[0, 1]
ax.hist(bt_tasks["bt_score"], bins=60, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(bt_tasks["bt_score"].mean(), color="red", linestyle="--", label=f"mean={bt_tasks['bt_score'].mean():.3f}")
ax.set_xlabel("BT score (rescaled)")
ax.set_ylabel("Number of tasks")
ax.set_title("Task-Level (after cutoff)")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)

# Occupation-level before
occ_no_cutoff = pd.DataFrame({"onet_code": bt_tasks["onet_code"], "s": scores_no_cutoff}).groupby("onet_code")["s"].mean()
ax = axes[1, 0]
ax.hist(occ_no_cutoff.values, bins=50, edgecolor="black", alpha=0.7, color="teal")
ax.axvline(occ_no_cutoff.mean(), color="red", linestyle="--", label=f"mean={occ_no_cutoff.mean():.3f}")
ax.set_xlabel("BT score")
ax.set_ylabel("Number of occupations")
ax.set_title("Occupation-Level (before cutoff)")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)

# Occupation-level after
ax = axes[1, 1]
ax.hist(bt_scores["task_exposure_bt_mean"].values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(bt_scores["task_exposure_bt_mean"].mean(), color="red", linestyle="--",
           label=f"mean={bt_scores['task_exposure_bt_mean'].mean():.3f}")
ax.set_xlabel("BT score (rescaled)")
ax.set_ylabel("Number of occupations")
ax.set_title("Occupation-Level (after cutoff)")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)

fig.suptitle("Score Distributions: Before and After Physical-Task Floor Cutoff (0.1)", fontsize=14)
fig.tight_layout()
plt.show()

# %%
# Occupations with all tasks zeroed
occ_zero = bt_scores[bt_scores["task_exposure_bt_mean"] == 0]
if len(occ_zero) > 0:
    occ_zero_titled = add_titles(occ_zero)
    print(f"Occupations with score = 0 (all tasks below cutoff): {len(occ_zero)}")
    for _, row in occ_zero_titled.iterrows():
        n_tasks = len(bt_tasks[bt_tasks["onet_code"] == row["onet_code"]])
        print(f"  {row['onet_code']}  ({n_tasks} tasks) {row['title']}")

# %% [markdown]
# ## Cutoff experimentation
#
# Adjust `CUTOFF` below and rerun to see how different floor cutoffs affect
# the task and occupation score distributions. The raw BT theta values are
# fixed; only the post-processing normalization changes.

# %%
CUTOFF = 0.1  # <-- change this to experiment

exp_scores = normalize_scores(bt_tasks["bt_theta"].values, floor_cutoff=CUTOFF)

# Task-level stats
n_zeroed = (exp_scores == 0).sum()
print(f"Cutoff: {CUTOFF}")
print(f"Tasks zeroed: {n_zeroed:,} ({n_zeroed/len(exp_scores)*100:.1f}%)")
print(f"Task scores: mean={exp_scores.mean():.4f}, std={exp_scores.std():.4f}, "
      f"range=[{exp_scores.min():.4f}, {exp_scores.max():.4f}]")

# Occupation-level aggregation
_exp_df = bt_tasks[["onet_code", "task_importance"]].copy()
_exp_df["bt_score"] = exp_scores
_g = _exp_df.groupby("onet_code")
_mean = _g["bt_score"].mean()
_exp_df["_w"] = _exp_df["bt_score"] * _exp_df["task_importance"]
_wnum = _g["_w"].sum()
_isum = _g["task_importance"].sum()
_imp_w = (_wnum / _isum).fillna(_mean)

exp_occ = pd.DataFrame({
    "onet_code": _mean.index,
    "bt_mean": _mean.values,
    "bt_imp_weighted": _imp_w.values,
})

n_occ_zero = (exp_occ["bt_mean"] == 0).sum()
print(f"\nOccupations: {len(exp_occ)}")
print(f"Occupations at 0: {n_occ_zero}")
print(f"Occupation scores: mean={exp_occ['bt_mean'].mean():.4f}, std={exp_occ['bt_mean'].std():.4f}, "
      f"range=[{exp_occ['bt_mean'].min():.4f}, {exp_occ['bt_mean'].max():.4f}]")

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Task-level: no cutoff
ax = axes[0, 0]
ax.hist(scores_no_cutoff, bins=60, edgecolor="black", alpha=0.7, color="teal")
ax.axvline(CUTOFF, color="black", linestyle=":", linewidth=2, label=f"cutoff={CUTOFF}")
ax.set_xlabel("BT score")
ax.set_ylabel("Number of tasks")
ax.set_title("Task-Level (no cutoff)")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)

# Task-level: with cutoff
ax = axes[0, 1]
ax.hist(exp_scores, bins=60, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(exp_scores.mean(), color="red", linestyle="--", label=f"mean={exp_scores.mean():.3f}")
ax.set_xlabel("BT score (rescaled)")
ax.set_ylabel("Number of tasks")
ax.set_title(f"Task-Level (cutoff={CUTOFF})")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)

# Occupation-level: no cutoff
occ_no = pd.DataFrame({"onet_code": bt_tasks["onet_code"], "s": scores_no_cutoff}).groupby("onet_code")["s"].mean()
ax = axes[1, 0]
ax.hist(occ_no.values, bins=50, edgecolor="black", alpha=0.7, color="teal")
ax.set_xlabel("BT score")
ax.set_ylabel("Number of occupations")
ax.set_title("Occupation-Level (no cutoff)")
ax.set_xlim(0, 1)

# Occupation-level: with cutoff
ax = axes[1, 1]
ax.hist(exp_occ["bt_mean"].values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
ax.axvline(exp_occ["bt_mean"].mean(), color="red", linestyle="--",
           label=f"mean={exp_occ['bt_mean'].mean():.3f}")
ax.set_xlabel("BT score (rescaled)")
ax.set_ylabel("Number of occupations")
ax.set_title(f"Occupation-Level (cutoff={CUTOFF})")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)

fig.suptitle(f"Cutoff Experiment: floor_cutoff={CUTOFF}", fontsize=14)
fig.tight_layout()
plt.show()

# %%
# Show occupations that would be zeroed at this cutoff
exp_zero = exp_occ[exp_occ["bt_mean"] == 0]
if len(exp_zero) > 0:
    exp_zero_titled = exp_zero.merge(onet_titles, on="onet_code", how="left")
    print(f"Occupations with score = 0 at cutoff={CUTOFF}: {len(exp_zero)}")
    for _, row in exp_zero_titled.iterrows():
        n_tasks = len(bt_tasks[bt_tasks["onet_code"] == row["onet_code"]])
        print(f"  {row['onet_code']}  ({n_tasks} tasks) {row['title']}")
else:
    print(f"No occupations fully zeroed at cutoff={CUTOFF}")

# Top/bottom 10 with this cutoff
exp_titled = exp_occ.merge(onet_titles, on="onet_code", how="left").sort_values("bt_mean")
print(f"\nBottom 10 (cutoff={CUTOFF}):")
for _, row in exp_titled.head(10).iterrows():
    print(f"  {row['bt_mean']:.4f}  {row['title']}")
print(f"\nTop 10 (cutoff={CUTOFF}):")
for _, row in exp_titled.tail(10).iterrows():
    print(f"  {row['bt_mean']:.4f}  {row['title']}")

# %% [markdown]
# ## Summary
#
# Key findings from the BT pairwise comparison approach compared to absolute
# 3-level classification.
