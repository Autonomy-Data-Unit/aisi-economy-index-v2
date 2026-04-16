# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.score_task_exposure_bt
#
# Pairwise comparison scoring of O\*NET tasks by AI exposure using
# Bradley-Terry model fitting. Instead of absolute 3-level classification,
# the LLM compares pairs of tasks and judges which is more affected by AI.
# A Bradley-Terry model recovers continuous latent scores from the binary
# outcomes. Multi-round adaptive sampling focuses comparisons on items with
# similar latent scores (highest uncertainty).
#
# Node variables:
# - `llm_model` (inherited): Model key from llm_models.toml (default gpt-4.1-mini)
# - `llm_batch_size` (global): Number of prompts per LLM call
# - `llm_max_new_tokens` (inherited): Max tokens per LLM response
# - `n_rounds` (per-node): Number of adaptive sampling rounds
# - `comparisons_per_item_r1` (per-node): Comparisons per item in round 1
# - `comparisons_per_item_r2` (per-node): Comparisons per item in round 2
# - `comparisons_per_item_r3` (per-node): Comparisons per item in round 3
# - `system_prompt` (per-node): Path in config/prompt_library/
# - `user_prompt` (per-node): Path in config/prompt_library/

# %%
#|default_exp score_task_exposure_bt
#|export_as_func true

# %%
#|top_export
from typing import Literal

from pydantic import BaseModel


class PairwiseResult(BaseModel):
    more_exposed: Literal["A", "B", "tie"]

# %%
#|set_func_signature
async def main(ctx, print) -> "pd.DataFrame":
    """Score O*NET tasks by AI exposure using pairwise BT model."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_api'
set_node_func_args('score_task_exposure_bt', run_name=run_name)
show_node_vars('score_task_exposure_bt', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils import strict_format, load_prompt, allm_generate
from ai_index.utils.scoring import OnetScoreSet
from ai_index.utils.bradley_terry import (
    fit_bradley_terry,
    generate_random_pairs,
    generate_adaptive_pairs,
    normalize_scores,
)

# %%
#|export
llm_model = ctx.vars["llm_model"]
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]
batch_size = ctx.vars["llm_batch_size"]
max_new_tokens = ctx.vars["llm_max_new_tokens"]
temperature = ctx.vars["temperature"]
top_p = ctx.vars["top_p"]
top_k = ctx.vars["top_k"]
n_rounds = ctx.vars["n_rounds"]
comparisons_per_item = [
    ctx.vars["comparisons_per_item_r1"],
    ctx.vars["comparisons_per_item_r2"],
    ctx.vars["comparisons_per_item_r3"],
]

SYSTEM_PROMPT = load_prompt(ctx.vars["system_prompt"])
USER_PROMPT_TEMPLATE = load_prompt(ctx.vars["user_prompt"])

output_dir = const.onet_exposure_scores_path / "score_task_exposure_bt" / llm_model
output_dir.mkdir(parents=True, exist_ok=True)

# Skip if output already exists
scores_path = output_dir / "scores.csv"
if scores_path.exists():
    scores = pd.read_csv(scores_path)
    print(f"score_task_exposure_bt: output already exists ({len(scores)} occupations), skipping")
    scores #|func_return_line

# %% [markdown]
# ## Load O\*NET task statements

# %%
#|export
extract_dir = const.onet_store_path / "db_30_0_text"
onet_targets = pd.read_parquet(const.onet_targets_path)
valid_codes = set(onet_targets["O*NET-SOC Code"].tolist())
code_to_title = dict(zip(onet_targets["O*NET-SOC Code"], onet_targets["Title"]))

# Task statements
tasks_raw = pd.read_csv(
    extract_dir / "Task Statements.txt", sep="\t", header=0, encoding="utf-8", dtype=str,
)
tasks_df = tasks_raw[tasks_raw["O*NET-SOC Code"].isin(valid_codes)].copy()
tasks_df["Task ID"] = tasks_df["Task ID"].astype(int)

# Task ratings: Importance (IM)
ratings_raw = pd.read_csv(
    extract_dir / "Task Ratings.txt", sep="\t", header=0, encoding="utf-8", dtype=str,
)
ratings_raw["Data Value"] = pd.to_numeric(ratings_raw["Data Value"], errors="coerce")
ratings_raw["Task ID"] = ratings_raw["Task ID"].astype(int)

im_ratings = ratings_raw[ratings_raw["Scale ID"] == "IM"][
    ["O*NET-SOC Code", "Task ID", "Data Value"]
].rename(columns={"Data Value": "task_importance"})

tasks_df = tasks_df.merge(im_ratings, on=["O*NET-SOC Code", "Task ID"], how="left")
tasks_df["task_importance"] = tasks_df["task_importance"].fillna(0.0)

# %% [markdown]
# ## Build items list

# %%
#|export
items = []
for _, row in tasks_df.iterrows():
    items.append({
        "onet_code": row["O*NET-SOC Code"],
        "title": code_to_title[row["O*NET-SOC Code"]],
        "task_id": row["Task ID"],
        "task_text": row["Task"],
        "importance": float(row["task_importance"]),
    })

n_items = len(items)
print(f"score_task_exposure_bt: {n_items} items across "
      f"{tasks_df['O*NET-SOC Code'].nunique()} occupations")

rng = np.random.default_rng(42)

# %% [markdown]
# ## Multi-round pairwise comparison

# %%
#|export
all_comparisons = []
theta = np.zeros(n_items)

for round_num in range(1, n_rounds + 1):
    cpi_idx = min(round_num - 1, len(comparisons_per_item) - 1)
    cpi = comparisons_per_item[cpi_idx]

    # Generate pairs
    if round_num == 1:
        pairs = generate_random_pairs(n_items, cpi, rng)
    else:
        pairs = generate_adaptive_pairs(theta, cpi, rng)

    print(f"  Round {round_num}: {len(pairs)} pairs (target ~{cpi} per item)")

    # Build prompts with random A/B presentation order to avoid position bias
    prompts = []
    swapped = []
    for a_idx, b_idx in pairs:
        if rng.random() < 0.5:
            # Present in original order
            swapped.append(False)
            prompts.append(strict_format(
                USER_PROMPT_TEMPLATE,
                occupation_a=items[a_idx]["title"],
                task_a=items[a_idx]["task_text"],
                occupation_b=items[b_idx]["title"],
                task_b=items[b_idx]["task_text"],
            ))
        else:
            # Swap: present b as A and a as B
            swapped.append(True)
            prompts.append(strict_format(
                USER_PROMPT_TEMPLATE,
                occupation_a=items[b_idx]["title"],
                task_a=items[b_idx]["task_text"],
                occupation_b=items[a_idx]["title"],
                task_b=items[a_idx]["task_text"],
            ))

    # Call LLM in batches, with retry on failure (split into smaller sub-batches)
    async def _call_batch(batch_prompts):
        _sa = {}
        responses = await allm_generate(
            batch_prompts,
            model=llm_model,
            system_message=SYSTEM_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            json_schema=PairwiseResult.model_json_schema(),
            cache=sbatch_cache,
            time=sbatch_time,
            slurm_accounting=_sa,
        )
        return responses, _sa

    round_responses = []
    slurm_jobs = []
    batch_num = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_num += 1
        try:
            responses, _sa = await _call_batch(batch)
            if _sa:
                slurm_jobs.append(_sa)
            round_responses.extend(responses)
        except Exception as e:
            print(f"    batch {batch_num} failed ({type(e).__name__}), retrying in sub-batches of 1000...")
            sub_size = 1000
            for j in range(0, len(batch), sub_size):
                sub = batch[j : j + sub_size]
                sub_responses, _sa = await _call_batch(sub)
                if _sa:
                    slurm_jobs.append(_sa)
                round_responses.extend(sub_responses)
        print(f"    batch {batch_num}: {len(round_responses)}/{len(prompts)} done")

    # Parse results, unmapping swapped presentation order
    n_parsed = 0
    n_failed = 0
    for idx, raw in enumerate(round_responses):
        try:
            result = PairwiseResult.model_validate_json(raw)
            a_idx, b_idx = pairs[idx]
            outcome = result.more_exposed
            if swapped[idx]:
                # LLM saw (b, a), so "A" means b wins and "B" means a wins
                if outcome == "A":
                    outcome = "B"
                elif outcome == "B":
                    outcome = "A"
                # "tie" stays "tie"
            all_comparisons.append((a_idx, b_idx, outcome))
            n_parsed += 1
        except Exception:
            n_failed += 1

    print(f"  Round {round_num}: {n_parsed} parsed, {n_failed} failed")

    # Fit BT model with all comparisons accumulated so far
    theta = fit_bradley_terry(all_comparisons, n_items)

    print(f"  Round {round_num} theta: mean={theta.mean():.4f}, std={theta.std():.4f}, "
          f"range=[{theta.min():.4f}, {theta.max():.4f}]")

total_comparisons = len(all_comparisons)
print(f"score_task_exposure_bt: {total_comparisons} total comparisons across {n_rounds} rounds")

# %% [markdown]
# ## Normalize and build task-level results

# %%
#|export
task_scores = normalize_scores(theta, floor_cutoff=0.1)

task_results = pd.DataFrame({
    "onet_code": [items[i]["onet_code"] for i in range(n_items)],
    "occupation_title": [items[i]["title"] for i in range(n_items)],
    "task_id": [items[i]["task_id"] for i in range(n_items)],
    "task_text": [items[i]["task_text"] for i in range(n_items)],
    "task_importance": [items[i]["importance"] for i in range(n_items)],
    "bt_theta": theta,
    "bt_score": task_scores,
})

task_results.to_parquet(output_dir / "task_bt_scores.parquet", index=False)
print(f"score_task_exposure_bt: wrote {const.rel(output_dir / 'task_bt_scores.parquet')} "
      f"({n_items} rows)")

# Save all comparisons
comp_records = [
    {"item_a_idx": a, "item_b_idx": b, "outcome": o}
    for a, b, o in all_comparisons
]
comp_df = pd.DataFrame(comp_records)
comp_df.to_parquet(output_dir / "comparisons.parquet", index=False)
print(f"score_task_exposure_bt: wrote {const.rel(output_dir / 'comparisons.parquet')} "
      f"({len(comp_df)} comparisons)")

# %% [markdown]
# ## Aggregate to occupation level

# %%
#|export
g = task_results.groupby("onet_code")

mean_score = g["bt_score"].mean()
task_results["_weighted"] = task_results["bt_score"] * task_results["task_importance"]
weighted_num = g["_weighted"].sum()
imp_sum = g["task_importance"].sum()
imp_weighted = (weighted_num / imp_sum).fillna(mean_score)

scores = pd.DataFrame({
    "onet_code": mean_score.index,
    "task_exposure_bt_mean": mean_score.values,
    "task_exposure_bt_importance_weighted": imp_weighted.values,
})

# Ensure all valid codes present
all_codes = pd.DataFrame({"onet_code": sorted(valid_codes)})
scores = all_codes.merge(scores, on="onet_code", how="left")
n_missing = int(scores["task_exposure_bt_mean"].isna().sum())
if n_missing > 0:
    print(f"  warning: {n_missing} occupations have no tasks (scores will be NaN)")

print(f"score_task_exposure_bt: {len(scores)} occupations")
print(f"  task_exposure_bt_mean: mean={scores['task_exposure_bt_mean'].mean():.4f}, "
      f"std={scores['task_exposure_bt_mean'].std():.4f}")

# %% [markdown]
# ## Save task details

# %%
#|export
details_df = pd.DataFrame({
    "onet_code": g["bt_score"].count().index,
    "n_tasks": g["bt_score"].count().values,
    "mean_bt_score": g["bt_score"].mean().values,
    "std_bt_score": g["bt_score"].std().fillna(0.0).values,
    "min_bt_score": g["bt_score"].min().values,
    "max_bt_score": g["bt_score"].max().values,
})
details_df.to_parquet(output_dir / "task_details.parquet", index=False)
print(f"score_task_exposure_bt: wrote {const.rel(output_dir / 'task_details.parquet')}")

# %% [markdown]
# ## Save OnetScoreSet and return

# %%
#|export
score_set = OnetScoreSet(name="task_exposure_bt", scores=scores)
score_set.save(output_dir)
print(f"score_task_exposure_bt: wrote {const.rel(output_dir / 'scores.csv')}")

if slurm_jobs:
    import json as _json
    _meta = {"slurm_jobs": slurm_jobs, "slurm_total_seconds": sum(j.get("elapsed_seconds", 0) for j in slurm_jobs)}
    with open(output_dir / "score_meta.json", "w") as _f:
        _json.dump(_meta, _f, indent=2)

scores #|func_return_line

# %% [markdown]
# ## Sample output

# %%
merged = scores.merge(
    onet_targets[["O*NET-SOC Code", "Title"]],
    left_on="onet_code", right_on="O*NET-SOC Code", how="left",
)

print(f"\nTop 10 most exposed occupations (BT):")
top = merged.nlargest(10, "task_exposure_bt_mean")
for _, row in top.iterrows():
    print(f"  {row['onet_code']}  mean={row['task_exposure_bt_mean']:.3f}  "
          f"imp_w={row['task_exposure_bt_importance_weighted']:.3f}  {row['Title']}")

print(f"\nBottom 10 least exposed occupations (BT):")
bot = merged.nsmallest(10, "task_exposure_bt_mean")
for _, row in bot.iterrows():
    print(f"  {row['onet_code']}  mean={row['task_exposure_bt_mean']:.3f}  "
          f"imp_w={row['task_exposure_bt_importance_weighted']:.3f}  {row['Title']}")
