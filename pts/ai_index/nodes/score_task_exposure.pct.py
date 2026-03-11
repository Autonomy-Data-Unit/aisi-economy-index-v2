# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.score_task_exposure
#
# Classify each O\*NET task statement via LLM into a 3-level AI exposure scale,
# then aggregate to occupation level.
#
# - Level 0 (NO CHANGE): physical, in-person, sensory, relationship-essential
# - Level 1 (HUMAN + LLM COLLABORATION): ≥30% productivity gain, human judgment essential
# - Level 2 (LLM INDEPENDENT): end-to-end with minimal human oversight
#
# Node variables:
# - `llm_model` (inherited global): Model key from llm_models.toml
# - `llm_batch_size` (global): Number of prompts per LLM call
# - `llm_max_new_tokens` (global): Max tokens per LLM response
# - `system_prompt` (per-node): Path in prompt_library/
# - `user_prompt` (per-node): Path in prompt_library/

# %%
#|default_exp nodes.score_task_exposure
#|export_as_func true

# %%
#|top_export
from pydantic import BaseModel, field_validator


class TaskExposureModel(BaseModel):
    occupation: str
    task: str
    exposure: int
    confidence_0to1: float

    @field_validator("exposure")
    @classmethod
    def exposure_in_range(cls, v):
        if v not in (0, 1, 2):
            raise ValueError(f"exposure must be 0, 1, or 2, got {v}")
        return v

    @field_validator("confidence_0to1")
    @classmethod
    def confidence_in_range(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v

# %%
#|set_func_signature
async def main(ctx, print) -> "pd.DataFrame":
    """Classify O*NET tasks by AI exposure level and aggregate to occupations."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('score_task_exposure', run_name=run_name)
show_node_vars('score_task_exposure', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import pandas as pd

from ai_index import const
from ai_index.utils import strict_format, load_prompt, allm_generate
from ai_index.utils.scoring import OnetScoreSet

# %%
#|export
llm_model = ctx.vars["llm_model"]
batch_size = ctx.vars["llm_batch_size"]
max_new_tokens = ctx.vars["llm_max_new_tokens"]

SYSTEM_PROMPT = load_prompt(ctx.vars["system_prompt"])
USER_PROMPT_TEMPLATE = load_prompt(ctx.vars["user_prompt"])

output_dir = const.onet_exposure_scores_path / "score_task_exposure" / llm_model
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load O\*NET task statements
#
# Each task has an occupation code, task ID, and task description.
# Task Ratings provide Importance (IM) and Relevance (RT) scores.

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

print(f"score_task_exposure: {len(tasks_df)} tasks across "
      f"{tasks_df['O*NET-SOC Code'].nunique()} occupations")

# %% [markdown]
# ## Build prompts

# %%
#|export
prompts = []
for _, row in tasks_df.iterrows():
    occupation = code_to_title[row["O*NET-SOC Code"]]
    prompts.append(strict_format(
        USER_PROMPT_TEMPLATE,
        occupation=occupation,
        task=row["Task"],
    ))

print(f"score_task_exposure: built {len(prompts)} prompts")

# %% [markdown]
# ## Call LLM in batches

# %%
#|export
all_responses = []
for i in range(0, len(prompts), batch_size):
    batch = prompts[i : i + batch_size]
    responses = await allm_generate(
        batch,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        json_schema=TaskExposureModel.model_json_schema(),
    )
    all_responses.extend(responses)
    print(f"  batch {i // batch_size + 1}: {len(all_responses)}/{len(prompts)} done")

# %% [markdown]
# ## Parse responses and aggregate to occupation level

# %%
#|export
parsed = []
n_failed = 0
for idx, raw in enumerate(all_responses):
    try:
        result = TaskExposureModel.model_validate_json(raw)
        row = tasks_df.iloc[idx]
        parsed.append({
            "onet_code": row["O*NET-SOC Code"],
            "task_id": row["Task ID"],
            "exposure": result.exposure,
            "confidence": result.confidence_0to1,
            "task_importance": float(row["task_importance"]),
        })
    except Exception:
        n_failed += 1

results_df = pd.DataFrame(parsed)
print(f"score_task_exposure: {len(results_df)} parsed, {n_failed} failed")

# %% [markdown]
# ## Compute occupation-level scores

# %%
#|export
g = results_df.groupby("onet_code")

mean_exp = g["exposure"].mean()
results_df["_weighted"] = results_df["exposure"] * results_df["task_importance"]
weighted_num = g["_weighted"].sum()
imp_sum = g["task_importance"].sum()
imp_weighted = (weighted_num / imp_sum).fillna(mean_exp)

# Normalize from 0-2 scale to [0, 1]
scores = pd.DataFrame({
    "onet_code": mean_exp.index,
    "task_exposure_mean": (mean_exp / 2.0).clip(0, 1).values,
    "task_exposure_importance_weighted": (imp_weighted / 2.0).clip(0, 1).values,
})

# Ensure all valid codes are present
all_codes = pd.DataFrame({"onet_code": sorted(valid_codes)})
scores = all_codes.merge(scores, on="onet_code", how="left").fillna(0.0)

print(f"score_task_exposure: {len(scores)} occupations")
print(f"  task_exposure_mean: mean={scores['task_exposure_mean'].mean():.4f}")
print(f"  task_exposure_importance_weighted: mean={scores['task_exposure_importance_weighted'].mean():.4f}")

# %% [markdown]
# ## Save detailed task breakdown

# %%
#|export
level_counts = g["exposure"].value_counts().unstack(fill_value=0)
for lvl in (0, 1, 2):
    if lvl not in level_counts.columns:
        level_counts[lvl] = 0

details_df = pd.DataFrame({
    "onet_code": g["exposure"].count().index,
    "n_tasks": g["exposure"].count().values,
    "n_level_0": level_counts[0].values,
    "n_level_1": level_counts[1].values,
    "n_level_2": level_counts[2].values,
    "mean_confidence": g["confidence"].mean().values,
})
n = details_df["n_tasks"]
details_df["pct_level_0"] = details_df["n_level_0"] / n * 100
details_df["pct_level_1"] = details_df["n_level_1"] / n * 100
details_df["pct_level_2"] = details_df["n_level_2"] / n * 100

details_df.to_parquet(output_dir / "task_details.parquet", index=False)
print(f"score_task_exposure: wrote {const.rel(output_dir / 'task_details.parquet')}")

# %% [markdown]
# ## Save OnetScoreSet and return

# %%
#|export
score_set = OnetScoreSet(name="task_exposure", scores=scores)
score_set.save(output_dir)
print(f"score_task_exposure: wrote {const.rel(output_dir / 'scores.csv')}")

scores #|func_return_line

# %% [markdown]
# ## Sample output

# %%
merged = scores.merge(
    onet_targets[["O*NET-SOC Code", "Title"]],
    left_on="onet_code", right_on="O*NET-SOC Code", how="left",
)

print(f"\nTop 10 most task-exposed occupations:")
top = merged.nlargest(10, "task_exposure_mean")
for _, row in top.iterrows():
    print(f"  {row['onet_code']}  mean={row['task_exposure_mean']:.3f}  "
          f"imp_w={row['task_exposure_importance_weighted']:.3f}  {row['Title']}")

print(f"\nBottom 10 least task-exposed occupations:")
bot = merged.nsmallest(10, "task_exposure_mean")
for _, row in bot.iterrows():
    print(f"  {row['onet_code']}  mean={row['task_exposure_mean']:.3f}  "
          f"imp_w={row['task_exposure_importance_weighted']:.3f}  {row['Title']}")
