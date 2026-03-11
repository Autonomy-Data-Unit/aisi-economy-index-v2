# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.score_felten
#
# Compute Felten AIOE scores per O\*NET occupation using the ability-application
# relatedness methodology from Felten et al. (2021).
#
# 1. Loads the ability-application relatedness matrix (52 abilities × 10 AI
#    applications) from the bundled AIOE Data Appendix (Appendix D).
# 2. Loads O\*NET Abilities table for importance and level scores.
# 3. Computes per-ability AI exposure as a progress-weighted average of
#    relatedness scores across AI applications.
# 4. Builds occupation-ability weight matrix: α × norm(importance) + (1-α) × norm(level).
# 5. Computes per-occupation exposure as a weighted average of ability exposures.
#
# Node variables:
# - `felten_alpha` (float): Weight for importance vs level (default 0.5)
# - `felten_scenario` (str): Progress scenario name (default "baseline_2025")

# %%
#|default_exp nodes.score_felten
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print) -> "pd.DataFrame":
    """Compute Felten AIOE scores per O*NET occupation."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('score_felten', run_name=run_name)
show_node_vars('score_felten', run_name=run_name)

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
from ai_index.utils.scoring import OnetScoreSet

# %%
#|export
alpha = ctx.vars["felten_alpha"]
scenario = ctx.vars["felten_scenario"]

output_dir = const.onet_exposure_scores_path / "score_felten"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Progress scenarios
#
# Weights reflecting estimated AI capability advancement per application area.
# Higher weight = more progress since Felten's 2021 baseline.

# %%
#|export
PROGRESS_SCENARIOS = {
    "felten_original": {
        "abstract_strategy_games": 1.0,
        "realtime_video_games": 1.0,
        "image_recognition": 1.0,
        "visual_question_answering": 1.0,
        "image_generation": 1.0,
        "reading_comprehension": 1.0,
        "language_modeling": 1.0,
        "translation": 1.0,
        "speech_recognition": 1.0,
        "instrumental_track_recognition": 1.0,
    },
    "baseline_2025": {
        "abstract_strategy_games": 1.0,
        "realtime_video_games": 1.3,
        "image_recognition": 1.1,
        "visual_question_answering": 1.5,
        "image_generation": 1.5,
        "reading_comprehension": 1.4,
        "language_modeling": 1.5,
        "translation": 1.3,
        "speech_recognition": 1.1,
        "instrumental_track_recognition": 1.2,
    },
    "conservative_2025": {
        "abstract_strategy_games": 0.90,
        "realtime_video_games": 0.50,
        "image_recognition": 0.82,
        "visual_question_answering": 0.72,
        "image_generation": 0.78,
        "reading_comprehension": 0.82,
        "language_modeling": 0.88,
        "translation": 0.78,
        "speech_recognition": 0.88,
        "instrumental_track_recognition": 0.60,
    },
    "genai_only": {
        "abstract_strategy_games": 0.0,
        "realtime_video_games": 0.0,
        "image_recognition": 0.0,
        "visual_question_answering": 0.0,
        "image_generation": 1.0,
        "reading_comprehension": 0.0,
        "language_modeling": 1.0,
        "translation": 0.0,
        "speech_recognition": 0.0,
        "instrumental_track_recognition": 0.0,
    },
}

progress = PROGRESS_SCENARIOS[scenario]
print(f"score_felten: scenario={scenario}, alpha={alpha}")

# %% [markdown]
# ## Load ability-application relatedness matrix
#
# From Felten et al. (2021) AIOE Data Appendix, Appendix D.
# 52 O\*NET abilities × 10 AI application areas, values in [0, 1].

# %%
#|export
AIOE_APPENDIX_PATH = const.inputs_path / "AIOE_DataAppendix.xlsx"

COLUMN_MAPPING = {
    "Abstract Strategy Games": "abstract_strategy_games",
    "Real-Time Video Games": "realtime_video_games",
    "Image Recognition": "image_recognition",
    "Visual Question Answering": "visual_question_answering",
    "Generating Images": "image_generation",
    "Reading Comprehension": "reading_comprehension",
    "Language Modeling": "language_modeling",
    "Translation": "translation",
    "Speech Recognition": "speech_recognition",
    "Instrumental Track Recognition": "instrumental_track_recognition",
}

ability_app_matrix = pd.read_excel(AIOE_APPENDIX_PATH, sheet_name="Appendix D")
ability_app_matrix = ability_app_matrix.rename(columns={"Unnamed: 0": "ability"})
ability_app_matrix = ability_app_matrix.set_index("ability")
ability_app_matrix = ability_app_matrix.rename(columns=COLUMN_MAPPING)
# Felten appendix uses "Determination" but O*NET 30.0 uses "Discrimination"
ability_app_matrix = ability_app_matrix.rename(index={
    "Visual Color Determination": "Visual Color Discrimination",
})

print(f"  ability-application matrix: {ability_app_matrix.shape}")

# %% [markdown]
# ## Load O\*NET Abilities

# %%
#|export
extract_dir = const.onet_store_path / "db_30_0_text"
onet_targets = pd.read_parquet(const.onet_targets_path)
valid_codes = set(onet_targets["O*NET-SOC Code"].tolist())

abilities_raw = pd.read_csv(
    extract_dir / "Abilities.txt", sep="\t", header=0, encoding="utf-8", dtype=str,
)
abilities_df = abilities_raw[abilities_raw["O*NET-SOC Code"].isin(valid_codes)].copy()
abilities_df["Data Value"] = pd.to_numeric(abilities_df["Data Value"], errors="coerce")
print(f"  abilities: {len(abilities_df)} rows, {len(valid_codes)} occupations")

# %% [markdown]
# ## Step 1: Compute per-ability AI exposure
#
# E[ability] = Σ(Relatedness[ability, app] × Progress[app]) / Σ(Progress[app])

# %%
#|export
apps = [a for a in progress if a in ability_app_matrix.columns]
P = pd.Series({app: progress[app] for app in apps})
A = ability_app_matrix[apps]

# Filter out zero-sum progress (e.g. genai_only has many zeros)
nonzero_apps = [a for a in apps if P[a] > 0]
if not nonzero_apps:
    raise ValueError(f"Progress scenario '{scenario}' has all-zero weights")

P_nz = P[nonzero_apps]
A_nz = A[nonzero_apps]

ability_exposure = (A_nz @ P_nz) / P_nz.sum()
ability_exposure.name = "ability_exposure"
print(f"  ability exposure: {len(ability_exposure)} abilities, "
      f"mean={ability_exposure.mean():.4f}")

# %% [markdown]
# ## Step 2: Build occupation-ability weight matrix
#
# Weight = α × (importance / 5) + (1 - α) × (level / 7)

# %%
#|export
# Pivot importance and level separately
im_df = abilities_df[abilities_df["Scale ID"] == "IM"].copy()
lv_df = abilities_df[abilities_df["Scale ID"] == "LV"].copy()

im_pivot = im_df.pivot_table(
    index="O*NET-SOC Code", columns="Element Name", values="Data Value", aggfunc="first",
)
lv_pivot = lv_df.pivot_table(
    index="O*NET-SOC Code", columns="Element Name", values="Data Value", aggfunc="first",
)

# Align columns
common_abilities = im_pivot.columns.intersection(lv_pivot.columns)
im_norm = (im_pivot[common_abilities] / 5.0).clip(0, 1).fillna(0)
lv_norm = (lv_pivot[common_abilities] / 7.0).clip(0, 1).fillna(0)

weight_matrix = alpha * im_norm + (1 - alpha) * lv_norm
print(f"  weight matrix: {weight_matrix.shape}")

# %% [markdown]
# ## Step 3: Compute per-occupation exposure
#
# E[occ] = Σ(Weight[occ, ability] × E[ability]) / Σ(Weight[occ, ability])

# %%
#|export
# Align abilities between weight matrix and ability exposure
common = weight_matrix.columns.intersection(ability_exposure.index)
print(f"  matched {len(common)} abilities between weight matrix and exposure scores")

W = weight_matrix[common]
E = ability_exposure[common]

numerator = W @ E
denominator = W.sum(axis=1).replace(0, np.nan)
occupation_exposure = (numerator / denominator).clip(0, 1)

# Build output DataFrame
scores = pd.DataFrame({
    "onet_code": occupation_exposure.index,
    "felten_score": occupation_exposure.values,
})
scores["felten_score"] = scores["felten_score"].astype(float)

print(f"score_felten: {len(scores)} occupations, "
      f"mean={scores['felten_score'].mean():.4f}, "
      f"std={scores['felten_score'].std():.4f}")

# %% [markdown]
# ## Save and return

# %%
#|export
score_set = OnetScoreSet(name="felten", scores=scores)
score_set.save(output_dir)
print(f"score_felten: wrote {output_dir / 'scores.csv'}")

scores #|func_return_line

# %% [markdown]
# ## Sample output

# %%
print(f"\nTop 10 most AI-exposed occupations:")
top = scores.nlargest(10, "felten_score").merge(
    onet_targets[["O*NET-SOC Code", "Title"]],
    left_on="onet_code", right_on="O*NET-SOC Code", how="left",
)
for _, row in top.iterrows():
    print(f"  {row['onet_code']}  {row['felten_score']:.4f}  {row['Title']}")

print(f"\nBottom 10 least AI-exposed occupations:")
bot = scores.nsmallest(10, "felten_score").merge(
    onet_targets[["O*NET-SOC Code", "Title"]],
    left_on="onet_code", right_on="O*NET-SOC Code", how="left",
)
for _, row in bot.iterrows():
    print(f"  {row['onet_code']}  {row['felten_score']:.4f}  {row['Title']}")
