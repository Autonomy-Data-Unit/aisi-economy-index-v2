# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.score_presence
#
# Compute humanness/presence scores per O\*NET occupation across three
# dimensions: physical, emotional, and creative.
#
# Each dimension is defined by a curated set of O\*NET elements (Work Context,
# Work Activities, Skills). Elements are normalized to [0, 1] and averaged
# per dimension per occupation.
#
# - Physical: manual dexterity, spatial presence, bodily activity
# - Emotional: social judgment, conflict, leadership, relationships
# - Creative: autonomy, originality, writing
#
# Node variables: none (element sets are methodology, not configuration).

# %%
#|default_exp nodes.score_presence
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print) -> "pd.DataFrame":
    """Compute humanness/presence scores per O*NET occupation."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('score_presence', run_name=run_name)
show_node_vars('score_presence', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Dimension element definitions
#
# Curated sets of O\*NET Element IDs for each humanness dimension.
# Sources: Work Context (4.C.\*), Work Activities / GWAs (4.A.\*), Skills (2.\*).

# %%
#|export
import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils.scoring import OnetScoreSet

# %%
#|export
output_dir = const.onet_exposure_scores_path / "score_presence"
output_dir.mkdir(parents=True, exist_ok=True)

# %%
#|export
PHYSICAL_PRESENCE = {
    "work_context": [
        "4.C.2.a.3",      # Physical Proximity
        "4.C.1.a.2.l",    # Face-to-Face Discussions
        "4.C.1.a.4",      # Contact With Others
        "4.C.2.d.1.b",    # Spend Time Standing
        "4.C.2.d.1.g",    # Spend Time Using Your Hands
        "4.C.2.d.1.h",    # Spend Time Bending or Twisting Body
        "4.C.2.d.1.f",    # Spend Time Keeping or Regaining Balance
        "4.C.2.d.1.d",    # Spend Time Walking or Running
        "4.C.2.d.1.i",    # Spend Time Making Repetitive Motions
        "4.C.2.d.1.e",    # Spend Time Kneeling, Crouching, Stooping
        "4.C.1.b.1.e",    # Work With or Contribute to a Work Group or Team
        "4.C.2.c.1.b",    # Exposed to Disease or Infections
        "4.C.2.a.1.c",    # Outdoors, Exposed to All Weather
        "4.C.1.a.2.c",    # Public Speaking
        "4.C.2.e.1.d",    # Wear Common Protective Equipment
    ],
    "gwas": [
        "4.A.3.a.1",      # Performing General Physical Activities
        "4.A.4.a.8",      # Performing for or Working Directly with the Public
        "4.A.3.a.2",      # Handling and Moving Objects
        "4.A.3.a.4",      # Operating Vehicles, Mechanized Devices, or Equipment
        "4.A.1.b.2",      # Inspecting Equipment, Structures, or Materials
        "4.A.3.a.3",      # Controlling Machines and Processes
    ],
    "skills": [
        "2.B.1.b",        # Coordination
    ],
}

EMOTIONAL_PRESENCE = {
    "work_context": [
        "4.C.1.d.2",      # Dealing With Unpleasant, Angry, or Discourteous People
        "4.C.1.d.1",      # Conflict Situations
        "4.C.1.d.3",      # Dealing with Violent or Physically Aggressive People
        "4.C.1.b.1.f",    # Deal With External Customers or the Public
        "4.C.1.c.1",      # Health and Safety of Other Workers
        "4.C.1.c.2",      # Work Outcomes and Results of Other Workers
        "4.C.3.a.1",      # Consequence of Error
        "4.C.3.a.2.a",    # Impact of Decisions on Co-workers or Company Results
        "4.C.1.b.1.g",    # Coordinate or Lead Others
        "4.C.3.d.1",      # Time Pressure
    ],
    "gwas": [
        "4.A.4.a.5",      # Assisting and Caring for Others
        "4.A.4.a.7",      # Resolving Conflicts and Negotiating with Others
        "4.A.4.a.4",      # Establishing and Maintaining Interpersonal Relationships
        "4.A.4.b.5",      # Coaching and Developing Others
        "4.A.4.b.4",      # Guiding, Directing, and Motivating Subordinates
        "4.A.4.b.3",      # Training and Teaching Others
        "4.A.4.b.6",      # Providing Consultation and Advice to Others
        "4.A.4.a.1",      # Interpreting the Meaning of Information for Others
        "4.A.4.b.2",      # Developing and Building Teams
        "4.A.4.a.3",      # Communicating with People Outside the Organization
        "4.A.4.a.6",      # Selling or Influencing Others
        "4.A.4.a.2",      # Communicating with Supervisors, Peers, or Subordinates
    ],
    "skills": [
        "2.B.1.a",        # Social Perceptiveness
        "2.B.1.f",        # Service Orientation
        "2.B.1.d",        # Negotiation
        "2.A.1.b",        # Active Listening
        "2.A.1.d",        # Speaking
        "2.B.1.c",        # Persuasion
        "2.B.1.e",        # Instructing
        "2.B.4.e",        # Judgment and Decision Making
    ],
}

CREATIVE_EXPRESSION = {
    "work_context": [
        "4.C.3.a.4",      # Freedom to Make Decisions
        "4.C.3.b.8",      # Determine Tasks, Priorities and Goals
        "4.C.3.c.1",      # Level of Competition
    ],
    "gwas": [
        "4.A.2.b.2",      # Thinking Creatively
    ],
    "skills": [
        "2.A.1.c",        # Writing
    ],
}

DIMENSIONS = {
    "physical": PHYSICAL_PRESENCE,
    "emotional": EMOTIONAL_PRESENCE,
    "creative": CREATIVE_EXPRESSION,
}

# %% [markdown]
# ## Load O\*NET tables

# %%
#|export
extract_dir = const.onet_store_path / "db_30_0_text"
onet_targets = pd.read_parquet(const.onet_targets_path)
valid_codes = set(onet_targets["O*NET-SOC Code"].tolist())
print(f"score_presence: {len(valid_codes)} valid occupation codes")


def _load_onet_table(name):
    return pd.read_csv(extract_dir / f"{name}.txt", sep="\t", header=0, encoding="utf-8", dtype=str)

# %% [markdown]
# ## Load and prepare data
#
# Skills, Work Activities, and Work Context tables.
# Standardize column names and filter to valid occupation codes.

# %%
#|export
# Skills (also used for Abilities — same column structure)
skills_raw = _load_onet_table("Skills")
skills_df = skills_raw[skills_raw["O*NET-SOC Code"].isin(valid_codes)].copy()
skills_df["Data Value"] = pd.to_numeric(skills_df["Data Value"], errors="coerce")

# Work Activities (GWAs)
gwas_raw = _load_onet_table("Work Activities")
gwas_df = gwas_raw[gwas_raw["O*NET-SOC Code"].isin(valid_codes)].copy()
gwas_df["Data Value"] = pd.to_numeric(gwas_df["Data Value"], errors="coerce")

# Work Context
wc_raw = _load_onet_table("Work Context")
wc_df = wc_raw[wc_raw["O*NET-SOC Code"].isin(valid_codes)].copy()
wc_df["Data Value"] = pd.to_numeric(wc_df["Data Value"], errors="coerce")

print(f"  Skills: {len(skills_df)} rows")
print(f"  Work Activities: {len(gwas_df)} rows")
print(f"  Work Context: {len(wc_df)} rows")

# %% [markdown]
# ## Scoring functions
#
# For Skills/GWAs: $\text{score} = \frac{\text{norm}(IM) + \text{norm}(LV)}{2}$
#
# For Work Context: $\text{score} = \frac{CX - 1}{4}$

# %%
#|export
def _score_im_lv(df, element_ids):
    """Score elements that have Importance (IM) and Level (LV) scales."""
    im = df[(df["Element ID"].isin(element_ids)) & (df["Scale ID"] == "IM")].copy()
    lv = df[(df["Element ID"].isin(element_ids)) & (df["Scale ID"] == "LV")].copy()

    if im.empty:
        return pd.DataFrame(columns=["O*NET-SOC Code", "score"])

    # Normalize: IM is 1-5 → [0,1], LV is 0-7 → [0,1]
    im["norm"] = ((im["Data Value"] - 1) / 4).clip(0, 1)
    lv["norm"] = (lv["Data Value"] / 7).clip(0, 1)

    # Merge IM and LV per (soc, element)
    im_agg = im.groupby(["O*NET-SOC Code", "Element ID"])["norm"].mean().reset_index()
    im_agg.columns = ["O*NET-SOC Code", "Element ID", "norm_im"]

    lv_agg = lv.groupby(["O*NET-SOC Code", "Element ID"])["norm"].mean().reset_index()
    lv_agg.columns = ["O*NET-SOC Code", "Element ID", "norm_lv"]

    merged = im_agg.merge(lv_agg, on=["O*NET-SOC Code", "Element ID"], how="left")
    merged["element_score"] = (merged["norm_im"] + merged["norm_lv"].fillna(0)) / 2

    # Average across elements per SOC
    return merged.groupby("O*NET-SOC Code")["element_score"].mean().reset_index(name="score")


def _score_work_context(df, element_ids):
    """Score Work Context elements (CX scale: 1-5 → [0,1])."""
    filtered = df[
        (df["Element ID"].isin(element_ids)) & (df["Scale ID"] == "CX")
    ].copy()

    if filtered.empty:
        return pd.DataFrame(columns=["O*NET-SOC Code", "score"])

    filtered["element_score"] = ((filtered["Data Value"] - 1) / 4).clip(0, 1)
    return filtered.groupby("O*NET-SOC Code")["element_score"].mean().reset_index(name="score")


def _compute_dimension(dimension_elements):
    """Compute a single dimension score by averaging across sources."""
    source_scores = []

    # Work Context elements
    wc_ids = dimension_elements.get("work_context", [])
    if wc_ids:
        s = _score_work_context(wc_df, wc_ids)
        if not s.empty:
            source_scores.append(s.rename(columns={"score": "wc"}))

    # GWA elements (from Work Activities table)
    gwa_ids = dimension_elements.get("gwas", [])
    if gwa_ids:
        s = _score_im_lv(gwas_df, gwa_ids)
        if not s.empty:
            source_scores.append(s.rename(columns={"score": "gwa"}))

    # Skill elements (from Skills table)
    skill_ids = dimension_elements.get("skills", [])
    if skill_ids:
        s = _score_im_lv(skills_df, skill_ids)
        if not s.empty:
            source_scores.append(s.rename(columns={"score": "skill"}))

    # Merge and average across sources
    result = pd.DataFrame({"O*NET-SOC Code": sorted(valid_codes)})
    for s in source_scores:
        result = result.merge(s, on="O*NET-SOC Code", how="left")

    score_cols = [c for c in result.columns if c != "O*NET-SOC Code"]
    result["dimension_score"] = result[score_cols].mean(axis=1)
    return result[["O*NET-SOC Code", "dimension_score"]]

# %% [markdown]
# ## Compute all dimensions

# %%
#|export
scores = pd.DataFrame({"onet_code": sorted(valid_codes)})

for dim_name, dim_elements in DIMENSIONS.items():
    dim_scores = _compute_dimension(dim_elements)
    dim_scores = dim_scores.rename(columns={
        "O*NET-SOC Code": "onet_code",
        "dimension_score": f"presence_{dim_name}",
    })
    scores = scores.merge(dim_scores, on="onet_code", how="left")
    col = f"presence_{dim_name}"
    n_elements = sum(len(v) for v in dim_elements.values())
    print(f"  {dim_name}: {n_elements} elements, "
          f"mean={scores[col].mean():.3f}, std={scores[col].std():.3f}")

# Composite: average of the 3 dimensions
scores["presence_composite"] = scores[
    ["presence_physical", "presence_emotional", "presence_creative"]
].mean(axis=1)

print(f"  composite: mean={scores['presence_composite'].mean():.3f}, "
      f"std={scores['presence_composite'].std():.3f}")

# %% [markdown]
# ## Save and return

# %%
#|export
score_set = OnetScoreSet(name="presence", scores=scores)
score_set.save(output_dir)
print(f"score_presence: wrote {const.rel(output_dir / 'scores.csv')} ({len(scores)} occupations)")

scores #|func_return_line

# %% [markdown]
# ## Sample output

# %%
print(f"\nScore distributions:")
for col in ["presence_physical", "presence_emotional", "presence_creative", "presence_composite"]:
    print(f"  {col}: min={scores[col].min():.3f}, max={scores[col].max():.3f}, "
          f"mean={scores[col].mean():.3f}")

print(f"\nTop 5 by composite:")
top = scores.nlargest(5, "presence_composite")
for _, row in top.iterrows():
    print(f"  {row['onet_code']}  P={row['presence_physical']:.2f}  "
          f"E={row['presence_emotional']:.2f}  C={row['presence_creative']:.2f}  "
          f"comp={row['presence_composite']:.2f}")

print(f"\nBottom 5 by composite:")
bot = scores.nsmallest(5, "presence_composite")
for _, row in bot.iterrows():
    print(f"  {row['onet_code']}  P={row['presence_physical']:.2f}  "
          f"E={row['presence_emotional']:.2f}  C={row['presence_creative']:.2f}  "
          f"comp={row['presence_composite']:.2f}")
