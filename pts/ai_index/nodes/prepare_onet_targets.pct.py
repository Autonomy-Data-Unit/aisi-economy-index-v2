# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.prepare_onet_targets
#
# Filter O*NET occupations and build text descriptions for embedding.
#
# 1. Optionally removes 33 public-sector-only occupations (894 → 861)
#    that are not expected on commercial job boards.
# 2. Builds text columns per occupation for downstream embedding:
#    - "Description" = O*NET summary description
#    - "Top_Tasks" = top tasks ranked by direct importance (Task Ratings IM scale)
#    - "Alternate_Titles" = alternate job titles from O*NET
#
# Node variables:
# - `onet_exclude_public_sector` (per-node): Remove 33 public-sector occupations (default true)
# - `onet_top_n` (per-node): Top-N tasks per occupation (default 10)

# %%
#|default_exp nodes.prepare_onet_targets
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print):
    """Filter O*NET occupations and build text descriptions for embedding."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('prepare_onet_targets', run_name=run_name)
show_node_vars('prepare_onet_targets', run_name=run_name)

# %% [markdown]
# # Function body

# %%
#|export
import pandas as pd

from ai_index import const

# %% [markdown]
# ## Public-sector exclusion list
#
# 33 O*NET occupations with statutory authority, coercive power, or
# public-monopoly delivery that do not appear on commercial job boards.

# %%
#|export
PUBLIC_SECTOR_TITLES = [
    # Justice / Courts (8)
    "Judges, Magistrate Judges, and Magistrates",
    "Administrative Law Judges, Adjudicators, and Hearing Officers",
    "Judicial Law Clerks",
    "Coroners",
    "Bailiffs",
    "Probation Officers and Correctional Treatment Specialists",
    "Court, Municipal, and License Clerks",
    "Tax Examiners and Collectors, and Revenue Agents",
    # Policing / Security (10)
    "First-Line Supervisors of Police and Detectives",
    "Detectives and Criminal Investigators",
    "Police Identification and Records Officers",
    "Intelligence Analysts",
    "Fish and Game Wardens",
    "Parking Enforcement Workers",
    "Police and Sheriff's Patrol Officers",
    "Customs and Border Protection Officers",
    "Transit and Railroad Police",
    "Transportation Security Screeners",
    # Fire / Emergency (6)
    "Emergency Management Directors",
    "First-Line Supervisors of Firefighting and Prevention Workers",
    "Firefighters",
    "Fire Inspectors and Investigators",
    "Forest Fire Inspectors and Prevention Specialists",
    "Public Safety Telecommunicators",
    # Corrections (2)
    "First-Line Supervisors of Correctional Officers",
    "Correctional Officers and Jailers",
    # Postal (4)
    "Postmasters and Mail Superintendents",
    "Postal Service Clerks",
    "Postal Service Mail Carriers",
    "Postal Service Mail Sorters, Processors, and Processing Machine Operators",
    # Government (1)
    "Eligibility Interviewers, Government Programs",
    # Inspection (1)
    "Government Property Inspectors and Investigators",
    # Infrastructure (1)
    "Air Traffic Controllers",
]

# %% [markdown]
# ## Read node variables

# %%
#|export
exclude_public_sector = ctx.vars["onet_exclude_public_sector"]
top_n = ctx.vars["onet_top_n"]

# %% [markdown]
# ## Load O*NET tables from disk

# %%
#|export
extract_dir = const.onet_store_path / "db_30_0_text"

def _load_onet_table(name):
    return pd.read_csv(extract_dir / f"{name}.txt", sep="\t", header=0, encoding="utf-8", dtype=str)

occupation_data = _load_onet_table("Occupation Data")
task_statements = _load_onet_table("Task Statements")
task_ratings = _load_onet_table("Task Ratings")
alternate_titles = _load_onet_table("Alternate Titles")

# %% [markdown]
# ## Build top tasks
#
# Rank tasks by their direct Importance (IM) score from Task Ratings,
# not by the skill-sum proxy used in the old pipeline. This surfaces
# the actually important tasks for each occupation.

# %%
#|export
# Join task ratings (IM scale) with task text
tr_im = task_ratings[task_ratings["Scale ID"] == "IM"].copy()
tr_im["Data Value"] = pd.to_numeric(tr_im["Data Value"], errors="coerce")
tr_with_text = tr_im.merge(
    task_statements[["O*NET-SOC Code", "Task ID", "Task"]],
    on=["O*NET-SOC Code", "Task ID"],
)

# Top-N tasks per occupation by direct importance
top_tasks = (
    tr_with_text.sort_values(["O*NET-SOC Code", "Data Value"], ascending=[True, False])
    .groupby("O*NET-SOC Code")
    .head(top_n)
)
tasks_grouped = top_tasks.groupby("O*NET-SOC Code")["Task"].apply(list).reset_index()
tasks_grouped.rename(columns={"Task": "Top_Tasks"}, inplace=True)

# %% [markdown]
# ## Build alternate titles

# %%
#|export
alt_grouped = (
    alternate_titles
    .groupby("O*NET-SOC Code")["Alternate Title"]
    .apply(list)
    .reset_index()
)
alt_grouped.rename(columns={"Alternate Title": "Alternate_Titles"}, inplace=True)

# %% [markdown]
# ## Assemble output DataFrame

# %%
#|export
occ_df = occupation_data.copy()
occ_df = occ_df.merge(tasks_grouped, on="O*NET-SOC Code", how="left")
occ_df = occ_df.merge(alt_grouped, on="O*NET-SOC Code", how="left")
occ_df["Top_Tasks"] = occ_df["Top_Tasks"].apply(lambda x: x if isinstance(x, list) else [])
occ_df["Alternate_Titles"] = occ_df["Alternate_Titles"].apply(lambda x: x if isinstance(x, list) else [])

# Drop occupations with no tasks (mostly "All Other" catch-all categories
# and military roles that produce poor embeddings without task context)
occ_df = occ_df[occ_df["Top_Tasks"].apply(len) > 0]

onet_targets = occ_df[["O*NET-SOC Code", "Title", "Description", "Top_Tasks", "Alternate_Titles"]].copy()

# %% [markdown]
# ## Filter public sector occupations

# %%
#|export
if exclude_public_sector:
    exclude_set = set(PUBLIC_SECTOR_TITLES)
    before = len(onet_targets)
    onet_targets = onet_targets[~onet_targets["Title"].str.strip().isin(exclude_set)].reset_index(drop=True)
    n_dropped = before - len(onet_targets)
    print(f"prepare_onet_targets: dropped {n_dropped} public-sector occupations ({before} → {len(onet_targets)})")

print(f"prepare_onet_targets: {len(onet_targets)} occupations with text descriptions (top_n={top_n})")

onet_targets.to_parquet(const.onet_targets_path, index=False)
print(f"prepare_onet_targets: wrote {const.rel(const.onet_targets_path)}")

# %% [markdown]
# ## Sample output

# %%
print(f"Columns: {list(onet_targets.columns)}")
print(f"Shape: {onet_targets.shape}")
for _, row in onet_targets.head(2).iterrows():
    print(f"\n--- {row['Title']} ---")
    print(f"  Description: {row['Description'][:120]}...")
    print(f"  Top tasks ({len(row['Top_Tasks'])}): {', '.join(row['Top_Tasks'][:3])}...")
    print(f"  Alternate titles ({len(row['Alternate_Titles'])}): {', '.join(row['Alternate_Titles'][:5])}...")
