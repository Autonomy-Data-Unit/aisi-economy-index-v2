# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build O*NET Descriptions
#
# Build standard occupation descriptions (894 rows: SOC code, title, description,
# tasks/skills text) from raw O*NET data. Used as input to both the embedding
# and cosine similarity nodes.

# %%
#|default_exp nodes.build_onet_descriptions
#|export_as_func true

# %%
#|set_func_signature
def build_onet_descriptions(onet_tables, ctx, print) -> {"descriptions": dict}:
    """Build standard occupation descriptions from raw O*NET data."""
    ...

# %%
#|export
import pandas as pd

top_n = 10

occupation_data = onet_tables["Occupation Data"]
task_statements = onet_tables["Task Statements"]
skills = onet_tables["Skills"]
work_activities = onet_tables["Work Activities"]

print(f"build_onet_descriptions: {len(occupation_data)} occupations, top_n={top_n}")

# %%
#|export
# Top tasks per occupation (ranked by IM-scale skill scores)
occupation_tasks = occupation_data[["O*NET-SOC Code", "Title"]].merge(
    task_statements[["O*NET-SOC Code", "Task"]], on="O*NET-SOC Code", how="left"
)
occupation_tasks_skills = occupation_tasks.merge(
    skills[["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"]],
    on="O*NET-SOC Code", how="left"
)
occupation_tasks_skills = occupation_tasks_skills[occupation_tasks_skills["Scale ID"] == "IM"]
occupation_tasks_skills["Data Value"] = occupation_tasks_skills["Data Value"].astype(float)

grouped = occupation_tasks_skills.groupby(["Title", "Task"])["Data Value"].sum().reset_index()
top_tasks = (
    grouped.sort_values(["Title", "Data Value"], ascending=[True, False])
    .groupby("Title").head(top_n)
)
tasks_grouped = top_tasks.groupby("Title")["Task"].apply(list).reset_index()
tasks_grouped.rename(columns={"Task": "Top_Tasks"}, inplace=True)

# %%
#|export
# Top skills per occupation (IM scale)
occ_skills = occupation_data[["O*NET-SOC Code", "Title"]].merge(
    skills[["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"]],
    on="O*NET-SOC Code", how="left"
)
occ_skills = occ_skills[occ_skills["Scale ID"] == "IM"].copy()
occ_skills["Data Value"] = pd.to_numeric(occ_skills["Data Value"], errors="coerce")
occ_skills = occ_skills.dropna(subset=["Data Value"])
grouped_skills = occ_skills.groupby(["Title", "Element Name"], as_index=False)["Data Value"].sum()
top_skills = (
    grouped_skills.sort_values(["Title", "Data Value"], ascending=[True, False])
    .groupby("Title").head(top_n)
)
skills_grouped = top_skills.groupby("Title")["Element Name"].apply(list).reset_index()
skills_grouped.rename(columns={"Element Name": "Top_Skills"}, inplace=True)

# %%
#|export
# Top work activities per occupation (IM scale)
occ_activities = occupation_data[["O*NET-SOC Code", "Title"]].merge(
    work_activities[["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"]],
    on="O*NET-SOC Code", how="left"
)
occ_activities = occ_activities[occ_activities["Scale ID"] == "IM"]
occ_activities["Data Value"] = occ_activities["Data Value"].astype(float)
grouped_activities = occ_activities.groupby(["Title", "Element Name"])["Data Value"].sum().reset_index()
top_activities = (
    grouped_activities.sort_values(["Title", "Data Value"], ascending=[True, False])
    .groupby("Title").head(top_n)
)
activities_grouped = top_activities.groupby("Title")["Element Name"].apply(list).reset_index()
activities_grouped.rename(columns={"Element Name": "Top_Activities"}, inplace=True)

# %%
#|export
# Merge tasks + activities + skills, build combined text fields
merged = tasks_grouped.merge(activities_grouped, on="Title", how="outer")
merged = merged.merge(skills_grouped, on="Title", how="outer")
for col in ["Top_Tasks", "Top_Activities", "Top_Skills"]:
    merged[col] = merged[col].apply(lambda x: x if isinstance(x, list) else [])
merged["Combined"] = merged["Top_Tasks"] + merged["Top_Activities"] + merged["Top_Skills"]
merged.index = merged["Title"]

# Attach to occupation data and build output
occ = occupation_data.copy()
occ["Combined"] = occ["Title"].map(merged["Combined"].to_dict())
occ = occ.dropna(subset=["Combined"])

# Build role description: "Title - Description"
occ["role_description"] = occ["Title"] + " - " + occ["Description"].astype(str)
# Build task description: "Title - task1, task2, ..."
occ["task_description"] = occ["Title"] + " - " + occ["Combined"].apply(
    lambda items: ", ".join(str(x) for x in items)
)

soc_codes = occ["O*NET-SOC Code"].tolist()
titles = occ["Title"].tolist()
role_descriptions = occ["role_description"].tolist()
task_descriptions = occ["task_description"].tolist()

print(f"build_onet_descriptions: built {len(titles)} occupation descriptions")
return {"descriptions": {
    "soc_codes": soc_codes,
    "titles": titles,
    "role_descriptions": role_descriptions,
    "task_descriptions": task_descriptions,
}}
