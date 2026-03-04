# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build O*NET Descriptions
#
# For each of the 894 O*NET occupations, build two text fields for embedding:
# - `role_text`: Title + Description
# - `task_text`: Title + top-10 tasks + top-10 skills + top-10 work activities (by importance)

# %%
#|default_exp nodes.build_onet_desc
#|export_as_func true

# %%
#|set_func_signature
def main(onet_tables, ctx, print) -> {"onet_desc": dict}:
    """Build role and task description texts for O*NET occupations."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args()

# %%
#|export
import pandas as pd

from ai_index.const import pipeline_store_path

run_name = ctx.vars["run_name"]
store_dir = pipeline_store_path / run_name / "build_onet_desc"
output_path = store_dir / "onet_descriptions.parquet"

# %%
#|export
if output_path.exists():
    print(f"build_onet_desc: loading cached descriptions from {output_path}")
    desc_df = pd.read_parquet(output_path)
else:
    # Load the core occupation table
    occ_data = onet_tables["Occupation Data"]
    task_stmts = onet_tables["Task Statements"]
    skills = onet_tables["Skills"]
    work_acts = onet_tables["Work Activities"]

    # Build role_text: Title - Description
    occ = occ_data[["O*NET-SOC Code", "Title", "Description"]].copy()
    occ["role_text"] = occ["Title"] + " - " + occ["Description"]

    # Helper: get top-N items by importance (IM scale) for each occupation
    def _top_n_by_importance(df, name_col, n=10):
        im = df[df["Scale ID"] == "IM"].copy()
        im["Data Value"] = im["Data Value"].astype(float)
        # Group by occupation + item, sum importance scores
        grouped = im.groupby(["O*NET-SOC Code", name_col])["Data Value"].sum().reset_index()
        # Rank within each occupation, take top N
        grouped["rank"] = grouped.groupby("O*NET-SOC Code")["Data Value"].rank(
            ascending=False, method="first"
        )
        top = grouped[grouped["rank"] <= n]
        # Aggregate to comma-separated string per occupation
        return (
            top.sort_values(["O*NET-SOC Code", "rank"])
            .groupby("O*NET-SOC Code")[name_col]
            .agg(", ".join)
            .reset_index()
            .rename(columns={name_col: f"top_{name_col}"})
        )

    top_tasks = _top_n_by_importance(task_stmts, "Task", n=10)
    top_skills = _top_n_by_importance(skills, "Element Name", n=10)
    top_skills = top_skills.rename(columns={"top_Element Name": "top_skills"})
    top_acts = _top_n_by_importance(work_acts, "Element Name", n=10)
    top_acts = top_acts.rename(columns={"top_Element Name": "top_activities"})

    # Merge all into occupation table
    desc_df = occ[["O*NET-SOC Code", "Title", "role_text"]].copy()
    desc_df = desc_df.merge(top_tasks, on="O*NET-SOC Code", how="left")
    desc_df = desc_df.merge(top_skills, on="O*NET-SOC Code", how="left")
    desc_df = desc_df.merge(top_acts, on="O*NET-SOC Code", how="left")

    # Build task_text: Title - [tasks, skills, activities]
    parts = []
    for col in ["top_Task", "top_skills", "top_activities"]:
        parts.append(desc_df[col].fillna(""))
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.where(p == "", combined + ", " + p)
        combined = combined.where(combined != "", p)
    desc_df["task_text"] = desc_df["Title"] + " - " + combined

    desc_df = desc_df[["O*NET-SOC Code", "Title", "role_text", "task_text"]].copy()

    # Save
    store_dir.mkdir(parents=True, exist_ok=True)
    desc_df.to_parquet(output_path, compression="snappy")
    print(f"build_onet_desc: saved {len(desc_df)} occupation descriptions to {output_path}")

print(f"build_onet_desc: {len(desc_df)} occupations")

# %%
#|export
onet_desc = {
    "soc_codes": desc_df["O*NET-SOC Code"].tolist(),
    "titles": desc_df["Title"].tolist(),
    "role_texts": desc_df["role_text"].tolist(),
    "task_texts": desc_df["task_text"].tolist(),
    "store_path": str(output_path),
}

{"onet_desc": onet_desc}  #|func_return_line
