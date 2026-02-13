# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Load Job Ads
#
# Load job advertisement dataset from Adzuna extraction parquet files.
# Each parquet has columns: job_description, llm_output, parsed, error.
# The `parsed` dict contains: short_description, tasks (array), skills (array), domain.

# %%
#|default_exp nodes.load_job_ads
#|export_as_func true

# %%
#|set_func_signature
def load_job_ads(ctx, print) -> {"job_ads": dict}:
    """Load job advertisement dataset."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args(load_job_ads)

# %%
#|export
from pathlib import Path
from importlib import resources

import pandas as pd

store_dir = Path(resources.files("ai_index")).parent / "ai_index" / "store" / "inputs"

# Find all month parquet files
parquet_files = sorted(store_dir.glob("adzuna_extraction_month*_combined.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No adzuna parquet files found in {store_dir}")
print(f"load_job_ads: found {len(parquet_files)} month files")

# Load and concatenate
dfs = []
for pf in parquet_files:
    df = pd.read_parquet(pf)
    dfs.append(df)
    print(f"  {pf.name}: {len(df)} rows")
raw = pd.concat(dfs, ignore_index=True)
print(f"load_job_ads: {len(raw)} total rows")

# %%
#|export
# Filter rows with valid parsed data
valid_mask = raw["parsed"].apply(lambda x: isinstance(x, dict))
raw = raw[valid_mask].reset_index(drop=True)
print(f"load_job_ads: {len(raw)} rows after filtering invalid parsed")

# Extract fields from parsed dicts
job_ids = [f"JOB{i:07d}" for i in range(len(raw))]

role_text = []
taskskill_text = []
short_desc = []
tasks_and_skills = []
domains = []

for _, row in raw.iterrows():
    p = row["parsed"]
    sd = p.get("short_description", "")
    tasks = list(p.get("tasks", []))
    sk = list(p.get("skills", []))
    domain = p.get("domain", "")

    # role_text: short description (used for role embedding)
    role_text.append(sd)
    # taskskill_text: tasks + skills joined (used for task embedding)
    taskskill_text.append(", ".join(tasks + sk))
    short_desc.append(sd)
    tasks_and_skills.append(", ".join(tasks + sk))
    domains.append(domain)

print(f"load_job_ads: extracted {len(job_ids)} job ads")
{"job_ads": {  #|func_return_line
    "job_ids": job_ids,
    "role_text": role_text,
    "taskskill_text": taskskill_text,
    "short_desc": short_desc,
    "tasks_and_skills": tasks_and_skills,
    "domains": domains,
}}
