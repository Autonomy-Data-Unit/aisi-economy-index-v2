# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM Summarise
#
# Run LLM to extract structured summaries from job ads.
# Takes raw job ad text and produces structured fields:
# short_description, tasks, skills, domain, level, automation_prof_score.
#
# Uses the same system prompt and Pydantic schema as the old pipeline's
# Stage 1 (Meta-Llama-3.1-8B-Instruct), but routes through
# `ai_index.utils.llm_generate` which supports api/local/sbatch modes.

# %%
#|default_exp nodes.llm_summarise
#|export_as_func true

# %%
#|top_export
import numpy as np

# %%
#|set_func_signature
def main(ctx, print, ad_ids: np.ndarray) -> {
    'summary_meta': dict
}:
    """Run LLM to extract structured summaries from job ads."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
set_node_func_args(run_name='test_local')
show_node_vars(run_name='test_local')

# %% [markdown]
# # Function body

# %%
#|export
import json
from typing import List

from pydantic import BaseModel, ValidationError

from ai_index import const
from ai_index.utils import get_ads_by_id, get_adzuna_conn, llm_generate

# %% [markdown]
# ## Pydantic schema for LLM output

# %%
#|export
class JobInfoModel(BaseModel):
    short_description: str
    tasks: List[str]
    skills: List[str]
    domain: str
    level: str
    automation_prof_score: int

# %% [markdown]
# ## Prompt templates
#
# Identical to the old pipeline's Stage 1 (`run_llm_extract.py`).

# %%
#|export
SYSTEM_PROMPT = (
    "You are a human resources highly-accurate data extraction bot. "
    "Extract the following details from the job advertisement provided by the user. "
    "You MUST NOT include more than 5 tasks or 5 skills. Stop the list at 5. Do not write more. "
    "- 'level': classify as 'Entry-Level' if the job requires <3 years experience or mentions 'junior'/'entry'; otherwise 'Experienced'. "
    "- 'automation_prof_score': integer 0-10 estimating AI automation risk. "
    "0 = AI-proof (requires physical/manual presence, creativity, leadership, or deep social judgment). "
    "10 = highly automatable by AI (routine, repetitive non-manual tasks like data entry, scheduling). "
    "Manual labour (e.g. cleaning, lifting, warehouse, driving) should usually score 0-3, "
    "since AI alone cannot replace them."
)

USER_TEMPLATE = """Extract:
1. Short job description
2. Bullet list of up to 5 key tasks
3. Bullet list of up to 5 key skills
4. The domain or industry
5. The level (Entry-Level if <3 years experience or junior, else Experienced)
6. Automation proof score (0=AI proof, 10=highly automatable non-manual tasks)

Job Ad:
{job_text}
"""

# %% [markdown]
# ## JSON parsing

# %%
#|export
def _parse_llm_output(raw: str) -> tuple[dict | None, str | None]:
    """Parse structured LLM JSON output into a validated JobInfoModel dict.

    Returns (parsed_dict, error_string). Exactly one is None.
    """
    try:
        validated = JobInfoModel.model_validate_json(raw)
        validated.tasks = validated.tasks[:5]
        validated.skills = validated.skills[:5]
        return validated.model_dump(), None
    except (ValidationError, Exception) as e:
        return None, f"{type(e).__name__}: {e}"

# %% [markdown]
# ## Read ads from DuckDB

# %%
#|export
if ad_ids is not None:
    ad_table = get_ads_by_id(
        ad_ids.tolist(),
        columns=["id", "title", "category_name", "description"],
    )
else:
    conn = get_adzuna_conn(read_only=True)
    ad_table = conn.execute(
        "SELECT id, title, category_name, description FROM ads"
    ).fetch_arrow_table()
    conn.close()

n_ads = len(ad_table)
print(f"llm_summarise: processing {n_ads} ads")

# %% [markdown]
# ## Build prompts

# %%
#|export
ids = ad_table.column("id").to_pylist()
titles = ad_table.column("title").to_pylist()
categories = ad_table.column("category_name").to_pylist()
descriptions = ad_table.column("description").to_pylist()

prompts = []
for title, category, description in zip(titles, categories, descriptions):
    job_text = f"{title or ''}\n{category or ''}\n\n{(description or '')[:1200]}"
    prompts.append(USER_TEMPLATE.format(job_text=job_text))

# %% [markdown]
# ## Run LLM generation

# %%
#|export
llm_model = ctx.vars["llm_model"]
print(f"llm_summarise: calling llm_generate with model={llm_model!r} for {len(prompts)} prompts")

responses = llm_generate(
    prompts,
    model=llm_model,
    system_message=SYSTEM_PROMPT,
    max_new_tokens=220,
    json_schema=JobInfoModel.model_json_schema(),
)

# %% [markdown]
# ## Parse outputs and write results

# %%
#|export
import pandas as pd

records = []
n_success = 0
n_failed = 0

for ad_id, raw_response in zip(ids, responses):
    parsed, error = _parse_llm_output(raw_response)
    record = {
        "id": ad_id,
        "llm_output": raw_response,
        "error": error,
    }
    if parsed:
        record.update({
            "short_description": parsed["short_description"],
            "tasks": json.dumps(parsed["tasks"]),
            "skills": json.dumps(parsed["skills"]),
            "domain": parsed["domain"],
            "level": parsed["level"],
            "automation_prof_score": parsed["automation_prof_score"],
        })
        n_success += 1
    else:
        record.update({
            "short_description": None,
            "tasks": None,
            "skills": None,
            "domain": None,
            "level": None,
            "automation_prof_score": None,
        })
        n_failed += 1
    records.append(record)

df = pd.DataFrame(records)

run_name = ctx.vars["run_name"]
output_dir = const.pipeline_store_path / run_name
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "summaries.parquet"
df.to_parquet(output_path, index=False)

print(f"llm_summarise: {n_success} succeeded, {n_failed} failed out of {n_ads}")
print(f"llm_summarise: wrote {output_path}")

summary_meta = {
    "parquet_path": str(output_path),
    "n_total": n_ads,
    "n_success": n_success,
    "n_failed": n_failed,
}
summary_meta #|func_return_line
