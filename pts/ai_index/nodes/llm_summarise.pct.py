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
# Processes ads in chunks to handle arbitrarily large datasets without
# loading everything into memory. Writes results incrementally to DuckDB
# and supports resuming from a previous partial run. Schema changes in
# JobInfoModel are auto-detected and trigger a full reprocessing.
#
# Node variables:
# - `llm_model` (global): Model key from llm_models.toml
# - `llm_batch_size` (global): Number of prompts per LLM call (default 1000)
# - `llm_max_new_tokens` (global): Max tokens per LLM response (default 220)
# - `summarise_resume` (per-node): Resume from previous partial run (default true)
# - `summarise_max_retries` (per-node): Number of retry rounds for failed ads (default 0)
# - `llm_max_concurrent_batches` (global): Max concurrent batch LLM calls (default 1)

# %%
#|default_exp nodes.llm_summarise
#|export_as_func true

# %%
#|top_export
import numpy as np

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: np.ndarray) -> {
    'summary_meta': dict
}:
    """Run LLM to extract structured summaries from job ads."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_sbatch'
set_node_func_args('llm_summarise', run_name=run_name)
show_node_vars('llm_summarise', run_name=run_name)

# %% [markdown]
# # Function body

# %%
#|export
import json
from typing import List

import pandas as pd
from pydantic import BaseModel

from ai_index import const
from ai_index.utils import ResultStore, run_batched, strict_format, allm_generate, get_adzuna_conn, get_all_ad_ids

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
# ## Work function

# %%
#|export
def _validate_response(raw: str) -> str | None:
    """Validate an LLM response against JobInfoModel.

    Returns None if valid, or an error string.
    """
    try:
        JobInfoModel.model_validate_json(raw)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"

# %% [markdown]
# ## Read node variables

# %%
#|export
run_name = ctx.vars["run_name"]
llm_model = ctx.vars["llm_model"]
batch_size = ctx.vars["llm_batch_size"]
max_new_tokens = ctx.vars["llm_max_new_tokens"]
resume = ctx.vars["summarise_resume"]
max_retries = ctx.vars["summarise_max_retries"]
max_concurrent = ctx.vars["llm_max_concurrent_batches"]

output_dir = const.pipeline_store_path / run_name / "llm_summarise"
output_dir.mkdir(parents=True, exist_ok=True)
db_path = output_dir / "summaries.duckdb"

# %% [markdown]
# ## Define work function and run batched

# %%
#|export
json_schema = JobInfoModel.model_json_schema()

async def _work_fn(chunk_ids):
    """Fetch ads, build prompts, call LLM, validate, return DataFrame."""
    ads_conn = get_adzuna_conn(read_only=True)
    try:
        ads_conn.execute("CREATE OR REPLACE TEMP TABLE _chunk_ids (id BIGINT)")
        ads_conn.executemany("INSERT INTO _chunk_ids VALUES (?)", [(i,) for i in chunk_ids])
        ad_table = ads_conn.execute(
            "SELECT a.id, a.title, a.category_name, a.description "
            "FROM ads a JOIN _chunk_ids c ON a.id = c.id"
        ).fetch_arrow_table()
    finally:
        ads_conn.close()

    prompts = []
    ids_ordered = ad_table.column("id").to_pylist()
    for i in range(ad_table.num_rows):
        title = ad_table.column("title")[i].as_py()
        category = ad_table.column("category_name")[i].as_py()
        description = ad_table.column("description")[i].as_py()
        job_text = f"{title or ''}\n{category or ''}\n\n{(description or '')[:1200]}"
        prompts.append(strict_format(USER_TEMPLATE, job_text=job_text))

    responses = await allm_generate(
        prompts,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        json_schema=json_schema,
    )

    records = []
    for ad_id, response in zip(ids_ordered, responses):
        error = _validate_response(response)
        records.append({"id": ad_id, "data": response, "error": error})
    return pd.DataFrame(records)

all_ids = ad_ids.tolist() if ad_ids is not None else get_all_ad_ids()

store = ResultStore(db_path, {
    "id": "BIGINT NOT NULL",
    "data": "VARCHAR NOT NULL",
    "error": "VARCHAR",
})

summary_meta = await run_batched(
    all_ids, store, _work_fn,
    batch_size=batch_size,
    max_concurrent=max_concurrent,
    max_retries=max_retries,
    resume=resume,
    node_name="llm_summarise",
    print_fn=print,
)
store.close()

# %% [markdown]
# ## Write metadata

# %%
#|export
print(f"llm_summarise: wrote {db_path}")

meta_path = output_dir / "summary_meta.json"
with open(meta_path, "w") as f:
    json.dump(summary_meta, f, indent=2)
print(f"llm_summarise: wrote {meta_path}")

summary_meta #|func_return_line

# %% [markdown]
# ## Sample output

# %%
import duckdb
_conn = duckdb.connect(str(db_path), read_only=True)
_sample = _conn.execute("SELECT id, data, error FROM results LIMIT 3").fetchdf()
_conn.close()
for _, row in _sample.iterrows():
    print(f"\n--- Ad {row['id']} ---")
    if row["error"]:
        print(f"ERROR: {row['error']}")
    else:
        print(json.loads(row["data"]))
