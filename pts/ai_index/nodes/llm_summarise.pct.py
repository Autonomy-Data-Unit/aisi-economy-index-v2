# ---
# jupyter:
#   kernelspec:
#     display_name: ai-index (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.llm_summarise
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
# - `system_prompt` (per-node): Prompt library path for system prompt (default "llm_summarise/main/system")
# - `user_prompt` (per-node): Prompt library path for user template (default "llm_summarise/main/user")

# %%
#|default_exp nodes.llm_summarise
#|export_as_func true

# %%
#|top_export
from typing import List

import numpy as np
from pydantic import BaseModel

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: np.ndarray) -> {
    'successful_ad_ids': list[int]
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

import pandas as pd

from ai_index import const
from ai_index.utils import ResultStore, run_batched, strict_format, load_prompt, allm_generate, extract_json, is_reasoning_model, get_adzuna_conn, get_all_ad_ids

# %% [markdown]
# ## Pydantic schema for LLM output

# %%
#|top_export
class JobInfoModel(BaseModel):
    short_description: str
    tasks: List[str]
    skills: List[str]
    domain: str
    level: str

# %% [markdown]
# ## Prompt templates
#
# Loaded from the prompt library. Paths configured via node variables
# `system_prompt` and `user_prompt`.

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
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]
batch_size = ctx.vars["llm_batch_size"]
max_new_tokens = ctx.vars["llm_max_new_tokens"]
temperature = ctx.vars["temperature"]
top_p = ctx.vars["top_p"]
top_k = ctx.vars["top_k"]
resume = ctx.vars["summarise_resume"]
max_retries = ctx.vars["summarise_max_retries"]
raise_on_failure = ctx.vars["summarise_raise_on_failure"]
max_concurrent = ctx.vars["llm_max_concurrent_batches"]
duckdb_memory_limit = ctx.vars["duckdb_memory_limit"]

_is_reasoning = is_reasoning_model(llm_model)

SYSTEM_PROMPT = load_prompt(ctx.vars["system_prompt"])
USER_PROMPT_TEMPLATE = load_prompt(ctx.vars["user_prompt"])

output_dir = const.pipeline_store_path / run_name / "llm_summarise"
output_dir.mkdir(parents=True, exist_ok=True)
db_path = output_dir / "summaries.duckdb"

# %% [markdown]
# ## Define work function and run batched

# %%
#|export
json_schema = JobInfoModel.model_json_schema()
_slurm_jobs = []

async def _work_fn(chunk_ids):
    """Fetch ads, build prompts, call LLM, validate, return DataFrame."""
    ads_conn = get_adzuna_conn(read_only=True, memory_limit=duckdb_memory_limit)
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
        prompts.append(strict_format(USER_PROMPT_TEMPLATE, job_text=job_text))

    _sa = {}
    responses = await allm_generate(
        prompts,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        json_schema=json_schema,
        cache=sbatch_cache,
        time=sbatch_time,
        slurm_accounting=_sa,
    )
    if _sa: _slurm_jobs.append(_sa)

    records = []
    for ad_id, response in zip(ids_ordered, responses):
        if _is_reasoning:
            parsed = extract_json(response)
            if parsed is None:
                records.append({"id": ad_id, "data": response, "error": "Failed to extract JSON from reasoning model output"})
                continue
            response = json.dumps(parsed)
        error = _validate_response(response)
        records.append({"id": ad_id, "data": response, "error": error})
    return pd.DataFrame(records)

all_ids = ad_ids.tolist() if ad_ids is not None else get_all_ad_ids()

store = ResultStore(db_path, {
    "id": "BIGINT NOT NULL",
    "data": "VARCHAR NOT NULL",
    "error": "VARCHAR",
}, memory_limit=duckdb_memory_limit)

summary_meta = await run_batched(
    all_ids, store, _work_fn,
    batch_size=batch_size,
    max_concurrent=max_concurrent,
    max_retries=max_retries,
    resume=resume,
    node_name="llm_summarise",
    print_fn=print,
    raise_on_failure=raise_on_failure,
)
store.close()
del store

# %% [markdown]
# ## Write result metadata

# %%
#|export
summary_meta["slurm_jobs"] = _slurm_jobs
summary_meta["slurm_total_seconds"] = sum(j.get("elapsed_seconds", 0) for j in _slurm_jobs)
print(f"llm_summarise: wrote {const.rel(db_path)}")

meta_path = output_dir / "summary_meta.json"
with open(meta_path, "w") as f:
    json.dump(summary_meta, f, indent=2)
print(f"llm_summarise: wrote {const.rel(meta_path)}")

# %% [markdown]
# ## Return results

# %%
#|export
failed_set = set(summary_meta["failed_ids"])
successful_ad_ids = [i for i in all_ids if i not in failed_set]
successful_ad_ids; #|func_return_line

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
