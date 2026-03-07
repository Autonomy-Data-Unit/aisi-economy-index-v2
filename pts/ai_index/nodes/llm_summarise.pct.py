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
set_node_func_args(run_name='test_local')
show_node_vars(run_name='test_local')

# %% [markdown]
# # Function body

# %%
#|export
import asyncio
import json
from typing import List

import pandas as pd
from pydantic import BaseModel, ValidationError

from ai_index import const
from ai_index.utils import LLMResultStore, allm_generate, get_adzuna_conn, get_all_ad_ids

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
# ## Helpers

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


async def _fetch_and_call_llm(chunk_ids, llm_model, max_new_tokens, json_schema):
    """Fetch ads from DuckDB, build prompts, call LLM asynchronously.

    Opens its own read-only connection so multiple chunks can run concurrently.
    Returns (ids_ordered, responses) — raw strings, not yet validated.
    """
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
        prompts.append(USER_TEMPLATE.format(job_text=job_text))

    responses = await allm_generate(
        prompts,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        json_schema=json_schema,
    )

    return ids_ordered, responses


def _build_results_df(ids: list, responses: list[str]) -> tuple[pd.DataFrame, int, int]:
    """Validate responses and build a DataFrame for LLMResultStore.

    Returns (df with id/data/error columns, n_success, n_failed).
    """
    records = []
    n_success = n_failed = 0
    for ad_id, response in zip(ids, responses):
        error = _validate_response(response)
        records.append({"id": ad_id, "data": response, "error": error})
        if error:
            n_failed += 1
        else:
            n_success += 1
    return pd.DataFrame(records), n_success, n_failed

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

# %%
ctx.vars["llm_max_concurrent_batches"]

# %% [markdown]
# ## Determine which ads to process

# %%
#|export
store = LLMResultStore(db_path)

all_ids = ad_ids.tolist() if ad_ids is not None else None
if all_ids is None:
    all_ids = get_all_ad_ids()

if resume:
    done_ids = store.done_ids()
    remaining_ids = [i for i in all_ids if i not in done_ids]
    if done_ids:
        print(f"llm_summarise: resuming — {len(done_ids)} already done, {len(remaining_ids)} remaining")
else:
    store.clear()
    remaining_ids = all_ids

n_total = len(all_ids)
print(f"llm_summarise: {n_total} total ads, {len(remaining_ids)} to process (batch_size={batch_size}, max_concurrent={max_concurrent})")

# %% [markdown]
# ## Process in chunks

# %%
#|export
json_schema = JobInfoModel.model_json_schema()
n_success = 0
n_failed = 0
sem = asyncio.Semaphore(max_concurrent)

chunks = [
    remaining_ids[i : i + batch_size]
    for i in range(0, len(remaining_ids), batch_size)
]
n_chunks = len(chunks)

async def _process_chunk(chunk_ids, chunk_num):
    async with sem:
        print(f"llm_summarise: chunk {chunk_num}/{n_chunks} ({len(chunk_ids)} ads)")
        ids_ordered, responses = await _fetch_and_call_llm(
            chunk_ids, llm_model, max_new_tokens, json_schema,
        )
        chunk_df, chunk_ok, chunk_err = _build_results_df(ids_ordered, responses)
        print(f"llm_summarise: chunk {chunk_num} done — {chunk_ok} ok, {chunk_err} failed")
        return chunk_df, chunk_ok, chunk_err

for coro in asyncio.as_completed([
    _process_chunk(chunk_ids, i + 1)
    for i, chunk_ids in enumerate(chunks)
]):
    chunk_df, chunk_ok, chunk_err = await coro
    store.insert(chunk_df)
    n_success += chunk_ok
    n_failed += chunk_err

# %% [markdown]
# ## Retry failed ads

# %%
#|export
for retry_num in range(1, max_retries + 1):
    retry_ids = store.failed_ids()
    if not retry_ids:
        print(f"llm_summarise: no failures to retry")
        break

    print(f"llm_summarise: retry {retry_num}/{max_retries} — {len(retry_ids)} failed ads")
    store.delete_ids(retry_ids)

    retry_chunks = [
        retry_ids[i : i + batch_size]
        for i in range(0, len(retry_ids), batch_size)
    ]
    retry_ok = retry_err = 0

    async def _retry_chunk(chunk_ids):
        async with sem:
            ids_ordered, responses = await _fetch_and_call_llm(
                chunk_ids, llm_model, max_new_tokens, json_schema,
            )
            return _build_results_df(ids_ordered, responses)

    for coro in asyncio.as_completed([
        _retry_chunk(chunk_ids) for chunk_ids in retry_chunks
    ]):
        chunk_df, chunk_ok, chunk_err = await coro
        store.insert(chunk_df)
        retry_ok += chunk_ok
        retry_err += chunk_err

    print(f"llm_summarise: retry {retry_num} done — {retry_ok} recovered, {retry_err} still failed")

# %% [markdown]
# ## Return summary

# %%
#|export
n_success, n_failed = store.counts()
failed_ids = store.failed_ids()
store.close()

print(f"llm_summarise: {n_success} succeeded, {n_failed} failed out of {n_total}")
if failed_ids:
    print(f"llm_summarise: failed IDs: {failed_ids[:20]}{'...' if len(failed_ids) > 20 else ''}")
print(f"llm_summarise: wrote {db_path}")

summary_meta = {
    "db_path": str(db_path),
    "n_total": n_total,
    "n_success": n_success,
    "n_failed": n_failed,
    "failed_ids": failed_ids,
}

meta_path = output_dir / "summary_meta.json"
with open(meta_path, "w") as f:
    json.dump(summary_meta, f, indent=2)
print(f"llm_summarise: wrote {meta_path}")

summary_meta #|func_return_line
