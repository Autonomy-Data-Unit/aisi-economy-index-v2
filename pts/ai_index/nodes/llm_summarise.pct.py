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
# loading everything into memory. Writes results incrementally to parquet
# and supports resuming from a previous partial run.
#
# Node variables:
# - `llm_model` (global): Model key from llm_models.toml
# - `llm_batch_size` (global): Number of prompts per LLM call (default 1000)
# - `llm_max_new_tokens` (global): Max tokens per LLM response (default 220)
# - `summarise_resume` (per-node): Resume from previous partial run (default true)

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
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, ValidationError

from ai_index import const
from ai_index.utils import get_adzuna_conn, llm_generate

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


def _responses_to_df(chunk_ids: list, responses: list[str]) -> tuple[pd.DataFrame, int, int]:
    """Parse LLM responses into a DataFrame. Returns (df, n_success, n_failed)."""
    records = []
    n_success = 0
    n_failed = 0
    for ad_id, raw_response in zip(chunk_ids, responses):
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
    return pd.DataFrame(records), n_success, n_failed

# %% [markdown]
# ## Read node variables

# %%
#|export
run_name = ctx.vars["run_name"]
llm_model = ctx.vars["llm_model"]
batch_size = int(ctx.vars["llm_batch_size"])
max_new_tokens = int(ctx.vars["llm_max_new_tokens"])
resume = ctx.vars.get("summarise_resume", True)

output_dir = const.pipeline_store_path / run_name
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "summaries.parquet"

# %% [markdown]
# ## Determine which ads to process

# %%
#|export
all_ids = ad_ids.tolist() if ad_ids is not None else None

if all_ids is None:
    conn = get_adzuna_conn(read_only=True)
    all_ids = conn.execute("SELECT id FROM ads").fetchnumpy()["id"].tolist()
    conn.close()

# Resume: skip IDs already processed
if resume and output_path.exists():
    done_df = pd.read_parquet(output_path, columns=["id"])
    done_ids = set(done_df["id"].tolist())
    remaining_ids = [i for i in all_ids if i not in done_ids]
    print(f"llm_summarise: resuming — {len(done_ids)} already done, {len(remaining_ids)} remaining")
else:
    remaining_ids = all_ids
    done_ids = set()

n_total = len(all_ids)
print(f"llm_summarise: {n_total} total ads, {len(remaining_ids)} to process (batch_size={batch_size})")

# %% [markdown]
# ## Process in chunks

# %%
#|export
json_schema = JobInfoModel.model_json_schema()
n_success = 0
n_failed = 0

conn = get_adzuna_conn(read_only=True)

for chunk_start in range(0, len(remaining_ids), batch_size):
    chunk_ids = remaining_ids[chunk_start : chunk_start + batch_size]
    chunk_num = chunk_start // batch_size + 1
    n_chunks = (len(remaining_ids) + batch_size - 1) // batch_size
    print(f"llm_summarise: chunk {chunk_num}/{n_chunks} ({len(chunk_ids)} ads)")

    # Read this chunk from DuckDB using a temp table to avoid huge IN clauses
    conn.execute("CREATE OR REPLACE TEMP TABLE _chunk_ids (id BIGINT)")
    conn.executemany("INSERT INTO _chunk_ids VALUES (?)", [(i,) for i in chunk_ids])
    ad_table = conn.execute(
        "SELECT a.id, a.title, a.category_name, a.description "
        "FROM ads a JOIN _chunk_ids c ON a.id = c.id"
    ).fetch_arrow_table()

    # Build prompts for this chunk
    prompts = []
    ids_ordered = ad_table.column("id").to_pylist()
    for i in range(ad_table.num_rows):
        title = ad_table.column("title")[i].as_py()
        category = ad_table.column("category_name")[i].as_py()
        description = ad_table.column("description")[i].as_py()
        job_text = f"{title or ''}\n{category or ''}\n\n{(description or '')[:1200]}"
        prompts.append(USER_TEMPLATE.format(job_text=job_text))

    # Run LLM
    responses = llm_generate(
        prompts,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        json_schema=json_schema,
    )

    # Parse and write incrementally
    chunk_df, chunk_success, chunk_failed = _responses_to_df(ids_ordered, responses)
    n_success += chunk_success
    n_failed += chunk_failed

    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        chunk_df = pd.concat([existing_df, chunk_df], ignore_index=True)
    chunk_df.to_parquet(output_path, index=False)

    print(f"llm_summarise: chunk {chunk_num} done — {chunk_success} ok, {chunk_failed} failed")

conn.close()

# %% [markdown]
# ## Return summary

# %%
#|export
# Count totals including previously resumed data
if done_ids:
    done_df = pd.read_parquet(output_path, columns=["error"])
    n_success = int((done_df["error"].isna()).sum())
    n_failed = int((done_df["error"].notna()).sum())

print(f"llm_summarise: {n_success} succeeded, {n_failed} failed out of {n_total}")
print(f"llm_summarise: wrote {output_path}")

summary_meta = {
    "parquet_path": str(output_path),
    "n_total": n_total,
    "n_success": n_success,
    "n_failed": n_failed,
}
summary_meta #|func_return_line
