# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.llm_filter_candidates
#
# Run LLM negative selection to filter cosine candidates.
#
# For each job ad, builds a prompt with the ad's context (title, category,
# description excerpt) and its top-N candidate occupations from cosine matching.
# The LLM identifies which candidates to DROP, keeping the functional matches.
#
# Reads from `cosine_candidates/candidates.parquet` and produces both:
# - `filtered_matches.parquet`: kept candidates with cosine_score
# - `dropped_matches.parquet`: dropped candidates with cosine_score and drop reason
#
# Node variables:
# - `llm_model` (global): Model key from llm_models.toml
# - `llm_batch_size` (global): Number of prompts per LLM call
# - `llm_max_new_tokens` (global): Max tokens per LLM response
# - `llm_max_concurrent_batches` (global): Max concurrent batch LLM calls
# - `filter_resume` (per-node): Resume from previous partial run
# - `filter_max_retries` (per-node): Retry rounds for failed ads
# - `system_prompt` (per-node): Path in prompt_library/
# - `user_prompt` (per-node): Path in prompt_library/
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.llm_filter_candidates
#|export_as_func true

# %%
#|top_export
import json
from typing import List

from pydantic import BaseModel, field_validator

class FilterResponseModel(BaseModel):
    keep: List[int]

    @field_validator("keep")
    @classmethod
    def keep_indices_positive(cls, v):
        if len(v) < 1:
            raise ValueError("must keep at least 1 candidate")
        for idx in v:
            if idx < 1:
                raise ValueError(f"keep indices must be 1-based positive integers, got {idx}")
        return v

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int]) -> {
    'successful_ad_ids': list[int]
}:
    """Run LLM negative selection to filter cosine candidates."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('llm_filter_candidates', run_name=run_name)
show_node_vars('llm_filter_candidates', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import duckdb
import pandas as pd

from ai_index import const
from ai_index.utils import (
    ResultStore, run_batched, strict_format, load_prompt, allm_generate,
    extract_json, is_reasoning_model, uses_structured_output,
    get_adzuna_conn, duckdb_connect_retry,
)

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
max_concurrent = ctx.vars["llm_max_concurrent_batches"]
resume = ctx.vars["filter_resume"]
max_retries = ctx.vars["filter_max_retries"]
raise_on_failure = ctx.vars["filter_raise_on_failure"]
duckdb_memory_limit = ctx.vars["duckdb_memory_limit"]

_is_reasoning = is_reasoning_model(llm_model)
_use_structured_output = uses_structured_output(llm_model)

_system_prompt_key = ctx.vars["system_prompt"]
_user_prompt_key = ctx.vars["user_prompt"]
if not _use_structured_output:
    suffix = "_reasoning" if _is_reasoning else "_unstructured"
    _system_prompt_key += suffix
    _user_prompt_key += suffix

SYSTEM_PROMPT = load_prompt(_system_prompt_key)
USER_PROMPT_TEMPLATE = load_prompt(_user_prompt_key)

output_dir = const.pipeline_store_path / run_name / "llm_filter_candidates"
output_dir.mkdir(parents=True, exist_ok=True)
db_path = output_dir / "filter_results.duckdb"

# %% [markdown]
# ## Prepare data connections
#
# Data is loaded per-chunk inside `_work_fn` to avoid holding all matches
# and raw ads in memory at once.

# %%
#|export
matches_path = const.pipeline_store_path / run_name / "cosine_candidates" / "candidates.parquet"

_matches_conn = duckdb.connect()  # in-memory
_matches_conn.execute(f"CREATE VIEW candidates AS SELECT * FROM read_parquet('{matches_path}')")
_ads_conn = get_adzuna_conn(read_only=True, memory_limit=duckdb_memory_limit)

# Build O*NET candidate text for the LLM prompt: description + top 5 tasks.
# No alternate titles (they blow up the context without helping the LLM's
# keep/drop decision; the O*NET title is already shown separately).
_onet_targets = pd.read_parquet(const.onet_targets_path)

def _build_onet_text(row):
    parts = [row["Description"]]
    tasks = row["Top_Tasks"]
    if len(tasks) > 0:
        parts.append("Key tasks: " + "; ".join(tasks[:5]))
    return " ".join(parts)

_onet_descriptions = dict(zip(_onet_targets["O*NET-SOC Code"], _onet_targets.apply(_build_onet_text, axis=1)))

print(f"llm_filter: {len(ad_ids)} ads to process")
print(f"llm_filter: reading candidates from {const.rel(matches_path)}")


def _load_chunk_context(chunk_ids):
    """Load matches and raw ads for a chunk of ad IDs."""
    id_list = ",".join(str(int(i)) for i in chunk_ids)

    # Matches from parquet
    chunk_matches = _matches_conn.execute(
        f"SELECT * FROM candidates WHERE ad_id IN ({id_list}) ORDER BY ad_id, rank"
    ).fetchdf()
    matches_by_ad = {}
    for ad_id, group in chunk_matches.groupby("ad_id"):
        matches_by_ad[int(ad_id)] = group.to_dict("records")

    # Raw ads
    _ads_conn.execute(f"CREATE OR REPLACE TEMP TABLE _chunk_ids AS SELECT unnest([{id_list}]::BIGINT[]) AS id")
    raw_rows = _ads_conn.execute(
        "SELECT a.id, a.title, a.category_name, a.description FROM ads a JOIN _chunk_ids c ON a.id = c.id"
    ).fetchall()
    raw_ads_by_id = {int(r[0]): {"title": r[1], "category_name": r[2], "description": r[3]} for r in raw_rows}

    return matches_by_ad, raw_ads_by_id

# %% [markdown]
# ## Define work function
#
# For each chunk of ad IDs, build prompts and call the LLM.

# %%
#|export
def _build_prompt(ad_id, candidates, raw_ad):
    """Build the negative selection prompt for one ad."""
    candidate_lines = []
    for i, c in enumerate(candidates):
        desc = _onet_descriptions[c["onet_code"]]
        candidate_lines.append(f"{i+1}. {c['onet_title']}: {desc}" if desc else f"{i+1}. {c['onet_title']}")
    candidates_str = "\n".join(candidate_lines)
    # Cap at 6000 chars: covers p95+ of ads (median 2217, p95 5544, p99 7955).
    # The old limit of 1200 truncated 82% of ads. With 20 candidates at ~300
    # chars each, the total prompt stays under ~3K tokens for most ads, well
    # within all models' context windows (smallest is gemma-4b at 8K).
    full_ad_excerpt = (raw_ad["description"] or "")[:6000].strip()

    return strict_format(
        USER_PROMPT_TEMPLATE,
        n_candidates=len(candidates),
        job_ad_title=raw_ad["title"] or "",
        job_sector_category=raw_ad["category_name"] or "",
        full_ad_excerpt=full_ad_excerpt,
        candidates_str=candidates_str,
    )


def _validate_response(raw: str, n_candidates: int) -> str | None:
    """Validate an LLM filter response. Returns None if valid, or an error string."""
    try:
        parsed = FilterResponseModel.model_validate_json(raw)
        # Check indices are in valid range
        for idx in parsed.keep:
            if idx < 1 or idx > n_candidates:
                return f"keep index {idx} out of range [1, {n_candidates}]"
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


_slurm_jobs = []

async def _work_fn(chunk_ids):
    """Load chunk context, build prompts, call LLM, validate, return DataFrame."""
    matches_by_ad, raw_ads_by_id = _load_chunk_context(chunk_ids)

    prompts = []
    n_candidates_per_ad = []
    for ad_id in chunk_ids:
        candidates = matches_by_ad[ad_id]
        prompts.append(_build_prompt(ad_id, candidates, raw_ads_by_id[ad_id]))
        n_candidates_per_ad.append(len(candidates))

    _sa = {}
    schema = FilterResponseModel.model_json_schema() if _use_structured_output else None
    responses = await allm_generate(
        prompts,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        json_schema=schema,
        cache=sbatch_cache,
        time=sbatch_time,
        slurm_accounting=_sa,
    )
    if _sa: _slurm_jobs.append(_sa)

    records = []
    for ad_id, response, n_cands in zip(chunk_ids, responses, n_candidates_per_ad):
        if _is_reasoning or not _use_structured_output:
            parsed = extract_json(response, validator=FilterResponseModel.model_validate)
            if parsed is None:
                records.append({"id": ad_id, "data": response, "error": "Failed to extract valid JSON from model output"})
                continue
            response = json.dumps(parsed)
        error = _validate_response(response, n_cands)
        records.append({"id": ad_id, "data": response, "error": error})
    return pd.DataFrame(records)

# %% [markdown]
# ## Run batched LLM calls

# %%
#|export
store = ResultStore(db_path, {
    "id": "BIGINT NOT NULL",
    "data": "VARCHAR NOT NULL",
    "error": "VARCHAR",
}, memory_limit=duckdb_memory_limit)

filter_meta = await run_batched(
    ad_ids, store, _work_fn,
    batch_size=batch_size,
    max_concurrent=max_concurrent,
    max_retries=max_retries,
    resume=resume,
    node_name="llm_filter_candidates",
    print_fn=print,
    raise_on_failure=raise_on_failure,
)
store.close()
del store
filter_meta["slurm_jobs"] = _slurm_jobs
filter_meta["slurm_total_seconds"] = sum(j.get("elapsed_seconds", 0) for j in _slurm_jobs)
print(f"llm_filter: wrote {const.rel(db_path)}")

meta_path = output_dir / "filter_meta.json"
with open(meta_path, "w") as f:
    json.dump(filter_meta, f, indent=2)
print(f"llm_filter: wrote {const.rel(meta_path)}")

# %% [markdown]
# ## Build filtered and dropped matches
#
# Apply the LLM drop decisions to the cosine match results.
# Write both kept and dropped candidates in chunks to avoid accumulating
# all rows in memory.

# %%
#|export
import pyarrow as pa
import pyarrow.parquet as pq

filter_conn = duckdb.connect(str(db_path), read_only=True)
filter_rows = filter_conn.execute(
    "SELECT id, data FROM results WHERE error IS NULL"
).fetchall()
filter_conn.close()

filtered_schema = pa.schema([
    ("ad_id", pa.int64()),
    ("rank", pa.int32()),
    ("onet_code", pa.string()),
    ("onet_title", pa.string()),
    ("cosine_score", pa.float32()),
])
dropped_schema = pa.schema([
    ("ad_id", pa.int64()),
    ("onet_code", pa.string()),
    ("onet_title", pa.string()),
    ("cosine_score", pa.float32()),
    ("original_rank", pa.int32()),
])

filtered_path = output_dir / "filtered_matches.parquet"
dropped_path = output_dir / "dropped_matches.parquet"
filtered_writer = pq.ParquetWriter(filtered_path, filtered_schema)
dropped_writer = pq.ParquetWriter(dropped_path, dropped_schema)

total_kept = 0
total_dropped = 0

FILTER_CHUNK_SIZE = 5000
for chunk_start in range(0, len(filter_rows), FILTER_CHUNK_SIZE):
    chunk = filter_rows[chunk_start:chunk_start + FILTER_CHUNK_SIZE]
    chunk_ad_ids = [int(row[0]) for row in chunk]

    # Load matches for this chunk from parquet
    id_list = ",".join(str(i) for i in chunk_ad_ids)
    chunk_matches = _matches_conn.execute(
        f"SELECT * FROM candidates WHERE ad_id IN ({id_list}) ORDER BY ad_id, rank"
    ).fetchdf()
    matches_by_ad = {}
    for ad_id, group in chunk_matches.groupby("ad_id"):
        matches_by_ad[int(ad_id)] = group.to_dict("records")

    kept_rows = []
    drop_rows = []
    for ad_id_raw, data_str in chunk:
        ad_id = int(ad_id_raw)
        if ad_id not in matches_by_ad:
            continue
        parsed = json.loads(data_str)
        keep_set = set(parsed["keep"])  # 1-based indices
        candidates = matches_by_ad[ad_id]

        rank = 0
        for i, c in enumerate(candidates):
            if (i + 1) in keep_set:
                kept_rows.append({
                    "ad_id": ad_id,
                    "rank": rank,
                    "onet_code": c["onet_code"],
                    "onet_title": c["onet_title"],
                    "cosine_score": float(c["cosine_score"]),
                })
                rank += 1
            else:
                drop_rows.append({
                    "ad_id": ad_id,
                    "onet_code": c["onet_code"],
                    "onet_title": c["onet_title"],
                    "cosine_score": float(c["cosine_score"]),
                    "original_rank": i,
                })

    if kept_rows:
        filtered_writer.write_table(pa.Table.from_pylist(kept_rows, schema=filtered_schema))
        total_kept += len(kept_rows)
    if drop_rows:
        dropped_writer.write_table(pa.Table.from_pylist(drop_rows, schema=dropped_schema))
        total_dropped += len(drop_rows)

filtered_writer.close()
dropped_writer.close()
_matches_conn.close()
_ads_conn.close()

failed_set = set(filter_meta["failed_ids"])
successful_ad_ids = [i for i in ad_ids if i not in failed_set]

print(f"llm_filter: {total_kept} kept, {total_dropped} dropped for {len(successful_ad_ids)} ads")
print(f"  mean candidates kept: {total_kept / max(len(successful_ad_ids), 1):.1f}")
print(f"  filtered: {filtered_path}")
print(f"  dropped: {dropped_path}")

successful_ad_ids #|func_return_line
