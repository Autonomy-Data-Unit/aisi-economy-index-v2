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
# Run LLM negative selection to filter cosine match candidates.
#
# For each job ad, builds a prompt with the ad's context (title, sector, domain,
# tasks, raw description excerpt) and its top-K candidate occupations from cosine
# matching. The LLM identifies which candidates to DROP, keeping 2-3 functional
# matches per ad.
#
# 1. Loads cosine match results from `cosine_match/matches.parquet`.
# 2. Loads LLM summaries from `llm_summarise/summaries.duckdb` (tasks, skills, domain).
# 3. Loads raw ad metadata from Adzuna DuckDB (title, category, description).
# 4. Builds prompts and runs the LLM.
# 5. Parses `{"drop": [...]}` responses to produce filtered matches.
# 6. Saves filtered matches to `llm_filter/filtered_matches.parquet`.
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
    drop: List[int]

    @field_validator("drop")
    @classmethod
    def drop_indices_positive(cls, v):
        for idx in v:
            if idx < 1:
                raise ValueError(f"drop indices must be 1-based positive integers, got {idx}")
        return v

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int]) -> {
    'successful_ad_ids': list[int]
}:
    """Run LLM negative selection to filter cosine match candidates."""
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
from ai_index.nodes.llm_summarise import JobInfoModel
from ai_index.utils import (
    ResultStore, run_batched, strict_format, load_prompt, allm_generate,
    get_adzuna_conn,
)

# %%
#|export
run_name = ctx.vars["run_name"]
llm_model = ctx.vars["llm_model"]
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]
batch_size = ctx.vars["llm_batch_size"]
max_new_tokens = ctx.vars["llm_max_new_tokens"]
max_concurrent = ctx.vars["llm_max_concurrent_batches"]
resume = ctx.vars["filter_resume"]
max_retries = ctx.vars["filter_max_retries"]
raise_on_failure = ctx.vars["filter_raise_on_failure"]

SYSTEM_PROMPT = load_prompt(ctx.vars["system_prompt"])
USER_PROMPT_TEMPLATE = load_prompt(ctx.vars["user_prompt"])

output_dir = const.pipeline_store_path / run_name / "llm_filter_candidates"
output_dir.mkdir(parents=True, exist_ok=True)
db_path = output_dir / "filter_results.duckdb"

# %% [markdown]
# ## Prepare data connections
#
# Data is loaded per-chunk inside `_work_fn` to avoid holding all matches,
# summaries, and raw ads in memory at once.

# %%
#|export
matches_path = const.pipeline_store_path / run_name / "cosine_match" / "matches.parquet"
summaries_db = const.pipeline_store_path / run_name / "llm_summarise" / "summaries.duckdb"

_matches_conn = duckdb.connect()  # in-memory, queries parquet directly
_summaries_conn = duckdb.connect(str(summaries_db), read_only=True)
_ads_conn = get_adzuna_conn(read_only=True)

print(f"llm_filter: {len(ad_ids)} ads to process")


def _load_chunk_context(chunk_ids):
    """Load matches, summaries, and raw ads for a chunk of ad IDs."""
    id_list = ",".join(str(int(i)) for i in chunk_ids)

    # Matches from parquet
    chunk_matches = _matches_conn.execute(
        f"SELECT * FROM read_parquet('{matches_path}') WHERE ad_id IN ({id_list}) ORDER BY ad_id, rank"
    ).fetchdf()
    matches_by_ad = {}
    for ad_id, group in chunk_matches.groupby("ad_id"):
        matches_by_ad[int(ad_id)] = group.to_dict("records")

    # Summaries
    summary_rows = _summaries_conn.execute(
        f"SELECT id, data FROM results WHERE error IS NULL AND id IN ({id_list})"
    ).fetchall()
    summaries_by_ad = {int(rid): JobInfoModel.model_validate_json(data) for rid, data in summary_rows}

    # Raw ads
    _ads_conn.execute(f"CREATE OR REPLACE TEMP TABLE _chunk_ids AS SELECT unnest([{id_list}]::BIGINT[]) AS id")
    raw_rows = _ads_conn.execute(
        "SELECT a.id, a.title, a.category_name, a.description FROM ads a JOIN _chunk_ids c ON a.id = c.id"
    ).fetchall()
    raw_ads_by_id = {int(r[0]): {"title": r[1], "category_name": r[2], "description": r[3]} for r in raw_rows}

    return matches_by_ad, summaries_by_ad, raw_ads_by_id

# %% [markdown]
# ## Define work function
#
# For each chunk of ad IDs, build prompts and call the LLM.

# %%
#|export
def _build_prompt(ad_id, candidates, summary, raw_ad):
    """Build the negative selection prompt for one ad."""
    tasks_str = ", ".join(summary.tasks + summary.skills)[:800]
    candidates_str = "\n".join(
        f"{i+1}. {c['onet_title']}" for i, c in enumerate(candidates)
    )
    full_ad_excerpt = (raw_ad["description"] or "")[:700].strip()

    return strict_format(
        USER_PROMPT_TEMPLATE,
        n_candidates=len(candidates),
        job_ad_title=raw_ad["title"] or "",
        job_sector_category=raw_ad["category_name"] or "",
        domain=summary.domain,
        tasks_str=tasks_str,
        full_ad_excerpt=full_ad_excerpt,
        candidates_str=candidates_str,
    )


def _validate_response(raw: str, n_candidates: int) -> str | None:
    """Validate an LLM filter response. Returns None if valid, or an error string."""
    try:
        parsed = FilterResponseModel.model_validate_json(raw)
        # Check indices are in valid range
        for idx in parsed.drop:
            if idx < 1 or idx > n_candidates:
                return f"drop index {idx} out of range [1, {n_candidates}]"
        # Check at least 1 candidate is kept
        n_kept = n_candidates - len(set(parsed.drop))
        if n_kept < 1:
            return f"would drop all candidates ({n_candidates} dropped, 0 kept)"
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


_slurm_jobs = []

async def _work_fn(chunk_ids):
    """Load chunk context, build prompts, call LLM, validate, return DataFrame."""
    matches_by_ad, summaries_by_ad, raw_ads_by_id = _load_chunk_context(chunk_ids)

    prompts = []
    n_candidates_per_ad = []
    for ad_id in chunk_ids:
        candidates = matches_by_ad[ad_id]
        prompts.append(_build_prompt(ad_id, candidates, summaries_by_ad[ad_id], raw_ads_by_id[ad_id]))
        n_candidates_per_ad.append(len(candidates))

    _sa = {}
    responses = await allm_generate(
        prompts,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        json_schema=FilterResponseModel.model_json_schema(),
        cache=sbatch_cache,
        time=sbatch_time,
        slurm_accounting=_sa,
    )
    if _sa: _slurm_jobs.append(_sa)

    records = []
    for ad_id, response, n_cands in zip(chunk_ids, responses, n_candidates_per_ad):
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
})

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
filter_meta["slurm_jobs"] = _slurm_jobs
filter_meta["slurm_total_seconds"] = sum(j.get("elapsed_seconds", 0) for j in _slurm_jobs)
print(f"llm_filter: wrote {const.rel(db_path)}")

meta_path = output_dir / "filter_meta.json"
with open(meta_path, "w") as f:
    json.dump(filter_meta, f, indent=2)
print(f"llm_filter: wrote {const.rel(meta_path)}")

# %% [markdown]
# ## Build filtered matches
#
# Apply the LLM drop decisions to the cosine match results.
# Processes in chunks to avoid loading all matches into memory.

# %%
#|export
filter_conn = duckdb.connect(str(db_path), read_only=True)
filter_rows = filter_conn.execute(
    "SELECT id, data FROM results WHERE error IS NULL"
).fetchall()
filter_conn.close()

FILTER_CHUNK_SIZE = 5000
filtered_rows = []
for chunk_start in range(0, len(filter_rows), FILTER_CHUNK_SIZE):
    chunk = filter_rows[chunk_start:chunk_start + FILTER_CHUNK_SIZE]
    chunk_ad_ids = [int(row[0]) for row in chunk]

    # Load matches for this chunk from parquet
    id_list = ",".join(str(i) for i in chunk_ad_ids)
    chunk_matches = _matches_conn.execute(
        f"SELECT * FROM read_parquet('{matches_path}') WHERE ad_id IN ({id_list}) ORDER BY ad_id, rank"
    ).fetchdf()
    matches_by_ad = {}
    for ad_id, group in chunk_matches.groupby("ad_id"):
        matches_by_ad[int(ad_id)] = group.to_dict("records")

    for ad_id_raw, data_str in chunk:
        ad_id = int(ad_id_raw)
        if ad_id not in matches_by_ad:
            continue
        parsed = json.loads(data_str)
        drop_set = set(parsed["drop"])  # 1-based indices
        candidates = matches_by_ad[ad_id]
        kept = [c for i, c in enumerate(candidates) if (i + 1) not in drop_set]
        for rank, c in enumerate(kept):
            filtered_rows.append({
                "ad_id": ad_id,
                "rank": rank,
                "onet_code": c["onet_code"],
                "onet_title": c["onet_title"],
                "role_score": c["role_score"],
                "taskskill_score": c["taskskill_score"],
                "combined_score": c["combined_score"],
            })

_matches_conn.close()
_summaries_conn.close()
_ads_conn.close()

filtered_df = pd.DataFrame(filtered_rows)
filtered_path = output_dir / "filtered_matches.parquet"
filtered_df.to_parquet(filtered_path, index=False)

failed_set = set(filter_meta["failed_ids"])
successful_ad_ids = [i for i in ad_ids if i not in failed_set]

print(f"llm_filter: {len(filtered_df)} filtered match rows for {len(successful_ad_ids)} ads")
print(f"  mean candidates kept: {len(filtered_df) / max(len(successful_ad_ids), 1):.1f}")
print(f"  output: {filtered_path}")

successful_ad_ids #|func_return_line

# %% [markdown]
# ## Sample filtered matches

# %%
import duckdb
from ai_index.utils import get_adzuna_conn
from ai_index.nodes.llm_summarise import JobInfoModel

sample_ids = ad_ids[:5]
id_list = ",".join(str(int(i)) for i in sample_ids)

# Load sample context
conn = get_adzuna_conn(read_only=True)
conn.execute(f"CREATE OR REPLACE TEMP TABLE _sample_ids AS SELECT unnest([{id_list}]::BIGINT[]) AS id")
raw_ads = {int(r[0]): {"title": r[1], "category_name": r[2], "description": r[3]}
           for r in conn.execute(
    "SELECT a.id, a.title, a.category_name, a.description FROM ads a JOIN _sample_ids s ON a.id = s.id"
).fetchall()}
conn.close()

sconn = duckdb.connect(str(summaries_db), read_only=True)
summaries = {int(r[0]): JobInfoModel.model_validate_json(r[1])
             for r in sconn.execute(
    f"SELECT id, data FROM results WHERE error IS NULL AND id IN ({id_list})"
).fetchall()}
sconn.close()

mconn = duckdb.connect()
before_df = mconn.execute(
    f"SELECT * FROM read_parquet('{matches_path}') WHERE ad_id IN ({id_list}) ORDER BY ad_id, rank"
).fetchdf()
mconn.close()

for ad_id in sample_ids:
    ad_id = int(ad_id)
    raw = raw_ads[ad_id]
    summary = summaries[ad_id]
    before = before_df[before_df["ad_id"] == ad_id].sort_values("rank")
    after = filtered_df[filtered_df["ad_id"] == ad_id].sort_values("rank")
    print(f"\n{'━'*80}")
    print(f"Ad {ad_id}: {raw['title']}")
    print(f"  Sector: {raw['category_name']}")
    print(f"  Domain: {summary.domain} | Level: {summary.level}")
    print(f"  Summary: {summary.short_description}")
    print(f"  Tasks: {', '.join(summary.tasks)}")
    print(f"  Skills: {', '.join(summary.skills)}")
    desc = (raw["description"] or "")[:200].strip()
    if desc:
        print(f"  Description: {desc}...")
    print(f"{'─'*80}")
    print(f"  BEFORE ({len(before)} candidates):")
    for _, row in before.iterrows():
        kept = row["onet_title"] in after["onet_title"].values
        marker = "KEEP" if kept else "DROP"
        print(f"    #{row['rank']+1}  [{marker}]  {row['onet_title']:<45s}  "
              f"combined={row['combined_score']:.4f}")
    print(f"  AFTER ({len(after)} candidates):")
    for _, row in after.iterrows():
        print(f"    #{row['rank']+1}  {row['onet_title']:<45s}  "
              f"combined={row['combined_score']:.4f}")
