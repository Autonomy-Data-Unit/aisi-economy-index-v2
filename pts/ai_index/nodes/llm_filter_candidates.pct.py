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
    'ad_ids': list[int]
}:
    """Run LLM negative selection to filter cosine match candidates."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dotenv import load_dotenv; load_dotenv()
from dev_utils import set_node_func_args
set_node_func_args('llm_filter_candidates')

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
batch_size = ctx.vars["llm_batch_size"]
max_new_tokens = ctx.vars["llm_max_new_tokens"]
max_concurrent = ctx.vars["llm_max_concurrent_batches"]
resume = ctx.vars["filter_resume"]
max_retries = ctx.vars["filter_max_retries"]

SYSTEM_PROMPT = load_prompt(ctx.vars["system_prompt"])
USER_PROMPT_TEMPLATE = load_prompt(ctx.vars["user_prompt"])

output_dir = const.pipeline_store_path / run_name / "llm_filter_candidates"
output_dir.mkdir(parents=True, exist_ok=True)
db_path = output_dir / "filter_results.duckdb"

# %% [markdown]
# ## Load cosine match results

# %%
#|export
matches_path = const.pipeline_store_path / run_name / "cosine_match" / "matches.parquet"
matches_df = pd.read_parquet(matches_path)

ad_ids_set = set(ad_ids)
matches_df = matches_df[matches_df["ad_id"].isin(ad_ids_set)]

# Group candidates per ad: {ad_id: [{"rank": ..., "onet_title": ..., ...}, ...]}
matches_by_ad = {}
for ad_id, group in matches_df.groupby("ad_id"):
    matches_by_ad[int(ad_id)] = group.sort_values("rank").to_dict("records")

print(f"llm_filter: loaded {len(matches_df)} match rows for {len(matches_by_ad)} ads")

# %% [markdown]
# ## Load LLM summaries

# %%
#|export
summaries_db = const.pipeline_store_path / run_name / "llm_summarise" / "summaries.duckdb"
conn = duckdb.connect(str(summaries_db), read_only=True)
summary_rows = conn.execute(
    "SELECT id, data FROM results WHERE error IS NULL ORDER BY id"
).fetchall()
conn.close()

summaries_by_ad = {}
for row_id, data_str in summary_rows:
    if row_id in ad_ids_set:
        summaries_by_ad[row_id] = JobInfoModel.model_validate_json(data_str)

print(f"llm_filter: loaded {len(summaries_by_ad)} summaries")

# %% [markdown]
# ## Load raw ad metadata

# %%
#|export
ads_conn = get_adzuna_conn(read_only=True)
ads_conn.execute("CREATE OR REPLACE TEMP TABLE _filter_ids (id BIGINT)")
ads_conn.executemany("INSERT INTO _filter_ids VALUES (?)", [(i,) for i in ad_ids])
raw_ads = ads_conn.execute(
    "SELECT a.id, a.title, a.category_name, a.description "
    "FROM ads a JOIN _filter_ids f ON a.id = f.id"
).fetchall()
ads_conn.close()

raw_ads_by_id = {row[0]: {"title": row[1], "category_name": row[2], "description": row[3]} for row in raw_ads}
print(f"llm_filter: loaded {len(raw_ads_by_id)} raw ads")

# %% [markdown]
# ## Define work function
#
# For each chunk of ad IDs, build prompts and call the LLM.

# %%
#|export
def _build_prompt(ad_id: int) -> str:
    """Build the negative selection prompt for one ad."""
    candidates = matches_by_ad[ad_id]
    summary = summaries_by_ad[ad_id]
    raw = raw_ads_by_id[ad_id]

    tasks_str = ", ".join(summary.tasks + summary.skills)[:800]
    candidates_str = "\n".join(
        f"{i+1}. {c['onet_title']}" for i, c in enumerate(candidates)
    )
    full_ad_excerpt = (raw["description"] or "")[:700].strip()

    return strict_format(
        USER_PROMPT_TEMPLATE,
        n_candidates=len(candidates),
        job_ad_title=raw["title"] or "",
        job_sector_category=raw["category_name"] or "",
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


async def _work_fn(chunk_ids):
    """Build prompts, call LLM, validate, return DataFrame."""
    prompts = []
    n_candidates_per_ad = []
    for ad_id in chunk_ids:
        prompts.append(_build_prompt(ad_id))
        n_candidates_per_ad.append(len(matches_by_ad[ad_id]))

    responses = await allm_generate(
        prompts,
        model=llm_model,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
    )

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
)
store.close()
print(f"llm_filter: wrote {db_path}")

meta_path = output_dir / "filter_meta.json"
with open(meta_path, "w") as f:
    json.dump(filter_meta, f, indent=2)
print(f"llm_filter: wrote {meta_path}")

# %% [markdown]
# ## Build filtered matches
#
# Apply the LLM drop decisions to the cosine match results.

# %%
#|export
filter_conn = duckdb.connect(str(db_path), read_only=True)
filter_rows = filter_conn.execute(
    "SELECT id, data FROM results WHERE error IS NULL"
).fetchall()
filter_conn.close()

filtered_rows = []
for ad_id, data_str in filter_rows:
    ad_id = int(ad_id)
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
from ai_index.utils import get_adzuna_conn

conn = get_adzuna_conn(read_only=True)
conn.execute("CREATE OR REPLACE TEMP TABLE _sample_ids (id BIGINT)")
sample_ids = ad_ids[:5]
conn.executemany("INSERT INTO _sample_ids VALUES (?)", [(int(i),) for i in sample_ids])
ad_titles = dict(conn.execute(
    "SELECT a.id, a.title FROM ads a JOIN _sample_ids s ON a.id = s.id"
).fetchall())
conn.close()

for ad_id in sample_ids:
    ad_id = int(ad_id)
    raw = raw_ads_by_id[ad_id]
    summary = summaries_by_ad[ad_id]
    before = matches_df[matches_df["ad_id"] == ad_id].sort_values("rank")
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
