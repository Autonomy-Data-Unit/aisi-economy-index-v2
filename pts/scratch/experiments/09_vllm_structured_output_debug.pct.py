# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # vLLM Structured Output Debug
#
# Investigation into why some LLM models (qwen-14b, qwen-32b) produce
# truncated JSON output (`{`) in the llm_filter_candidates node despite
# `json_schema` enforcement via vLLM's `StructuredOutputsParams`.
#
# ## Key findings from calibration runs:
# - qwen-7b: 1/1000 failures (0.1%)
# - qwen-14b: 412/1000 failures (41.2%)
# - qwen-32b: 227/1000 failures (22.7%)
# - All failures return exactly `{` (1 char) as the response
# - Retries don't recover most failures (deterministic per prompt)
# - Prompt tokens ~570, max_model_len=4096, so context limit is NOT the issue
#
# ## Hypotheses:
# 1. vLLM's guided decoding (outlines/xgrammar) crashes for certain prompts
#    on larger models, returning partial output
# 2. The StructuredOutputsParams API isn't being applied correctly
# 3. Model-specific tokenizer issue with the chat template
# 4. Prompt content triggers an edge case in the guided decoding FSM
#
# ## Test plan:
# 1. Run the same prompt through qwen-7b and qwen-32b on Isambard
# 2. Test with and without json_schema to isolate guided decoding
# 3. Check if the failure is prompt-specific (same ads always fail)

# %%
import json
import duckdb
import pandas as pd
from pathlib import Path

from ai_index.const import pipeline_store_path
from ai_index.utils import get_adzuna_conn, allm_generate
from ai_index.utils.prompts import load_prompt
from ai_index.utils.batch import strict_format

# %% [markdown]
# ## 1. Load failed ad IDs from calibration runs

# %%
# Find all cal__ runs with filter results
failed_by_model = {}
for run_dir in sorted(pipeline_store_path.iterdir()):
    if not run_dir.name.startswith("cal__"):
        continue
    meta_path = run_dir / "llm_filter_candidates" / "filter_meta.json"
    db_path = run_dir / "llm_filter_candidates" / "filter_results.duckdb"
    if not meta_path.exists() or not db_path.exists():
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    n_failed = len(meta["failed_ids"])
    n_total = meta["n_total"]

    # Extract LLM model from run name: cal__<llm>__<embed>[__<rerank>]
    parts = run_dir.name.split("__")
    llm = parts[1] if len(parts) >= 3 else "unknown"

    # Get response lengths for failed
    conn = duckdb.connect(str(db_path), read_only=True)
    failed_data = conn.execute(
        "SELECT id, LENGTH(data) as len, data FROM results WHERE error IS NOT NULL"
    ).fetchall()
    conn.close()

    response_lengths = [r[1] for r in failed_data]
    sample_responses = [(r[0], r[2]) for r in failed_data[:3]]

    failed_by_model[llm] = {
        "run_name": run_dir.name,
        "n_total": n_total,
        "n_failed": n_failed,
        "rate": f"{100*n_failed/n_total:.1f}%",
        "response_lengths": response_lengths,
        "sample_responses": sample_responses,
        "failed_ids": meta["failed_ids"][:20],
    }

    print(f"{llm}: {n_failed}/{n_total} ({100*n_failed/n_total:.1f}%) "
          f"response_lens={set(response_lengths)}")

# %% [markdown]
# ## 2. Build a test prompt from a consistently-failing ad
#
# Find an ad ID that fails across multiple models.

# %%
from collections import Counter

all_failed_ids = []
for model, info in failed_by_model.items():
    all_failed_ids.extend(info["failed_ids"])

id_counts = Counter(all_failed_ids)
# Find IDs that fail in multiple models
common_failures = [aid for aid, count in id_counts.most_common(20) if count > 1]
print(f"Ad IDs failing in multiple models: {common_failures[:10]}")

# If none fail in multiple models, just pick from the highest-failure model
if not common_failures:
    worst_model = max(failed_by_model, key=lambda m: failed_by_model[m]["n_failed"])
    test_ad_ids = failed_by_model[worst_model]["failed_ids"][:5]
    print(f"Using failed IDs from {worst_model}: {test_ad_ids}")
else:
    test_ad_ids = common_failures[:5]

# %% [markdown]
# ## 3. Reconstruct the exact prompt for a failing ad

# %%
SYSTEM_PROMPT = load_prompt("llm_filter/v2/system")
USER_PROMPT_TEMPLATE = load_prompt("llm_filter/v2/user")

# Load candidates for test ads from any available cal__ run
run_with_data = next(iter(failed_by_model.values()))["run_name"]
candidates_path = pipeline_store_path / run_with_data / "cosine_candidates" / "candidates.parquet"
candidates_df = pd.read_parquet(candidates_path)

# Load raw ads
ads_conn = get_adzuna_conn(read_only=True)
id_list = ",".join(str(i) for i in test_ad_ids)
raw_ads = {}
for row in ads_conn.execute(
    f"SELECT id, title, category_name, description FROM ads WHERE id IN ({id_list})"
).fetchall():
    raw_ads[row[0]] = {"title": row[1], "category_name": row[2], "description": row[3]}
ads_conn.close()

# Build prompts
test_prompts = []
for ad_id in test_ad_ids:
    if ad_id not in raw_ads:
        continue
    ad_candidates = candidates_df[candidates_df["ad_id"] == ad_id].sort_values("rank")
    candidates_str = "\n".join(
        f"{i+1}. {row['onet_title']}" for i, (_, row) in enumerate(ad_candidates.iterrows())
    )
    raw_ad = raw_ads[ad_id]
    full_ad_excerpt = (raw_ad["description"] or "")[:1200].strip()

    prompt = strict_format(
        USER_PROMPT_TEMPLATE,
        n_candidates=len(ad_candidates),
        job_ad_title=raw_ad["title"] or "",
        job_sector_category=raw_ad["category_name"] or "",
        full_ad_excerpt=full_ad_excerpt,
        candidates_str=candidates_str,
    )
    test_prompts.append((ad_id, prompt))
    print(f"Ad {ad_id}: prompt length={len(prompt)} chars, {len(ad_candidates)} candidates")

# %% [markdown]
# ## 4. Test: Run the same prompt with json_schema ON vs OFF
#
# This isolates whether the issue is in guided decoding or the model itself.

# %%
from ai_index.nodes.llm_filter_candidates import FilterResponseModel

schema = FilterResponseModel.model_json_schema()

# Test with a model that fails a lot (if available on sbatch)
test_model = "qwen-7b-sbatch"  # Start with one that works
ad_id, prompt = test_prompts[0]

print(f"Testing ad {ad_id} with {test_model}")
print(f"Prompt: {prompt[:200]}...")
print()

# With schema
result_with = await allm_generate(
    [prompt],
    model=test_model,
    system_message=SYSTEM_PROMPT,
    max_new_tokens=2048,
    json_schema=schema,
)
print(f"With json_schema: {result_with[0][:200]}")

# Without schema
result_without = await allm_generate(
    [prompt],
    model=test_model,
    system_message=SYSTEM_PROMPT,
    max_new_tokens=2048,
)
print(f"Without json_schema: {result_without[0][:200]}")

# %% [markdown]
# ## 5. Test with the failing model
#
# Run the same prompt through the model that has high failure rates.

# %%
test_model_fail = "qwen-32b-sbatch"  # Or whichever model has failures

print(f"Testing ad {ad_id} with {test_model_fail}")

# With schema
result_fail_with = await allm_generate(
    [prompt],
    model=test_model_fail,
    system_message=SYSTEM_PROMPT,
    max_new_tokens=2048,
    json_schema=schema,
)
print(f"With json_schema: {result_fail_with[0][:200]}")

# Without schema
result_fail_without = await allm_generate(
    [prompt],
    model=test_model_fail,
    system_message=SYSTEM_PROMPT,
    max_new_tokens=2048,
)
print(f"Without json_schema: {result_fail_without[0][:200]}")

# %% [markdown]
# ## 6. Batch test: run all failing prompts without schema
#
# If the model produces valid JSON without guided decoding, the issue is
# in vLLM's structured output. If it produces garbage, the model itself
# can't follow the instruction.

# %%
all_test_prompts = [p for _, p in test_prompts]
all_test_ids = [aid for aid, _ in test_prompts]

results_no_schema = await allm_generate(
    all_test_prompts,
    model=test_model_fail,
    system_message=SYSTEM_PROMPT,
    max_new_tokens=2048,
)

for ad_id, response in zip(all_test_ids, results_no_schema):
    print(f"\nAd {ad_id} (no schema):")
    print(f"  {response[:300]}")
    # Try to parse as JSON
    try:
        parsed = json.loads(response)
        print(f"  Valid JSON: {parsed}")
    except json.JSONDecodeError:
        # Try extract_json
        from ai_index.utils import extract_json
        extracted = extract_json(response)
        if extracted:
            print(f"  Extracted JSON: {extracted}")
        else:
            print(f"  Not parseable as JSON")
