# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM Filter Failure Diagnosis
#
# The qwen-14b-sbatch model had a 41% failure rate in the LLM filter during
# calibration (412/1000 ads failed after 3 retries). The qwen-7b-sbatch only
# had 1 failure. This notebook diagnoses the issue by:
#
# 1. Loading the failed responses from filter_results.duckdb
# 2. Categorising the failure modes (malformed JSON, wrong schema, out-of-range, etc.)
# 3. Comparing successful vs failed responses to identify patterns
# 4. Testing fixes (prompt tweaks, schema enforcement) on a small sample

# %%
import json
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from ai_index.const import pipeline_store_path
from ai_index.nodes.llm_filter_candidates import FilterResponseModel
from ai_index.utils import extract_json

# %% [markdown]
# ## 1. Load filter results from the most recent calibration run
#
# The calibration run stores all LLM responses (including failures) in
# `filter_results.duckdb` with columns: id, data, error.

# %%
# Find the most recent calibration filter results
cal_filter_dir = pipeline_store_path / "calibration" / "llm_filter_candidates"
db_path = cal_filter_dir / "filter_results.duckdb"

if not db_path.exists():
    # Try to find any validation run with failures
    for d in sorted(pipeline_store_path.iterdir(), reverse=True):
        candidate = d / "llm_filter_candidates" / "filter_results.duckdb"
        if candidate.exists():
            db_path = candidate
            break

print(f"Loading from: {db_path}")

conn = duckdb.connect(str(db_path), read_only=True)
all_results = conn.execute("SELECT id, data, error FROM results").fetchdf()
conn.close()

n_total = len(all_results)
n_ok = all_results["error"].isna().sum()
n_failed = all_results["error"].notna().sum()
print(f"Total: {n_total}, OK: {n_ok}, Failed: {n_failed} ({100*n_failed/n_total:.1f}%)")

# %% [markdown]
# ## 2. Categorise failure modes
#
# Group errors by type to understand what's going wrong.

# %%
failed = all_results[all_results["error"].notna()].copy()

if len(failed) == 0:
    print("No failures to diagnose!")
else:
    # Categorise errors
    def categorise_error(error_str):
        if "ValidationError" in error_str:
            if "must keep at least 1" in error_str:
                return "empty_keep"
            if "keep indices must be 1-based" in error_str:
                return "zero_or_negative_index"
            return "validation_error"
        if "JSONDecodeError" in error_str:
            return "malformed_json"
        if "out of range" in error_str:
            return "index_out_of_range"
        if "Failed to extract JSON" in error_str:
            return "no_json_in_reasoning"
        return "other"

    failed["error_category"] = failed["error"].apply(categorise_error)
    category_counts = failed["error_category"].value_counts()
    print("Error categories:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} ({100*count/len(failed):.1f}%)")

# %% [markdown]
# ## 3. Inspect sample failed responses
#
# Look at actual LLM outputs to understand the failure pattern.

# %%
if len(failed) > 0:
    for category in failed["error_category"].unique():
        cat_samples = failed[failed["error_category"] == category].head(3)
        print(f"\n{'='*80}")
        print(f"Category: {category} ({len(failed[failed['error_category'] == category])} total)")
        print(f"{'='*80}")
        for _, row in cat_samples.iterrows():
            print(f"\n--- Ad {row['id']} ---")
            print(f"Error: {row['error'][:200]}")
            # Show first 500 chars of the response
            data = row["data"]
            print(f"Response ({len(data)} chars):")
            print(data[:500])
            if len(data) > 500:
                print("...")

# %% [markdown]
# ## 4. Attempt JSON extraction from failed responses
#
# Some models wrap JSON in markdown code blocks or add preamble text.
# Try extract_json to see if the JSON is present but wrapped.

# %%
if len(failed) > 0:
    n_recoverable = 0
    recovery_examples = []

    for _, row in failed.iterrows():
        parsed = extract_json(row["data"])
        if parsed is not None:
            try:
                model = FilterResponseModel.model_validate(parsed)
                n_recoverable += 1
                if len(recovery_examples) < 5:
                    recovery_examples.append({
                        "ad_id": row["id"],
                        "extracted": parsed,
                        "original_error": row["error_category"],
                    })
            except Exception:
                pass

    print(f"Recoverable with extract_json: {n_recoverable}/{len(failed)} ({100*n_recoverable/len(failed):.1f}%)")

    if recovery_examples:
        print("\nRecovery examples:")
        for ex in recovery_examples:
            print(f"  Ad {ex['ad_id']}: {ex['extracted']} (was: {ex['original_error']})")

# %% [markdown]
# ## 5. Compare successful response patterns
#
# Look at what successful responses look like to understand the gap.

# %%
ok = all_results[all_results["error"].isna()].copy()

if len(ok) > 0:
    # Parse successful responses
    keep_lengths = []
    for _, row in ok.head(100).iterrows():
        parsed = json.loads(row["data"])
        keep_lengths.append(len(parsed["keep"]))

    print(f"Successful responses (sample of {len(keep_lengths)}):")
    print(f"  Mean candidates kept: {sum(keep_lengths)/len(keep_lengths):.1f}")
    print(f"  Min: {min(keep_lengths)}, Max: {max(keep_lengths)}")

    # Show a few
    print("\nSample successful responses:")
    for _, row in ok.head(5).iterrows():
        print(f"  Ad {row['id']}: {row['data'][:100]}")

# %% [markdown]
# ## 6. Check if json_schema enforcement is working
#
# When json_schema is passed to vLLM, it should constrain output to valid JSON.
# If failures are happening despite schema enforcement, the schema might not
# be reaching the model, or the model might be ignoring it.

# %%
print("FilterResponseModel JSON schema:")
print(json.dumps(FilterResponseModel.model_json_schema(), indent=2))

# %% [markdown]
# ## 7. Cross-model comparison
#
# If multiple calibration results exist, compare failure rates across models.

# %%
import os

results_dir = Path("store/calibration/results")
if results_dir.exists():
    model_stats = []
    for result_file in sorted(results_dir.glob("*.json")):
        with open(result_file) as f:
            r = json.load(f)
        llm = r["llm_model"]
        filter_meta = r["nodes"].get("llm_filter_candidates", {})
        n = filter_meta.get("n", 0)
        model_stats.append({
            "llm_model": llm,
            "embed_model": r["embedding_model"],
            "rerank_model": r.get("rerank_model", "(default)"),
            "n_ads": n,
            "file": result_file.name,
        })

    if model_stats:
        stats_df = pd.DataFrame(model_stats)
        print("Calibrated models:")
        print(stats_df.to_string(index=False))
    else:
        print("No calibration results found.")
else:
    print(f"Results directory not found: {results_dir}")

# %% [markdown]
# ## 8. Per-model failure analysis from filter_meta.json
#
# Check the filter_meta.json files from different calibration runs
# for failure counts.

# %%
for run_dir in sorted(pipeline_store_path.iterdir()):
    meta_path = run_dir / "llm_filter_candidates" / "filter_meta.json"
    if not meta_path.exists():
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    n_total = meta["n_total"]
    n_failed = len(meta["failed_ids"])
    if n_failed > 0:
        print(f"{run_dir.name}: {n_failed}/{n_total} failed ({100*n_failed/n_total:.1f}%)")
