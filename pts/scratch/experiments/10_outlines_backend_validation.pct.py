# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Structured Output Bypass Validation
#
# Tests whether disabling `json_schema` (structured output) and using
# `extract_json` with pydantic validation fixes the `{` truncation bug
# for misbehaving models.
#
# Reconstructs the same 1000 prompts from a calibration run and re-runs
# them through the model WITHOUT `json_schema`. Responses are parsed via
# `extract_json` with `FilterResponseModel.model_validate` as the validator.
# Compares failure rates against the original calibration (which used json_schema).
#
# Usage:
#     uv run python pts/scratch/experiments/10_outlines_backend_validation.pct.py [model_key]
#
# Defaults to `qwen-32b-sbatch` if no model key is provided.

# %%
import asyncio
import json
import sys
import time

import duckdb
import pandas as pd

from ai_index.const import pipeline_store_path
from ai_index.utils import get_adzuna_conn, allm_generate, extract_json
from ai_index.utils.prompts import load_prompt
from ai_index.utils.batch import strict_format
from ai_index.nodes.llm_filter_candidates import FilterResponseModel

# %%
MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen-32b-sbatch"
print(f"Model: {MODEL_KEY}")

# %% [markdown]
# ## 1. Find calibration run and load original failure data

# %%
cal_run_dir = None
for run_dir in sorted(pipeline_store_path.iterdir()):
    if not run_dir.name.startswith("cal__"):
        continue
    parts = run_dir.name.split("__")
    if len(parts) >= 3 and parts[1] == MODEL_KEY:
        meta_path = run_dir / "llm_filter_candidates" / "filter_meta.json"
        if meta_path.exists():
            cal_run_dir = run_dir
            break

if cal_run_dir is None:
    print(f"ERROR: No calibration run found for {MODEL_KEY}")
    sys.exit(1)

print(f"Calibration run: {cal_run_dir.name}")

with open(cal_run_dir / "llm_filter_candidates" / "filter_meta.json") as f:
    original_meta = json.load(f)

original_failed = set(original_meta["failed_ids"])
original_total = original_meta["n_total"]
print(f"Original (with json_schema): {len(original_failed)}/{original_total} failures "
      f"({100 * len(original_failed) / original_total:.1f}%)")

# %% [markdown]
# ## 2. Load candidates and raw ads

# %%
candidates_path = cal_run_dir / "cosine_candidates" / "candidates.parquet"
candidates_df = pd.read_parquet(candidates_path)
all_ad_ids = sorted(candidates_df["ad_id"].unique())
print(f"Candidates: {len(all_ad_ids)} ads")

ads_conn = get_adzuna_conn(read_only=True)
id_list = ",".join(str(int(i)) for i in all_ad_ids)
raw_ads = {}
for row in ads_conn.execute(
    f"SELECT id, title, category_name, description FROM ads WHERE id IN ({id_list})"
).fetchall():
    raw_ads[int(row[0])] = {"title": row[1], "category_name": row[2], "description": row[3]}
ads_conn.close()
print(f"Raw ads: {len(raw_ads)} loaded")

# %% [markdown]
# ## 3. Build prompts (same construction as llm_filter_candidates node)

# %%
SYSTEM_PROMPT = load_prompt("llm_filter/v2/system_unstructured")
USER_PROMPT_TEMPLATE = load_prompt("llm_filter/v2/user_unstructured")

prompts = []
prompt_ad_ids = []
n_candidates_per_ad = []

for ad_id in all_ad_ids:
    if ad_id not in raw_ads:
        continue
    ad_candidates = candidates_df[candidates_df["ad_id"] == ad_id].sort_values("rank")
    candidates_list = ad_candidates.to_dict("records")
    candidates_str = "\n".join(
        f"{i+1}. {c['onet_title']}" for i, c in enumerate(candidates_list)
    )
    raw_ad = raw_ads[ad_id]
    full_ad_excerpt = (raw_ad["description"] or "")[:1200].strip()

    prompt = strict_format(
        USER_PROMPT_TEMPLATE,
        n_candidates=len(candidates_list),
        job_ad_title=raw_ad["title"] or "",
        job_sector_category=raw_ad["category_name"] or "",
        full_ad_excerpt=full_ad_excerpt,
        candidates_str=candidates_str,
    )
    prompts.append(prompt)
    prompt_ad_ids.append(ad_id)
    n_candidates_per_ad.append(len(candidates_list))

print(f"Built {len(prompts)} prompts")

# %% [markdown]
# ## 4. Run all prompts WITHOUT json_schema
#
# The model has `structured_output = false` in `llm_models.toml`, so the
# pipeline node would skip `json_schema`. Here we replicate that by not
# passing `json_schema` at all.

# %%
async def run_test():
    print(f"\nRunning {len(prompts)} prompts through {MODEL_KEY} (no json_schema)...")
    t0 = time.time()
    results = await allm_generate(
        prompts,
        model=MODEL_KEY,
        system_message=SYSTEM_PROMPT,
        max_new_tokens=2048,
        time="00:15:00",
    )
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")
    return results


results = asyncio.run(run_test())

# %% [markdown]
# ## 5. Parse results with extract_json + pydantic validation

# %%
new_failures = []
fixed_ads = []

for ad_id, response, n_cands in zip(prompt_ad_ids, results, n_candidates_per_ad):
    parsed = extract_json(response, validator=FilterResponseModel.model_validate)
    if parsed is None:
        new_failures.append((ad_id, response[:100], "extract_json returned None"))
        continue
    # Check index ranges
    try:
        for idx in parsed["keep"]:
            if idx < 1 or idx > n_cands:
                raise ValueError(f"keep index {idx} out of range [1, {n_cands}]")
        if ad_id in original_failed:
            fixed_ads.append(ad_id)
    except Exception as e:
        new_failures.append((ad_id, response[:100], str(e)))

n_new = len(new_failures)
n_total = len(prompts)
n_old = len(original_failed)

print(f"\n{'=' * 60}")
print(f"RESULTS: {MODEL_KEY}")
print(f"{'=' * 60}")
print(f"Original failures (json_schema): {n_old}/{original_total} ({100 * n_old / original_total:.1f}%)")
print(f"New failures (no schema):        {n_new}/{n_total} ({100 * n_new / n_total:.1f}%)")
print(f"Fixed (was failing, now OK):      {len(fixed_ads)}")

if n_old > 0:
    reduction = 100 * (n_old - n_new) / n_old
    print(f"Failure reduction:               {reduction:+.1f}%")

if new_failures:
    repeat = [(aid, s, e) for aid, s, e in new_failures if aid in original_failed]
    regressions = [(aid, s, e) for aid, s, e in new_failures if aid not in original_failed]

    if repeat:
        print(f"\nRepeat failures ({len(repeat)}):")
        for ad_id, snippet, err in repeat[:5]:
            print(f"  ad {ad_id}: {snippet!r}")
            print(f"    error: {err}")

    if regressions:
        print(f"\nNew regressions ({len(regressions)}):")
        for ad_id, snippet, err in regressions[:5]:
            print(f"  ad {ad_id}: {snippet!r}")
            print(f"    error: {err}")
else:
    print(f"\nAll {n_old} previous failures are now fixed.")
