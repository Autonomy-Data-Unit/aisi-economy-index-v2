# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM Filter Candidates
#
# Use LLM negative selection to filter cosine similarity candidates.
# The LLM identifies irrelevant O*NET matches given the job ad context,
# keeping 1-5 candidates per ad.

# %%
#|default_exp nodes.llm_filter_candidates
#|export_as_func true

# %%
#|set_func_signature
def main(candidates_meta, ads_manifest, ctx, print) -> {"filtered_meta": dict}:
    """LLM negative selection to filter cosine similarity candidates."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args()

# %%
#|export
import json
import re
from pathlib import Path

import pandas as pd

from ai_index.const import pipeline_store_path
from ai_index.utils import llm_generate

run_name = ctx.vars["run_name"]
execution_mode = ctx.vars["execution_mode"]
llm_model = ctx.vars["llm_model"]

MAX_KEEP = 5
store_dir = pipeline_store_path / run_name / ctx.node_name

# %%
#|export
PROMPT_TEMPLATE = """You ONLY output one JSON object.
No English. No sentences. No explanation.

Task: Filter out candidates that are IRRELEVANT to the job description.

HARD RULES:
- You MUST keep between 1 and {max_keep} candidates only.
- Therefore you MUST drop ALL other candidates (even if somewhat related).
- PRIORITIZE TASKS & DUTIES over the general Industry.
- If uncertain, KEEP FEWER (1 is acceptable).
- IMPORTANT: Do not drop all candidates.

Output JSON format (1-based indices):
{{"drop":[3,7]}}

Job:
Sector: {sector}
Description:
{description}
Candidates:
{candidates_list}"""

_JSON_RE = re.compile(r"\{(?:[^\{}]|(?:\{[^\{}]*\}))*\}", flags=re.DOTALL)

# %%
#|export
def _build_prompt(title, description, category, candidates):
    """Build prompt for a single job ad."""
    desc_text = f"{title}. {description[:1200]}" if description else title
    cand_lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidates))
    return PROMPT_TEMPLATE.format(
        max_keep=MAX_KEEP,
        sector=category or "Unknown",
        description=desc_text,
        candidates_list=cand_lines,
    )

def _parse_drop_response(response, n_candidates):
    """Parse LLM drop response, return set of 0-based indices to drop."""
    match = _JSON_RE.search(response)
    if not match:
        return set()  # keep all if can't parse
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    drop_indices = parsed.get("drop", [])
    if not isinstance(drop_indices, list):
        return set()
    # Convert 1-based to 0-based, validate bounds
    drop_set = set()
    for x in drop_indices:
        try:
            xi = int(x) - 1
            if 0 <= xi < n_candidates:
                drop_set.add(xi)
        except (ValueError, TypeError):
            continue
    # Safety: never drop all
    if len(drop_set) >= n_candidates:
        drop_set.discard(0)  # always keep first candidate
    return drop_set

# %%
#|export
# Build lookup from ads_manifest for reading raw ad data
ads_paths = {}
for m in ads_manifest["months"]:
    key = (m["year"], m["filename"])
    ads_paths[key] = m["path"]

# %%
#|export
month_metas = []
LLM_BATCH_SIZE = 64

for month_info in candidates_meta["months"]:
    year = month_info["year"]
    filename = month_info["filename"]
    candidates_path = Path(month_info["path"])
    expected_count = month_info["row_count"]

    out_dir = store_dir / year
    out_path = out_dir / filename

    # Check cache
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["job_id"])
        if len(existing) >= expected_count:
            print(f"llm_filter: {year}/{filename} — {len(existing)} cached, skipping")
            month_metas.append({
                "year": year,
                "filename": filename,
                "path": str(out_path),
                "row_count": len(existing),
            })
            continue

    # Load candidates and raw ad data
    cand_df = pd.read_parquet(candidates_path)
    ads_key = (year, filename)
    ads_path = ads_paths.get(ads_key)
    if ads_path is None:
        print(f"llm_filter: {year}/{filename} — no ads path found, skipping")
        continue
    ads_df = pd.read_parquet(ads_path, columns=["id", "title", "description", "category_name"])
    ads_df["id"] = ads_df["id"].astype(str)
    ads_lookup = ads_df.set_index("id")

    # Build prompts
    prompts = []
    prompt_indices = []
    for idx, row in cand_df.iterrows():
        job_id = row["job_id"]
        candidates = row["candidate_titles"]
        if job_id in ads_lookup.index:
            ad = ads_lookup.loc[job_id]
            title = ad.get("title", "") or ""
            description = ad.get("description", "") or ""
            category = ad.get("category_name", "") or ""
        else:
            title, description, category = "", "", ""
        prompts.append(_build_prompt(title, description, category, candidates))
        prompt_indices.append(idx)

    # Call LLM in batches
    print(f"llm_filter: {year}/{filename} — filtering {len(prompts)} ads with {llm_model} (mode={execution_mode})")
    all_responses = []
    for b0 in range(0, len(prompts), LLM_BATCH_SIZE):
        batch = prompts[b0:b0 + LLM_BATCH_SIZE]
        responses = llm_generate(batch, mode=execution_mode, model=llm_model)
        all_responses.extend(responses)
        if b0 + LLM_BATCH_SIZE < len(prompts):
            print(f"  batch {b0//LLM_BATCH_SIZE + 1}: {len(all_responses)}/{len(prompts)}")

    # Parse responses and build output
    rows = []
    for idx, response in zip(prompt_indices, all_responses):
        row = cand_df.loc[idx]
        candidates = row["candidate_titles"]
        soc_codes = row["candidate_soc_codes"]
        scores = row["candidate_scores"]

        drop_set = _parse_drop_response(response, len(candidates))

        kept_titles = [t for i, t in enumerate(candidates) if i not in drop_set][:MAX_KEEP]
        kept_codes = [c for i, c in enumerate(soc_codes) if i not in drop_set][:MAX_KEEP]
        kept_scores = [s for i, s in enumerate(scores) if i not in drop_set][:MAX_KEEP]

        rows.append({
            "job_id": row["job_id"],
            "kept_soc_codes": kept_codes,
            "kept_titles": kept_titles,
            "kept_scores": kept_scores,
        })

    df_out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, compression="snappy")
    print(f"llm_filter: {year}/{filename} — saved {len(df_out)} filtered results")

    month_metas.append({
        "year": year,
        "filename": filename,
        "path": str(out_path),
        "row_count": len(df_out),
    })

total = sum(m["row_count"] for m in month_metas)
print(f"llm_filter: {total} total ads filtered across {len(month_metas)} months")

# %%
#|export
filtered_meta = {
    "months": month_metas,
    "llm_model": llm_model,
    "total_ads": total,
}

{"filtered_meta": filtered_meta}  #|func_return_line
