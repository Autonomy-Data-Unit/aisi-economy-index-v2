# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compute Weights
#
# Normalize kept candidate scores to sum to 1.0 per job ad.
# Produces the final per-job occupation weights for exposure scoring.

# %%
#|default_exp nodes.compute_weights
#|export_as_func true

# %%
#|set_func_signature
def main(filtered_meta, ctx, print) -> {"weighted_codes_meta": dict}:
    """Normalize kept candidate scores to per-job occupation weights."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args()

# %%
#|export
from pathlib import Path

import pandas as pd

from ai_index.const import pipeline_store_path

run_name = ctx.vars["run_name"]
store_dir = pipeline_store_path / run_name / ctx.node_name

# %%
#|export
month_metas = []

for month_info in filtered_meta["months"]:
    year = month_info["year"]
    filename = month_info["filename"]
    filtered_path = Path(month_info["path"])
    expected_count = month_info["row_count"]

    out_dir = store_dir / year
    out_path = out_dir / filename

    # Check cache
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        n_jobs = existing["job_id"].nunique()
        if n_jobs >= expected_count:
            print(f"compute_weights: {year}/{filename} — {n_jobs} jobs cached, skipping")
            month_metas.append({
                "year": year,
                "filename": filename,
                "path": str(out_path),
                "n_jobs": n_jobs,
                "n_rows": len(existing),
            })
            continue

    # Load filtered results
    filt_df = pd.read_parquet(filtered_path)

    # Explode to one row per (job_id, soc_code) and normalize
    rows = []
    for _, row in filt_df.iterrows():
        job_id = row["job_id"]
        codes = row["kept_soc_codes"]
        scores = row["kept_scores"]

        if len(codes) == 0:
            continue

        total = sum(scores)
        if total <= 0:
            # Equal weights if all scores zero
            w = 1.0 / len(codes)
            for code in codes:
                rows.append({"job_id": job_id, "soc_code": code, "weight": w})
        else:
            for code, score in zip(codes, scores):
                rows.append({"job_id": job_id, "soc_code": code, "weight": score / total})

    df_out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, compression="snappy")

    n_jobs = df_out["job_id"].nunique()
    print(f"compute_weights: {year}/{filename} — {n_jobs} jobs, {len(df_out)} rows")

    month_metas.append({
        "year": year,
        "filename": filename,
        "path": str(out_path),
        "n_jobs": n_jobs,
        "n_rows": len(df_out),
    })

total_jobs = sum(m["n_jobs"] for m in month_metas)
total_rows = sum(m["n_rows"] for m in month_metas)
print(f"compute_weights: {total_jobs} jobs, {total_rows} rows across {len(month_metas)} months")

# %%
#|export
weighted_codes_meta = {
    "months": month_metas,
    "total_jobs": total_jobs,
    "total_rows": total_rows,
}

{"weighted_codes_meta": weighted_codes_meta}  #|func_return_line
