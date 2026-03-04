# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compute Cosine Similarity
#
# For each job ad, compute top-K cosine similarity matches against O*NET
# occupations. Merges role and task top-K lists: occupations appearing in both
# get their scores averaged, then takes top-10 overall candidates.

# %%
#|default_exp nodes.compute_cosine_sim
#|export_as_func true

# %%
#|set_func_signature
def main(onet_embed_meta, job_embed_meta, ctx, print) -> {"candidates_meta": dict}:
    """Compute top-K cosine similarity between job ads and O*NET occupations."""
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

import numpy as np
import pandas as pd

from ai_index.const import pipeline_store_path

run_name = ctx.vars["run_name"]
topk = int(ctx.vars["topk"])

store_dir = pipeline_store_path / run_name / "compute_cosine_sim"

# %%
#|export
# Load O*NET embeddings
onet_data = np.load(onet_embed_meta["store_path"])
onet_role = onet_data["role_embeddings"].astype(np.float32)
onet_task = onet_data["task_embeddings"].astype(np.float32)
onet_soc_codes = onet_embed_meta["soc_codes"]
onet_titles = onet_embed_meta["titles"]

print(f"compute_cosine_sim: loaded O*NET embeddings: {onet_role.shape}")

# %%
#|export
def _merge_topk(role_indices, role_scores, task_indices, task_scores, k, soc_codes, titles):
    """Merge role and task top-K results for a batch of job ads.

    For occupations appearing in both lists, average scores.
    Returns top-10 candidates per job ad.
    """
    n = len(role_indices)
    results = []
    for i in range(n):
        combo = {}
        for j in range(k):
            idx = int(role_indices[i, j])
            combo.setdefault(idx, []).append(float(role_scores[i, j]))
        for j in range(k):
            idx = int(task_indices[i, j])
            combo.setdefault(idx, []).append(float(task_scores[i, j]))

        # Average and sort
        candidates = sorted(
            ((idx, sum(scores) / len(scores)) for idx, scores in combo.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        results.append([
            (soc_codes[idx], titles[idx], score)
            for idx, score in candidates
        ])
    return results

# %%
#|export
month_metas = []

for month_info in job_embed_meta["months"]:
    year = month_info["year"]
    filename = month_info["filename"]
    embed_path = Path(month_info["embed_path"])
    expected_count = month_info["row_count"]

    out_dir = store_dir / year
    out_path = out_dir / filename.replace(".npz", ".parquet")

    # Check cache
    if out_path.exists():
        existing = pd.read_parquet(out_path, columns=["job_id"])
        if len(existing) >= expected_count:
            print(f"compute_cosine_sim: {year}/{filename} — {len(existing)} cached, skipping")
            month_metas.append({
                "year": year,
                "filename": filename,
                "path": str(out_path),
                "row_count": len(existing),
            })
            continue

    # Load job embeddings
    job_data = np.load(embed_path, allow_pickle=True)
    job_ids = job_data["job_ids"]
    job_role = job_data["role_embeddings"].astype(np.float32)
    job_task = job_data["task_embeddings"].astype(np.float32)

    print(f"compute_cosine_sim: {year}/{filename} — computing top-{topk} for {len(job_ids)} ads")

    # Compute cosine similarity (dot product since vectors are L2-normalized)
    # Process in batches to limit memory
    batch_size = 8192
    all_results = []

    for b0 in range(0, len(job_ids), batch_size):
        b1 = min(b0 + batch_size, len(job_ids))
        jr = job_role[b0:b1]
        jt = job_task[b0:b1]

        sim_r = jr @ onet_role.T  # (batch, n_occ)
        sim_t = jt @ onet_task.T

        # Top-K
        role_top_idx = np.argpartition(-sim_r, topk, axis=1)[:, :topk]
        task_top_idx = np.argpartition(-sim_t, topk, axis=1)[:, :topk]

        # Get actual scores for top-K indices
        role_top_scores = np.take_along_axis(sim_r, role_top_idx, axis=1)
        task_top_scores = np.take_along_axis(sim_t, task_top_idx, axis=1)

        # Sort within top-K by score descending
        r_order = np.argsort(-role_top_scores, axis=1)
        t_order = np.argsort(-task_top_scores, axis=1)
        role_top_idx = np.take_along_axis(role_top_idx, r_order, axis=1)
        role_top_scores = np.take_along_axis(role_top_scores, r_order, axis=1)
        task_top_idx = np.take_along_axis(task_top_idx, t_order, axis=1)
        task_top_scores = np.take_along_axis(task_top_scores, t_order, axis=1)

        batch_results = _merge_topk(
            role_top_idx, role_top_scores,
            task_top_idx, task_top_scores,
            topk, onet_soc_codes, onet_titles,
        )
        all_results.extend(batch_results)

    # Build output DataFrame
    rows = []
    for job_id, candidates in zip(job_ids, all_results):
        soc_codes_list = [c[0] for c in candidates]
        titles_list = [c[1] for c in candidates]
        scores_list = [c[2] for c in candidates]
        rows.append({
            "job_id": str(job_id),
            "candidate_soc_codes": soc_codes_list,
            "candidate_titles": titles_list,
            "candidate_scores": scores_list,
        })

    df_out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, compression="snappy")
    print(f"compute_cosine_sim: {year}/{filename} — saved {len(df_out)} results")

    month_metas.append({
        "year": year,
        "filename": filename,
        "path": str(out_path),
        "row_count": len(df_out),
    })

total = sum(m["row_count"] for m in month_metas)
print(f"compute_cosine_sim: {total} total ads matched across {len(month_metas)} months")

# %%
#|export
candidates_meta = {
    "months": month_metas,
    "topk": topk,
    "total_ads": total,
}

{"candidates_meta": candidates_meta}  #|func_return_line
