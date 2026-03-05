#!/bin/bash
#SBATCH --job-name=z_build_df_missing_desc
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --gpus=0

set -euo pipefail

source /projects/a5u/adu_dev/aisi-economy-index/.venv/bin/activate

python - << 'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store/isambard/202601/adzuna_2022_all_months")

OUT_DIR = Path("/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store/AISI_demo/stage_3/prod/llm_negative_selection")
OUT_PARQUET = OUT_DIR / "df_missing_with_desc.parquet"
OUT_NPZ = OUT_DIR / "df_missing_with_desc.npz"  # uncompressed (fast)

def canon_id(x):
    s = "" if x is None else str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return str(int(float(s)))
    except Exception:
        return s

# ------------------------------------------------------------
# 1) Build df_missing from missing_job_ids.txt (months 01-12)
# ------------------------------------------------------------
rows = []
for m in range(1, 13):
    month = f"adzuna_month{m:02d}"
    missing_path = ROOT / f"{month}_npz" / f"{month}_missing_job_ids.txt"
    if not missing_path.exists():
        print("Missing:", missing_path, flush=True)
        continue

    ids = [x.strip() for x in missing_path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    rows.extend((month, jid) for jid in ids)

df_missing = pd.DataFrame(rows, columns=["month", "job_id"])
print("Raw rows:", len(df_missing), flush=True)

# dedup on job_id (keeps first month occurrence)
df_missing = df_missing.drop_duplicates("job_id").reset_index(drop=True)
print("Dedup rows:", len(df_missing), flush=True)

df_missing["job_id_norm"] = df_missing["job_id"].map(canon_id)
df_missing["job_description"] = ""

# ------------------------------------------------------------
# 2) Fill descriptions from monthly base NPZ files
# ------------------------------------------------------------
for month, g in tqdm(df_missing.groupby("month"), desc="Months", unit="month"):

    base_npz = ROOT / f"{month}_npz" / f"{month}.npz"
    if not base_npz.exists():
        print("Missing base npz:", base_npz, flush=True)
        continue

    z = np.load(base_npz, allow_pickle=True)

    if "id" in z.files:
        ids = z["id"]
    elif "job_ids" in z.files:
        ids = z["job_ids"]
    else:
        raise KeyError(f"No id field in {base_npz.name}. Have: {z.files}")

    if "description" in z.files:
        desc = z["description"]
    elif "job_description" in z.files:
        desc = z["job_description"]
    else:
        raise KeyError(f"No description field in {base_npz.name}. Have: {z.files}")

    ids_norm = np.array([canon_id(x) for x in ids], dtype=object)

    order = np.argsort(ids_norm)
    ids_sorted = ids_norm[order]
    desc_sorted = desc[order]

    want = g["job_id_norm"].values
    pos = np.searchsorted(ids_sorted, want)
    hit = (pos < len(ids_sorted)) & (ids_sorted[pos] == want)

    out = np.full(len(want), "", dtype=object)
    out[hit] = desc_sorted[pos[hit]]

    df_missing.loc[g.index, "job_description"] = out

    print(f"{month}: matched {int(hit.sum()):,} / {len(want):,}", flush=True)

filled = (df_missing["job_description"].astype(str).str.strip() != "").mean()
print("Filled share:", float(filled), flush=True)

# ------------------------------------------------------------
# 3) Save outputs (fast)
# ------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Saving parquet:", OUT_PARQUET, flush=True)
df_missing.to_parquet(OUT_PARQUET, index=False)

print("Saving npz (uncompressed):", OUT_NPZ, flush=True)
np.savez(
    OUT_NPZ,
    month=df_missing["month"].values.astype(str),
    job_id=df_missing["job_id"].values.astype(str),
    job_id_norm=df_missing["job_id_norm"].values.astype(str),
    job_description=df_missing["job_description"].astype(object).values
)

print("Saved:", OUT_PARQUET, flush=True)
print("Saved:", OUT_NPZ, flush=True)
PY