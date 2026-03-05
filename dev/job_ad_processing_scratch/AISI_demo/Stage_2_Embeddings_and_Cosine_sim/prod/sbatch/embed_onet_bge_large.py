import os, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import trange

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path("/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store/AISI_demo/stage_2_embeddings_and_cosines/prod/embeddings/")
RUN_DIR  = BASE_DIR / "run_embed_onet_bge"
RUN_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "standard_df_onet_occupations_description_activities_and_tasks.csv"

OUT_ROLE = RUN_DIR / "embeddings_role_onet_bge_large.npy"
OUT_TASK = RUN_DIR / "embeddings_tasks_onet_bge_large.npy"
OUT_IDS  = RUN_DIR / "onet_row_idx_bge.npy"  # keeps alignment with filtered rows

TMP_PREFIX = RUN_DIR / "tmp_chunk_onet"

# ------------------------------------------------------------
# PARAMS
# ------------------------------------------------------------
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-large-en-v1.5")

def main():
    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    roles = df["Job Role Description"].fillna("").astype(str).values
    tasks = df["Work Activities/Tasks/Skills"].fillna("").astype(str).values

    mask = roles != ""
    roles = roles[mask]
    tasks = tasks[mask]
    keep_idx = np.nonzero(mask)[0]  # original row indices

    n = len(roles)
    print(f"ONET rows after drop empty roles: {n}")
    print("CUDA available:", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA not visible (are you on GPU node and correct env?)"

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------
    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )

    # ------------------------------------------------------------
    # RESUME LOGIC
    # ------------------------------------------------------------
    role_chunks, task_chunks, idx_chunks = [], [], []
    start_idx = 0

    tmp_dir = str(RUN_DIR)
    tmp_base = Path(TMP_PREFIX).name

    completed = []
    for f in os.listdir(tmp_dir):
        if not (f.startswith(f"{tmp_base}_role_") and f.endswith(".npy")):
            continue
        m = re.search(r"_role_(\d+)\.npy$", f)
        if not m:
            continue
        i = int(m.group(1))
        if Path(f"{TMP_PREFIX}_task_{i}.npy").exists() and Path(f"{TMP_PREFIX}_idx_{i}.npy").exists():
            completed.append(i)

    if completed:
        completed.sort()
        start_idx = completed[-1] + CHUNK_SIZE
        print(f"Resuming from index {start_idx} ({len(completed)} chunks)")
        for i in completed:
            role_chunks.append(np.load(f"{TMP_PREFIX}_role_{i}.npy"))
            task_chunks.append(np.load(f"{TMP_PREFIX}_task_{i}.npy"))
            idx_chunks.append(np.load(f"{TMP_PREFIX}_idx_{i}.npy"))

    # ------------------------------------------------------------
    # EMBEDDING LOOP
    # ------------------------------------------------------------
    for i in trange(start_idx, n, CHUNK_SIZE, desc="Embedding ONET chunks"):
        sl = slice(i, min(i + CHUNK_SIZE, n))

        role_emb = model.encode(
            roles[sl],
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        task_emb = model.encode(
            tasks[sl],
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        role_chunks.append(role_emb)
        task_chunks.append(task_emb)
        idx_chunks.append(keep_idx[sl])

        np.save(f"{TMP_PREFIX}_role_{i}.npy", role_emb)
        np.save(f"{TMP_PREFIX}_task_{i}.npy", task_emb)
        np.save(f"{TMP_PREFIX}_idx_{i}.npy", keep_idx[sl])

        print(f"chunk {i}:{sl.stop} wrote role {role_emb.shape} task {task_emb.shape}")

    # ------------------------------------------------------------
    # FINAL PACK
    # ------------------------------------------------------------
    role_embeds = np.vstack(role_chunks)
    task_embeds = np.vstack(task_chunks)
    keep_idx_all = np.concatenate(idx_chunks)

    np.save(OUT_ROLE, role_embeds)
    np.save(OUT_TASK, task_embeds)
    np.save(OUT_IDS, keep_idx_all)

    print("Saved:")
    print(" ", OUT_ROLE, role_embeds.shape, role_embeds.dtype)
    print(" ", OUT_TASK, task_embeds.shape, task_embeds.dtype)
    print(" ", OUT_IDS, keep_idx_all.shape, keep_idx_all.dtype)

if __name__ == "__main__":
    main()
