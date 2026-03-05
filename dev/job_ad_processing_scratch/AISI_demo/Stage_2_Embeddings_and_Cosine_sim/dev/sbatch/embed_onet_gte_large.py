import os, re, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import trange

# =============================================================================
# CONFIG (GTE only)
# =============================================================================
BASE_DIR = Path(
    "/projects/a5u/adu_dev/aisi-economy-index/"
    "aisi_economy_index/store/AISI_demo/stage_2_embeddings_and_cosines/dev/embeddings/"
)

RUN_DIR = BASE_DIR / "run_embed_onet_gte"
RUN_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = BASE_DIR / "standard_df_onet_occupations_description_activities_and_tasks.csv"

MODEL_NAME = os.environ.get("MODEL_NAME", "Alibaba-NLP/gte-large-en-v1.5")
assert "gte" in MODEL_NAME.lower(), f"This script is GTE-only. Got MODEL_NAME={MODEL_NAME}"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))

MODEL_TAG = "gte_large_en_v1p5"

OUT_ROLE = RUN_DIR / f"embeddings_role_onet_{MODEL_TAG}.npy"
OUT_TASK = RUN_DIR / f"embeddings_tasks_onet_{MODEL_TAG}.npy"
OUT_IDS  = RUN_DIR / f"onet_row_idx_{MODEL_TAG}.npy"

TMP_PREFIX = RUN_DIR / f"tmp_chunk_onet_{MODEL_TAG}"


# =============================================================================
# UTILS
# =============================================================================
def clean_text_array(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in arr:
        if x is None:
            out.append("")
            continue
        s = str(x).strip()
        if s.lower() in {"nan", "none", "null"}:
            s = ""
        out.append(s)
    return np.array(out, dtype=object)

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def pairwise_cos_stats(X: np.ndarray, name: str, n_pairs: int = 20000, seed: int = 0) -> None:
    X = X.astype(np.float32)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 2:
        print(name, "pairwise cos: <n too small>")
        return
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    same = (i == j)
    j[same] = (j[same] + 1) % n
    cos = (X[i] * X[j]).sum(axis=1)
    print(
        f"{name} pairwise cos: mean {float(cos.mean()):.6f} std {float(cos.std()):.6f} "
        f"p01 {float(np.quantile(cos, 0.01)):.6f} p50 {float(np.quantile(cos, 0.50)):.6f} "
        f"p99 {float(np.quantile(cos, 0.99)):.6f}"
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("start:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("MODEL_NAME:", MODEL_NAME)
    assert CSV_PATH.exists(), f"Missing CSV: {CSV_PATH}"

    df = pd.read_csv(CSV_PATH)

    # Avoid .astype(str) before cleaning to prevent 'nan' strings
    roles_raw = df["Job Role Description"].values
    tasks_raw = df["Work Activities/Tasks/Skills"].values

    roles_strip = clean_text_array(roles_raw)
    tasks_strip = clean_text_array(tasks_raw)

    # Drop empty roles (always)
    mask = roles_strip != ""
    roles = roles_strip[mask]
    tasks = tasks_strip[mask]
    keep_idx = np.nonzero(mask)[0].astype(np.int32)

    # Patch empty tasks so you don't embed empty strings
    roles = np.array([r if r else "unknown role" for r in roles], dtype=object)
    tasks = np.array([t if t else "unknown tasks/skills" for t in tasks], dtype=object)

    n = len(roles)
    print("ONET rows after drop empty roles:", n)
    assert torch.cuda.is_available(), "CUDA not visible"
    print("GPU:", torch.cuda.get_device_name(0))

    # GTE: NO prefixes
    roles_in = roles
    tasks_in = tasks

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    )

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

    for i in trange(start_idx, n, CHUNK_SIZE, desc="Embedding ONET chunks (GTE)"):
        sl = slice(i, min(i + CHUNK_SIZE, n))

        role_emb = model.encode(
            roles_in[sl].tolist(),
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        task_emb = model.encode(
            tasks_in[sl].tolist(),
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

    role_embeds = np.vstack(role_chunks)
    task_embeds = np.vstack(task_chunks)
    keep_idx_all = np.concatenate(idx_chunks)

    # Quick sanity diagnostics
    print("SANITY:")
    print(" role norms min/med/max:",
          float(np.linalg.norm(role_embeds, axis=1).min()),
          float(np.median(np.linalg.norm(role_embeds, axis=1))),
          float(np.linalg.norm(role_embeds, axis=1).max()))
    print(" task norms min/med/max:",
          float(np.linalg.norm(task_embeds, axis=1).min()),
          float(np.median(np.linalg.norm(task_embeds, axis=1))),
          float(np.linalg.norm(task_embeds, axis=1).max()))
    pairwise_cos_stats(role_embeds, "ONET role_embeds (GTE)")
    pairwise_cos_stats(task_embeds, "ONET task_embeds (GTE)")

    np.save(OUT_ROLE, role_embeds)
    np.save(OUT_TASK, task_embeds)
    np.save(OUT_IDS, keep_idx_all)

    print("Saved:")
    print(" ", OUT_ROLE, role_embeds.shape, role_embeds.dtype, "sha256=", _sha256_file(OUT_ROLE))
    print(" ", OUT_TASK, task_embeds.shape, task_embeds.dtype, "sha256=", _sha256_file(OUT_TASK))
    print(" ", OUT_IDS,  keep_idx_all.shape, keep_idx_all.dtype,  "sha256=", _sha256_file(OUT_IDS))
    print("done minutes:", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
