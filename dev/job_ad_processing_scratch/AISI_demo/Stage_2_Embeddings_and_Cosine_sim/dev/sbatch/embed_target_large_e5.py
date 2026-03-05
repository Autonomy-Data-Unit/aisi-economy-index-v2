import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Updated to the E5-Large-v2 model
MODEL_NAME = "intfloat/e5-large-v2"

def shard_slice(n: int, shard_id: int, n_shards: int) -> slice:
    base = n // n_shards
    rem = n % n_shards
    start = shard_id * base + min(shard_id, rem)
    end = start + base + (1 if shard_id < rem else 0)
    return slice(start, end)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    in_npz = Path(args.in_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.month}_shard{args.shard_id:02d}_of{args.n_shards:02d}.npz"
    if out_path.exists():
        print("Already exists, skipping:", out_path)
        return

    assert in_npz.exists(), f"Missing input {in_npz}"
    assert torch.cuda.is_available(), "No CUDA visible"

    data = np.load(in_npz, allow_pickle=True)
    job_ids = data["job_ids"].astype(str)
    
    # E5-v2 REQUIREMENT: Prepend "passage: " to all text entries
    roles = ["query: " + str(x) for x in data["role_text"]]
    taskskills = ["query: " + str(x) for x in data["taskskill_text"]]

    n = len(job_ids)
    sl = shard_slice(n, args.shard_id, args.n_shards)

    job_ids = job_ids[sl]
    roles = roles[sl]
    taskskills = taskskills[sl]

    print(f"Month={args.month} shard={args.shard_id}/{args.n_shards} rows={len(job_ids)}")

    dtype = torch.float16
    model = SentenceTransformer(MODEL_NAME, device="cuda", model_kwargs={"torch_dtype": dtype})

    # Encode with the prefixes included
    role_emb = model.encode(roles, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)
    task_emb = model.encode(taskskills, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)

    np.savez(
        out_path,
        job_ids=job_ids.astype(object),
        role_embeds=role_emb,
        taskskill_embeds=task_emb,
        month=np.array([args.month] * len(job_ids), dtype=object),
        shard_id=np.int32(args.shard_id),
        n_shards=np.int32(args.n_shards),
    )
    print("Wrote:", out_path, "role", role_emb.shape, "task", task_emb.shape)

if __name__ == "__main__":
    main()