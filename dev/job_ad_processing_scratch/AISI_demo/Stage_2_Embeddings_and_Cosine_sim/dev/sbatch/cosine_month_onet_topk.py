import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange


def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x: [n,d]
    n = torch.linalg.norm(x, dim=1, keepdim=True)
    n = torch.clamp(n, min=eps)
    return x / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month-emb-npz", dest="month_emb_npz", required=True, help="adzuna_monthXX_embeds.npz")
    ap.add_argument("--onet-role-npy", required=True)
    ap.add_argument("--onet-task-npy", required=True)
    ap.add_argument("--titles-npz", required=True, help="npz containing titles array (e.g. aspectt_vectors.npz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--month", required=True)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--chunk-rows", type=int, default=80000)   # safe start
    ap.add_argument("--job-batch", type=int, default=16384)    # inner matmul batch
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not visible"

    out_dir = Path(args.out_dir) / args.month
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"[SETUP] device={device} dtype={dtype} topk={args.topk} chunk_rows={args.chunk_rows} job_batch={args.job_batch}")
    t0 = time.time()

    # ---- titles
    titles_npz = np.load(args.titles_npz, allow_pickle=True)
    onet_titles = titles_npz["titles"]

    # ---- ONET to GPU (normalised)
    role_onet = np.load(args.onet_role_npy).astype("float32")
    task_onet = np.load(args.onet_task_npy).astype("float32")

    role_onet_t = torch.tensor(role_onet, device=device, dtype=torch.float32)
    task_onet_t = torch.tensor(task_onet, device=device, dtype=torch.float32)

    role_onet_t = l2_normalize_rows(role_onet_t).to(dtype)
    task_onet_t = l2_normalize_rows(task_onet_t).to(dtype)

    del role_onet, task_onet
    gc.collect()
    torch.cuda.empty_cache()

    n_onet = role_onet_t.shape[0]
    print(f"[ONET] n={n_onet} dim={role_onet_t.shape[1]}")

    # ---- Month embeds (CPU arrays)
    d = np.load(args.month_emb_npz, allow_pickle=True)
    job_ids = d["job_ids"].astype(object)
    role_cpu = d["role_embeds"]         # expected float16 already
    task_cpu = d["taskskill_embeds"]    # expected float16 already

    n = len(job_ids)
    print(f"[MONTH] {args.month} rows={n} role_dtype={role_cpu.dtype} task_dtype={task_cpu.dtype}")

    # We will normalise on GPU per batch. Avoid CPU float32 expansion.

    part = 0
    for c0 in trange(0, n, args.chunk_rows, desc=f"month={args.month}"):
        c1 = min(c0 + args.chunk_rows, n)

        # output part path
        out_path = out_dir / f"part_{part:04d}.npz"
        if out_path.exists():
            part += 1
            continue

        # allocate output buffers on CPU
        rows = c1 - c0
        topk = args.topk

        role_idx = np.empty((rows, topk), dtype=np.int32)
        role_val = np.empty((rows, topk), dtype=np.float16)
        task_idx = np.empty((rows, topk), dtype=np.int32)
        task_val = np.empty((rows, topk), dtype=np.float16)

        # stream inside chunk in smaller job-batches to keep matmul temps bounded
        out_off = 0
        for b0 in range(c0, c1, args.job_batch):
            b1 = min(b0 + args.job_batch, c1)
            bs = b1 - b0

            jr = torch.tensor(role_cpu[b0:b1], device=device, dtype=dtype)
            jt = torch.tensor(task_cpu[b0:b1], device=device, dtype=dtype)

            # normalise
            jr = l2_normalize_rows(jr.to(torch.float32)).to(dtype)
            jt = l2_normalize_rows(jt.to(torch.float32)).to(dtype)

            sim_r = jr @ role_onet_t.T
            sim_t = jt @ task_onet_t.T

            vr, ir = torch.topk(sim_r, k=topk, dim=1)
            vt, it = torch.topk(sim_t, k=topk, dim=1)

            role_idx[out_off:out_off+bs] = ir.int().cpu().numpy()
            task_idx[out_off:out_off+bs] = it.int().cpu().numpy()
            role_val[out_off:out_off+bs] = vr.to(torch.float16).cpu().numpy()
            task_val[out_off:out_off+bs] = vt.to(torch.float16).cpu().numpy()

            out_off += bs

            del jr, jt, sim_r, sim_t, vr, ir, vt, it
            torch.cuda.empty_cache()

        # save part
        np.savez(
            out_path,
            job_ids=job_ids[c0:c1],
            role_topk_idx=role_idx,
            role_topk_val=role_val,
            task_topk_idx=task_idx,
            task_topk_val=task_val,
            month=np.array([args.month], dtype=object),
            chunk_start=np.int64(c0),
            chunk_end=np.int64(c1),
        )
        print(f"[WROTE] {out_path} rows={rows}")

        part += 1

    mins = (time.time() - t0) / 60
    print(f"[DONE] {args.month} minutes={mins:.1f}")
    print(f"[GPU] alloc_gb={torch.cuda.memory_allocated()/1024**3:.2f} reserved_gb={torch.cuda.memory_reserved()/1024**3:.2f}")


if __name__ == "__main__":
    main()
