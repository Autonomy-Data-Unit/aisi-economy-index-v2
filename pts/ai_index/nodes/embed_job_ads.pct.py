# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embed Job Ads
#
# Embed job ads with BGE-large sentence transformer
# (Nx1024 float16, role + task/skill text).

# %%
#|default_exp nodes.embed_job_ads
#|export_as_func true

# %%
#|set_func_signature
def embed_job_ads(job_ads, ctx, print) -> {"job_ad_embeddings": dict}:
    """Embed job ads with BGE-large."""
    ...

# %%
#|export
from ai_index.utils import maybe_run_remote

remote_result = maybe_run_remote(
    func_path="ai_index.nodes.embed_job_ads.embed_job_ads",
    inputs={"job_ads": job_ads},
    ctx=ctx, print_fn=print,
    job_name="embed_job_ads", time="02:00:00",
    required_models_from_vars=["embedding_model"],
)
if remote_result is not None:
    return remote_result

# %%
#|export
import numpy as np

execution_mode = ctx.vars.get("execution_mode", "local")
device = "cpu" if execution_mode == "api" else "cuda"

model_name = ctx.vars.get("embedding_model", "BAAI/bge-large-en-v1.5")
dtype = ctx.vars.get("embedding_dtype", "float16")
batch_size = ctx.vars.get("embed_job_ads_batch_size", 512)
chunk_size = ctx.vars.get("embed_job_ads_chunk_size", 20000)

job_ids = job_ads["job_ids"]
role_texts = job_ads["role_text"]
taskskill_texts = job_ads["taskskill_text"]

# Filter out rows with empty role text
valid = [(jid, rt, tt) for jid, rt, tt in zip(job_ids, role_texts, taskskill_texts) if rt and rt.strip()]
n_orig = len(job_ids)
n_valid = len(valid)
if n_valid < n_orig:
    print(f"embed_job_ads: filtered {n_orig - n_valid} empty role_text rows ({n_orig} -> {n_valid})")
    job_ids = [v[0] for v in valid]
    role_texts = [v[1] for v in valid]
    taskskill_texts = [v[2] for v in valid]

# Optional subsampling
sample_rate = float(ctx.vars.get("job_ads_sample_rate", "1.0") or "1.0")
seed_str = ctx.vars.get("job_ads_random_seed", "")
if 0 < sample_rate < 1.0:
    import random
    rng = random.Random(int(seed_str) if seed_str else None)
    n_sample = max(1, int(n_valid * sample_rate))
    indices = sorted(rng.sample(range(n_valid), n_sample))
    job_ids = [job_ids[i] for i in indices]
    role_texts = [role_texts[i] for i in indices]
    taskskill_texts = [taskskill_texts[i] for i in indices]
    n_valid = n_sample
    print(f"embed_job_ads: sampled {n_sample} ads (rate={sample_rate}, seed={seed_str or 'None'})")

print(f"embed_job_ads: encoding {n_valid} job ads with {model_name} (device={device}, dtype={dtype}, batch_size={batch_size}, chunk_size={chunk_size})")

from isambard_utils.models import load_embedding_model
model = load_embedding_model(model_name, device=device, dtype=dtype)

# %%
#|export
# Chunked encoding with progress logging
role_chunks = []
task_chunks = []
n_chunks = (n_valid + chunk_size - 1) // chunk_size

for i in range(0, n_valid, chunk_size):
    chunk_idx = i // chunk_size + 1
    end = min(i + chunk_size, n_valid)
    print(f"embed_job_ads: chunk {chunk_idx}/{n_chunks} [{i}:{end}]")

    role_emb = model.encode(role_texts[i:end], batch_size=batch_size)
    role_chunks.append(np.asarray(role_emb, dtype=np.float16))

    task_emb = model.encode(taskskill_texts[i:end], batch_size=batch_size)
    task_chunks.append(np.asarray(task_emb, dtype=np.float16))

    print(f"  chunk {chunk_idx} done")

role_embeddings = np.vstack(role_chunks) if len(role_chunks) > 1 else role_chunks[0]
task_embeddings = np.vstack(task_chunks) if len(task_chunks) > 1 else task_chunks[0]

assert role_embeddings.shape[0] == n_valid
assert task_embeddings.shape[0] == n_valid
assert not np.any(np.isnan(role_embeddings)), "NaN in role_embeddings"
assert not np.any(np.isnan(task_embeddings)), "NaN in task_embeddings"

print(f"embed_job_ads: done — {n_valid} ads, dim={role_embeddings.shape[1]}")
return {"job_ad_embeddings": {
    "role_embeddings": role_embeddings,
    "task_embeddings": task_embeddings,
    "job_ids": job_ids,
}}
