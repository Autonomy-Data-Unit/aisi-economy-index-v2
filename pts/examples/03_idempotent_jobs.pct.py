# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Idempotent Content-Addressed Jobs
#
# This notebook demonstrates the idempotent job caching system. Every remote
# job is identified by a deterministic hash of (operation + config + input data).
# Identical requests reuse cached results or attach to in-progress jobs.
#
# **Prerequisites:**
# - Active Clifton certificate (`ISAMBARD_HOST` set in `.env`)
# - SSH connectivity to Isambard

# %%
#|skip_evals
from dotenv import load_dotenv; load_dotenv()

from isambard_utils import IsambardConfig, acheck_connection
config = IsambardConfig.from_env()
print(f"Host: {config.ssh_host}")
print(f"Project dir: {config.project_dir}")

await acheck_connection(config=config)
print("SSH connection OK")

# %%
#|skip_evals
from isambard_utils.orchestrate import asetup_runner, arun_remote, TransferMode, compute_job_hash, aclear_job_cache
import asyncio
import numpy as np

# Setup once
await asetup_runner(config=config)

# %% [markdown]
# ## Embedding: cache miss → attach → cache hit

# %%
#|skip_evals
texts = ["The quick brown fox", "jumps over the lazy dog", "AI will transform the economy"]
embed_config = {"model": "BAAI/bge-large-en-v1.5", "device": "cuda", "dtype": "float16"}
embed_hash = compute_job_hash("embed", {"texts": texts}, embed_config)
print(f"Job hash: {embed_hash}")

# Clear any previous cache for this hash
await aclear_job_cache(embed_hash, config=config)

# %%
#|skip_evals
# Submit job 1 (cache miss)
task1 = asyncio.create_task(arun_remote(
    "embed", {"texts": texts}, embed_config,
    job_name="embed", setup=False, time="00:10:00",
    required_models=["BAAI/bge-large-en-v1.5"],
    isambard_config=config,
))

await asyncio.sleep(1)

# Submit job 2 (same params — should attach to running job)
task2 = asyncio.create_task(arun_remote(
    "embed", {"texts": texts}, embed_config,
    job_name="embed", setup=False, time="00:10:00",
    required_models=["BAAI/bge-large-en-v1.5"],
    isambard_config=config,
))

result1, result2 = await asyncio.gather(task1, task2)
print(f"Result 1 shape: {result1['embeddings'].shape}")
print(f"Result 2 shape: {result2['embeddings'].shape}")
assert np.array_equal(result1['embeddings'], result2['embeddings']), "Results should be identical"
print("Results are identical!")

# %%
#|skip_evals
# Submit job 3 (same params — should return from cache instantly)
import time
start = time.time()
result3 = await arun_remote(
    "embed", {"texts": texts}, embed_config,
    job_name="embed", setup=False, time="00:10:00",
    isambard_config=config,
)
elapsed = time.time() - start
print(f"Cache hit took {elapsed:.1f}s")
assert np.array_equal(result1['embeddings'], result3['embeddings']), "Cached result should match"
print("Cache hit result matches!")

# %% [markdown]
# ## LLM Generate: same pattern

# %%
#|skip_evals
prompts = ["What is AI?", "Explain machine learning in one sentence."]
llm_config = {"model": "Qwen/Qwen2.5-7B-Instruct", "max_new_tokens": 60,
              "dtype": "float16", "backend": "vllm"}
llm_hash = compute_job_hash("llm_generate", {"prompts": prompts}, llm_config)
print(f"Job hash: {llm_hash}")

await aclear_job_cache(llm_hash, config=config)

result = await arun_remote(
    "llm_generate", {"prompts": prompts}, llm_config,
    job_name="llm_generate", setup=False, time="00:15:00",
    required_models=["Qwen/Qwen2.5-7B-Instruct"],
    isambard_config=config,
)
print(f"Generated {len(result['responses'])} responses")
for i, r in enumerate(result['responses']):
    print(f"  [{i}]: {r[:80]}...")

# Cache hit
result2 = await arun_remote(
    "llm_generate", {"prompts": prompts}, llm_config,
    job_name="llm_generate", setup=False, time="00:15:00",
    isambard_config=config,
)
assert result['responses'] == result2['responses'], "Cache hit should match"
print("Cache hit matches!")

# %% [markdown]
# ## Cosine Top-K: same pattern

# %%
#|skip_evals
A = np.random.randn(5, 768).astype(np.float16)
B = np.random.randn(20, 768).astype(np.float16)
cosine_config = {"device": "cuda"}
cosine_hash = compute_job_hash("cosine_topk", {"A": A, "B": B, "k": 3}, cosine_config)
print(f"Job hash: {cosine_hash}")

await aclear_job_cache(cosine_hash, config=config)

result = await arun_remote(
    "cosine_topk", {"A": A, "B": B, "k": 3}, cosine_config,
    job_name="cosine_topk", setup=False, time="00:05:00",
    isambard_config=config,
)
print(f"Indices shape: {result['indices'].shape}")
print(f"Scores shape: {result['scores'].shape}")

# Cache hit
result2 = await arun_remote(
    "cosine_topk", {"A": A, "B": B, "k": 3}, cosine_config,
    job_name="cosine_topk", setup=False, time="00:05:00",
    isambard_config=config,
)
assert np.array_equal(result['indices'], result2['indices']), "Cache hit should match"
print("Cache hit matches!")
