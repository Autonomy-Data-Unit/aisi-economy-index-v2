# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Running Models on Isambard with `run_remote()`
#
# This notebook shows how to run embedding, cosine similarity, and LLM jobs
# on Isambard using `isambard_utils.orchestrate.run_remote()`.
#
# `run_remote()` handles the full lifecycle: deploying `llm_runner` on
# Isambard, transferring inputs, submitting a SBATCH job, polling for
# completion, and downloading results.
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
from isambard_utils.orchestrate import asetup_runner, arun_remote, TransferMode

# Setup once: syncs llm_runner + deps to Isambard
await asetup_runner(config=config)

# %% [markdown]
# ## Batch Embeddings
#
# `arun_remote("embed", ...)` serializes the texts, transfers them to
# Isambard, runs `python -m llm_runner embed` in a SBATCH job, and
# downloads the resulting numpy array.
#
# After calling `asetup_runner()` above, we pass `setup=False` to skip
# redundant setup on each call.

# %%
texts = [
    "Software engineer with experience in Python and machine learning",
    "Registered nurse specializing in pediatric care",
    "Financial analyst with expertise in risk modeling",
    "High school mathematics teacher",
    "Civil engineer designing highway infrastructure",
    "Data scientist working on natural language processing",
    "Chef specializing in French cuisine",
    "Electrician with commercial wiring experience",
    "Marketing manager for consumer electronics",
    "Pharmacist in a hospital setting",
]

embedding_model = "BAAI/bge-small-en-v1.5"

result = await arun_remote(
    "embed",
    inputs={"texts": texts},
    config_dict={"model_name": embedding_model, "dtype": "float16"},
    setup=False,
    transfer_modes=TransferMode.COMPRESSED,
    job_name="embed_demo",
    time="00:10:00",
    required_models=[embedding_model],
    isambard_config=config,
)

embeddings = result["embeddings"]
print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
print(f"First embedding (truncated): {embeddings[0][:8]}...")

# %% [markdown]
# ## Cosine Similarity Search
#
# Embed occupations, then run cosine top-K matching on GPU.

# %%
occupations = [
    "Software Developers",
    "Registered Nurses",
    "Financial Analysts",
    "Secondary School Teachers",
    "Civil Engineers",
    "Data Scientists",
    "Chefs and Head Cooks",
    "Electricians",
    "Marketing Managers",
    "Pharmacists",
]

occ_result = await arun_remote(
    "embed",
    inputs={"texts": occupations},
    config_dict={"model_name": embedding_model, "dtype": "float16"},
    setup=False,
    transfer_modes=TransferMode.COMPRESSED,
    job_name="embed_occ_demo",
    time="00:10:00",
    required_models=[embedding_model],
    isambard_config=config,
)

occ_embeddings = occ_result["embeddings"]
print(f"Occupation embeddings: {occ_embeddings.shape}")

# %% [markdown]
# For the cosine similarity job, the embedding arrays can be large. We
# use `TransferMode.COMPRESSED` which tar-gzips the data before piping over
# SSH, and stores it in a content-hashed directory (skips if already present).

# %%
cosine_result = await arun_remote(
    "cosine_topk",
    inputs={"A": embeddings, "B": occ_embeddings},
    config_dict={"k": 3},
    setup=False,
    transfer_modes=TransferMode.COMPRESSED,  # compressed + content-hashed
    job_name="cosine_demo",
    time="00:10:00",
    isambard_config=config,
)

print("Top-3 occupation matches for each job ad:\n")
for i, text in enumerate(texts):
    print(f"  {text[:50]}...")
    for j in range(3):
        idx = cosine_result["indices"][i, j]
        score = cosine_result["scores"][i, j]
        print(f"    {j+1}. {occupations[idx]} (score={score:.3f})")
    print()

# %% [markdown]
# ## Batch LLM Generation
#
# Run LLM inference on Isambard's GPU. The `required_models` parameter
# ensures the model is pre-cached on the login node before the SBATCH
# job starts (compute nodes have no internet access).

# %%
llm_model = "Qwen/Qwen2.5-7B-Instruct"

prompts = [
    "What is the capital of France? Answer in one word.",
    "What is 2 + 2? Answer in one word.",
    "Name one programming language. Answer in one word.",
    "What color is the sky? Answer in one word.",
    "Name a planet in our solar system. Answer in one word.",
    "What is the chemical symbol for water? Answer in one word.",
    "Name a continent. Answer in one word.",
    "What is the largest ocean? Answer in one word.",
    "Name a primary color. Answer in one word.",
    "What is the boiling point of water in Celsius? Answer in one number.",
]

llm_result = await arun_remote(
    "llm_generate",
    inputs={"prompts": prompts},
    config_dict={
        "model_name": llm_model,
        "dtype": "float16",
        "backend": "vllm",
        "max_new_tokens": 20,
        "system_message": "You are a helpful assistant. Give very brief answers.",
    },
    setup=False,
    transfer_modes=TransferMode.COMPRESSED,
    job_name="llm_demo",
    time="00:10:00",
    required_models=[llm_model],
    isambard_config=config,
)

for prompt, response in zip(prompts, llm_result["responses"]):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")

# %% [markdown]
# ## Transfer Modes
#
# `run_remote()` supports three transfer modes for inputs:
#
# | Mode | Description | Use case |
# |------|-------------|----------|
# | `DIRECT` | Tar + SSH pipe (default) | Small/one-off data |
# | `UPLOAD` | rsync to content-hashed dir | Large arrays, reused across jobs |
# | `COMPRESSED` | tar.gz + SSH pipe, content-hashed | Large data over slow links |
#
# You can set a single mode for all inputs, or per-input:
#
# ```python
# run_remote(
#     "cosine_topk",
#     inputs={"A": large_array, "B": small_array},
#     config_dict={"k": 5},
#     transfer_modes={
#         "A": TransferMode.UPLOAD,       # large, reused
#         "B": TransferMode.DIRECT,       # small, one-off
#     },
#     ...
# )
# ```
