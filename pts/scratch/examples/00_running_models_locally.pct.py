# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Local Inference with `llm_runner`
#
# This notebook demonstrates how to use the `llm_runner` package to run
# embeddings, LLM generation, and cosine similarity search locally on CPU.
#
# We use `device="cpu"` here so the notebook runs anywhere (models are
# downloaded on first use). On a machine with CUDA, switch to
# `device="cuda"` for GPU acceleration.

# %% [markdown]
# ## Batch Embeddings

# %%
from llm_runner import run_embeddings

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

embeddings = run_embeddings(
    texts,
    model_name="BAAI/bge-small-en-v1.5",  # small variant for demo (384-dim)
    device="cpu",
    dtype="float32",
    batch_size=4,
)

print(f"Input:  {len(texts)} texts")
print(f"Output: shape={embeddings.shape}, dtype={embeddings.dtype}")
print(f"\nFirst embedding (truncated): {embeddings[0][:8]}...")

# %% [markdown]
# ## Batch LLM Generation

# %%
from llm_runner import run_llm_generate

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

responses = run_llm_generate(
    prompts,
    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",  # tiny model for demo
    device="cpu",
    dtype="float32",
    backend="transformers",
    max_new_tokens=20,
    batch_size=4,
    system_message="You are a helpful assistant. Give very brief answers.",
)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")

# %% [markdown]
# ## Cosine Similarity Search
#
# Given two sets of embeddings, find the top-K most similar pairs.
# Here we embed job descriptions and O\*NET occupation titles, then
# find the best matches.

# %%
from llm_runner import run_cosine_topk

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

occ_embeddings = run_embeddings(
    occupations,
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
    dtype="float32",
)

result = run_cosine_topk(embeddings, occ_embeddings, k=3, device="cpu")

print("Top-3 occupation matches for each job ad:\n")
for i, text in enumerate(texts):
    print(f"  {text[:50]}...")
    for j in range(3):
        idx = result["indices"][i, j]
        score = result["scores"][i, j]
        print(f"    {j+1}. {occupations[idx]} (score={score:.3f})")
    print()

# %% [markdown]
# ## Serialization Round-Trip
#
# `llm_runner.serialization` handles persisting inputs/outputs for the
# CLI and remote orchestration. Numpy arrays go to `.npy`, JSON-serializable
# objects to `.json`, with a `_manifest.json` tracking types.

# %%
import tempfile
from pathlib import Path
from llm_runner import serialize, deserialize

data = {
    "embeddings": embeddings,
    "texts": texts,
    "config": {"model": "bge-small-en-v1.5", "dim": 384},
}

with tempfile.TemporaryDirectory() as tmp:
    serialize(data, Path(tmp))
    print("Serialized files:", sorted(p.name for p in Path(tmp).iterdir()))

    restored = deserialize(Path(tmp))
    print(f"\nRestored keys: {list(restored.keys())}")
    print(f"Embeddings match: {(restored['embeddings'] == embeddings).all()}")
    print(f"Texts match: {restored['texts'] == texts}")
