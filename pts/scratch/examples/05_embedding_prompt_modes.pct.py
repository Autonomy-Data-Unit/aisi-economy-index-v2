# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embedding Prompt Modes
#
# Demonstrates the three categories of prompt/instruction support in
# `embed_models.toml`, each with a model running on Isambard via sbatch.
#
# **Category 1: Fixed prefix** (e5-large-v2)
# The model requires specific prefixes: `"query: "` for queries and
# `"passage: "` for documents. Without these prefixes, embeddings are
# poor quality.
#
# **Category 2: Named prompt** (arctic-embed-l-v2)
# The model has built-in prompt names in its SentenceTransformer config.
# Passing `prompt_name="query"` applies the model's trained query format.
#
# **Category 3: Custom task instruction** (Qwen3-Embedding-8B)
# The model accepts a free-form task instruction via the `prompt` parameter.
# You can describe your specific task and the model adapts its embeddings.

# %%
import numpy as np
from ai_index.utils.embed import embed, aembed

# %%
# Sample texts for demonstration
queries = [
    "Registered Nurse needed for hospital ward, providing patient care and administering medications",
    "Python software developer with experience in data pipelines and cloud infrastructure",
    "HGV Class 2 driver for waste collection routes across South London",
]

documents = [
    "Registered Nurses: Assess patient health problems and needs, develop and implement nursing care plans, and maintain medical records. Administer medications and treatments.",
    "Software Developers: Research, design, and develop computer and network software. Analyze user needs and develop software solutions, applying principles of computer science and engineering.",
    "Refuse and Recyclable Material Collectors: Collect refuse or recyclable materials from homes and businesses using garbage trucks. May drive the truck along established routes.",
    "Carpenters: Construct, erect, install, or repair structures and fixtures made of wood and comparable materials.",
    "Data Scientists: Develop and implement methods, processes, and systems to consolidate and analyze large datasets and complex data types.",
]

# %% [markdown]
# ## Category 1: Fixed prefix (e5-large-v2)
#
# The `embed_models.toml` config for e5-large has:
# ```toml
# query_prefix = "query: "
# document_prefix = "passage: "
# ```
#
# These must always be applied. The node reads these from the config and
# prepends them before calling `embed()`.

# %%
# Simulate what a pipeline node would do: read prefixes from config and apply them
query_prefix = "query: "
document_prefix = "passage: "

e5_query_texts = [query_prefix + q for q in queries]
e5_doc_texts = [document_prefix + d for d in documents]

print("E5 query texts (first 80 chars):")
for t in e5_query_texts:
    print(f"  {t[:80]}...")

# %%
e5_q_embeds = await aembed(e5_query_texts, model="e5-large-sbatch", time="00:10:00")
e5_d_embeds = await aembed(e5_doc_texts, model="e5-large-sbatch", time="00:10:00")

e5_q_norm = e5_q_embeds / np.linalg.norm(e5_q_embeds, axis=1, keepdims=True)
e5_d_norm = e5_d_embeds / np.linalg.norm(e5_d_embeds, axis=1, keepdims=True)
e5_sims = e5_q_norm @ e5_d_norm.T

print(f"\nE5-large similarity matrix (queries x documents):")
print(f"{'':30s}", end="")
for j in range(len(documents)):
    print(f"  doc{j}", end="")
print()
for i, q in enumerate(queries):
    print(f"  {q[:28]:28s}", end="")
    for j in range(len(documents)):
        print(f"  {e5_sims[i,j]:.3f}", end="")
    print()

# %% [markdown]
# ## Category 2: Named prompt (arctic-embed-l-v2)
#
# The `embed_models.toml` config has:
# ```toml
# query_prompt_name = "query"
# ```
#
# This references a named prompt in the model's SentenceTransformer config.
# The model applies its trained query formatting internally. Documents get
# no prompt (no `document_prompt_name` set).

# %%
# prompt_name is passed directly to embed(), which passes it to SentenceTransformer.encode()
arctic_q_embeds = await aembed(queries, model="arctic-embed-l-sbatch",
                               prompt_name="query", time="00:10:00")
arctic_d_embeds = await aembed(documents, model="arctic-embed-l-sbatch",
                               time="00:10:00")  # no prompt for documents

arctic_q_norm = arctic_q_embeds / np.linalg.norm(arctic_q_embeds, axis=1, keepdims=True)
arctic_d_norm = arctic_d_embeds / np.linalg.norm(arctic_d_embeds, axis=1, keepdims=True)
arctic_sims = arctic_q_norm @ arctic_d_norm.T

print(f"\nArctic-embed similarity matrix (queries x documents):")
print(f"{'':30s}", end="")
for j in range(len(documents)):
    print(f"  doc{j}", end="")
print()
for i, q in enumerate(queries):
    print(f"  {q[:28]:28s}", end="")
    for j in range(len(documents)):
        print(f"  {arctic_sims[i,j]:.3f}", end="")
    print()

# %% [markdown]
# ## Category 3: Custom task instruction (Qwen3-Embedding-8B)
#
# The `embed_models.toml` config has:
# ```toml
# supports_prompt = true
# query_prompt_name = "query"
# ```
#
# The model supports both a default named prompt (`prompt_name="query"`) and
# a custom task instruction (`prompt="Instruct: ...\nQuery: "`). When you
# provide a custom `prompt`, it overrides the named prompt. The instruction
# tells the model what task you're doing, and it adapts its embeddings.

# %%
# Option A: use the default named prompt (generic retrieval instruction)
qwen_q_default = await aembed(queries, model="qwen3-embed-8b-sbatch",
                              prompt_name="query", time="00:10:00")
qwen_d = await aembed(documents, model="qwen3-embed-8b-sbatch",
                      time="00:10:00")

# Option B: use a custom task instruction specific to our domain
task_instruction = "Instruct: Given a job advertisement, retrieve the occupational classification that best describes the type of work\nQuery: "
qwen_q_custom = await aembed(queries, model="qwen3-embed-8b-sbatch",
                             prompt=task_instruction, time="00:10:00")

# Compare the two
qwen_d_norm = qwen_d / np.linalg.norm(qwen_d, axis=1, keepdims=True)

for label, q_embeds in [("default prompt_name", qwen_q_default), ("custom instruction", qwen_q_custom)]:
    q_norm = q_embeds / np.linalg.norm(q_embeds, axis=1, keepdims=True)
    sims = q_norm @ qwen_d_norm.T

    print(f"\nQwen3-Embed-8B [{label}] similarity matrix:")
    print(f"{'':30s}", end="")
    for j in range(len(documents)):
        print(f"  doc{j}", end="")
    print()
    for i, q in enumerate(queries):
        print(f"  {q[:28]:28s}", end="")
        for j in range(len(documents)):
            print(f"  {sims[i,j]:.3f}", end="")
        print()

# %% [markdown]
# ## How pipeline nodes should use these
#
# A node that embeds queries and documents reads the model config to decide
# which prompt mechanism to use:
#
# ```python
# from ai_index.utils._model_config import _load_model_config
# from ai_index.const import embed_models_config_path
#
# _, cfg = _load_model_config(embed_models_config_path, model_key)
#
# # Fixed prefixes (category 1): prepend to texts before calling embed()
# query_prefix = cfg.pop("query_prefix", "")
# document_prefix = cfg.pop("document_prefix", "")
# query_texts = [query_prefix + t for t in raw_queries]
# doc_texts = [document_prefix + t for t in raw_documents]
#
# # Named prompts (category 2): pass as prompt_name kwarg
# query_prompt_name = cfg.pop("query_prompt_name", None)
# document_prompt_name = cfg.pop("document_prompt_name", None)
#
# # Custom instruction (category 3): only if model supports it
# supports_prompt = cfg.pop("supports_prompt", False)
# task_prompt = ctx.vars["embed_task_prompt"] if supports_prompt else None
#
# q_embeds = embed(query_texts, model=model_key,
#                  prompt_name=query_prompt_name, prompt=task_prompt)
# d_embeds = embed(doc_texts, model=model_key,
#                  prompt_name=document_prompt_name)
# ```
