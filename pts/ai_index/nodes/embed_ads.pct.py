# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.embed_ads
#
# Embed raw job ad text directly using the configured embedding model.
# No LLM summarisation needed.
#
# 1. Loads raw ad text (title + description) from Adzuna DuckDB.
# 2. Reads prompt support config from `embed_models.toml` for the model:
#    - `query_prefix`: fixed string prepended unconditionally
#    - `query_prompt_name`: named prompt passed to SentenceTransformer
#    - `supports_prompt`: whether the model accepts a custom task instruction
# 3. Embeds in chunks with resume support via ResultStore.
# 4. Saves embeddings to `store/pipeline/{run_name}/embed_ads/`.
#
# Node variables:
# - `embedding_model` (global): Model key from embed_models.toml
# - `embed_task_prompt` (per-node): Custom task instruction (only used if
#   model has `supports_prompt = true`)
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.embed_ads
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: "np.ndarray") -> {
    'ad_ids': list[int]
}:
    """Embed raw job ad text directly."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('embed_ads', run_name=run_name)
show_node_vars('embed_ads', run_name=run_name)

# %% [markdown]
# # Function body

# %% [markdown]
# ## Read node variables

# %%
#|export
import json
import time

import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils import aembed, get_ads_by_id
from ai_index.utils._model_config import _load_model_config
from ai_index.utils.result_store import ResultStore

# %%
#|export
run_name = ctx.vars["run_name"]
embedding_model = ctx.vars["embedding_model"]
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]
embed_task_prompt = ctx.vars["embed_task_prompt"]
duckdb_memory_limit = ctx.vars["duckdb_memory_limit"]

output_dir = const.pipeline_store_path / run_name / "embed_ads"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Read prompt config from embed_models.toml

# %%
#|export
_, model_cfg = _load_model_config(const.embed_models_config_path, embedding_model)
query_prefix = model_cfg.get("query_prefix", "")
query_prompt_name = model_cfg.get("query_prompt_name", None)
supports_prompt = model_cfg.get("supports_prompt", False)

# Determine the prompt kwarg to pass to aembed()
embed_prompt_kwargs = {}
if supports_prompt and embed_task_prompt:
    embed_prompt_kwargs["prompt"] = embed_task_prompt
    print(f"embed_ads: using custom task prompt: {embed_task_prompt[:80]}...")
elif query_prompt_name:
    embed_prompt_kwargs["prompt_name"] = query_prompt_name
    print(f"embed_ads: using named prompt: {query_prompt_name}")

if query_prefix:
    print(f"embed_ads: applying query prefix: {query_prefix!r}")

# %% [markdown]
# ## Load raw ad texts

# %%
#|export
ad_ids_list = [int(i) for i in ad_ids]
ads_table = get_ads_by_id(ad_ids_list, columns=["title", "description"], memory_limit=duckdb_memory_limit)
ads_df = ads_table.to_pandas().set_index("id")

# Build texts: query_prefix + title + ". " + description
texts_by_id = {}
for ad_id in ad_ids_list:
    row = ads_df.loc[ad_id]
    title = str(row["title"] or "")
    desc = str(row["description"] or "")
    text = f"{title}. {desc}"
    texts_by_id[ad_id] = query_prefix + text

# Maintain order matching ad_ids
ordered_ids = [i for i in ad_ids_list if i in texts_by_id]
ordered_texts = [texts_by_id[i] for i in ordered_ids]

n_total = len(ordered_ids)
print(f"embed_ads: loaded {n_total} ad texts")
if n_total > 0:
    sample = ordered_texts[0][:120]
    print(f"  sample: {sample}...")

# %% [markdown]
# ## Embed in chunks
#
# Embeds ads in chunks and writes each chunk to DuckDB as BLOBs via ResultStore.
# Already-embedded ads (from a previous partial run) are skipped automatically.

# %%
#|export
CHUNK_SIZE = 5000

db_path = output_dir / "embeddings.duckdb"
store = ResultStore(db_path, {
    "id": "BIGINT NOT NULL",
    "embedding": "BLOB NOT NULL",
    "error": "VARCHAR",
}, memory_limit=duckdb_memory_limit)

done = store.done_ids()
remaining_indices = [i for i in range(n_total) if ordered_ids[i] not in done]
n_remaining = len(remaining_indices)
print(f"embed_ads: {len(done)} already embedded, {n_remaining} remaining")

n_chunks = (n_remaining + CHUNK_SIZE - 1) // CHUNK_SIZE

started_at = time.time()
slurm_jobs = []

for chunk_idx in range(n_chunks):
    start = chunk_idx * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, n_remaining)
    chunk_indices = remaining_indices[start:end]

    chunk_texts = [ordered_texts[i] for i in chunk_indices]

    _sa = {}
    embeddings = await aembed(
        chunk_texts, model=embedding_model,
        cache=sbatch_cache, time=sbatch_time,
        slurm_accounting=_sa,
        **embed_prompt_kwargs,
    )
    if _sa:
        slurm_jobs.append(_sa)

    df = pd.DataFrame({
        "id": [ordered_ids[i] for i in chunk_indices],
        "embedding": [row.astype(np.float32).tobytes() for row in embeddings],
        "error": [None] * len(chunk_indices),
    })
    store.insert(df)

    print(f"  chunk {chunk_idx + 1}/{n_chunks}: embedded {len(chunk_indices)} ads")

ended_at = time.time()

n_ok, n_err = store.counts()
store.close()
del store
print(f"embed_ads: done, {n_ok} succeeded, {n_err} failed")

embed_meta = {
    "n_total": n_total,
    "n_embedded": n_remaining,
    "n_skipped": len(done),
    "started_at": started_at,
    "ended_at": ended_at,
    "elapsed_seconds": ended_at - started_at,
    "slurm_jobs": slurm_jobs,
    "slurm_total_seconds": sum(j.get("elapsed_seconds", 0) for j in slurm_jobs),
}
meta_path = output_dir / "embed_meta.json"
with open(meta_path, "w") as f:
    json.dump(embed_meta, f, indent=2)
print(f"embed_ads: wrote {const.rel(meta_path)}")

ad_ids = ordered_ids
ad_ids #|func_return_line
