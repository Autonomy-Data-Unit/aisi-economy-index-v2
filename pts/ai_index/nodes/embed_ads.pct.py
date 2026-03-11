# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embed Ads
#
# Build text descriptions from LLM summaries and embed them.
#
# 1. Reads summaries from the `llm_summarise` DuckDB store.
# 2. Builds two text columns per ad:
#    - `role_text` = `"[{domain}] {short_description}"`
#    - `taskskill_text` = `"{short_description} - {tasks, skills joined}"`
# 3. Embeds both in chunks with the configured embedding model.
# 4. Saves embeddings as BLOBs in DuckDB via ResultStore (keyed by ad_id,
#    supports resume). Stored in `store/pipeline/{run_name}/embed_ads/`.
#
# Node variables:
# - `embedding_model` (global): Model key from embed_models.toml
# - `run_name` (global): Pipeline run name
# - `embed_chunk_size` (per-node): Number of ads to embed per chunk

# %%
#|default_exp nodes.embed_ads
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, successful_ad_ids: list[int]) -> {
    'ad_ids': list[int]
}:
    """Build text descriptions from LLM summaries and embed them."""
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

# %%
#|export
import duckdb
import numpy as np
import pandas as pd

from ai_index import const
from ai_index.nodes.llm_summarise import JobInfoModel
from ai_index.utils import aembed
from ai_index.utils.result_store import ResultStore

# %% [markdown]
# ## Read node variables

# %%
#|export
run_name = ctx.vars["run_name"]
embedding_model = ctx.vars["embedding_model"]
chunk_size = ctx.vars["embed_chunk_size"]

output_dir = const.pipeline_store_path / run_name / "embed_ads"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load summaries from DuckDB

# %%
#|export
summaries_db = const.pipeline_store_path / run_name / "llm_summarise" / "summaries.duckdb"
conn = duckdb.connect(str(summaries_db), read_only=True)
rows = conn.execute(
    "SELECT id, data FROM results WHERE error IS NULL ORDER BY id"
).fetchall()
conn.close()

ad_ids_set = set(successful_ad_ids)
rows = [(rid, data) for rid, data in rows if rid in ad_ids_set]
print(f"embed_ads: loaded {len(rows)} summaries from {summaries_db}")

# %% [markdown]
# ## Build text descriptions
#
# - **role_text**: `"[{domain}] {short_description}"` — domain-enriched role for embedding
# - **taskskill_text**: `"{short_description} - {tasks, skills}"` — detailed competencies

# %%
#|export
ad_ids = []
role_texts = []
taskskill_texts = []

for ad_id, data_str in rows:
    info = JobInfoModel.model_validate_json(data_str)

    role_texts.append(f"[{info.domain}] {info.short_description}")
    taskskill_texts.append(f"{info.short_description} - {', '.join(info.tasks + info.skills)}")
    ad_ids.append(ad_id)

n_total = len(ad_ids)
print(f"embed_ads: built texts for {n_total} ads")
print(f"  role_text sample: {role_texts[0][:100]}...")
print(f"  taskskill_text sample: {taskskill_texts[0][:100]}...")

# %% [markdown]
# ## Embed in chunks
#
# Embeds ads in chunks of `embed_chunk_size` and writes each chunk to DuckDB
# as BLOBs via ResultStore. Already-embedded ads (from a previous partial run)
# are skipped automatically.

# %%
#|export
db_path = output_dir / "embeddings.duckdb"
store = ResultStore(db_path, {
    "id": "BIGINT NOT NULL",
    "role": "BLOB NOT NULL",
    "taskskill": "BLOB NOT NULL",
    "error": "VARCHAR",
})

done = store.done_ids()
remaining_indices = [i for i in range(n_total) if ad_ids[i] not in done]
n_remaining = len(remaining_indices)
print(f"embed_ads: {len(done)} already embedded, {n_remaining} remaining")

n_chunks = (n_remaining + chunk_size - 1) // chunk_size

for chunk_idx in range(n_chunks):
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n_remaining)
    chunk_indices = remaining_indices[start:end]

    chunk_role_texts = [role_texts[i] for i in chunk_indices]
    chunk_taskskill_texts = [taskskill_texts[i] for i in chunk_indices]

    role_chunk = await aembed(chunk_role_texts, model=embedding_model)
    taskskill_chunk = await aembed(chunk_taskskill_texts, model=embedding_model)

    df = pd.DataFrame({
        "id": [ad_ids[i] for i in chunk_indices],
        "role": [row.astype(np.float32).tobytes() for row in role_chunk],
        "taskskill": [row.astype(np.float32).tobytes() for row in taskskill_chunk],
        "error": [None] * len(chunk_indices),
    })
    store.insert(df)

    print(f"  chunk {chunk_idx + 1}/{n_chunks}: embedded {len(chunk_indices)} ads")

n_ok, n_err = store.counts()
store.close()
print(f"embed_ads: done — {n_ok} succeeded, {n_err} failed")

ad_ids #|func_return_line
