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
# 3. Embeds both with the configured embedding model.
# 4. Saves embeddings as `.npy` files in `store/pipeline/{run_name}/embed_ads/`.
#
# Node variables:
# - `embedding_model` (global): Model key from embed_models.toml
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.embed_ads
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, successful_ad_ids: list[int]):
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
import json

import duckdb
import numpy as np

from ai_index import const
from ai_index.utils import aembed

# %% [markdown]
# ## Read node variables

# %%
#|export
run_name = ctx.vars["run_name"]
embedding_model = ctx.vars["embedding_model"]

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
    parsed = json.loads(data_str)
    domain = parsed["domain"]
    short_desc = parsed["short_description"]
    tasks = parsed["tasks"]
    skills = parsed["skills"]

    role_texts.append(f"[{domain}] {short_desc}")
    taskskill_texts.append(f"{short_desc} - {', '.join(tasks + skills)}")
    ad_ids.append(ad_id)

print(f"embed_ads: built texts for {len(ad_ids)} ads")
print(f"  role_text sample: {role_texts[0][:100]}...")
print(f"  taskskill_text sample: {taskskill_texts[0][:100]}...")

# %% [markdown]
# ## Embed

# %%
#|export
role_embeddings = await aembed(role_texts, model=embedding_model)
print(f"embed_ads: role embeddings shape: {role_embeddings.shape}")

taskskill_embeddings = await aembed(taskskill_texts, model=embedding_model)
print(f"embed_ads: taskskill embeddings shape: {taskskill_embeddings.shape}")

# %% [markdown]
# ## Save to disk

# %%
#|export
ad_ids_arr = np.array(ad_ids)
np.save(output_dir / "ad_ids.npy", ad_ids_arr)
np.save(output_dir / "role_embeddings.npy", role_embeddings)
np.save(output_dir / "taskskill_embeddings.npy", taskskill_embeddings)

print(f"embed_ads: wrote {output_dir}")
print(f"  ad_ids: {ad_ids_arr.shape}")
print(f"  role_embeddings: {role_embeddings.shape}")
print(f"  taskskill_embeddings: {taskskill_embeddings.shape}")
