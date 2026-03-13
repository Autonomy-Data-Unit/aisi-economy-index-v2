# ---
# jupyter:
#   kernelspec:
#     display_name: ai-index (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.embed_onet
#
# Embed O*NET occupation text descriptions for cosine matching against job ads.
#
# 1. Reads `onet_targets.parquet` (from `prepare_onet_targets`).
# 2. Embeds two columns:
#    - "Job Role Description" = `"{Title} - {Description}"`
#    - "Work Activities/Tasks/Skills" = `"{Title} - {top tasks, activities, skills}"`
# 3. Saves embeddings as `.npy` files in `store/pipeline/{run_name}/embed_onet/`.
#
# Node variables:
# - `embedding_model` (global): Model key from embed_models.toml
# - `run_name` (global): Pipeline run name

# %%
#|default_exp nodes.embed_onet
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print) -> bool:
    """Embed O*NET occupation text descriptions."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('embed_onet', run_name=run_name)
show_node_vars('embed_onet', run_name=run_name)

# %% [markdown]
# # Function body

# %%
#|export
import json
import time

import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils import aembed

# %% [markdown]
# ## Read node variables

# %%
#|export
run_name = ctx.vars["run_name"]
embedding_model = ctx.vars["embedding_model"]
sbatch_cache = ctx.vars["sbatch_cache"]

output_dir = const.pipeline_store_path / run_name / "embed_onet"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Check for existing embeddings

# %%
#|export
expected_files = [
    output_dir / "onet_codes.npy",
    output_dir / "onet_titles.npy",
    output_dir / "role_embeddings.npy",
    output_dir / "taskskill_embeddings.npy",
]

if all(f.exists() for f in expected_files):
    print(f"embed_onet: all output files exist in {const.rel(output_dir)}, skipping embedding")
    True #|func_return_line

# %% [markdown]
# ## Load O*NET targets

# %%
#|export
onet_targets = pd.read_parquet(const.onet_targets_path)
print(f"embed_onet: loaded {len(onet_targets)} occupations from {const.rel(const.onet_targets_path)}")

role_texts = onet_targets["Job Role Description"].tolist()
taskskill_texts = onet_targets["Work Activities/Tasks/Skills"].tolist()

print(f"  role sample: {role_texts[0][:100]}...")
print(f"  taskskill sample: {taskskill_texts[0][:100]}...")

# %% [markdown]
# ## Embed

# %%
#|export
started_at = time.time()
slurm_jobs = []

_sa1 = {}
role_embeddings = await aembed(role_texts, model=embedding_model, cache=sbatch_cache, slurm_accounting=_sa1)
if _sa1: slurm_jobs.append(_sa1)
print(f"embed_onet: role embeddings shape: {role_embeddings.shape}")

_sa2 = {}
taskskill_embeddings = await aembed(taskskill_texts, model=embedding_model, cache=sbatch_cache, slurm_accounting=_sa2)
if _sa2: slurm_jobs.append(_sa2)
print(f"embed_onet: taskskill embeddings shape: {taskskill_embeddings.shape}")

ended_at = time.time()

# %% [markdown]
# ## Save to disk

# %%
#|export
onet_codes = np.array(onet_targets["O*NET-SOC Code"].tolist(), dtype=str)
onet_titles = np.array(onet_targets["Title"].tolist(), dtype=str)
np.save(output_dir / "onet_codes.npy", onet_codes)
np.save(output_dir / "onet_titles.npy", onet_titles)
np.save(output_dir / "role_embeddings.npy", role_embeddings)
np.save(output_dir / "taskskill_embeddings.npy", taskskill_embeddings)

print(f"embed_onet: wrote {const.rel(output_dir)}")
print(f"  onet_codes: {onet_codes.shape}")
print(f"  role_embeddings: {role_embeddings.shape}")
print(f"  taskskill_embeddings: {taskskill_embeddings.shape}")

embed_meta = {
    "n_occupations": len(onet_targets),
    "started_at": started_at,
    "ended_at": ended_at,
    "elapsed_seconds": ended_at - started_at,
    "slurm_jobs": slurm_jobs,
    "slurm_total_seconds": sum(j.get("elapsed_seconds", 0) for j in slurm_jobs),
}
meta_path = output_dir / "embed_meta.json"
with open(meta_path, "w") as f:
    json.dump(embed_meta, f, indent=2)
print(f"embed_onet: wrote {const.rel(meta_path)}")

True #|func_return_line
