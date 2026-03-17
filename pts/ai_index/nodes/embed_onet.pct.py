# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.embed_onet
#
# Embed O*NET occupations as single rich documents.
#
# 1. Reads `onet_targets.parquet` (from `prepare_onet_targets`).
# 2. Builds a single rich text per occupation:
#    `"{Title}\n\n{Description}\n\nKey tasks and skills: {Work Activities/Tasks/Skills}"`
# 3. Reads document-side prompt config from `embed_models.toml`:
#    - `document_prefix`: fixed string prepended unconditionally
#    - `document_prompt_name`: named prompt passed to SentenceTransformer
# 4. Embeds all occupations in one call.
# 5. Saves embeddings + codes to `store/pipeline/{run_name}/embed_onet/`.
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
    """Embed O*NET occupations as single rich documents."""
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

# %% [markdown]
# ## Read node variables

# %%
#|export
import json
import time

import numpy as np
import pandas as pd

from ai_index import const
from ai_index.utils import aembed
from ai_index.utils._model_config import _load_model_config

# %%
#|export
run_name = ctx.vars["run_name"]
embedding_model = ctx.vars["embedding_model"]
sbatch_cache = ctx.vars["sbatch_cache"]
sbatch_time = ctx.vars["sbatch_time"]

output_dir = const.pipeline_store_path / run_name / "embed_onet"
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Check for existing embeddings

# %%
#|export
expected_files = [
    output_dir / "onet_codes.json",
    output_dir / "onet_embeddings.npy",
]

if all(f.exists() for f in expected_files):
    print(f"embed_onet: all output files exist in {const.rel(output_dir)}, skipping embedding")
    True #|func_return_line

# %% [markdown]
# ## Read prompt config from embed_models.toml

# %%
#|export
_, model_cfg = _load_model_config(const.embed_models_config_path, embedding_model)
document_prefix = model_cfg.get("document_prefix", "")
document_prompt_name = model_cfg.get("document_prompt_name", None)

embed_prompt_kwargs = {}
if document_prompt_name:
    embed_prompt_kwargs["prompt_name"] = document_prompt_name
    print(f"embed_onet: using named prompt: {document_prompt_name}")

if document_prefix:
    print(f"embed_onet: applying document prefix: {document_prefix!r}")

# %% [markdown]
# ## Load O*NET targets and build texts

# %%
#|export
onet_targets = pd.read_parquet(const.onet_targets_path)
print(f"embed_onet: loaded {len(onet_targets)} occupations from {const.rel(const.onet_targets_path)}")

onet_codes = onet_targets["O*NET-SOC Code"].tolist()

# Build a single rich text per occupation for embedding.
# Includes alternate titles (bridges vocabulary gap with job ads),
# description, and top tasks ranked by direct importance.
onet_texts = []
for _, row in onet_targets.iterrows():
    parts = [row['Title']]
    alt_titles = row['Alternate_Titles']
    if alt_titles:
        parts.append(f"Also known as: {', '.join(alt_titles)}")
    parts.append(row['Description'])
    top_tasks = row['Top_Tasks']
    if top_tasks:
        parts.append("Key tasks: " + "; ".join(top_tasks))
    text = "\n\n".join(parts)
    onet_texts.append(document_prefix + text)

print(f"  Built {len(onet_texts)} texts")
print(f"  Mean length: {np.mean([len(t) for t in onet_texts]):.0f} chars")
print(f"  Sample: {onet_texts[0][:120]}...")

# %% [markdown]
# ## Embed

# %%
#|export
started_at = time.time()
slurm_jobs = []

_sa = {}
onet_embeddings = await aembed(
    onet_texts, model=embedding_model,
    cache=sbatch_cache, time=sbatch_time,
    slurm_accounting=_sa,
    **embed_prompt_kwargs,
)
if _sa:
    slurm_jobs.append(_sa)
print(f"embed_onet: embeddings shape: {onet_embeddings.shape}")

ended_at = time.time()

# %% [markdown]
# ## Save to disk

# %%
#|export
np.save(output_dir / "onet_embeddings.npy", onet_embeddings.astype(np.float32))

with open(output_dir / "onet_codes.json", "w") as f:
    json.dump(onet_codes, f)

print(f"embed_onet: wrote {const.rel(output_dir)}")
print(f"  onet_codes: {len(onet_codes)} codes")
print(f"  onet_embeddings: {onet_embeddings.shape}")

embed_meta = {
    "n_occupations": len(onet_targets),
    "embedding_dim": int(onet_embeddings.shape[1]),
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
