# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embed O*NET
#
# Embed O*NET occupation texts (role_texts and task_texts) using the configured
# embedding model. Small enough (~894 texts) to always recompute quickly.

# %%
#|default_exp nodes.embed_onet
#|export_as_func true

# %%
#|set_func_signature
def main(onet_desc, ctx, print) -> {"onet_embed_meta": dict}:
    """Embed O*NET occupation descriptions."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args()

# %%
#|export
import numpy as np

from ai_index.const import pipeline_store_path
from ai_index.utils import embed

run_name = ctx.vars["run_name"]
embedding_model = ctx.vars["embedding_model"]

store_dir = pipeline_store_path / run_name / ctx.node_name
output_path = store_dir / "onet_embeddings.npz"

# %%
#|export
if output_path.exists():
    print(f"embed_onet: loading cached embeddings from {output_path}")
    data = np.load(output_path)
    role_embeddings = data["role_embeddings"]
    task_embeddings = data["task_embeddings"]
    soc_codes = data["soc_codes"]
else:
    role_texts = onet_desc["role_texts"]
    task_texts = onet_desc["task_texts"]
    soc_codes = np.array(onet_desc["soc_codes"])

    print(f"embed_onet: embedding {len(role_texts)} occupations with {embedding_model}")

    role_embeddings = embed(role_texts, model=embedding_model)
    task_embeddings = embed(task_texts, model=embedding_model)

    # L2-normalize
    role_embeddings = role_embeddings / np.maximum(np.linalg.norm(role_embeddings, axis=1, keepdims=True), 1e-12)
    task_embeddings = task_embeddings / np.maximum(np.linalg.norm(task_embeddings, axis=1, keepdims=True), 1e-12)

    # Save
    store_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        role_embeddings=role_embeddings.astype(np.float16),
        task_embeddings=task_embeddings.astype(np.float16),
        soc_codes=soc_codes,
    )
    print(f"embed_onet: saved embeddings {role_embeddings.shape} to {output_path}")

print(f"embed_onet: role={role_embeddings.shape}, task={task_embeddings.shape}")

# %%
#|export
onet_embed_meta = {
    "store_path": str(output_path),
    "embedding_model": embedding_model,
    "n_occupations": len(soc_codes),
    "embedding_dim": int(role_embeddings.shape[1]),
    "soc_codes": onet_desc["soc_codes"],
    "titles": onet_desc["titles"],
}

{"onet_embed_meta": onet_embed_meta}  #|func_return_line
