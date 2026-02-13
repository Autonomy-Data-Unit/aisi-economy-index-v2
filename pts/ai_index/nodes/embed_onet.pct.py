# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embed O*NET
#
# Embed O*NET occupations with BGE-large sentence transformer
# (894x1024 float16, role + task descriptions).

# %%
#|default_exp nodes.embed_onet
#|export_as_func true

# %%
#|set_func_signature
def embed_onet(descriptions, ctx, print) -> {"onet_embeddings": dict}:
    """Embed O*NET occupations with BGE-large."""
    ...

# %%
#|export
from ai_index.utils import maybe_run_remote

remote_result = maybe_run_remote(
    func_path="ai_index.nodes.embed_onet.embed_onet",
    inputs={"descriptions": descriptions},
    ctx=ctx, print_fn=print,
    job_name="embed_onet", time="01:00:00",
    required_models_from_vars=["embedding_model"],
)
if remote_result is not None:
    return remote_result

# %%
#|export
import numpy as np

execution_mode = ctx.vars.get("execution_mode", "local")
device = "cpu" if execution_mode == "api" else "cuda"

model_name = ctx.vars.get("embedding_model", "BAAI/bge-large-en-v1.5")
dtype = ctx.vars.get("embedding_dtype", "float16")
batch_size = ctx.vars.get("embed_onet_batch_size", 64)

titles = descriptions["titles"]
role_texts = descriptions["role_descriptions"]
task_texts = descriptions["task_descriptions"]
n = len(titles)
print(f"embed_onet: encoding {n} occupations with {model_name} (device={device}, dtype={dtype}, batch_size={batch_size})")

from isambard_utils.models import load_embedding_model
model = load_embedding_model(model_name, device=device, dtype=dtype, offline=(execution_mode != "api"))

# %%
#|export
print(f"embed_onet: encoding role descriptions...")
role_embeddings = model.encode(role_texts, batch_size=batch_size)
role_embeddings = np.asarray(role_embeddings, dtype=np.float16)
print(f"  role_embeddings shape={role_embeddings.shape}, dtype={role_embeddings.dtype}")

print(f"embed_onet: encoding task descriptions...")
task_embeddings = model.encode(task_texts, batch_size=batch_size)
task_embeddings = np.asarray(task_embeddings, dtype=np.float16)
print(f"  task_embeddings shape={task_embeddings.shape}, dtype={task_embeddings.dtype}")

# Sanity checks
assert role_embeddings.shape[0] == n, f"Expected {n} rows, got {role_embeddings.shape[0]}"
assert task_embeddings.shape[0] == n, f"Expected {n} rows, got {task_embeddings.shape[0]}"
assert not np.any(np.isnan(role_embeddings)), "NaN in role_embeddings"
assert not np.any(np.isnan(task_embeddings)), "NaN in task_embeddings"

print(f"embed_onet: done — {n} occupations, dim={role_embeddings.shape[1]}")
return {"onet_embeddings": {
    "role_embeddings": role_embeddings,
    "task_embeddings": task_embeddings,
    "titles": titles,
}}
