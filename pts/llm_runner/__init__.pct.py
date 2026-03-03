# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # llm_runner
#
# Lightweight, general-purpose inference package. Knows how to load models and
# run inference (embed, LLM generate, cosine top-K). No knowledge of SSH, Slurm,
# or Isambard. Can run locally (CPU/CUDA) or be invoked as a CLI by SBATCH jobs.

# %%
#|default_exp __init__

# %%
#|export
from llm_runner.models import (
    EmbeddingModel, LLM, VllmLLM, ApiLLM,
    load_embedding_model, load_llm, set_model_env,
)
from llm_runner.embed import run_embeddings
from llm_runner.llm import run_llm_generate
from llm_runner.cosine import run_cosine_topk
from llm_runner.serialization import serialize, deserialize
