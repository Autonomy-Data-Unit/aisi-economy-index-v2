# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embeddings

# %%
#|default_exp utils.embed

# %%
#|export
import asyncio

import numpy as np

from ai_index.const import embed_models_config_path
from ai_index.utils._model_config import _resolve_model_args, _split_remote_kwargs

def embed(
    texts: list[str],
    *,
    model: str,
    **kwargs,
) -> np.ndarray:
    """Embed texts using a named model key from embed_models.toml.

    The model key determines the execution mode (api/local/sbatch).
    Any explicit **kwargs override config values.

    Pass slurm_accounting={} to collect Slurm resource accounting data
    (sbatch mode only). The dict is populated in-place with timing and
    resource fields from sacct.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    mode, model_name, cfg = _resolve_model_args(embed_models_config_path, model, kwargs)
    slurm_accounting = cfg.pop("slurm_accounting", None)

    if mode == "local":
        from llm_runner.embed import run_embeddings
        return run_embeddings(texts, model_name=model_name, **cfg)

    elif mode == "api":
        from adulib.llm import batch_embeddings
        batch_size = cfg.pop("batch_size")
        embeddings, _ = batch_embeddings(model_name, input=texts, batch_size=batch_size, **cfg)
        return np.array(embeddings)

    elif mode == "sbatch":
        from isambard_utils.orchestrate import run_remote
        remote_kw = _split_remote_kwargs(cfg)
        cfg["model_name"] = model_name
        result = run_remote(
            "embed",
            inputs={"texts": texts},
            config_dict=cfg,
            required_models=[model_name],
            **remote_kw,
        )
        if slurm_accounting is not None and "_slurm_accounting" in result:
            slurm_accounting.update(result["_slurm_accounting"])
        return result["embeddings"]

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

# %%
#|export
async def aembed(
    texts: list[str],
    *,
    model: str,
    **kwargs,
) -> np.ndarray:
    """Async version of embed.

    For local mode, runs run_embeddings in a thread.
    For api mode, uses adulib's native async_batch_embeddings.
    For sbatch mode, uses arun_remote.

    Pass slurm_accounting={} to collect Slurm resource accounting data
    (sbatch mode only).
    """
    mode, model_name, cfg = _resolve_model_args(embed_models_config_path, model, kwargs)
    slurm_accounting = cfg.pop("slurm_accounting", None)

    if mode == "local":
        from llm_runner.embed import run_embeddings
        return await asyncio.to_thread(run_embeddings, texts, model_name=model_name, **cfg)

    elif mode == "api":
        from adulib.llm import async_batch_embeddings
        batch_size = cfg.pop("batch_size")
        embeddings, _ = await async_batch_embeddings(model_name, input=texts, batch_size=batch_size, **cfg)
        return np.array(embeddings)

    elif mode == "sbatch":
        from isambard_utils.orchestrate import arun_remote
        remote_kw = _split_remote_kwargs(cfg)
        cfg["model_name"] = model_name
        result = await arun_remote(
            "embed",
            inputs={"texts": texts},
            config_dict=cfg,
            required_models=[model_name],
            **remote_kw,
        )
        if slurm_accounting is not None and "_slurm_accounting" in result:
            slurm_accounting.update(result["_slurm_accounting"])
        return result["embeddings"]

    else:
        raise ValueError(f"Unknown mode: {mode!r}")
