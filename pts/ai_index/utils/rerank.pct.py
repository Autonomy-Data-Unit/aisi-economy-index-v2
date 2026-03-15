# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rerank
#
# Cross-encoder reranking: given queries and candidate documents, score each
# (query, document) pair and return top-K indices and scores per query.

# %%
#|default_exp utils.rerank

# %%
#|export
import asyncio

import numpy as np

from ..const import rerank_models_config_path
from ..utils._model_config import _resolve_model_args, _split_remote_kwargs

def _rerank_api(queries, documents, top_k, model_name):
    """Rerank using a hosted API (e.g. Voyage, Cohere) via litellm."""
    import litellm
    import time

    n_queries = len(queries)
    all_indices = np.zeros((n_queries, top_k), dtype=np.int64)
    all_scores = np.zeros((n_queries, top_k), dtype=np.float32)

    for qi in range(n_queries):
        for attempt in range(5):
            try:
                result = litellm.rerank(
                    model=model_name, query=queries[qi],
                    documents=documents, top_n=top_k,
                )
                for rank, r in enumerate(result.results[:top_k]):
                    all_indices[qi, rank] = r["index"]
                    all_scores[qi, rank] = r["relevance_score"]
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise

    return {"indices": all_indices, "scores": all_scores}


async def _arerank_api(queries, documents, top_k, model_name):
    """Async API reranking, one query at a time with rate limit pacing."""
    import litellm
    import time

    n_queries = len(queries)
    all_indices = np.zeros((n_queries, top_k), dtype=np.int64)
    all_scores = np.zeros((n_queries, top_k), dtype=np.float32)

    for qi in range(n_queries):
        for attempt in range(5):
            try:
                result = await litellm.arerank(
                    model=model_name, query=queries[qi],
                    documents=documents, top_n=top_k,
                )
                for rank, r in enumerate(result.results[:top_k]):
                    all_indices[qi, rank] = r["index"]
                    all_scores[qi, rank] = r["relevance_score"]
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 15 * (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    raise

    return {"indices": all_indices, "scores": all_scores}


def rerank(
    queries: list[str],
    documents: list[str],
    top_k: int = 10,
    *,
    model: str,
    **kwargs,
) -> dict:
    """Rerank documents for each query using a named model key from rerank_models.toml.

    The model key determines the execution mode (api/local/sbatch).
    Any explicit **kwargs override config values.

    Pass slurm_accounting={} to collect Slurm resource accounting data
    (sbatch mode only).

    Returns:
        Dict with "indices" (n_queries, top_k) and "scores" (n_queries, top_k).
    """
    mode, model_name, cfg = _resolve_model_args(rerank_models_config_path, model, kwargs)
    slurm_accounting = cfg.pop("slurm_accounting", None)

    if mode == "api":
        return _rerank_api(queries, documents, top_k, model_name)

    elif mode == "local":
        from llm_runner.rerank import run_rerank
        return run_rerank(queries, documents, top_k, model_name=model_name, **cfg)

    elif mode == "sbatch":
        from isambard_utils.orchestrate import run_remote
        remote_kw = _split_remote_kwargs(cfg)
        cfg["model_name"] = model_name
        cfg["top_k"] = top_k
        result = run_remote(
            "rerank",
            inputs={"queries": queries, "documents": documents},
            config_dict=cfg,
            required_models=[model_name],
            **remote_kw,
        )
        if slurm_accounting is not None and "_slurm_accounting" in result:
            slurm_accounting.update(result.pop("_slurm_accounting"))
        else:
            result.pop("_slurm_accounting", None)
        return result

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

# %%
#|export
async def arerank(
    queries: list[str],
    documents: list[str],
    top_k: int = 10,
    *,
    model: str,
    **kwargs,
) -> dict:
    """Async version of rerank.

    For local mode, runs in a thread. For sbatch, uses arun_remote.

    Pass slurm_accounting={} to collect Slurm resource accounting data
    (sbatch mode only).
    """
    mode, model_name, cfg = _resolve_model_args(rerank_models_config_path, model, kwargs)
    slurm_accounting = cfg.pop("slurm_accounting", None)

    if mode == "api":
        return await _arerank_api(queries, documents, top_k, model_name)

    elif mode == "local":
        from llm_runner.rerank import run_rerank
        return await asyncio.to_thread(run_rerank, queries, documents, top_k,
                                       model_name=model_name, **cfg)

    elif mode == "sbatch":
        from isambard_utils.orchestrate import arun_remote
        remote_kw = _split_remote_kwargs(cfg)
        cfg["model_name"] = model_name
        cfg["top_k"] = top_k
        result = await arun_remote(
            "rerank",
            inputs={"queries": queries, "documents": documents},
            config_dict=cfg,
            required_models=[model_name],
            **remote_kw,
        )
        if slurm_accounting is not None and "_slurm_accounting" in result:
            slurm_accounting.update(result.pop("_slurm_accounting"))
        else:
            result.pop("_slurm_accounting", None)
        return result

    else:
        raise ValueError(f"Unknown mode: {mode!r}")
