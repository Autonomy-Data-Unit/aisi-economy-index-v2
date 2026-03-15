# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # run_calibration
#
# Run a calibration pipeline to measure per-ad timing for all compute-heavy nodes.
#
# Usage:
#     uv run run-calibration <llm_model_key> <embedding_model_key> [<rerank_model_key>]
#
# Example:
#     uv run run-calibration qwen-7b-sbatch bge-large-sbatch qwen3-reranker-8b-sbatch
#
# Runs the pipeline with the 'calibration' run definition from config/run_defs.toml
# (sample_n=1000, shorter sbatch times, resume=false for LLM nodes).
# Saves a single results JSON per run to store/calibration/results/.

# %%
#|default_exp run_calibration

# %%
#|export
import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from ai_index.const import calibration_results_path, pipeline_store_path, run_defs_path

RESULTS_DIR = calibration_results_path
RUN_NAME = "calibration"

# %%
#|export
def _build_run_defs(llm_model_key: str, embedding_model_key: str,
                    rerank_model_key: str | None = None,
                    run_name: str = RUN_NAME) -> dict:
    """Load run_defs.toml and inject dynamic model keys into the calibration run."""
    from ai_index.run_pipeline import _load_run_defs
    import copy

    run_defs = _load_run_defs(run_defs_path)
    cal_config = copy.deepcopy(run_defs["runs"][RUN_NAME])
    cal_config["llm_model"] = llm_model_key
    cal_config["embedding_model"] = embedding_model_key
    if rerank_model_key is not None:
        cal_config.setdefault("rerank_candidates", {})["rerank_model"] = rerank_model_key
    run_defs["runs"][run_name] = cal_config
    return run_defs

# %%
#|export
def _read_meta(run_name: str, node_name: str, meta_filename: str) -> dict | None:
    """Read a node's meta JSON file from the pipeline store."""
    meta_path = pipeline_store_path / run_name / node_name / meta_filename
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)

# %%
#|export
def _extract_timing(meta: dict, n_key: str) -> dict:
    """Extract timing fields from a node meta dict."""
    n = meta[n_key]
    slurm_total = meta.get("slurm_total_seconds", 0)
    wall_clock = meta.get("elapsed_seconds", 0)
    elapsed = slurm_total if slurm_total > 0 else wall_clock

    result = {
        "n": n,
        "wall_clock_seconds": wall_clock,
        "slurm_seconds": slurm_total if slurm_total > 0 else None,
        "elapsed_seconds": elapsed,
        "seconds_per_ad": elapsed / n if n > 0 else 0.0,
    }

    slurm_jobs = meta.get("slurm_jobs", [])
    total_node_hours = sum(j.get("node_hours", 0) for j in slurm_jobs)
    if total_node_hours > 0:
        result["node_hours"] = total_node_hours

    return result

# %%
#|export
def _extract_fixed_timing(meta: dict) -> dict:
    """Extract timing from a fixed-cost node (e.g. embed_onet)."""
    slurm_total = meta.get("slurm_total_seconds", 0)
    wall_clock = meta.get("elapsed_seconds", 0)
    elapsed = slurm_total if slurm_total > 0 else wall_clock

    result = {
        "wall_clock_seconds": wall_clock,
        "slurm_seconds": slurm_total if slurm_total > 0 else None,
        "elapsed_seconds": elapsed,
    }

    slurm_jobs = meta.get("slurm_jobs", [])
    total_node_hours = sum(j.get("node_hours", 0) for j in slurm_jobs)
    if total_node_hours > 0:
        result["node_hours"] = total_node_hours

    return result

# %%
#|export
def _print_timing(name: str, timing: dict) -> None:
    elapsed = timing["elapsed_seconds"]
    slurm = timing.get("slurm_seconds")
    source = "slurm" if slurm else "wall-clock"
    if "seconds_per_ad" in timing:
        spa = timing["seconds_per_ad"]
        n = timing["n"]
        print(f"  {name:30s} {elapsed:>8.1f}s total  {spa:.4f}s/ad  (n={n}, {source})")
    else:
        print(f"  {name:30s} {elapsed:>8.1f}s total  (fixed, {source})")

# %%
#|export
async def run_calibration(llm_model_key: str, embedding_model_key: str,
                          rerank_model_key: str | None = None,
                          *, run_name: str = RUN_NAME) -> None:
    from ai_index.run_pipeline import run_pipeline_async

    run_defs = _build_run_defs(llm_model_key, embedding_model_key, rerank_model_key, run_name)
    sample_n = run_defs["runs"][run_name].get("sample_n", run_defs["defaults"]["sample_n"])

    calibration_store = pipeline_store_path / run_name
    if calibration_store.exists():
        print(f"calibration: cleaning {calibration_store}")
        shutil.rmtree(calibration_store)

    print(f"calibration: llm={llm_model_key}, embed={embedding_model_key}, "
          f"rerank={rerank_model_key or '(default)'}, sample_n={sample_n}")
    await run_pipeline_async(run_name, run_defs=run_defs)

    timestamp = datetime.now(timezone.utc).isoformat()

    # Collect timing from all compute-heavy nodes
    nodes = {}

    # Per-ad nodes
    for node_name, meta_file, n_key in [
        ("embed_ads", "embed_meta.json", "n_embedded"),
        ("cosine_candidates", "cosine_meta.json", "n_total"),
        ("llm_filter_candidates", "filter_meta.json", "n_total"),
        ("rerank_candidates", "rerank_meta.json", "n_total"),
    ]:
        meta = _read_meta(run_name, node_name, meta_file)
        if meta is not None:
            nodes[node_name] = _extract_timing(meta, n_key)

    # Fixed-cost nodes
    for node_name, meta_file in [
        ("embed_onet", "embed_meta.json"),
    ]:
        meta = _read_meta(run_name, node_name, meta_file)
        if meta is not None:
            nodes[node_name] = _extract_fixed_timing(meta)

    result = {
        "llm_model": llm_model_key,
        "embedding_model": embedding_model_key,
        "rerank_model": rerank_model_key,
        "sample_n": sample_n,
        "timestamp": timestamp,
        "nodes": nodes,
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{embedding_model_key}__{llm_model_key}.json"
    if rerank_model_key:
        filename = f"{embedding_model_key}__{rerank_model_key}__{llm_model_key}.json"
    result_path = RESULTS_DIR / filename
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\ncalibration: results written to {result_path}")
    for name, timing in nodes.items():
        _print_timing(name, timing)

# %%
#|export
def main():
    run_name = RUN_NAME
    argv = sys.argv[1:]
    if "--run-name" in argv:
        idx = argv.index("--run-name")
        run_name = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2:]
    if len(argv) < 2 or len(argv) > 3:
        print(
            "Usage: uv run run-calibration <llm_model_key> <embedding_model_key> [<rerank_model_key>] [--run-name NAME]",
            file=sys.stderr,
        )
        sys.exit(1)
    llm_model_key = argv[0]
    embedding_model_key = argv[1]
    rerank_model_key = argv[2] if len(argv) > 2 else None
    asyncio.run(run_calibration(llm_model_key, embedding_model_key, rerank_model_key, run_name=run_name))
