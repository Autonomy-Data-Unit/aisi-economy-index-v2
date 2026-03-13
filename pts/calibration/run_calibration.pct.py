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
# Run a calibration pipeline to measure per-ad timing for LLM and embedding models.
#
# Usage:
#     uv run run-calibration <llm_model_key> <embedding_model_key>
#
# Example:
#     uv run run-calibration qwen-7b-sbatch bge-large-sbatch
#
# Runs the pipeline with the 'calibration' run definition from config/run_defs.toml
# (sample_n=1000, shorter sbatch times, resume=false for LLM nodes).
# Collects timing data from LLM and embedding nodes and writes separate results to:
#   store/calibration/results/llm/<llm_model_key>.json
#   store/calibration/results/embed/<embedding_model_key>.json
#
# Timing data includes both wall-clock times (from Python time.time()) and Slurm
# accounting times (from sacct). The Slurm times reflect actual GPU execution and
# exclude transfer/queue overhead, making them more accurate for cost estimation.

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

LLM_RESULTS_DIR = calibration_results_path / "llm"
EMBED_RESULTS_DIR = calibration_results_path / "embed"
RUN_NAME = "calibration"

# %%
#|export
def _build_run_defs(llm_model_key: str, embedding_model_key: str) -> dict:
    """Load run_defs.toml and inject dynamic model keys into the calibration run."""
    from ai_index.run_pipeline import _load_run_defs

    run_defs = _load_run_defs(run_defs_path)
    run_defs["runs"][RUN_NAME]["llm_model"] = llm_model_key
    run_defs["runs"][RUN_NAME]["embedding_model"] = embedding_model_key
    return run_defs

# %%
#|export
def _read_meta(run_name: str, node_name: str, meta_filename: str) -> dict | None:
    """Read a node's meta JSON file from the pipeline store."""
    meta_path = pipeline_store_path / run_name / node_name / meta_filename
    if not meta_path.exists():
        print(f"warning: {meta_path} not found, skipping")
        return None
    with open(meta_path) as f:
        return json.load(f)

# %%
#|export
def _extract_timing(meta: dict, n_key: str) -> dict:
    """Extract timing fields from a node meta dict.

    Uses slurm_total_seconds (actual GPU time from sacct) when available,
    falling back to elapsed_seconds (wall-clock including transfer overhead).
    """
    n = meta[n_key]
    slurm_total = meta.get("slurm_total_seconds", 0)
    wall_clock = meta.get("elapsed_seconds", 0)

    # Prefer slurm timing (excludes transfer/queue overhead)
    elapsed = slurm_total if slurm_total > 0 else wall_clock

    result = {
        "n": n,
        "wall_clock_seconds": wall_clock,
        "slurm_seconds": slurm_total if slurm_total > 0 else None,
        "elapsed_seconds": elapsed,
        "seconds_per_ad": elapsed / n if n > 0 else 0.0,
    }

    # Include node hours if available
    slurm_jobs = meta.get("slurm_jobs", [])
    total_node_hours = sum(j.get("node_hours", 0) for j in slurm_jobs)
    if total_node_hours > 0:
        result["node_hours"] = total_node_hours

    return result

# %%
#|export
def _extract_fixed_timing(meta: dict) -> dict:
    """Extract timing fields from a fixed-cost node meta dict (e.g. embed_onet)."""
    slurm_total = meta.get("slurm_total_seconds", 0)
    wall_clock = meta.get("elapsed_seconds", 0)
    elapsed = slurm_total if slurm_total > 0 else wall_clock

    result = {
        "n_occupations": meta["n_occupations"],
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
def _write_result(output_dir: Path, filename: str, data: dict) -> Path:
    """Write a result JSON file, creating directories as needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path

# %%
#|export
def _print_timing(name: str, timing: dict) -> None:
    """Print a single node timing summary line."""
    elapsed = timing["elapsed_seconds"]
    slurm = timing.get("slurm_seconds")
    source = "slurm" if slurm else "wall-clock"
    if "seconds_per_ad" in timing:
        spa = timing["seconds_per_ad"]
        n = timing["n"]
        print(f"  {name}: {elapsed:.1f}s total, {spa:.3f}s/ad (n={n}, {source})")
    else:
        print(f"  {name}: {elapsed:.1f}s total ({source})")

# %%
#|export
async def run_calibration(llm_model_key: str, embedding_model_key: str) -> None:
    from ai_index.run_pipeline import run_pipeline_async

    run_defs = _build_run_defs(llm_model_key, embedding_model_key)
    # Effective sample_n: run-level override takes precedence over defaults
    sample_n = run_defs["runs"][RUN_NAME].get("sample_n", run_defs["defaults"]["sample_n"])

    # Clean previous calibration run to ensure fresh timing
    calibration_store = pipeline_store_path / RUN_NAME
    if calibration_store.exists():
        print(f"calibration: cleaning {calibration_store}")
        shutil.rmtree(calibration_store)

    print(f"calibration: llm_model={llm_model_key}, embedding_model={embedding_model_key}, sample_n={sample_n}")
    await run_pipeline_async(RUN_NAME, run_defs=run_defs)

    timestamp = datetime.now(timezone.utc).isoformat()

    # --- LLM results ---
    llm_nodes = {}

    summarise_meta = _read_meta(RUN_NAME, "llm_summarise", "summary_meta.json")
    if summarise_meta is not None:
        llm_nodes["llm_summarise"] = _extract_timing(summarise_meta, "n_total")

    filter_meta = _read_meta(RUN_NAME, "llm_filter_candidates", "filter_meta.json")
    if filter_meta is not None:
        llm_nodes["llm_filter_candidates"] = _extract_timing(filter_meta, "n_total")

    llm_result = {
        "model_key": llm_model_key,
        "sample_n": sample_n,
        "timestamp": timestamp,
        "nodes": llm_nodes,
    }
    llm_path = _write_result(LLM_RESULTS_DIR, f"{llm_model_key}.json", llm_result)

    # --- Embed results ---
    embed_nodes = {}

    embed_ads_meta = _read_meta(RUN_NAME, "embed_ads", "embed_meta.json")
    if embed_ads_meta is not None:
        embed_nodes["embed_ads"] = _extract_timing(embed_ads_meta, "n_embedded")

    embed_onet_meta = _read_meta(RUN_NAME, "embed_onet", "embed_meta.json")
    if embed_onet_meta is not None:
        embed_nodes["embed_onet"] = _extract_fixed_timing(embed_onet_meta)

    embed_result = {
        "model_key": embedding_model_key,
        "sample_n": sample_n,
        "timestamp": timestamp,
        "nodes": embed_nodes,
    }
    embed_path = _write_result(EMBED_RESULTS_DIR, f"{embedding_model_key}.json", embed_result)

    # Print summary
    print(f"\ncalibration: LLM results written to {llm_path}")
    for name, timing in llm_nodes.items():
        _print_timing(name, timing)

    print(f"\ncalibration: embed results written to {embed_path}")
    for name, timing in embed_nodes.items():
        _print_timing(name, timing)

# %%
#|export
def main():
    if len(sys.argv) != 3:
        print(
            "Usage: uv run run-calibration <llm_model_key> <embedding_model_key>",
            file=sys.stderr,
        )
        sys.exit(1)
    llm_model_key = sys.argv[1]
    embedding_model_key = sys.argv[2]
    asyncio.run(run_calibration(llm_model_key, embedding_model_key))
