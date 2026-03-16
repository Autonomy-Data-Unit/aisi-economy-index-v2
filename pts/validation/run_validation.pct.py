# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # run_validation
#
# Run a single validation pipeline for a given run definition and model pair.
#
# Usage:
#     uv run run-validation <run_def_name> <llm_model_key> <embed_model_key> [--force]
#
# Example:
#     uv run run-validation validation qwen-7b-sbatch bge-large-sbatch
#
# Runs the pipeline with the named run definition from config/run_defs.toml.
# Each (run_def, llm, embed) triple gets its own run directory
# at store/pipeline/val__<run_def>__<llm>__<embed>/.
#
# Unlike calibration, does NOT clean the store before running (resume=true for
# expensive runs). Use --force to skip the completion check and re-run.

# %%
#|default_exp run_validation

# %%
#|export
import asyncio
import copy
import sys
from pathlib import Path

from ai_index.const import pipeline_store_path, run_defs_path

# %%
#|export
def _make_run_name(run_def: str, llm: str, embed: str) -> str:
    """Build a validation run name from run definition and model keys."""
    return f"val__{run_def}__{llm}__{embed}"

# %%
#|export
def _build_run_defs(run_def: str, llm: str, embed: str) -> tuple[dict, str]:
    """Load run_defs.toml, deep-copy the named run definition, and inject model keys.

    Args:
        run_def: Name of the run definition in run_defs.toml (e.g. 'validation').
        llm: LLM model key.
        embed: Embedding model key.

    Returns (run_defs, run_name) where run_defs has a new entry at
    runs[run_name] with the injected model keys.
    """
    from ai_index.run_pipeline import _load_run_defs

    run_defs = _load_run_defs(run_defs_path)
    run_name = _make_run_name(run_def, llm, embed)

    # Deep-copy the named run definition into a new run entry
    template = run_defs["runs"][run_def]
    run_entry = copy.deepcopy(template)
    run_entry["llm_model"] = llm
    run_entry["embedding_model"] = embed
    run_defs["runs"][run_name] = run_entry

    return run_defs, run_name

# %%
#|export
def _is_run_complete(run_name: str) -> bool:
    """Check if a validation run has completed (final ad_exposure.parquet exists)."""
    exposure = pipeline_store_path / run_name / "compute_job_ad_exposure" / "ad_exposure.parquet"
    return exposure.exists()

# %%
#|export
async def run_validation(run_def: str, llm: str, embed: str, force: bool = False) -> None:
    """Run a single validation pipeline for the given run definition and model pair."""
    from ai_index.run_pipeline import run_pipeline_async

    run_name = _make_run_name(run_def, llm, embed)

    if not force and _is_run_complete(run_name):
        print(f"validation: {run_name} already complete (use --force to re-run)")
        return

    run_defs, run_name = _build_run_defs(run_def, llm, embed)
    sample_n = run_defs["runs"][run_name].get("sample_n", run_defs["defaults"]["sample_n"])

    print(f"validation: run_name={run_name}, run_def={run_def}, llm={llm}, embed={embed}, sample_n={sample_n}")
    await run_pipeline_async(run_name, run_defs=run_defs)
    print(f"validation: {run_name} complete")

# %%
#|export
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    force = "--force" in flags

    if len(args) != 3:
        print(
            "Usage: uv run run-validation <run_def_name> <llm_model_key> <embed_model_key> [--force]",
            file=sys.stderr,
        )
        sys.exit(1)

    run_def_name = args[0]
    llm_model_key = args[1]
    embed_model_key = args[2]
    asyncio.run(run_validation(run_def_name, llm_model_key, embed_model_key, force=force))
