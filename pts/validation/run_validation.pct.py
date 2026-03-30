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
import sys
from pathlib import Path

from ai_index.utils.pipeline import make_run_name, build_run_defs, is_run_complete, get_sample_n

# %%
#|export
def _make_run_name(run_def: str, llm: str, embed: str, rerank: str | None = None) -> str:
    """Build a validation run name from run definition and model keys."""
    return make_run_name("val", run_def, llm, embed, rerank)

def _build_val_run_defs(run_def: str, llm: str, embed: str, rerank: str | None = None) -> tuple[dict, str]:
    """Build run_defs for a validation run. Returns (run_defs, run_name)."""
    run_name = _make_run_name(run_def, llm, embed, rerank)
    overrides = {"llm_model": llm, "embedding_model": embed}
    if rerank is not None:
        overrides["rerank_candidates"] = {"rerank_model": rerank}
    run_defs = build_run_defs(run_def, run_name, overrides)
    return run_defs, run_name

def _is_run_complete(run_name: str) -> bool:
    """Check if a validation run has completed."""
    return is_run_complete(run_name)

# %%
#|export
async def run_validation(run_def: str, llm: str, embed: str, rerank: str | None = None, force: bool = False) -> None:
    """Run a single validation pipeline for the given run definition and model triple."""
    from ai_index.run_pipeline import run_pipeline_async

    run_name = _make_run_name(run_def, llm, embed, rerank)

    if not force and _is_run_complete(run_name):
        print(f"validation: {run_name} already complete (use --force to re-run)")
        return

    run_defs, run_name = _build_val_run_defs(run_def, llm, embed, rerank)
    sample_n = get_sample_n(run_defs, run_name)

    print(f"validation: run_name={run_name}, run_def={run_def}, llm={llm}, embed={embed}, "
          f"rerank={rerank or '(default)'}, sample_n={sample_n}")
    await run_pipeline_async(run_name, run_defs=run_defs)
    print(f"validation: {run_name} complete")

# %%
#|export
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    force = "--force" in flags

    if len(args) < 3 or len(args) > 4:
        print(
            "Usage: uv run run-validation <run_def_name> <llm_model_key> <embed_model_key> [<rerank_model_key>] [--force]",
            file=sys.stderr,
        )
        sys.exit(1)

    run_def_name = args[0]
    llm_model_key = args[1]
    embed_model_key = args[2]
    rerank_model_key = args[3] if len(args) > 3 else None
    asyncio.run(run_validation(run_def_name, llm_model_key, embed_model_key, rerank_model_key, force=force))
