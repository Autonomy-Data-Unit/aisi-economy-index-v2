# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # run_all
#
# Orchestrate all validation runs for the crossed model sensitivity design.
#
# Usage:
#     uv run validate-all <run_def_name> [--dry-run] [--force]
#
# Example:
#     uv run validate-all validation --dry-run
#
# Reads the crossed design from config/validation.toml and runs each model pair
# through the pipeline using the named run definition from config/run_defs.toml.
# Skips pairs that have already completed (unless --force).
#
# The crossed design:
# - Arm 1: Fix embedding (bge-large-sbatch), vary all 11 calibrated LLMs
# - Arm 2: Fix LLM (qwen-7b-sbatch), vary all 10 calibrated embeddings
# - 20 unique runs (the reference pair appears in both arms, de-duplicated)

# %%
#|default_exp run_all

# %%
#|export
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from ai_index.const import validation_config_path
from validation.run_validation import _make_run_name, _is_run_complete

# %%
#|export
def _load_validation_config() -> dict:
    """Load the validation crossed-design config from config/validation.toml."""
    with open(validation_config_path, "rb") as f:
        return tomllib.load(f)

# %%
#|export
def plan_runs(config: dict) -> list[tuple[str, str]]:
    """Generate crossed (llm, embed) pairs from the validation config.

    Arm 1: fix embedding, vary LLMs.
    Arm 2: fix LLM, vary embeddings.
    De-duplicates the reference pair that appears in both arms.
    """
    fixed_embed = config["fixed_embedding"]
    fixed_llm = config["fixed_llm"]
    llm_models = config["llm_models"]
    embed_models = config["embed_models"]

    seen = set()
    pairs = []

    # Arm 1: fix embedding, vary LLMs
    for llm in llm_models:
        key = (llm, fixed_embed)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    # Arm 2: fix LLM, vary embeddings
    for embed in embed_models:
        key = (fixed_llm, embed)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    return pairs

# %%
#|export
def _completed_runs(run_def: str, pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """Return the subset of pairs whose validation runs are already complete."""
    return {
        (llm, embed) for llm, embed in pairs
        if _is_run_complete(_make_run_name(run_def, llm, embed))
    }

# %%
#|export
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    dry_run = "--dry-run" in flags
    force = "--force" in flags

    if len(args) != 1:
        print(
            "Usage: uv run validate-all <run_def_name> [--dry-run] [--force]",
            file=sys.stderr,
        )
        sys.exit(1)

    run_def = args[0]

    config = _load_validation_config()
    pairs = plan_runs(config)
    done = _completed_runs(run_def, pairs)

    print(f"Run definition: {run_def}")
    print(f"Validation runs: {len(pairs)} total, {len(done)} complete, {len(pairs) - len(done)} remaining")
    if done:
        print(f"  Completed: {', '.join(_make_run_name(run_def, l, e) for l, e in sorted(done))}")

    remaining = [(l, e) for l, e in pairs if force or (l, e) not in done]

    if not remaining:
        print("\nAll validation runs already complete.")
        return

    print(f"\nPlanned runs ({len(remaining)}):")
    for i, (llm, embed) in enumerate(remaining, 1):
        status = "(done, --force)" if (llm, embed) in done else "(pending)"
        print(f"  {i:>2}. {_make_run_name(run_def, llm, embed)} {status}")

    if dry_run:
        print("\n--dry-run: no validation runs executed.")
        return

    run_validation_bin = shutil.which("run-validation")
    if run_validation_bin is None:
        print("ERROR: 'run-validation' not found on PATH. Is the package installed?", file=sys.stderr)
        sys.exit(1)

    print()
    failures = []
    for i, (llm, embed) in enumerate(remaining, 1):
        run_name = _make_run_name(run_def, llm, embed)
        print(f"{'=' * 70}")
        print(f"Run {i}/{len(remaining)}: {run_name}")
        print(f"{'=' * 70}")

        cmd = [run_validation_bin, run_def, llm, embed]
        if force:
            cmd.append("--force")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: run-validation failed for {run_name} (exit code {result.returncode})")
            failures.append((run_name, result.returncode))

        print()

    if failures:
        print(f"\n{len(failures)}/{len(remaining)} validation runs failed:")
        for run_name, code in failures:
            print(f"  - {run_name} (exit code {code})")
        sys.exit(1)

    print(f"All {len(remaining)} validation runs complete.")
