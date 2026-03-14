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
#     uv run validate-all <run_def_name> [--dry-run] [--force] [--concurrency N]
#
# Example:
#     uv run validate-all validation --dry-run
#     uv run validate-all validation_5k --concurrency 4
#
# Reads the crossed design from config/validation.toml and runs each model pair
# through the pipeline using the named run definition from config/run_defs.toml.
# Skips pairs that have already completed (unless --force).
#
# The crossed design:
# - Arm 1: For each fixed embedding, vary all calibrated LLMs
# - Arm 2: For each fixed LLM, vary all calibrated embeddings
# - De-duplicated across all arms

# %%
#|default_exp run_all

# %%
#|export
import asyncio
import sys
import tomllib
from pathlib import Path

from ai_index.const import validation_config_path
from validation.run_validation import _make_run_name, _is_run_complete, run_validation

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

    Arm 1: for each fixed embedding, vary LLMs.
    Arm 2: for each fixed LLM, vary embeddings.
    De-duplicates pairs that appear in multiple arms.
    """
    fixed_embeddings = config["fixed_embeddings"]
    fixed_llms = config["fixed_llms"]
    llm_models = config["llm_models"]
    embed_models = config["embed_models"]

    seen = set()
    pairs = []

    # Arm 1: fix embedding, vary LLMs
    for fixed_embed in fixed_embeddings:
        for llm in llm_models:
            key = (llm, fixed_embed)
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    # Arm 2: fix LLM, vary embeddings
    for fixed_llm in fixed_llms:
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
def _parse_concurrency(argv: list[str]) -> tuple[int, list[str]]:
    """Extract --concurrency N from argv. Returns (concurrency, remaining_argv)."""
    remaining = []
    concurrency = 1
    i = 0
    while i < len(argv):
        if argv[i] == "--concurrency" and i + 1 < len(argv):
            concurrency = int(argv[i + 1])
            i += 2
        elif argv[i].startswith("--concurrency="):
            concurrency = int(argv[i].split("=", 1)[1])
            i += 1
        else:
            remaining.append(argv[i])
            i += 1
    return concurrency, remaining

# %%
#|export
async def _run_all(
    run_def: str,
    remaining: list[tuple[str, str]],
    *,
    concurrency: int = 1,
    force: bool = False,
) -> list[tuple[str, Exception]]:
    """Run validation for all pairs with bounded concurrency. Returns list of (run_name, error) failures."""
    sem = asyncio.Semaphore(concurrency)
    failures: list[tuple[str, Exception]] = []

    async def _run_one(i: int, llm: str, embed: str):
        run_name = _make_run_name(run_def, llm, embed)
        async with sem:
            print(f"\n{'=' * 70}")
            print(f"Run {i}/{len(remaining)}: {run_name}")
            print(f"{'=' * 70}", flush=True)
            try:
                await run_validation(run_def, llm, embed, force=force)
            except Exception as e:
                print(f"\nERROR: validation failed for {run_name}: {e}")
                failures.append((run_name, e))

    tasks = [_run_one(i, llm, embed) for i, (llm, embed) in enumerate(remaining, 1)]
    await asyncio.gather(*tasks)
    return failures

# %%
#|export
def main():
    concurrency, argv = _parse_concurrency(sys.argv[1:])
    args = [a for a in argv if not a.startswith("--")]
    flags = [a for a in argv if a.startswith("--")]
    dry_run = "--dry-run" in flags
    force = "--force" in flags

    if len(args) != 1:
        print(
            "Usage: uv run validate-all <run_def_name> [--dry-run] [--force] [--concurrency N]",
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

    print(f"\nPlanned runs ({len(remaining)}, concurrency={concurrency}):")
    for i, (llm, embed) in enumerate(remaining, 1):
        status = "(done, --force)" if (llm, embed) in done else "(pending)"
        print(f"  {i:>2}. {_make_run_name(run_def, llm, embed)} {status}")

    if dry_run:
        print("\n--dry-run: no validation runs executed.")
        return

    failures = asyncio.run(_run_all(run_def, remaining, concurrency=concurrency, force=force))

    if failures:
        print(f"\n{len(failures)}/{len(remaining)} validation runs failed:")
        for run_name, exc in failures:
            print(f"  - {run_name}: {exc}")
        sys.exit(1)

    print(f"\nAll {len(remaining)} validation runs complete.")
