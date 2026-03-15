# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # calibrate_all
#
# Run calibrations for all sbatch models, skipping those already calibrated.
#
# Usage:
#     uv run calibrate-all [--dry-run] [--sequential]
#
# Reads all sbatch models from embed_models.toml and llm_models.toml, checks
# which already have results in store/calibration/results/, and pairs
# uncalibrated models into the minimum number of pipeline runs.

# %%
#|default_exp calibrate_all

# %%
#|export
import asyncio
import shutil
import sys
import tomllib
from pathlib import Path

from ai_index.const import calibration_results_path, llm_models_config_path, embed_models_config_path, pipeline_store_path
from calibration.run_calibration import run_calibration, RESULTS_DIR

# %%
#|export
def _get_sbatch_keys(config_path: Path) -> list[str]:
    """Return all model keys with mode='sbatch' from a model TOML config."""
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    return [
        key for key, entry in cfg["models"].items()
        if entry["mode"] == "sbatch"
    ]

# %%
#|export
def _calibrated_keys(results_dir: Path) -> tuple[set[str], set[str]]:
    """Return (llm_keys, embed_keys) that already have calibration results."""
    done_llm = set()
    done_embed = set()
    if not results_dir.exists():
        return done_llm, done_embed
    for path in results_dir.glob("*.json"):
        import json
        with open(path) as f:
            r = json.load(f)
        done_llm.add(r["llm_model"])
        done_embed.add(r["embedding_model"])
    return done_llm, done_embed

# %%
#|export
def plan_runs(
    all_llm: list[str],
    all_embed: list[str],
    done_llm: set[str],
    done_embed: set[str],
) -> list[tuple[str, str]]:
    """Plan (llm_key, embed_key) pairs to cover all uncalibrated models."""
    need_llm = [k for k in all_llm if k not in done_llm]
    need_embed = [k for k in all_embed if k not in done_embed]

    pairs = []

    while need_llm and need_embed:
        pairs.append((need_llm.pop(0), need_embed.pop(0)))

    if need_llm:
        fallback_embed = next(iter(done_embed), all_embed[0])
        for llm_key in need_llm:
            pairs.append((llm_key, fallback_embed))

    if need_embed:
        fallback_llm = next(iter(done_llm), all_llm[0])
        for embed_key in need_embed:
            pairs.append((fallback_llm, embed_key))

    return pairs

# %%
#|export
async def _run_all(pairs: list[tuple[str, str]], *, parallel: bool = True) -> list[tuple[str, str]]:
    """Run calibration for all pairs. Returns list of failed pairs."""
    failures = []

    async def _run_one(i: int, llm: str, embed: str):
        run_name = f"_calibration_{i}" if parallel else "calibration"
        print(f"\n{'=' * 70}")
        print(f"Run {i}/{len(pairs)}: {llm} + {embed}")
        print(f"{'=' * 70}", flush=True)
        try:
            await run_calibration(llm, embed, run_name=run_name)
        except Exception as e:
            print(f"\nERROR: calibration failed for {llm} + {embed}: {e}")
            failures.append((llm, embed))

    tasks = [_run_one(i, llm, embed) for i, (llm, embed) in enumerate(pairs, 1)]
    if parallel:
        await asyncio.gather(*tasks)
    else:
        for task in tasks:
            await task

    return failures

# %%
#|export
def main():
    dry_run = "--dry-run" in sys.argv
    sequential = "--sequential" in sys.argv

    all_llm = _get_sbatch_keys(llm_models_config_path)
    all_embed = _get_sbatch_keys(embed_models_config_path)
    done_llm, done_embed = _calibrated_keys(RESULTS_DIR)

    print(f"LLM models:   {len(all_llm)} total, {len(done_llm)} calibrated, {len(all_llm) - len(done_llm)} remaining")
    print(f"Embed models: {len(all_embed)} total, {len(done_embed)} calibrated, {len(all_embed) - len(done_embed)} remaining")

    pairs = plan_runs(all_llm, all_embed, done_llm, done_embed)

    if not pairs:
        print("\nAll models already calibrated.")
        return

    mode = "sequential" if sequential else "parallel"
    print(f"\nPlanned runs ({len(pairs)}, {mode}):")
    for i, (llm, embed) in enumerate(pairs, 1):
        llm_new = "(new)" if llm not in done_llm else "(done)"
        embed_new = "(new)" if embed not in done_embed else "(done)"
        print(f"  {i:>2}. {llm} {llm_new}  +  {embed} {embed_new}")

    if dry_run:
        print("\n--dry-run: no calibrations executed.")
        return

    print()
    failures = asyncio.run(_run_all(pairs, parallel=not sequential))

    if not sequential:
        for i in range(1, len(pairs) + 1):
            tmp_store = pipeline_store_path / f"_calibration_{i}"
            if tmp_store.exists():
                shutil.rmtree(tmp_store)

    if failures:
        print(f"\n{len(failures)}/{len(pairs)} calibration runs failed:")
        for llm, embed in failures:
            print(f"  - {llm} + {embed}")
        sys.exit(1)

    print(f"All {len(pairs)} calibration runs complete.")
