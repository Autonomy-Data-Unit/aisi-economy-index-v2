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
# which already have results in store/calibration/results/{llm,embed}/, and pairs
# uncalibrated models into the minimum number of pipeline runs. Each run
# calibrates one LLM model and one embedding model simultaneously.
#
# When one model type has more uncalibrated entries than the other, the excess
# are paired with an already-calibrated model (or the first model of the other
# type if none are calibrated yet).

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

LLM_RESULTS_DIR = calibration_results_path / "llm"
EMBED_RESULTS_DIR = calibration_results_path / "embed"

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
def _calibrated_keys(results_dir: Path) -> set[str]:
    """Return model keys that already have calibration result files."""
    if not results_dir.exists():
        return set()
    return {p.stem for p in results_dir.glob("*.json")}

# %%
#|export
def plan_runs(
    all_llm: list[str],
    all_embed: list[str],
    done_llm: set[str],
    done_embed: set[str],
) -> list[tuple[str, str]]:
    """Plan (llm_key, embed_key) pairs to cover all uncalibrated models.

    Pairs uncalibrated LLMs with uncalibrated embeds first, then pairs any
    remaining with an already-calibrated model from the other type.
    """
    need_llm = [k for k in all_llm if k not in done_llm]
    need_embed = [k for k in all_embed if k not in done_embed]

    pairs = []

    # Pair uncalibrated LLMs with uncalibrated embeds
    while need_llm and need_embed:
        pairs.append((need_llm.pop(0), need_embed.pop(0)))

    # Remaining uncalibrated LLMs: pair with a calibrated embed (or first embed)
    if need_llm:
        fallback_embed = next(iter(done_embed), all_embed[0])
        for llm_key in need_llm:
            pairs.append((llm_key, fallback_embed))

    # Remaining uncalibrated embeds: pair with a calibrated LLM (or first LLM)
    if need_embed:
        fallback_llm = next(iter(done_llm), all_llm[0])
        for embed_key in need_embed:
            pairs.append((fallback_llm, embed_key))

    return pairs

# %%
#|export
async def _run_all(pairs: list[tuple[str, str]], *, parallel: bool = True) -> list[tuple[str, str]]:
    """Run calibration for all pairs as subprocesses. Returns list of failed (llm, embed) pairs.

    Each calibration runs in a separate process to avoid in-process DuckDB
    contention (DuckDB rejects concurrent connections with different
    configurations within the same process). Cross-process, DuckDB uses file
    locking which handles contention gracefully.
    """
    run_calibration_bin = shutil.which("run-calibration")
    if run_calibration_bin is None:
        print("ERROR: 'run-calibration' not found on PATH. Is the package installed?", file=sys.stderr)
        sys.exit(1)

    failures = []

    async def _run_one(i: int, llm: str, embed: str):
        run_name = f"_calibration_{i}" if parallel else "calibration"
        print(f"\n{'=' * 70}")
        print(f"Run {i}/{len(pairs)}: {llm} + {embed}")
        print(f"{'=' * 70}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            run_calibration_bin, llm, embed, "--run-name", run_name,
        )
        returncode = await proc.wait()
        if returncode != 0:
            print(f"\nERROR: calibration failed for {llm} + {embed} (exit code {returncode})")
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
    done_llm = _calibrated_keys(LLM_RESULTS_DIR)
    done_embed = _calibrated_keys(EMBED_RESULTS_DIR)

    print(f"LLM models:   {len(all_llm)} total, {len(done_llm)} calibrated, {len(all_llm) - len(done_llm)} remaining")
    print(f"Embed models: {len(all_embed)} total, {len(done_embed)} calibrated, {len(all_embed) - len(done_embed)} remaining")

    if done_llm:
        print(f"  LLM done:   {', '.join(sorted(done_llm))}")
    if done_embed:
        print(f"  Embed done: {', '.join(sorted(done_embed))}")

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

    # Clean up temporary pipeline store directories from parallel runs
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
