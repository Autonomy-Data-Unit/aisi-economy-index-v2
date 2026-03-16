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
import sys
import tomllib
from pathlib import Path

from ai_index.const import calibration_results_path, llm_models_config_path, embed_models_config_path, rerank_models_config_path
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
def _calibrated_keys(results_dir: Path) -> tuple[set[str], set[str], set[str]]:
    """Return (llm_keys, embed_keys, rerank_keys) that already have calibration results."""
    done_llm = set()
    done_embed = set()
    done_rerank = set()
    if not results_dir.exists():
        return done_llm, done_embed, done_rerank
    for path in results_dir.glob("*.json"):
        import json
        with open(path) as f:
            r = json.load(f)
        done_llm.add(r["llm_model"])
        done_embed.add(r["embedding_model"])
        if r.get("rerank_model"):
            done_rerank.add(r["rerank_model"])
    return done_llm, done_embed, done_rerank

# %%
#|export
def plan_runs(
    all_llm: list[str],
    all_embed: list[str],
    all_rerank: list[str],
    done_llm: set[str],
    done_embed: set[str],
    done_rerank: set[str],
) -> list[tuple[str, str, str | None]]:
    """Plan (llm_key, embed_key, rerank_key) triples to cover all uncalibrated models.

    Pairs uncalibrated LLM, embed, and rerank models together to minimise
    the number of pipeline runs. Each run calibrates one model from each
    category. When one category runs out, the remaining models use a
    fallback (an already-calibrated key or the first key).
    """
    need_llm = [k for k in all_llm if k not in done_llm]
    need_embed = [k for k in all_embed if k not in done_embed]
    need_rerank = [k for k in all_rerank if k not in done_rerank]

    triples = []

    # Zip all three lists together as far as they go
    while need_llm and need_embed and need_rerank:
        triples.append((need_llm.pop(0), need_embed.pop(0), need_rerank.pop(0)))

    # Two lists remaining
    while need_llm and need_embed:
        triples.append((need_llm.pop(0), need_embed.pop(0), None))

    while need_llm and need_rerank:
        fallback_embed = next(iter(done_embed), all_embed[0])
        triples.append((need_llm.pop(0), fallback_embed, need_rerank.pop(0)))

    while need_embed and need_rerank:
        fallback_llm = next(iter(done_llm), all_llm[0])
        triples.append((fallback_llm, need_embed.pop(0), need_rerank.pop(0)))

    # One list remaining
    if need_llm:
        fallback_embed = next(iter(done_embed), all_embed[0])
        for llm_key in need_llm:
            triples.append((llm_key, fallback_embed, None))

    if need_embed:
        fallback_llm = next(iter(done_llm), all_llm[0])
        for embed_key in need_embed:
            triples.append((fallback_llm, embed_key, None))

    if need_rerank:
        fallback_llm = next(iter(done_llm), all_llm[0])
        fallback_embed = next(iter(done_embed), all_embed[0])
        for rerank_key in need_rerank:
            triples.append((fallback_llm, fallback_embed, rerank_key))

    return triples

# %%
#|export
async def _run_all(triples: list[tuple[str, str, str | None]], *, concurrency: int = 2) -> list[tuple[str, str, str | None]]:
    """Run calibration for all triples with bounded concurrency. Returns list of failed triples."""
    sem = asyncio.Semaphore(concurrency)
    failures = []

    async def _run_one(i: int, llm: str, embed: str, rerank: str | None):
        rerank_label = rerank or "(default)"
        async with sem:
            print(f"\n{'=' * 70}")
            print(f"Run {i}/{len(triples)}: {llm} + {embed} + {rerank_label}")
            print(f"{'=' * 70}", flush=True)
            try:
                await run_calibration(llm, embed, rerank)
            except Exception as e:
                print(f"\nERROR: calibration failed for {llm} + {embed} + {rerank_label}: {e}")
                failures.append((llm, embed, rerank))

    tasks = [_run_one(i, llm, embed, rerank) for i, (llm, embed, rerank) in enumerate(triples, 1)]
    await asyncio.gather(*tasks)

    return failures

# %%
#|export
def _parse_concurrency(argv: list[str]) -> tuple[int, list[str]]:
    """Extract --concurrency N from argv. Returns (concurrency, remaining_argv)."""
    remaining = []
    concurrency = 2
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


def main():
    concurrency, argv = _parse_concurrency(sys.argv[1:])
    dry_run = "--dry-run" in argv

    all_llm = _get_sbatch_keys(llm_models_config_path)
    all_embed = _get_sbatch_keys(embed_models_config_path)
    all_rerank = _get_sbatch_keys(rerank_models_config_path)
    done_llm, done_embed, done_rerank = _calibrated_keys(RESULTS_DIR)

    print(f"LLM models:    {len(all_llm)} total, {len(done_llm)} calibrated, {len(all_llm) - len(done_llm)} remaining")
    print(f"Embed models:  {len(all_embed)} total, {len(done_embed)} calibrated, {len(all_embed) - len(done_embed)} remaining")
    print(f"Rerank models: {len(all_rerank)} total, {len(done_rerank)} calibrated, {len(all_rerank) - len(done_rerank)} remaining")

    triples = plan_runs(all_llm, all_embed, all_rerank, done_llm, done_embed, done_rerank)

    if not triples:
        print("\nAll models already calibrated.")
        return

    print(f"\nPlanned runs ({len(triples)}, concurrency={concurrency}):")
    for i, (llm, embed, rerank) in enumerate(triples, 1):
        llm_tag = "(new)" if llm not in done_llm else "(done)"
        embed_tag = "(new)" if embed not in done_embed else "(done)"
        rerank_label = rerank or "(default)"
        rerank_tag = "(new)" if rerank and rerank not in done_rerank else "(done)" if rerank else ""
        print(f"  {i:>2}. {llm} {llm_tag}  +  {embed} {embed_tag}  +  {rerank_label} {rerank_tag}")

    if dry_run:
        print("\n--dry-run: no calibrations executed.")
        return

    print()
    failures = asyncio.run(_run_all(triples, concurrency=concurrency))

    if failures:
        print(f"\n{len(failures)}/{len(triples)} calibration runs failed:")
        for llm, embed, rerank in failures:
            print(f"  - {llm} + {embed} + {rerank or '(default)'}")
        sys.exit(1)

    print(f"All {len(triples)} calibration runs complete.")
