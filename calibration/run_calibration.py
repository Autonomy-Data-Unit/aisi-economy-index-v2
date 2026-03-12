"""Run a calibration pipeline to measure per-ad timing for LLM and embedding models.

Usage:
    uv run python calibration/run_calibration.py <llm_model_key> <embedding_model_key>

Example:
    uv run python calibration/run_calibration.py qwen-7b-sbatch bge-large-sbatch

Runs the pipeline with sample_n=1000 ads (configurable in calibration/run_defs.toml),
collects timing data from LLM and embedding nodes, and writes results to
calibration/results/<llm_model>__<embed_model>.json.
"""

import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CALIBRATION_DIR = Path(__file__).parent
RESULTS_DIR = CALIBRATION_DIR / "results"
CALIBRATION_RUN_DEFS_PATH = CALIBRATION_DIR / "run_defs.toml"
RUN_NAME = "calibration"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _build_run_defs(llm_model_key: str, embedding_model_key: str) -> dict:
    """Load main run_defs.toml, overlay calibration overrides, and inject dynamic run."""
    from ai_index.const import run_defs_path as main_run_defs_path
    from ai_index.run_pipeline import _load_run_defs

    base = _load_run_defs(main_run_defs_path)
    overrides = _load_run_defs(CALIBRATION_RUN_DEFS_PATH)
    run_defs = _deep_merge(base, overrides)

    run_defs.setdefault("runs", {})[RUN_NAME] = {
        "llm_model": llm_model_key,
        "embedding_model": embedding_model_key,
    }
    return run_defs


def _read_meta(run_name: str, node_name: str, meta_filename: str) -> dict | None:
    """Read a node's meta JSON file from the pipeline store."""
    from ai_index.const import pipeline_store_path

    meta_path = pipeline_store_path / run_name / node_name / meta_filename
    if not meta_path.exists():
        print(f"warning: {meta_path} not found, skipping")
        return None
    with open(meta_path) as f:
        return json.load(f)


def _extract_llm_timing(meta: dict) -> dict:
    """Extract timing fields from a run_batched meta dict."""
    n_total = meta["n_total"]
    elapsed = meta["elapsed_seconds"]
    return {
        "n_total": n_total,
        "n_success": meta["n_success"],
        "n_failed": meta["n_failed"],
        "started_at": meta["started_at"],
        "ended_at": meta["ended_at"],
        "elapsed_seconds": elapsed,
        "seconds_per_ad": elapsed / n_total if n_total > 0 else 0.0,
    }


def _extract_embed_ads_timing(meta: dict) -> dict:
    """Extract timing fields from embed_ads meta dict."""
    n_embedded = meta["n_embedded"]
    elapsed = meta["elapsed_seconds"]
    return {
        "n_total": meta["n_total"],
        "n_embedded": n_embedded,
        "n_skipped": meta["n_skipped"],
        "started_at": meta["started_at"],
        "ended_at": meta["ended_at"],
        "elapsed_seconds": elapsed,
        "seconds_per_ad": elapsed / n_embedded if n_embedded > 0 else 0.0,
    }


def _extract_embed_onet_timing(meta: dict) -> dict:
    """Extract timing fields from embed_onet meta dict."""
    return {
        "n_occupations": meta["n_occupations"],
        "started_at": meta["started_at"],
        "ended_at": meta["ended_at"],
        "elapsed_seconds": meta["elapsed_seconds"],
    }


async def run_calibration(llm_model_key: str, embedding_model_key: str) -> None:
    from ai_index.const import pipeline_store_path
    from ai_index.run_pipeline import run_pipeline_async

    run_defs = _build_run_defs(llm_model_key, embedding_model_key)
    sample_n = run_defs["defaults"]["sample_n"]

    # Clean previous calibration run to ensure fresh timing
    calibration_store = pipeline_store_path / RUN_NAME
    if calibration_store.exists():
        print(f"calibration: cleaning {calibration_store}")
        shutil.rmtree(calibration_store)

    print(f"calibration: llm_model={llm_model_key}, embedding_model={embedding_model_key}, sample_n={sample_n}")
    await run_pipeline_async(RUN_NAME, run_defs=run_defs)

    # Collect timing from meta files
    nodes = {}

    summarise_meta = _read_meta(RUN_NAME, "llm_summarise", "summary_meta.json")
    if summarise_meta is not None:
        nodes["llm_summarise"] = _extract_llm_timing(summarise_meta)

    filter_meta = _read_meta(RUN_NAME, "llm_filter_candidates", "filter_meta.json")
    if filter_meta is not None:
        nodes["llm_filter_candidates"] = _extract_llm_timing(filter_meta)

    embed_ads_meta = _read_meta(RUN_NAME, "embed_ads", "embed_meta.json")
    if embed_ads_meta is not None:
        nodes["embed_ads"] = _extract_embed_ads_timing(embed_ads_meta)

    embed_onet_meta = _read_meta(RUN_NAME, "embed_onet", "embed_meta.json")
    if embed_onet_meta is not None:
        nodes["embed_onet"] = _extract_embed_onet_timing(embed_onet_meta)

    result = {
        "llm_model_key": llm_model_key,
        "embedding_model_key": embedding_model_key,
        "sample_n": sample_n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_name = f"{llm_model_key}__{embedding_model_key}"
    output_path = RESULTS_DIR / f"{result_name}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\ncalibration: results written to {output_path}")

    # Print summary
    for name, timing in nodes.items():
        elapsed = timing["elapsed_seconds"]
        if "seconds_per_ad" in timing:
            spa = timing["seconds_per_ad"]
            n = timing.get("n_total", timing.get("n_embedded", "?"))
            print(f"  {name}: {elapsed:.1f}s total, {spa:.3f}s/ad (n={n})")
        else:
            print(f"  {name}: {elapsed:.1f}s total")


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: uv run python calibration/run_calibration.py <llm_model_key> <embedding_model_key>",
            file=sys.stderr,
        )
        sys.exit(1)
    llm_model_key = sys.argv[1]
    embedding_model_key = sys.argv[2]
    asyncio.run(run_calibration(llm_model_key, embedding_model_key))


if __name__ == "__main__":
    main()
