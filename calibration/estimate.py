"""Estimate GPU-hours for a full pipeline run based on calibration results.

Usage:
    uv run python calibration/estimate.py [N_ADS]

Reads all calibration result JSONs from calibration/results/ and prints
estimated GPU-hours per model per node. Default N_ADS is 30,000,000.

When Slurm accounting data is available (from sacct), estimates use actual
GPU execution time. Otherwise falls back to wall-clock time (which includes
transfer overhead and may overestimate).
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_N_ADS = 30_000_000

# Nodes with per-ad scaling (seconds_per_ad field)
PER_AD_NODES = ["llm_summarise", "llm_filter_candidates", "embed_ads"]
# Nodes with fixed cost (no per-ad scaling)
FIXED_NODES = ["embed_onet"]


def load_results() -> list[dict]:
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def estimate_hours(seconds_per_ad: float, n_ads: int) -> float:
    return seconds_per_ad * n_ads / 3600


def main():
    n_ads = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_ADS

    results = load_results()
    if not results:
        print(f"No calibration results found in {RESULTS_DIR}/")
        print("Run: uv run python calibration/run_calibration.py <llm_model> <embed_model>")
        sys.exit(1)

    # Header
    print(f"\nGPU-hour estimates for {n_ads:,} ads")
    print("=" * 100)

    # Column widths
    model_w = max(
        len(f"{r['llm_model_key']} / {r['embedding_model_key']}") for r in results
    ) + 2
    node_w = max(len(n) for n in PER_AD_NODES + FIXED_NODES) + 2

    # Table header
    header = f"{'Models':<{model_w}} {'Node':<{node_w}} {'s/ad':>8} {'Est. hours':>12} {'NHR':>8} {'Cal. N':>8} {'Source':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        model_label = f"{r['llm_model_key']} / {r['embedding_model_key']}"
        nodes = r["nodes"]
        total_hours = 0.0
        total_nhr = 0.0

        for node_name in PER_AD_NODES:
            if node_name not in nodes:
                print(f"{model_label:<{model_w}} {node_name:<{node_w}} {'n/a':>8}")
                continue

            timing = nodes[node_name]
            spa = timing["seconds_per_ad"]
            hours = estimate_hours(spa, n_ads)
            total_hours += hours
            n_cal = timing["n"]
            source = "slurm" if timing.get("slurm_seconds") else "clock"

            # Estimate node-hours (0.25 NHR per GPU-hour for 1-GPU jobs)
            nhr = hours * 0.25
            total_nhr += nhr

            print(f"{model_label:<{model_w}} {node_name:<{node_w}} {spa:>8.4f} {hours:>12.1f} {nhr:>8.1f} {n_cal:>8,} {source:>8}")
            model_label = ""  # only print model on first row

        for node_name in FIXED_NODES:
            if node_name not in nodes:
                continue

            timing = nodes[node_name]
            fixed_hours = timing["elapsed_seconds"] / 3600
            total_hours += fixed_hours
            source = "slurm" if timing.get("slurm_seconds") else "clock"
            nhr = fixed_hours * 0.25
            total_nhr += nhr

            print(f"{model_label:<{model_w}} {node_name:<{node_w}} {'fixed':>8} {fixed_hours:>12.3f} {nhr:>8.3f} {'':>8} {source:>8}")
            model_label = ""

        print(f"{'':>{model_w}} {'TOTAL':<{node_w}} {'':>8} {total_hours:>12.1f} {total_nhr:>8.1f}")
        print()


if __name__ == "__main__":
    main()
