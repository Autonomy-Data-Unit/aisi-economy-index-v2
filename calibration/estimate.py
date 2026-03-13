"""Estimate GPU-hours for a full pipeline run based on calibration results.

Usage:
    uv run python calibration/estimate.py [N_ADS] [SAMPLE_N]

Arguments:
    N_ADS     Total ads to estimate for (default: actual count from adzuna.duckdb)
    SAMPLE_N  Override calibration sample size for per-ad rate calculation

Reads calibration results from calibration/results/llm/ and
calibration/results/embed/, reporting LLM and embedding models separately.

When Slurm accounting data is available (from sacct), estimates use actual
GPU execution time. Otherwise falls back to wall-clock time (which includes
transfer overhead and may overestimate).
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
LLM_RESULTS_DIR = RESULTS_DIR / "llm"
EMBED_RESULTS_DIR = RESULTS_DIR / "embed"

# LLM nodes with per-ad scaling
LLM_NODES = ["llm_summarise", "llm_filter_candidates"]
# Embed nodes with per-ad scaling
EMBED_PER_AD_NODES = ["embed_ads"]
# Embed nodes with fixed cost
EMBED_FIXED_NODES = ["embed_onet"]


def _count_ads() -> int:
    """Count total ads in the Adzuna DuckDB database."""
    import duckdb
    from ai_index.const import inputs_path
    con = duckdb.connect(str(inputs_path / "adzuna.duckdb"), read_only=True)
    count = con.sql("SELECT COUNT(*) FROM ads").fetchone()[0]
    con.close()
    return count


def _load_results(results_dir: Path) -> list[dict]:
    results = []
    if results_dir.exists():
        for path in sorted(results_dir.glob("*.json")):
            with open(path) as f:
                results.append(json.load(f))
    return results


def _estimate_hours(seconds_per_ad: float, n_ads: int) -> float:
    return seconds_per_ad * n_ads / 3600


def _print_table(title: str, results: list[dict], per_ad_nodes: list[str],
                 fixed_nodes: list[str], n_ads: int, sample_n_override: int | None) -> None:
    if not results:
        print(f"\n{title}: no results found")
        return

    model_w = max(len(r["model_key"]) for r in results) + 2
    node_w = max(len(n) for n in per_ad_nodes + fixed_nodes) + 2

    print(f"\n{title}")
    print("=" * 90)
    header = f"{'Model':<{model_w}} {'Node':<{node_w}} {'s/ad':>8} {'Est. hours':>12} {'NHR':>8} {'Cal. N':>8} {'Source':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        model_label = r["model_key"]
        nodes = r["nodes"]
        total_hours = 0.0
        total_nhr = 0.0

        for node_name in per_ad_nodes:
            if node_name not in nodes:
                print(f"{model_label:<{model_w}} {node_name:<{node_w}} {'n/a':>8}")
                model_label = ""
                continue

            timing = nodes[node_name]
            n_cal = sample_n_override if sample_n_override else timing["n"]
            spa = timing["elapsed_seconds"] / n_cal
            hours = _estimate_hours(spa, n_ads)
            total_hours += hours
            source = "slurm" if timing.get("slurm_seconds") else "clock"
            nhr = hours * 0.25

            total_nhr += nhr
            print(f"{model_label:<{model_w}} {node_name:<{node_w}} {spa:>8.4f} {hours:>12.1f} {nhr:>8.1f} {n_cal:>8,} {source:>8}")
            model_label = ""

        for node_name in fixed_nodes:
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


def main():
    n_ads = int(sys.argv[1]) if len(sys.argv) > 1 else _count_ads()
    sample_n_override = int(sys.argv[2]) if len(sys.argv) > 2 else None

    llm_results = _load_results(LLM_RESULTS_DIR)
    embed_results = _load_results(EMBED_RESULTS_DIR)

    if not llm_results and not embed_results:
        print(f"No calibration results found in {RESULTS_DIR}/")
        print("Run: uv run python calibration/run_calibration.py <llm_model> <embed_model>")
        sys.exit(1)

    print(f"\nGPU-hour estimates for {n_ads:,} ads")

    _print_table("LLM Models", llm_results, LLM_NODES, [], n_ads, sample_n_override)
    _print_table("Embedding Models", embed_results, EMBED_PER_AD_NODES, EMBED_FIXED_NODES, n_ads, sample_n_override)


if __name__ == "__main__":
    main()
