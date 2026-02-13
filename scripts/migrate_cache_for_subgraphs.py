#!/usr/bin/env python3
"""Migrate netrun cache directories after subgraph decomposition.

Renames .cache/netrun/<old_name>/ -> .cache/netrun/<subgraph.old_name>/
for all nodes that moved into subgraphs. Also handles renames from the
initial decomposition (data_prep.X -> exposure_scores.X, etc.).

Safe to run multiple times (skips already-migrated entries).
"""

import shutil
from pathlib import Path

CACHE_DIR = Path(".cache/netrun")

# Mapping: old_name -> new_prefixed_name
# Includes both the original flat names and the first-round prefixed names.
RENAMES = {
    # === From flat (original) names ===
    # data_prep subgraph
    "fetch_onet": "data_prep.fetch_onet",
    "load_job_ads": "data_prep.load_job_ads",
    # exposure_scores subgraph (O*NET processing + scoring)
    "bc_onet_tables": "exposure_scores.bc_onet_tables",
    "build_onet_descriptions": "exposure_scores.build_onet_descriptions",
    "build_onet_eval_dfs": "exposure_scores.build_onet_eval_dfs",
    "score_task_exposure": "exposure_scores.score_task_exposure",
    "bc_abilities": "exposure_scores.bc_abilities",
    "score_presence": "exposure_scores.score_presence",
    "score_felten": "exposure_scores.score_felten",
    "aggregate_soc_exposure": "exposure_scores.aggregate_soc_exposure",
    # job_ad_matching subgraph
    "bc_descriptions": "job_ad_matching.bc_descriptions",
    "embed_onet": "job_ad_matching.embed_onet",
    "embed_job_ads": "job_ad_matching.embed_job_ads",
    "compute_cosine_similarity": "job_ad_matching.compute_cosine_similarity",
    "llm_filter_candidates": "job_ad_matching.llm_filter_candidates",
    # benchmark_exposure_scores subgraph
    "benchmark_exposure": "benchmark_exposure_scores.benchmark_exposure",
    # generate_index subgraph
    "combine_job_exposure": "generate_index.combine_job_exposure",
    # index_analysis subgraph
    "bc_job_exposure_index": "index_analysis.bc_job_exposure_index",
    "aggregate_geography": "index_analysis.aggregate_geography",
    "compute_summary_stats": "index_analysis.compute_summary_stats",

    # === From first-round prefixed names (data_prep/matching/exposure/output) ===
    # data_prep nodes that moved
    "data_prep.bc_onet_tables": "exposure_scores.bc_onet_tables",
    "data_prep.build_onet_descriptions": "exposure_scores.build_onet_descriptions",
    "data_prep.build_onet_eval_dfs": "exposure_scores.build_onet_eval_dfs",
    # matching -> job_ad_matching
    "matching.bc_descriptions": "job_ad_matching.bc_descriptions",
    "matching.embed_onet": "job_ad_matching.embed_onet",
    "matching.embed_job_ads": "job_ad_matching.embed_job_ads",
    "matching.compute_cosine_similarity": "job_ad_matching.compute_cosine_similarity",
    "matching.llm_filter_candidates": "job_ad_matching.llm_filter_candidates",
    # exposure -> exposure_scores
    "exposure.score_task_exposure": "exposure_scores.score_task_exposure",
    "exposure.bc_abilities": "exposure_scores.bc_abilities",
    "exposure.score_presence": "exposure_scores.score_presence",
    "exposure.score_felten": "exposure_scores.score_felten",
    "exposure.aggregate_soc_exposure": "exposure_scores.aggregate_soc_exposure",
    # output -> split across subgraphs
    "output.bc_exposure_scores": "bc_exposure_scores",  # moved to parent
    "output.benchmark_exposure": "benchmark_exposure_scores.benchmark_exposure",
    "output.combine_job_exposure": "generate_index.combine_job_exposure",
    "output.bc_job_exposure_index": "index_analysis.bc_job_exposure_index",
    "output.aggregate_geography": "index_analysis.aggregate_geography",
    "output.compute_summary_stats": "index_analysis.compute_summary_stats",
}


def migrate():
    if not CACHE_DIR.exists():
        print(f"Cache directory {CACHE_DIR} does not exist. Nothing to migrate.")
        return

    migrated = 0
    skipped = 0
    for old_name, new_name in RENAMES.items():
        old_path = CACHE_DIR / old_name
        new_path = CACHE_DIR / new_name
        if old_path.exists():
            if new_path.exists():
                print(f"  SKIP {old_name} -> {new_name} (target already exists)")
                skipped += 1
                continue
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"  MOVED {old_name} -> {new_name}")
            migrated += 1
        else:
            if new_path.exists():
                skipped += 1
            else:
                print(f"  SKIP {old_name} (not cached)")
                skipped += 1

    print(f"\nDone: {migrated} migrated, {skipped} skipped.")


if __name__ == "__main__":
    migrate()
