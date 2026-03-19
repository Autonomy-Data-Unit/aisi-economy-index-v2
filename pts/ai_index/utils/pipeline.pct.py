# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # utils.pipeline
#
# Shared utilities for constructing and running pipeline configurations.
# Used by the calibration and validation modules to build dynamic run
# definitions from a base config + model key overrides.

# %%
#|default_exp utils.pipeline

# %%
#|export
import copy
from pathlib import Path

from ai_index.const import run_defs_path, pipeline_store_path


def make_run_name(prefix: str, *parts: str) -> str:
    """Build a run name from a prefix and parts joined by double underscores.

    Example: make_run_name("val", "validation_5k", "qwen-7b-sbatch", "bge-large-sbatch")
    -> "val__validation_5k__qwen-7b-sbatch__bge-large-sbatch"
    """
    return "__".join([prefix] + [p for p in parts if p is not None])


def build_run_defs(base_run: str, run_name: str, overrides: dict | None = None) -> dict:
    """Load run_defs.toml, deep-copy a base run section, apply overrides, and register
    the result under a new run name.

    Overrides follow the same structure as a [runs.X] section in run_defs.toml:
    scalar values override global node vars, dict values deep-merge into per-node
    var subtables (preserving base values not present in the override).

    Args:
        base_run: Name of the base run definition in run_defs.toml (e.g. "calibration").
        run_name: Name to register the new run under.
        overrides: Dict of overrides to apply on top of the base run definition.

    Returns:
        The full run_defs dict with the new run registered at runs[run_name].
    """
    from ai_index.run_pipeline import _load_run_defs

    run_defs = _load_run_defs(run_defs_path)
    template = run_defs["runs"][base_run]
    run_entry = copy.deepcopy(template)

    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict):
                # Deep-merge per-node overrides: preserve base keys, override specified ones
                run_entry.setdefault(k, {}).update(v)
            else:
                run_entry[k] = v

    run_defs["runs"][run_name] = run_entry
    return run_defs


def is_run_complete(run_name: str, marker_path: str = "compute_job_ad_exposure/exposure_meta.json") -> bool:
    """Check if a pipeline run has completed by looking for a marker file.

    Uses *_meta.json files as completion markers since these are written last
    by each node, after all parquet/data files. A missing meta file means the
    node was interrupted mid-write and the data files may be corrupt.

    Args:
        run_name: The pipeline run name (directory under store/pipeline/).
        marker_path: Relative path within the run directory to check.
            Default checks for the final exposure meta (last node to complete).
    """
    return (pipeline_store_path / run_name / marker_path).exists()


def check_run_integrity(run_name: str) -> list[str]:
    """Verify that all expected meta files (completion markers) exist for a pipeline run.

    Each node writes a *_meta.json after its data files. A missing meta file
    indicates an interrupted write, meaning the corresponding data files may be
    corrupt and the node should be re-run.

    Returns a list of issues (empty if everything is OK).
    """
    run_dir = pipeline_store_path / run_name
    if not run_dir.exists():
        return [f"run directory does not exist: {run_dir}"]

    expected_metas = [
        "cosine_candidates/cosine_meta.json",
        "llm_filter_candidates/filter_meta.json",
        "rerank_candidates/rerank_meta.json",
        "compute_job_ad_exposure/exposure_meta.json",
    ]

    issues = []
    for rel_path in expected_metas:
        p = run_dir / rel_path
        if not p.exists():
            # Check if the parent node directory exists at all (in-progress vs corrupt)
            node_dir = p.parent
            if node_dir.exists():
                issues.append(f"INCOMPLETE {rel_path} (node dir exists but meta missing, likely interrupted)")
            else:
                issues.append(f"MISSING {rel_path}")
    return issues


def get_sample_n(run_defs: dict, run_name: str) -> int:
    """Read sample_n from a run definition, falling back to defaults."""
    run_entry = run_defs["runs"][run_name]
    if "sample_n" in run_entry:
        return run_entry["sample_n"]
    return run_defs["defaults"]["sample_n"]


def clean_incomplete_nodes(run_name: str, *, dry_run: bool = True) -> list[str]:
    """Find and optionally delete node directories that lack a *_meta.json completion marker.

    A node directory without a meta file was interrupted mid-write and its data
    files may be corrupt. Deleting it allows the pipeline to re-run that node cleanly.

    Args:
        run_name: The pipeline run name (directory under store/pipeline/).
        dry_run: If True (default), only report what would be deleted. If False, delete.

    Returns:
        List of node directories that were (or would be) deleted.
    """
    import shutil

    run_dir = pipeline_store_path / run_name
    if not run_dir.exists():
        return []

    # Map of node directory name -> expected meta file name
    node_metas = {
        "cosine_candidates": "cosine_meta.json",
        "llm_filter_candidates": "filter_meta.json",
        "rerank_candidates": "rerank_meta.json",
        "compute_job_ad_exposure": "exposure_meta.json",
        "embed_ads": "embed_meta.json",
        "embed_onet": "embed_meta.json",
    }

    deleted = []
    for node_name, meta_file in node_metas.items():
        node_dir = run_dir / node_name
        if node_dir.exists() and not (node_dir / meta_file).exists():
            deleted.append(str(node_dir))
            if not dry_run:
                shutil.rmtree(node_dir)
    return deleted


def clean_store_main():
    """CLI entry point: scan store/pipeline/ for incomplete node directories and clean them."""
    import argparse

    import fnmatch

    parser = argparse.ArgumentParser(description="Clean incomplete node directories from store/pipeline/")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run)")
    parser.add_argument("pattern", nargs="?", default=None, help="Glob pattern to filter run names (e.g. 'val__*', 'cal__*__bge-large-*')")
    args = parser.parse_args()

    all_runs = sorted(d.name for d in pipeline_store_path.iterdir() if d.is_dir())
    if args.pattern:
        run_names = [r for r in all_runs if fnmatch.fnmatch(r, args.pattern)]
    else:
        run_names = all_runs

    total_deleted = 0
    for run_name in run_names:
        removed = clean_incomplete_nodes(run_name, dry_run=not args.apply)
        for path in removed:
            action = "DELETED" if args.apply else "WOULD DELETE"
            print(f"  {action}: {path}")
            total_deleted += 1

    if total_deleted == 0:
        print("All node directories have meta files. Nothing to clean.")
    elif not args.apply:
        print(f"\nDry run: {total_deleted} directories would be deleted. Pass --apply to delete.")
