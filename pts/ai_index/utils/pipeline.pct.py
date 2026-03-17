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


def is_run_complete(run_name: str, marker_path: str = "compute_job_ad_exposure/ad_exposure.parquet") -> bool:
    """Check if a pipeline run has completed by looking for a marker file.

    Args:
        run_name: The pipeline run name (directory under store/pipeline/).
        marker_path: Relative path within the run directory to check.
            Default checks for the final exposure output.
    """
    return (pipeline_store_path / run_name / marker_path).exists()


def get_sample_n(run_defs: dict, run_name: str) -> int:
    """Read sample_n from a run definition, falling back to defaults."""
    run_entry = run_defs["runs"][run_name]
    if "sample_n" in run_entry:
        return run_entry["sample_n"]
    return run_defs["defaults"]["sample_n"]
