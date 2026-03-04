# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run Pipeline
#
# CLI entry point for running the AISI Economy Index pipeline end-to-end.
# Loads run definitions from `run_defs.toml` and overrides node_vars accordingly.

# %%
#|default_exp run_pipeline

# %%
#|export
import asyncio
import os
import tomllib
from importlib import resources
from pathlib import Path

from netrun.core import Net, NetConfig

# %%
#|export
def _load_run_defs(run_defs_path: Path) -> dict:
    """Load run definitions from TOML file."""
    with open(run_defs_path, "rb") as f:
        return tomllib.load(f)

# %%
#|export
def _resolve_run_defs(run_defs: dict, run_name: str) -> tuple[dict, dict]:
    """Resolve run_defs into (global_node_vars, per_node_vars) dicts.

    Merges [defaults] with [runs.<run_name>]. Subtables are per-node overrides,
    scalar values are global vars. Returns dicts compatible with
    NetConfig.from_file(global_node_vars=..., node_vars=...).
    """
    defaults = dict(run_defs.get("defaults", {}))
    runs = run_defs.get("runs", {})

    if run_name not in runs:
        available = ", ".join(sorted(runs.keys()))
        raise ValueError(f"Unknown run name {run_name!r}. Available: {available}")

    run_overrides = dict(runs[run_name])

    # Split defaults into globals vs per-node
    default_globals = {}
    default_node = {}
    for k, v in defaults.items():
        if isinstance(v, dict):
            default_node[k] = v
        else:
            default_globals[k] = v

    # Split run overrides into globals vs per-node
    run_globals = {}
    run_node = {}
    for k, v in run_overrides.items():
        if isinstance(v, dict):
            run_node[k] = v
        else:
            run_globals[k] = v

    # Merge: defaults <- run overrides
    merged_globals = {**default_globals, **run_globals}
    merged_globals["run_name"] = run_name

    # Convert values to strings for NodeVariable compatibility
    global_node_vars = {k: str(v) for k, v in merged_globals.items()}

    # Merge per-node: defaults <- run overrides
    all_node_names = set(default_node) | set(run_node)
    per_node_vars = {}
    for node_name in all_node_names:
        merged = {**default_node.get(node_name, {}), **run_node.get(node_name, {})}
        per_node_vars[node_name] = {k: str(v) for k, v in merged.items()}

    return global_node_vars, per_node_vars

# %%
#|export
async def run_pipeline_async(run_name: str | None = None):
    """Load and run the full pipeline, returning output queue results."""
    from ai_index.const import run_defs_path

    config_path = resources.files("ai_index.assets") / "netrun.json"

    # Load and resolve run definitions
    run_name = run_name or os.environ.get("RUN_NAME", "baseline")
    run_defs = _load_run_defs(run_defs_path)
    global_vars, node_vars = _resolve_run_defs(run_defs, run_name)

    config = NetConfig.from_file(
        str(config_path),
        global_node_vars=global_vars,
        node_vars=node_vars,
    )
    config.project_root_override = str(Path.cwd())
    print(f"run_pipeline: using run definition {run_name!r}")

    async with Net(config) as net:
        made_progress = True
        while made_progress:
            made_progress, _ = await net.run_until_blocked()

        results = net.flush_all_output_queues()
        for queue_name, outputs in results.items():
            print(f"\n=== Output queue: {queue_name} ({len(outputs)} packet(s)) ===")
            for i, output in enumerate(outputs):
                print(f"  [{i}] {output}")

    return results

# %%
#|export
def main():
    """Sync entry point for the run-pipeline CLI command."""
    asyncio.run(run_pipeline_async())
