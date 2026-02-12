# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Execution Mode Utilities
#
# Abstraction layer for running GPU nodes in different execution modes:
# - **local** / **deploy** — direct CUDA on current machine
# - **sbatch** — orchestrate from local, submit SBATCH jobs to Isambard
# - **api** — CPU-only: embeddings on CPU, LLM via API

# %%
#|default_exp utils

# %% [markdown]
# ## Types

# %%
#|export
from typing import Literal

ExecutionMode = Literal["local", "deploy", "sbatch", "api"]

# %% [markdown]
# ## Serialization (for sbatch mode data transfer)
#
# Node inputs/outputs are dicts containing numpy arrays and Python lists.
# For rsync transfer between local and Isambard:
# - `numpy.ndarray` values -> individual `.npy` files
# - Everything else -> `_meta.pkl` (pickle)
# - One level of nesting supported

# %%
#|export
import pickle
from pathlib import Path

import numpy as np


def serialize_node_data(data: dict, directory: Path) -> None:
    """Serialize a dict of node data (numpy arrays + Python objects) to a directory.

    Args:
        data: Dict mapping names to values. Values can be numpy arrays, Python
            objects, or nested dicts containing a mix of both.
        directory: Target directory (created if needed).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    meta = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.save(directory / f"{key}.npy", value)
            meta[key] = {"__type": "ndarray"}
        elif isinstance(value, dict):
            # One level of nesting
            sub_dir = directory / key
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub_meta = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    np.save(sub_dir / f"{sub_key}.npy", sub_value)
                    sub_meta[sub_key] = {"__type": "ndarray"}
                else:
                    sub_meta[sub_key] = {"__type": "pickle", "value": sub_value}
            meta[key] = {"__type": "nested", "meta": sub_meta}
        else:
            meta[key] = {"__type": "pickle", "value": value}

    with open(directory / "_meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_node_data(directory: Path) -> dict:
    """Deserialize node data from a directory written by serialize_node_data.

    Args:
        directory: Directory containing serialized data.

    Returns:
        Dict mapping names to values, matching the original structure.
    """
    directory = Path(directory)

    with open(directory / "_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    result = {}
    for key, info in meta.items():
        if info["__type"] == "ndarray":
            result[key] = np.load(directory / f"{key}.npy")
        elif info["__type"] == "nested":
            sub_dir = directory / key
            sub_result = {}
            for sub_key, sub_info in info["meta"].items():
                if sub_info["__type"] == "ndarray":
                    sub_result[sub_key] = np.load(sub_dir / f"{sub_key}.npy")
                else:
                    sub_result[sub_key] = sub_info["value"]
            result[key] = sub_result
        else:
            result[key] = info["value"]

    return result

# %% [markdown]
# ## Runner script generation
#
# Generates a Python script that runs on the Isambard compute node.
# It deserializes inputs, imports and calls the node function (with
# `execution_mode` forced to `"local"` to prevent recursion), and
# serializes outputs.

# %%
#|export
def _generate_runner_script(func_path: str, inputs_dir: str, outputs_dir: str,
                            status_file: str, node_vars: dict) -> str:
    """Generate a Python runner script for an sbatch compute node.

    Args:
        func_path: Dotted import path to the node function (e.g.
            "ai_index.nodes.embed_onet.embed_onet").
        inputs_dir: Remote path to directory with serialized inputs.
        outputs_dir: Remote path to directory for serialized outputs.
        status_file: Remote path for the JSON status file.
        node_vars: Node variables dict (execution_mode will be forced to "local").
    """
    # Force local execution mode on compute node to prevent sbatch recursion
    safe_vars = dict(node_vars)
    safe_vars["execution_mode"] = "local"

    module_path = ".".join(func_path.split(".")[:-1])
    func_name = func_path.split(".")[-1]

    return f'''\
#!/usr/bin/env python3
"""Auto-generated runner script for sbatch execution."""
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

def main():
    status = {{"state": "RUNNING"}}
    try:
        # Deserialize inputs
        from ai_index.utils import deserialize_node_data, serialize_node_data
        inputs = deserialize_node_data(Path({inputs_dir!r}))

        # Build ctx with forced local execution_mode
        node_vars = {safe_vars!r}
        ctx = SimpleNamespace(vars=node_vars)

        # Import and call node function
        from {module_path} import {func_name}
        result = {func_name}(**inputs, ctx=ctx, print=print)

        # Serialize outputs
        outputs_path = Path({outputs_dir!r})
        serialize_node_data(result, outputs_path)

        status = {{"state": "COMPLETED"}}
    except Exception as e:
        status = {{"state": "FAILED", "error": str(e), "traceback": traceback.format_exc()}}
        print(f"RUNNER ERROR: {{e}}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        with open({status_file!r}, "w") as f:
            json.dump(status, f)

if __name__ == "__main__":
    main()
'''

# %% [markdown]
# ## sbatch orchestration
#
# Handles the full lifecycle: env setup, model pre-caching, input
# serialization, upload, sbatch submit, wait, download, deserialization.

# %%
#|export
import json
import tempfile


def _run_sbatch(func_path: str, inputs: dict, node_vars: dict,
                job_name: str, time: str,
                required_models: list[str],
                print_fn) -> dict:
    """Run a node function remotely via Isambard sbatch.

    Args:
        func_path: Dotted import path to the node function.
        inputs: Dict of input port values.
        node_vars: All node variables from ctx.vars.
        job_name: Slurm job name.
        time: Slurm time limit (e.g. "02:00:00").
        required_models: HuggingFace model names to pre-cache on Isambard.
        print_fn: Print function for logging.

    Returns:
        Dict of output port values (deserialized from remote).
    """
    from isambard_utils import (
        IsambardConfig, setup, ensure_model, upload, download,
        upload_bytes, submit, wait, job_log, generate_sbatch, SbatchConfig,
    )
    from isambard_utils.ssh import run as ssh_run

    config = IsambardConfig.from_env()
    work_dir = f"{config.project_dir}/.gpu_jobs/{job_name}"
    inputs_dir = f"{work_dir}/inputs"
    outputs_dir = f"{work_dir}/outputs"
    runner_file = f"{work_dir}/runner.py"
    status_file = f"{work_dir}/status.json"

    # 1. Bootstrap remote environment
    print_fn(f"sbatch [{job_name}]: setting up remote environment...")
    setup(config=config)

    # 2. Pre-cache required models
    for model_name in required_models:
        print_fn(f"sbatch [{job_name}]: ensuring model {model_name}...")
        ensure_model(model_name, config=config)

    # 3. Serialize inputs locally
    print_fn(f"sbatch [{job_name}]: serializing inputs...")
    with tempfile.TemporaryDirectory() as tmp:
        local_inputs = Path(tmp) / "inputs"
        serialize_node_data(inputs, local_inputs)

        # 4. Create remote work dir and upload inputs
        print_fn(f"sbatch [{job_name}]: uploading inputs...")
        ssh_run(f"mkdir -p {inputs_dir} {outputs_dir}", config=config)
        upload(str(local_inputs) + "/", inputs_dir, config=config)

    # 5. Generate and upload runner script
    runner_code = _generate_runner_script(
        func_path, inputs_dir, outputs_dir, status_file, dict(node_vars),
    )
    upload_bytes(runner_code.encode(), runner_file, config=config)

    # 6. Generate and submit sbatch script
    sbatch_cfg = SbatchConfig(
        job_name=job_name,
        time=time,
        python_script=runner_file,
    )
    sbatch_script = generate_sbatch(sbatch_cfg, isambard_config=config)
    print_fn(f"sbatch [{job_name}]: submitting job...")
    job = submit(sbatch_script, config=config)
    print_fn(f"sbatch [{job_name}]: submitted job {job.job_id}")

    # 7. Wait for completion
    def _on_poll(status):
        if status:
            print_fn(f"sbatch [{job_name}]: job {job.job_id} state={status.get('state', '?')}")

    final = wait(job.job_id, config=config, on_poll=_on_poll)
    final_state = final.get("state", "UNKNOWN")
    print_fn(f"sbatch [{job_name}]: job finished with state={final_state}")

    if final_state != "COMPLETED":
        # Fetch logs for debugging
        stdout_log = job_log(job.job_id, config=config, stream="stdout", tail=50)
        stderr_log = job_log(job.job_id, config=config, stream="stderr", tail=50)
        raise RuntimeError(
            f"sbatch job {job_name} (id={job.job_id}) failed with state={final_state}.\n"
            f"--- stdout (last 50 lines) ---\n{stdout_log}\n"
            f"--- stderr (last 50 lines) ---\n{stderr_log}"
        )

    # 8. Check runner status file
    check = ssh_run(f"cat {status_file}", config=config, check=False)
    if check.returncode == 0:
        runner_status = json.loads(check.stdout)
        if runner_status.get("state") != "COMPLETED":
            raise RuntimeError(
                f"sbatch runner for {job_name} reported: {runner_status}"
            )

    # 9. Download outputs
    print_fn(f"sbatch [{job_name}]: downloading outputs...")
    with tempfile.TemporaryDirectory() as tmp:
        local_outputs = Path(tmp) / "outputs"
        local_outputs.mkdir()
        download(outputs_dir + "/", str(local_outputs), config=config)
        result = deserialize_node_data(local_outputs)

    # 10. Cleanup remote work dir
    ssh_run(f"rm -rf {work_dir}", config=config, check=False)
    print_fn(f"sbatch [{job_name}]: done")

    return result

# %% [markdown]
# ## Guard function
#
# Called at the top of each GPU node. Returns `None` for local/deploy (let the
# body run), or the full result dict for sbatch mode.

# %%
#|export
def maybe_run_remote(func_path: str, inputs: dict, ctx, print_fn,
                     job_name: str, time: str = "02:00:00",
                     required_models_from_vars: list[str] | None = None,
                     ) -> dict | None:
    """Guard function for GPU nodes — handles sbatch orchestration.

    Call at the top of each GPU node. For **local** and **deploy** modes,
    returns ``None`` (the existing node body should run). For **sbatch** mode,
    handles the full remote execution lifecycle and returns the result dict.

    **api** mode is not handled here — each node implements its own API path.

    Args:
        func_path: Dotted import path to this node's function
            (e.g. "ai_index.nodes.embed_onet.embed_onet").
        inputs: Dict mapping input port names to values.
        ctx: Node execution context (must have ``ctx.vars``).
        print_fn: Print function for logging.
        job_name: Slurm job name.
        time: Slurm time limit (default "02:00:00").
        required_models_from_vars: List of var names whose values are HuggingFace
            model IDs to pre-cache on Isambard (e.g. ["embedding_model"]).

    Returns:
        ``None`` if the node body should execute locally, or a dict of output
        port values if remote execution was performed.
    """
    mode = ctx.vars.get("execution_mode", "local")

    if mode in ("local", "deploy", "api"):
        return None

    if mode == "sbatch":
        required_models = []
        for var_name in (required_models_from_vars or []):
            model_name = ctx.vars.get(var_name)
            if model_name:
                required_models.append(model_name)

        return _run_sbatch(
            func_path=func_path,
            inputs=inputs,
            node_vars=dict(ctx.vars),
            job_name=job_name,
            time=time,
            required_models=required_models,
            print_fn=print_fn,
        )

    raise ValueError(f"Unknown execution_mode: {mode!r}. Expected one of: local, deploy, sbatch, api")
