# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Remote Orchestration
#
# High-level function `run_remote()` that orchestrates running `llm_runner`
# operations on Isambard: setup, data transfer, SBATCH submission, polling,
# result download.

# %%
#|default_exp orchestrate

# %%
#|export
import json
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import arun as async_ssh_run, _get_config, _run_sync

# %%
#|export
class TransferMode(str, Enum):
    """How to transfer input data to/from Isambard."""
    DIRECT = "direct"           # Tar + SSH pipe, no remote persistence
    UPLOAD = "upload"            # rsync to content-hashed folder, idempotent
    COMPRESSED = "compressed"   # tar.gz + SSH pipe to content-hashed folder, idempotent

# %%
#|export
async def asetup_runner(*, config: IsambardConfig | None = None, print_fn=print) -> None:
    """Ensure llm_runner is deployed on Isambard (idempotent, targeted).

    Uploads a minimal remote_pyproject.toml (llm_runner deps only) and
    src/llm_runner/. Creates dirs, installs uv if needed, and runs uv sync.

    Call this once before multiple arun_remote(..., setup=False) calls.

    Args:
        config: Isambard configuration.
        print_fn: Print function for progress logging.
    """
    import subprocess
    config = _get_config(config)
    from isambard_utils.env import _aensure_uv, _aensure_venv
    from isambard_utils.transfer import aupload as async_rsync_upload, aupload_bytes
    import llm_runner

    try:
        # 1. Create remote dirs
        await async_ssh_run(
            f"mkdir -p {config.project_dir}/src {config.hf_cache_dir} {config.logs_dir}",
            config=config, retries=3, print_fn=print_fn,
        )

        # 2. Ensure uv
        await _aensure_uv(config=config)

        # 3. Upload remote_pyproject.toml as pyproject.toml
        runner_src = Path(llm_runner.__path__[0])
        remote_pyproject = runner_src / "assets" / "remote_pyproject.toml"
        await aupload_bytes(remote_pyproject.read_bytes(),
                            f"{config.project_dir}/pyproject.toml", config=config)

        # 4. Rsync llm_runner source
        print_fn("runner setup: syncing llm_runner source...")
        await async_rsync_upload(
            str(runner_src) + "/", f"{config.project_dir}/src/llm_runner",
            config=config, exclude=["__pycache__", "*.pyc"],
        )

        # 5. Install deps
        print_fn("runner setup: installing dependencies...")
        await _aensure_venv(config=config)
        print_fn("runner setup: done")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Runner setup failed (exit {e.returncode}).\n"
            f"--- command ---\n{e.cmd}\n"
            f"--- stderr ---\n{e.stderr or '(empty)'}"
        ) from e

# %%
#|export
def setup_runner(*, config: IsambardConfig | None = None, print_fn=print) -> None:
    """Ensure llm_runner is deployed on Isambard (sync wrapper).

    See asetup_runner() for details.
    """
    _run_sync(asetup_runner(config=config, print_fn=print_fn))

# %%
#|export
async def arun_remote(
    operation: str,
    inputs: dict[str, Any],
    config_dict: dict,
    *,
    setup: bool = True,
    transfer_modes: dict[str, TransferMode] | TransferMode = TransferMode.DIRECT,
    output_transfer: TransferMode = TransferMode.DIRECT,
    job_name: str,
    time: str = "02:00:00",
    required_models: list[str] | None = None,
    isambard_config: IsambardConfig | None = None,
    print_fn=print,
) -> dict[str, Any]:
    """Run a llm_runner operation remotely on Isambard via SBATCH (async).

    Args:
        operation: Operation name ("embed", "llm_generate", "cosine_topk").
        inputs: Named inputs for the operation.
        config_dict: Operation config dict (model_name, dtype, etc.).
        setup: Run runner setup before job (default True). Set to False after
            calling asetup_runner() to skip redundant setup on repeat calls.
        transfer_modes: Transfer mode per input key, or a single mode for all.
        output_transfer: Transfer mode for downloading outputs.
        job_name: Slurm job name.
        time: Slurm time limit (e.g. "02:00:00").
        required_models: HuggingFace model names to pre-cache.
        isambard_config: Isambard configuration.
        print_fn: Print function for progress logging.

    Returns:
        Dict of operation outputs (deserialized).
    """
    from isambard_utils.models import aensure_model
    from isambard_utils.transfer import (
        aupload_tar_pipe, adownload_tar_pipe,
        aupload_idempotent, aupload_compressed,
        aupload_bytes, compute_content_hash,
    )
    from isambard_utils.slurm import asubmit, await_job, ajob_log
    from isambard_utils.sbatch import generate as generate_sbatch, SbatchConfig
    from llm_runner.serialization import serialize, deserialize

    ic = _get_config(isambard_config)
    work_dir = f"{ic.project_dir}/.runner_jobs/{job_name}"
    cache_dir = f"{ic.project_dir}/.runner_cache"
    outputs_remote = f"{work_dir}/outputs"

    # Normalize transfer_modes to per-key dict
    if isinstance(transfer_modes, TransferMode):
        transfer_modes = {key: transfer_modes for key in inputs}

    # 1. Setup runner
    if setup:
        print_fn(f"run_remote [{job_name}]: setting up runner...")
        await asetup_runner(config=ic, print_fn=print_fn)

    # 2. Pre-cache models
    for model_name in (required_models or []):
        print_fn(f"run_remote [{job_name}]: ensuring model {model_name}...")
        await aensure_model(model_name, config=ic)

    # 3. Transfer inputs
    print_fn(f"run_remote [{job_name}]: transferring inputs...")
    await async_ssh_run(f"mkdir -p {work_dir} {outputs_remote}", config=ic)
    manifest = {}

    for key, value in inputs.items():
        mode = transfer_modes.get(key, TransferMode.DIRECT)
        with tempfile.TemporaryDirectory() as tmp:
            # Serialize this single input
            local_dir = Path(tmp) / key
            serialize({key: value}, local_dir)

            if mode == TransferMode.DIRECT:
                remote_dir = f"{work_dir}/inputs/{key}"
                await aupload_tar_pipe(str(local_dir), remote_dir, config=ic)
                manifest[key] = remote_dir
            elif mode == TransferMode.UPLOAD:
                content_hash = compute_content_hash(str(local_dir))
                remote_dir = await aupload_idempotent(
                    str(local_dir), cache_dir, content_hash, config=ic,
                )
                manifest[key] = remote_dir
            elif mode == TransferMode.COMPRESSED:
                content_hash = compute_content_hash(str(local_dir))
                remote_dir = await aupload_compressed(
                    str(local_dir), cache_dir, content_hash, config=ic,
                )
                manifest[key] = remote_dir

    # 4. Write manifest
    manifest_remote = f"{work_dir}/manifest.json"
    await aupload_bytes(json.dumps(manifest).encode(), manifest_remote, config=ic)

    # 5. Submit SBATCH
    config_json = json.dumps(config_dict)
    python_command = (
        f"cd {ic.project_dir} && source .venv/bin/activate && "
        f"PYTHONPATH={ic.project_dir}/src "
        f"python -m llm_runner {operation} "
        f"--manifest {manifest_remote} "
        f"--outputs-dir {outputs_remote} "
        f"--config '{config_json}'"
    )
    sbatch_cfg = SbatchConfig(
        job_name=job_name,
        time=time,
        python_command=python_command,
    )
    sbatch_script = generate_sbatch(sbatch_cfg, isambard_config=ic)
    print_fn(f"run_remote [{job_name}]: submitting job...")
    job = await asubmit(sbatch_script, config=ic)
    print_fn(f"run_remote [{job_name}]: submitted job {job.job_id}")

    # 6. Poll
    def _on_poll(status):
        if status:
            print_fn(f"run_remote [{job_name}]: job {job.job_id} state={status.get('state', '?')}")

    final = await await_job(job.job_id, config=ic, on_poll=_on_poll)
    final_state = final.get("state", "UNKNOWN")
    print_fn(f"run_remote [{job_name}]: job finished with state={final_state}")

    if final_state != "COMPLETED":
        stdout_log = await ajob_log(job.job_id, config=ic, stream="stdout", tail=50)
        stderr_log = await ajob_log(job.job_id, config=ic, stream="stderr", tail=50)
        raise RuntimeError(
            f"run_remote job {job_name} (id={job.job_id}) failed with state={final_state}.\n"
            f"--- stdout (last 50 lines) ---\n{stdout_log}\n"
            f"--- stderr (last 50 lines) ---\n{stderr_log}"
        )

    # 7. Check status.json
    check = await async_ssh_run(f"cat {outputs_remote}/status.json", config=ic, check=False)
    if check.returncode == 0:
        runner_status = json.loads(check.stdout)
        if runner_status.get("state") != "COMPLETED":
            raise RuntimeError(
                f"run_remote runner for {job_name} reported: {runner_status}"
            )

    # 8. Download outputs
    print_fn(f"run_remote [{job_name}]: downloading outputs...")
    with tempfile.TemporaryDirectory() as tmp:
        local_outputs = Path(tmp) / "outputs"
        local_outputs.mkdir()

        if output_transfer == TransferMode.DIRECT:
            await adownload_tar_pipe(outputs_remote, str(local_outputs), config=ic)
        else:
            # For idempotent modes, use rsync download
            from isambard_utils.transfer import adownload
            await adownload(outputs_remote + "/", str(local_outputs), config=ic)

        result = deserialize(local_outputs)

    # 9. Cleanup work dir (but not cache)
    await async_ssh_run(f"rm -rf {work_dir}", config=ic, check=False)
    print_fn(f"run_remote [{job_name}]: done")

    return result

# %%
#|export
def run_remote(
    operation: str,
    inputs: dict[str, Any],
    config_dict: dict,
    *,
    setup: bool = True,
    transfer_modes: dict[str, TransferMode] | TransferMode = TransferMode.DIRECT,
    output_transfer: TransferMode = TransferMode.DIRECT,
    job_name: str,
    time: str = "02:00:00",
    required_models: list[str] | None = None,
    isambard_config: IsambardConfig | None = None,
    print_fn=print,
) -> dict[str, Any]:
    """Run a llm_runner operation remotely on Isambard via SBATCH.

    See arun_remote() for full documentation.
    """
    return _run_sync(arun_remote(
        operation, inputs, config_dict,
        setup=setup,
        transfer_modes=transfer_modes,
        output_transfer=output_transfer,
        job_name=job_name, time=time,
        required_models=required_models,
        isambard_config=isambard_config,
        print_fn=print_fn,
    ))
