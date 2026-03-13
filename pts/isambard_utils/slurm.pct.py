# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Slurm
#
# Submit, monitor, and wait for Slurm jobs on Isambard.

# %%
#|default_exp slurm

# %%
#|export
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import run as ssh_run, arun as async_ssh_run, _get_config, _run_sync

# %%
#|export
@dataclass
class SlurmJob:
    """Represents a submitted Slurm job."""
    job_id: str
    array_task_ids: list[int] | None = None

# %%
#|export
def _parse_array_spec(spec: str) -> list[int]:
    """Parse a Slurm array spec like '0-9' or '1,3,5' into a list of task IDs."""
    ids = []
    for part in spec.split(","):
        if "-" in part:
            parts = part.split("-")
            start, end = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) > 2 else 1
            ids.extend(range(start, end + 1, step))
        else:
            ids.append(int(part))
    return ids

# %%
#|export
async def asubmit(sbatch_script: str, *, config: IsambardConfig | None = None) -> SlurmJob:
    """Submit an SBATCH script and return the job (async).

    Uploads the script to a temp file on the remote, runs sbatch, and parses
    the job ID from stdout.

    Args:
        sbatch_script: Complete SBATCH script content as a string.
        config: Isambard configuration.
    """
    config = _get_config(config)
    scripts_dir = f"{config.project_dir}/.sbatch_scripts"
    await async_ssh_run(f"mkdir -p {scripts_dir}", config=config)

    # Upload script to remote temp file
    import hashlib
    script_hash = hashlib.md5(sbatch_script.encode()).hexdigest()[:8]
    remote_script = f"{scripts_dir}/job_{script_hash}.sh"

    from isambard_utils.transfer import aupload_bytes
    await aupload_bytes(sbatch_script.encode(), remote_script, config=config)

    # Submit via sbatch
    result = await async_ssh_run(f"sbatch {remote_script}", config=config)
    # Parse: "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Failed to parse sbatch output: {result.stdout}")

    job_id = match.group(1)

    # Check for array jobs
    array_ids = None
    if "--array" in sbatch_script:
        array_match = re.search(r"#SBATCH\s+--array=(\S+)", sbatch_script)
        if array_match:
            array_spec = array_match.group(1)
            array_ids = _parse_array_spec(array_spec)

    return SlurmJob(job_id=job_id, array_task_ids=array_ids)

# %%
#|export
def submit(sbatch_script: str, *, config: IsambardConfig | None = None) -> SlurmJob:
    """Submit an SBATCH script and return the job.

    Uploads the script to a temp file on the remote, runs sbatch, and parses
    the job ID from stdout.

    Args:
        sbatch_script: Complete SBATCH script content as a string.
        config: Isambard configuration.
    """
    return _run_sync(asubmit(sbatch_script, config=config))

# %%
#|export
async def astatus(job_id: str, *, config: IsambardConfig | None = None) -> dict:
    """Get job status via squeue (async).

    Returns a dict with job information, or empty dict if job is no longer in
    the queue (completed/failed).

    Args:
        job_id: Slurm job ID.
        config: Isambard configuration.
    """
    config = _get_config(config)
    result = await async_ssh_run(
        f"squeue --json -j {job_id}",
        config=config, check=False,
    )
    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)
        jobs = data.get("jobs", [])
        if not jobs:
            return {}
        job = jobs[0]
        state = job.get("job_state", "UNKNOWN")
        # squeue --json may return state as a list (e.g. ["RUNNING"])
        if isinstance(state, list):
            state = state[0] if state else "UNKNOWN"
        return {
            "job_id": str(job.get("job_id", job_id)),
            "state": state,
            "name": job.get("name", ""),
            "node": job.get("nodes", ""),
            "time": job.get("time", ""),
            "partition": job.get("partition", ""),
        }
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        import warnings
        warnings.warn(f"astatus: failed to parse squeue JSON for job {job_id}: {e}")
        return {}

# %%
#|export
def status(job_id: str, *, config: IsambardConfig | None = None) -> dict:
    """Get job status via squeue.

    Returns a dict with job information, or empty dict if job is no longer in
    the queue (completed/failed).

    Args:
        job_id: Slurm job ID.
        config: Isambard configuration.
    """
    return _run_sync(astatus(job_id, config=config))

# %%
#|export
async def _asacct_status(job_id: str, *, config: IsambardConfig | None = None) -> dict:
    """Get completed job status and resource accounting via sacct (async).

    Returns a dict with at minimum {"state": str}. When sacct provides
    accounting data, also includes timing and resource fields:
    - elapsed_seconds: int — wall-clock job duration (excludes queue wait)
    - start_time: int — Unix epoch when the job started running
    - end_time: int — Unix epoch when the job finished
    - allocated_cpus: int — number of CPUs allocated
    - allocated_gpus: int — number of GPUs allocated
    - node_hours: float — Isambard billing node-hours (0.25 NHR per GPU-hour)
    """
    config = _get_config(config)
    result = await async_ssh_run(
        f"sacct -j {job_id} --json",
        config=config, check=False,
    )
    if result.returncode != 0:
        return {"state": "UNKNOWN"}
    try:
        data = json.loads(result.stdout)
        jobs = data.get("jobs", [])
        if not jobs:
            return {"state": "UNKNOWN"}
        job = jobs[0]
        state = job.get("state", {})
        if isinstance(state, dict):
            state = state.get("current", ["UNKNOWN"])[0]

        result_dict = {
            "job_id": str(job.get("job_id", job_id)),
            "state": state,
            "exit_code": job.get("exit_code", {}).get("return_code", None),
        }

        # Extract timing from sacct time object
        time_info = job.get("time", {})
        if isinstance(time_info, dict):
            elapsed = time_info.get("elapsed")
            if isinstance(elapsed, (int, float)) and elapsed >= 0:
                result_dict["elapsed_seconds"] = int(elapsed)
            start = time_info.get("start")
            if isinstance(start, int) and start > 0:
                result_dict["start_time"] = start
            end = time_info.get("end")
            if isinstance(end, int) and end > 0:
                result_dict["end_time"] = end

        # Extract allocated resources from TRES
        tres = job.get("tres", {})
        allocated = tres.get("allocated", [])
        if isinstance(allocated, list):
            for item in allocated:
                if not isinstance(item, dict):
                    continue
                tres_type = item.get("type", "")
                tres_name = item.get("name", "")
                count = item.get("count", 0)
                if tres_type == "cpu":
                    result_dict["allocated_cpus"] = count
                elif tres_type == "gres" and tres_name == "gpu":
                    result_dict["allocated_gpus"] = count

        # Compute Isambard node-hours: 1 GPU = 0.25 NHR per wall-hour
        if "elapsed_seconds" in result_dict and "allocated_gpus" in result_dict:
            gpu_frac = result_dict["allocated_gpus"] / 4  # 4 GPUs per node
            result_dict["node_hours"] = gpu_frac * result_dict["elapsed_seconds"] / 3600

        return result_dict
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        import warnings
        warnings.warn(f"_asacct_status: failed to parse sacct JSON for job {job_id}: {e}")
        return {"state": "UNKNOWN"}

# %%
#|export
def _sacct_status(job_id: str, *, config: IsambardConfig | None = None) -> dict:
    """Get completed job status via sacct."""
    return _run_sync(_asacct_status(job_id, config=config))

# %%
#|export
async def await_job(job_id: str, *, config: IsambardConfig | None = None,
                    poll_interval: int = 15, timeout: int | None = None,
                    on_poll: Callable | None = None) -> dict:
    """Poll until a job completes, then return sacct summary (async).

    Args:
        job_id: Slurm job ID.
        config: Isambard configuration.
        poll_interval: Seconds between status checks.
        timeout: Maximum seconds to wait (None = no limit).
        on_poll: Optional callback receiving status dict each iteration.
    """
    config = _get_config(config)
    start = time.time()

    while True:
        if timeout and (time.time() - start) > timeout:
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

        job_status = await astatus(job_id, config=config)
        if on_poll:
            on_poll(job_status)

        if not job_status:
            # Job left the queue -- check sacct for final status
            return await _asacct_status(job_id, config=config)

        state = job_status.get("state", "")
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                      "OUT_OF_MEMORY", "PREEMPTED"):
            return await _asacct_status(job_id, config=config)

        await asyncio.sleep(poll_interval)

# %%
#|export
def wait(job_id: str, *, config: IsambardConfig | None = None,
         poll_interval: int = 15, timeout: int | None = None,
         on_poll: Callable | None = None) -> dict:
    """Poll until a job completes, then return sacct summary.

    Args:
        job_id: Slurm job ID.
        config: Isambard configuration.
        poll_interval: Seconds between status checks.
        timeout: Maximum seconds to wait (None = no limit).
        on_poll: Optional callback receiving status dict each iteration.
    """
    return _run_sync(await_job(job_id, config=config, poll_interval=poll_interval,
                               timeout=timeout, on_poll=on_poll))

# %%
#|export
async def ajob_state(job_id: str, *, config: IsambardConfig | None = None) -> str:
    """Get the effective state of a Slurm job via squeue then sacct (async).

    Checks squeue first (for running/pending jobs). If the job is no longer
    in the queue, falls back to sacct for the final state.

    Returns one of: "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED",
    "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", or "UNKNOWN".
    """
    config = _get_config(config)
    sq = await astatus(job_id, config=config)
    if sq:
        return sq.get("state", "UNKNOWN")
    sa = await _asacct_status(job_id, config=config)
    return sa.get("state", "UNKNOWN")

# %%
#|export
def job_state(job_id: str, *, config: IsambardConfig | None = None) -> str:
    """Get the effective state of a Slurm job via squeue then sacct.

    See ajob_state() for details.
    """
    return _run_sync(ajob_state(job_id, config=config))

# %%
#|export
async def acancel(job_id: str, *, config: IsambardConfig | None = None) -> None:
    """Cancel a running or pending job via scancel (async).

    Args:
        job_id: Slurm job ID to cancel.
        config: Isambard configuration.
    """
    config = _get_config(config)
    await async_ssh_run(f"scancel {job_id}", config=config)

# %%
#|export
def cancel(job_id: str, *, config: IsambardConfig | None = None) -> None:
    """Cancel a running or pending job via scancel.

    Args:
        job_id: Slurm job ID to cancel.
        config: Isambard configuration.
    """
    _run_sync(acancel(job_id, config=config))

# %%
#|export
async def ajob_log(job_id: str, *, config: IsambardConfig | None = None,
                   stream: str = "stdout", tail: int | None = None) -> str:
    """Read job stdout or stderr log file from Isambard (async).

    Args:
        job_id: Slurm job ID.
        config: Isambard configuration.
        stream: 'stdout' or 'stderr'.
        tail: If set, return only the last N lines.
    """
    config = _get_config(config)
    ext = "out" if stream == "stdout" else "err"
    pattern = f"{config.logs_dir}/*_{job_id}.{ext}"

    cat_cmd = f"cat {pattern}"
    if tail:
        cat_cmd = f"tail -n {tail} {pattern}"

    result = await async_ssh_run(cat_cmd, config=config, check=False)
    return result.stdout if result.returncode == 0 else ""

# %%
#|export
def job_log(job_id: str, *, config: IsambardConfig | None = None,
            stream: str = "stdout", tail: int | None = None) -> str:
    """Read job stdout or stderr log file from Isambard.

    Args:
        job_id: Slurm job ID.
        config: Isambard configuration.
        stream: 'stdout' or 'stderr'.
        tail: If set, return only the last N lines.
    """
    return _run_sync(ajob_log(job_id, config=config, stream=stream, tail=tail))
