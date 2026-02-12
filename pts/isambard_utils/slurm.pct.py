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
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import run as ssh_run, _get_config

# %%
#|export
@dataclass
class SlurmJob:
    """Represents a submitted Slurm job."""
    job_id: str
    array_task_ids: list[int] | None = None

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
    config = _get_config(config)
    scripts_dir = f"{config.project_dir}/.sbatch_scripts"
    ssh_run(f"mkdir -p {scripts_dir}", config=config)

    # Upload script to remote temp file
    import hashlib
    script_hash = hashlib.md5(sbatch_script.encode()).hexdigest()[:8]
    remote_script = f"{scripts_dir}/job_{script_hash}.sh"

    from isambard_utils.transfer import upload_bytes
    upload_bytes(sbatch_script.encode(), remote_script, config=config)

    # Submit via sbatch
    result = ssh_run(f"sbatch {remote_script}", config=config)
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
def status(job_id: str, *, config: IsambardConfig | None = None) -> dict:
    """Get job status via squeue.

    Returns a dict with job information, or empty dict if job is no longer in
    the queue (completed/failed).

    Args:
        job_id: Slurm job ID.
        config: Isambard configuration.
    """
    config = _get_config(config)
    result = ssh_run(
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
        return {
            "job_id": str(job.get("job_id", job_id)),
            "state": job.get("job_state", "UNKNOWN"),
            "name": job.get("name", ""),
            "node": job.get("nodes", ""),
            "time": job.get("time", ""),
            "partition": job.get("partition", ""),
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return {}

# %%
#|export
def _sacct_status(job_id: str, *, config: IsambardConfig | None = None) -> dict:
    """Get completed job status via sacct."""
    config = _get_config(config)
    result = ssh_run(
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
        return {
            "job_id": str(job.get("job_id", job_id)),
            "state": state,
            "exit_code": job.get("exit_code", {}).get("return_code", None),
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        return {"state": "UNKNOWN"}

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
    config = _get_config(config)
    start = time.time()

    while True:
        if timeout and (time.time() - start) > timeout:
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

        job_status = status(job_id, config=config)
        if on_poll:
            on_poll(job_status)

        if not job_status:
            # Job left the queue — check sacct for final status
            return _sacct_status(job_id, config=config)

        state = job_status.get("state", "")
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                      "OUT_OF_MEMORY", "PREEMPTED"):
            return _sacct_status(job_id, config=config)

        time.sleep(poll_interval)

# %%
#|export
def cancel(job_id: str, *, config: IsambardConfig | None = None) -> None:
    """Cancel a running or pending job via scancel.

    Args:
        job_id: Slurm job ID to cancel.
        config: Isambard configuration.
    """
    config = _get_config(config)
    ssh_run(f"scancel {job_id}", config=config)

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
    config = _get_config(config)
    ext = "out" if stream == "stdout" else "err"
    pattern = f"{config.logs_dir}/*_{job_id}.{ext}"

    cat_cmd = f"cat {pattern}"
    if tail:
        cat_cmd = f"tail -n {tail} {pattern}"

    result = ssh_run(cat_cmd, config=config, check=False)
    return result.stdout if result.returncode == 0 else ""
