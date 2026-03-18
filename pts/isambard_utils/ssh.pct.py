# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SSH
#
# Thin wrapper around subprocess SSH for running commands on the Isambard login node.

# %%
#|default_exp ssh

# %%
#|export
import asyncio
import subprocess
import shlex
from functools import partial
from pathlib import Path
from isambard_utils.config import IsambardConfig

_fprint = partial(print, flush=True)

# %%
#|export
def _get_config(config: IsambardConfig | None) -> IsambardConfig:
    """Get config from argument or environment."""
    if config is not None:
        return config
    return IsambardConfig.from_env()

# %%
#|export
def _run_sync(coro):
    """Run an async coroutine synchronously.

    Works both inside and outside an existing event loop (uses nest_asyncio
    when an event loop is already running, e.g. inside Jupyter).
    """
    try:
        asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
    except RuntimeError:
        pass
    return asyncio.run(coro)

# %%
#|export
def _build_ssh_cmd(config: IsambardConfig, *, timeout: int = 120) -> list[str]:
    """Build the base SSH command list."""
    connect_timeout = min(timeout, 30)
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
           "-o", f"ConnectTimeout={connect_timeout}",
           "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]
           # "-o", "IdentitiesOnly=yes", "-o", "PreferredAuthentications=publickey"]
    if config.ssh_user:
        cmd.extend(["-l", config.ssh_user])
    cmd.append(config.ssh_host)
    return cmd

# %%
#|export
def _run_once_sync(ssh_cmd: list[str], *, timeout: int,
                   capture: bool) -> subprocess.CompletedProcess:
    """Single SSH attempt using synchronous subprocess (no asyncio child watcher needed).

    Uses subprocess.run instead of asyncio.create_subprocess_exec to avoid
    the child watcher issue in thread pool workers (netrun issue #32) and
    event loop blocking issues when sync nodes run on the main loop.
    """
    try:
        return subprocess.run(
            ssh_cmd,
            capture_output=capture,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        raise


async def _arun_once(ssh_cmd: list[str], *, timeout: int,
                     capture: bool) -> subprocess.CompletedProcess:
    """Single SSH attempt. Runs subprocess in a thread to avoid blocking the event loop."""
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _run_once_sync(ssh_cmd, timeout=timeout, capture=capture),
    )

import os as _os

_SSH_TRANSIENT_EXIT = 255
_SSH_RETRY_DELAYS = [2, 5, 10, 30, 60]
_SSH_DEFAULT_RETRIES = int(_os.environ.get("ISAMBARD_SSH_RETRIES", "10"))

async def arun(cmd: str, *, config: IsambardConfig | None = None, timeout: int = 120,
               check: bool = True, capture: bool = True,
               retries: int = 0, ssh_retries: int = _SSH_DEFAULT_RETRIES,
               print_fn=_fprint) -> subprocess.CompletedProcess:
    """Run a command on the Isambard login node via SSH (async).

    Args:
        cmd: Shell command to execute remotely.
        config: Isambard configuration. Defaults to IsambardConfig.from_env().
        timeout: Timeout in seconds.
        check: Raise CalledProcessError on non-zero exit.
        capture: Capture stdout/stderr (True) or inherit terminal (False).
        retries: Number of retries on timeout (default 0 = no retries).
        ssh_retries: Number of retries on transient SSH errors, exit code 255
            (default 3). Set to 0 to disable.
        print_fn: Print function for retry logging.
    """
    config = _get_config(config)
    ssh_cmd = _build_ssh_cmd(config, timeout=timeout) + [cmd]

    last_exc = None
    for attempt in range(1 + retries):
        try:
            # Inner loop: retry on transient SSH connection errors (exit 255)
            for ssh_attempt in range(1 + ssh_retries):
                result = await _arun_once(ssh_cmd, timeout=timeout, capture=capture)
                if result.returncode == _SSH_TRANSIENT_EXIT and ssh_attempt < ssh_retries:
                    delay = _SSH_RETRY_DELAYS[min(ssh_attempt, len(_SSH_RETRY_DELAYS) - 1)]
                    print_fn(f"SSH connection error (exit 255), retrying in {delay}s ({ssh_attempt + 1}/{ssh_retries})...")
                    await asyncio.sleep(delay)
                    continue
                break
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, ssh_cmd,
                    output=result.stdout, stderr=result.stderr,
                )
            return result
        except subprocess.TimeoutExpired as e:
            last_exc = e
            if attempt < retries:
                print_fn(f"SSH timed out ({timeout}s), retrying ({attempt + 1}/{retries})...")
    raise last_exc

# %%
#|export
def run(cmd: str, *, config: IsambardConfig | None = None, timeout: int = 120,
        check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command on the Isambard login node via SSH.

    Args:
        cmd: Shell command to execute remotely.
        config: Isambard configuration. Defaults to IsambardConfig.from_env().
        timeout: Timeout in seconds.
        check: Raise CalledProcessError on non-zero exit.
        capture: Capture stdout/stderr (True) or inherit terminal (False).
    """
    return _run_sync(arun(cmd, config=config, timeout=timeout, check=check, capture=capture))

# %%
#|export
async def acheck_connection(config: IsambardConfig | None = None, *,
                            max_retries: int = 10, print_fn=_fprint) -> bool:
    """Test SSH connectivity with progressive timeout retries (async).

    Starts with a short timeout and increases it on each retry, up to
    max_retries attempts. Returns True if connection succeeds.
    """
    timeouts = [30, 30, 30, 30, 60, 60, 60, 60, 60, 60]
    for attempt in range(max_retries):
        timeout = timeouts[attempt] if attempt < len(timeouts) else 60
        try:
            result = await arun("echo ok", config=config, timeout=timeout, check=False)
            if result.returncode == 0 and "ok" in result.stdout:
                return True
        except subprocess.TimeoutExpired:
            pass
        except (FileNotFoundError, OSError):
            return False
        if attempt < max_retries - 1:
            print_fn(f"SSH connection attempt {attempt + 1}/{max_retries} timed out ({timeout}s), retrying...")
    return False

# %%
#|export
def check_connection(config: IsambardConfig | None = None) -> bool:
    """Test SSH connectivity. Returns True if connection succeeds."""
    return _run_sync(acheck_connection(config=config))

# %%
#|export
async def acheck_clifton_auth() -> bool:
    """Check if Clifton certificate is still valid (async)."""
    cert_path = Path.home() / ".ssh" / "config_clifton"
    if not cert_path.exists():
        return False
    try:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-F", str(cert_path), "a5u.aip2.isambard", "echo ok"]
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=15),
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False

# %%
#|export
def check_clifton_auth() -> bool:
    """Check if Clifton certificate is still valid by looking for recent cert files."""
    return _run_sync(acheck_clifton_auth())
