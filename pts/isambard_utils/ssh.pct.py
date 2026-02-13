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
from pathlib import Path
from isambard_utils.config import IsambardConfig

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
def _build_ssh_cmd(config: IsambardConfig) -> list[str]:
    """Build the base SSH command list."""
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if config.ssh_user:
        cmd.extend(["-l", config.ssh_user])
    cmd.append(config.ssh_host)
    return cmd

# %%
#|export
async def arun(cmd: str, *, config: IsambardConfig | None = None, timeout: int = 120,
               check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command on the Isambard login node via SSH (async).

    Args:
        cmd: Shell command to execute remotely.
        config: Isambard configuration. Defaults to IsambardConfig.from_env().
        timeout: Timeout in seconds.
        check: Raise CalledProcessError on non-zero exit.
        capture: Capture stdout/stderr (True) or inherit terminal (False).
    """
    config = _get_config(config)
    ssh_cmd = _build_ssh_cmd(config) + [cmd]

    if capture:
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_exec(*ssh_cmd)

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise subprocess.TimeoutExpired(ssh_cmd, timeout)

    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    result = subprocess.CompletedProcess(
        args=ssh_cmd, returncode=proc.returncode,
        stdout=stdout if capture else None,
        stderr=stderr if capture else None,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, ssh_cmd,
            output=result.stdout, stderr=result.stderr,
        )
    return result

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
async def acheck_connection(config: IsambardConfig | None = None) -> bool:
    """Test SSH connectivity (async). Returns True if connection succeeds."""
    try:
        result = await arun("echo ok", config=config, timeout=30, check=False)
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
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
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=15)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return False
        return proc.returncode == 0
    except (FileNotFoundError, OSError):
        return False

# %%
#|export
def check_clifton_auth() -> bool:
    """Check if Clifton certificate is still valid by looking for recent cert files."""
    return _run_sync(acheck_clifton_auth())
