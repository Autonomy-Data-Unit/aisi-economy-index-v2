# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # File Transfer
#
# rsync-based upload/download over SSH for Isambard.

# %%
#|default_exp transfer

# %%
#|export
import subprocess
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import _get_config

# %%
#|export
def _build_rsync_cmd(config: IsambardConfig, src: str, dst: str, *,
                     exclude: list[str] | None = None,
                     delete: bool = False,
                     dry_run: bool = False) -> list[str]:
    """Build an rsync command list."""
    cmd = ["rsync", "-avz", "--progress", "-e", "ssh"]
    if delete:
        cmd.append("--delete")
    if dry_run:
        cmd.append("--dry-run")
    for pattern in (exclude or []):
        cmd.extend(["--exclude", pattern])
    cmd.extend([src, dst])
    return cmd

# %%
#|export
def _remote_path(config: IsambardConfig, path: str) -> str:
    """Format a remote path as user@host:path or host:path."""
    if config.ssh_user:
        return f"{config.ssh_user}@{config.ssh_host}:{path}"
    return f"{config.ssh_host}:{path}"

# %%
#|export
def upload(local_path: str, remote_path: str, *,
           config: IsambardConfig | None = None,
           exclude: list[str] | None = None,
           delete: bool = False,
           dry_run: bool = False) -> None:
    """Upload local files/dirs to Isambard via rsync.

    Args:
        local_path: Local file or directory path.
        remote_path: Destination path on Isambard.
        config: Isambard configuration.
        exclude: rsync exclude patterns.
        delete: Delete remote files not present locally.
        dry_run: Show what would be transferred without doing it.
    """
    config = _get_config(config)
    cmd = _build_rsync_cmd(
        config, local_path, _remote_path(config, remote_path),
        exclude=exclude, delete=delete, dry_run=dry_run,
    )
    subprocess.run(cmd, check=True)

# %%
#|export
def download(remote_path: str, local_path: str, *,
             config: IsambardConfig | None = None,
             exclude: list[str] | None = None) -> None:
    """Download files/dirs from Isambard via rsync.

    Args:
        remote_path: Source path on Isambard.
        local_path: Local destination path.
        config: Isambard configuration.
        exclude: rsync exclude patterns.
    """
    config = _get_config(config)
    cmd = _build_rsync_cmd(
        config, _remote_path(config, remote_path), local_path,
        exclude=exclude,
    )
    subprocess.run(cmd, check=True)

# %%
#|export
def upload_bytes(data: bytes, remote_path: str, *,
                 config: IsambardConfig | None = None) -> None:
    """Upload in-memory bytes to a remote file via SSH stdin pipe.

    Args:
        data: Bytes to write to the remote file.
        remote_path: Destination file path on Isambard.
        config: Isambard configuration.
    """
    config = _get_config(config)
    ssh_cmd = ["ssh"]
    if config.ssh_user:
        ssh_cmd.extend(["-l", config.ssh_user])
    ssh_cmd.extend([config.ssh_host, f"cat > {remote_path}"])
    subprocess.run(ssh_cmd, input=data, check=True)
