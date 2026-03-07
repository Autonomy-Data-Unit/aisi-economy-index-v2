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
import asyncio
import subprocess
from pathlib import Path
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import _get_config, _run_sync

# %%
#|exporti
import os as _os

_SSH_TRANSIENT_EXIT = 255  # SSH connection error (reset, refused, timeout, etc.)
_DEFAULT_RETRIES = int(_os.environ.get("ISAMBARD_SSH_RETRIES", "10"))
_RETRY_DELAYS = [2, 5, 10, 30, 60]  # seconds between retries; last element repeats

async def _retry_on_ssh_error(coro_fn, *, retries: int = _DEFAULT_RETRIES):
    """Retry an async callable on transient SSH errors (exit code 255).

    Args:
        coro_fn: Zero-argument callable returning an awaitable.
        retries: Max number of retries (default 3).
    """
    last_exc = None
    for attempt in range(1 + retries):
        try:
            return await coro_fn()
        except subprocess.CalledProcessError as e:
            if e.returncode != _SSH_TRANSIENT_EXIT or attempt >= retries:
                raise
            last_exc = e
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            await asyncio.sleep(delay)
    raise last_exc

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
async def aupload(local_path: str, remote_path: str, *,
                  config: IsambardConfig | None = None,
                  exclude: list[str] | None = None,
                  delete: bool = False,
                  dry_run: bool = False) -> None:
    """Upload local files/dirs to Isambard via rsync (async).

    Retries on transient SSH errors (exit code 255).

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
    async def _attempt():
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    await _retry_on_ssh_error(_attempt)

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
    _run_sync(aupload(local_path, remote_path, config=config,
                       exclude=exclude, delete=delete, dry_run=dry_run))

# %%
#|export
async def adownload(remote_path: str, local_path: str, *,
                    config: IsambardConfig | None = None,
                    exclude: list[str] | None = None) -> None:
    """Download files/dirs from Isambard via rsync (async).

    Retries on transient SSH errors (exit code 255).

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
    async def _attempt():
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.communicate()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    await _retry_on_ssh_error(_attempt)

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
    _run_sync(adownload(remote_path, local_path, config=config, exclude=exclude))

# %%
#|export
async def aupload_bytes(data: bytes, remote_path: str, *,
                        config: IsambardConfig | None = None) -> None:
    """Upload in-memory bytes to a remote file via SSH stdin pipe (async).

    Retries on transient SSH errors (exit code 255).

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
    async def _attempt():
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd, stdin=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=data)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, ssh_cmd)
    await _retry_on_ssh_error(_attempt)

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
    _run_sync(aupload_bytes(data, remote_path, config=config))

# %% [markdown]
# ## Tar pipe transfers
#
# Fast streaming transfers via SSH pipe. No intermediate files on remote.

# %%
#|export
async def aupload_tar_pipe(local_dir: str, remote_dir: str, *,
                            config: IsambardConfig | None = None) -> None:
    """Upload a local directory to remote via tar + SSH pipe (async).

    Streams the contents without persistent intermediate files.
    Retries on transient SSH errors (exit code 255).

    Args:
        local_dir: Local directory to upload.
        remote_dir: Remote destination directory.
        config: Isambard configuration.
    """
    import os
    config = _get_config(config)
    remote = f"{config.ssh_host}" if not config.ssh_user else f"{config.ssh_user}@{config.ssh_host}"
    tar_cmd = ["tar", "cf", "-", "-C", local_dir, "."]
    ssh_cmd = ["ssh", remote, f"mkdir -p {remote_dir} && tar xf - -C {remote_dir}"]

    async def _attempt():
        read_fd, write_fd = os.pipe()
        try:
            tar_proc = await asyncio.create_subprocess_exec(
                *tar_cmd, stdout=write_fd,
            )
            os.close(write_fd); write_fd = -1
            ssh_proc = await asyncio.create_subprocess_exec(
                *ssh_cmd, stdin=read_fd,
            )
            os.close(read_fd); read_fd = -1
        except:
            if write_fd >= 0: os.close(write_fd)
            if read_fd >= 0: os.close(read_fd)
            raise

        await asyncio.gather(tar_proc.wait(), ssh_proc.wait())

        if ssh_proc.returncode != 0:
            raise subprocess.CalledProcessError(ssh_proc.returncode, ssh_cmd)

    await _retry_on_ssh_error(_attempt)

# %%
#|export
def upload_tar_pipe(local_dir: str, remote_dir: str, *,
                    config: IsambardConfig | None = None) -> None:
    """Upload a local directory to remote via tar + SSH pipe.

    Args:
        local_dir: Local directory to upload.
        remote_dir: Remote destination directory.
        config: Isambard configuration.
    """
    _run_sync(aupload_tar_pipe(local_dir, remote_dir, config=config))

# %%
#|export
async def adownload_tar_pipe(remote_dir: str, local_dir: str, *,
                              config: IsambardConfig | None = None) -> None:
    """Download a remote directory to local via tar + SSH pipe (async).

    Retries on transient SSH errors (exit code 255).

    Args:
        remote_dir: Remote source directory.
        local_dir: Local destination directory.
        config: Isambard configuration.
    """
    import os
    config = _get_config(config)
    os.makedirs(local_dir, exist_ok=True)
    remote = f"{config.ssh_host}" if not config.ssh_user else f"{config.ssh_user}@{config.ssh_host}"
    ssh_cmd = ["ssh", remote, f"tar cf - -C {remote_dir} ."]
    tar_cmd = ["tar", "xf", "-", "-C", local_dir]

    async def _attempt():
        read_fd, write_fd = os.pipe()
        try:
            ssh_proc = await asyncio.create_subprocess_exec(
                *ssh_cmd, stdout=write_fd,
            )
            os.close(write_fd); write_fd = -1
            tar_proc = await asyncio.create_subprocess_exec(
                *tar_cmd, stdin=read_fd,
            )
            os.close(read_fd); read_fd = -1
        except:
            if write_fd >= 0: os.close(write_fd)
            if read_fd >= 0: os.close(read_fd)
            raise

        await asyncio.gather(ssh_proc.wait(), tar_proc.wait())

        if ssh_proc.returncode != 0:
            raise subprocess.CalledProcessError(ssh_proc.returncode, ssh_cmd)
        if tar_proc.returncode != 0:
            raise subprocess.CalledProcessError(tar_proc.returncode, tar_cmd)

    await _retry_on_ssh_error(_attempt)

# %%
#|export
def download_tar_pipe(remote_dir: str, local_dir: str, *,
                      config: IsambardConfig | None = None) -> None:
    """Download a remote directory to local via tar + SSH pipe.

    Args:
        remote_dir: Remote source directory.
        local_dir: Local destination directory.
        config: Isambard configuration.
    """
    _run_sync(adownload_tar_pipe(remote_dir, local_dir, config=config))

# %% [markdown]
# ## Idempotent uploads (content-hashed)
#
# Upload to a content-hashed directory on remote. Skips if `.complete` marker exists.

# %%
#|export
import hashlib

def compute_content_hash(directory: str | Path) -> str:
    """Compute SHA256 hash over sorted (filename, file_bytes) pairs.

    Deterministic: same directory contents always produce the same hash.

    Args:
        directory: Local directory to hash.

    Returns:
        Hex string of the SHA256 digest.
    """
    directory = Path(directory)
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(directory))
            h.update(rel.encode())
            h.update(path.read_bytes())
    return h.hexdigest()

# %%
#|export
async def aupload_idempotent(local_dir: str, remote_base: str, content_hash: str, *,
                              config: IsambardConfig | None = None) -> str:
    """Upload to a content-hashed directory via rsync (idempotent, async).

    Skips upload if the remote directory already has a `.complete` marker.

    Args:
        local_dir: Local directory to upload.
        remote_base: Remote base directory (hash dir created under this).
        content_hash: Content hash string for the directory name.
        config: Isambard configuration.

    Returns:
        Remote path of the content-hashed directory.
    """
    from isambard_utils.ssh import arun as async_ssh_run
    config = _get_config(config)
    remote_path = f"{remote_base}/{content_hash}"

    # Check if already uploaded
    check = await async_ssh_run(
        f"test -f {remote_path}/.complete", config=config, check=False,
    )
    if check.returncode == 0:
        return remote_path

    # Upload via rsync
    await async_ssh_run(f"mkdir -p {remote_path}", config=config)
    local = local_dir.rstrip("/") + "/"
    await aupload(local, remote_path, config=config)

    # Mark complete
    await async_ssh_run(f"touch {remote_path}/.complete", config=config)
    return remote_path

# %%
#|export
def upload_idempotent(local_dir: str, remote_base: str, content_hash: str, *,
                      config: IsambardConfig | None = None) -> str:
    """Upload to a content-hashed directory via rsync (idempotent).

    Args:
        local_dir: Local directory to upload.
        remote_base: Remote base directory.
        content_hash: Content hash string.
        config: Isambard configuration.

    Returns:
        Remote path of the content-hashed directory.
    """
    return _run_sync(aupload_idempotent(local_dir, remote_base, content_hash, config=config))

# %%
#|export
async def aupload_compressed(local_dir: str, remote_base: str, content_hash: str, *,
                              config: IsambardConfig | None = None) -> str:
    """Upload via compressed tar + SSH pipe to a content-hashed directory (async).

    Idempotent — skips if `.complete` marker exists on remote.
    Retries on transient SSH errors (exit code 255).

    Args:
        local_dir: Local directory to upload.
        remote_base: Remote base directory.
        content_hash: Content hash string.
        config: Isambard configuration.

    Returns:
        Remote path of the content-hashed directory.
    """
    import tempfile, os
    from isambard_utils.ssh import arun as async_ssh_run
    config = _get_config(config)
    remote_path = f"{remote_base}/{content_hash}"

    # Check if already uploaded
    check = await async_ssh_run(
        f"test -f {remote_path}/.complete", config=config, check=False,
    )
    if check.returncode == 0:
        return remote_path

    # Create tar.gz locally
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tar_proc = await asyncio.create_subprocess_exec(
            "tar", "czf", tmp_path, "-C", local_dir, ".",
        )
        await tar_proc.communicate()
        if tar_proc.returncode != 0:
            raise subprocess.CalledProcessError(tar_proc.returncode, ["tar", "czf"])

        # Stream tar.gz to remote and extract
        remote = f"{config.ssh_host}" if not config.ssh_user else f"{config.ssh_user}@{config.ssh_host}"
        extract_cmd = (
            f"mkdir -p {remote_path} && "
            f"cat > /tmp/_upload_{content_hash[:16]}.tar.gz && "
            f"tar xzf /tmp/_upload_{content_hash[:16]}.tar.gz -C {remote_path} && "
            f"touch {remote_path}/.complete && "
            f"rm /tmp/_upload_{content_hash[:16]}.tar.gz"
        )
        ssh_cmd = ["ssh", remote, extract_cmd]

        async def _attempt():
            with open(tmp_path, "rb") as tar_file:
                ssh_proc = await asyncio.create_subprocess_exec(
                    *ssh_cmd, stdin=tar_file,
                )
                await ssh_proc.communicate()
            if ssh_proc.returncode != 0:
                raise subprocess.CalledProcessError(ssh_proc.returncode, ssh_cmd)

        await _retry_on_ssh_error(_attempt)
    finally:
        os.unlink(tmp_path)

    return remote_path

# %%
#|export
async def _aupload_compressed_direct(local_dir: str, remote_dir: str, *,
                                      config: IsambardConfig | None = None) -> None:
    """Upload via compressed tar + SSH pipe to a specific remote dir (async).

    Unlike aupload_compressed, this does NOT use content hashing or .complete
    markers. It simply tar.gz's the local directory and extracts it at the
    given remote path. Retries on transient SSH errors (exit code 255).

    Args:
        local_dir: Local directory to upload.
        remote_dir: Remote destination directory.
        config: Isambard configuration.
    """
    import tempfile, os
    from isambard_utils.ssh import arun as async_ssh_run
    config = _get_config(config)

    await async_ssh_run(f"mkdir -p {remote_dir}", config=config)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tar_proc = await asyncio.create_subprocess_exec(
            "tar", "czf", tmp_path, "-C", local_dir, ".",
        )
        await tar_proc.communicate()
        if tar_proc.returncode != 0:
            raise subprocess.CalledProcessError(tar_proc.returncode, ["tar", "czf"])

        remote = f"{config.ssh_host}" if not config.ssh_user else f"{config.ssh_user}@{config.ssh_host}"
        extract_cmd = f"tar xzf - -C {remote_dir}"
        ssh_cmd = ["ssh", remote, extract_cmd]

        async def _attempt():
            with open(tmp_path, "rb") as tar_file:
                ssh_proc = await asyncio.create_subprocess_exec(
                    *ssh_cmd, stdin=tar_file,
                )
                await ssh_proc.communicate()
            if ssh_proc.returncode != 0:
                raise subprocess.CalledProcessError(ssh_proc.returncode, ssh_cmd)

        await _retry_on_ssh_error(_attempt)
    finally:
        os.unlink(tmp_path)

# %%
#|export
def upload_compressed(local_dir: str, remote_base: str, content_hash: str, *,
                      config: IsambardConfig | None = None) -> str:
    """Upload via compressed tar + SSH pipe to a content-hashed directory.

    Args:
        local_dir: Local directory to upload.
        remote_base: Remote base directory.
        content_hash: Content hash string.
        config: Isambard configuration.

    Returns:
        Remote path of the content-hashed directory.
    """
    return _run_sync(aupload_compressed(local_dir, remote_base, content_hash, config=config))
