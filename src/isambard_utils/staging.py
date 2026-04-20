"""Stage data files on Isambard for efficient chunk-based processing.

Instead of uploading 500MB of serialized data per sbatch job, stage the raw
data files (parquet, JSON) once and let each job read its chunk directly
from the Lustre filesystem.

Files are stored under {project_dir}/.staged_data/{content_hash}/ with a
.complete marker for idempotent uploads.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import IsambardConfig
from .ssh import _get_config


@dataclass(frozen=True)
class StagedRef:
    """Reference to a file that has been staged on Isambard."""
    remote_path: str
    content_hash: str


def compute_file_hash(local_path: Path) -> str:
    """Compute SHA256 hash of a single file's contents."""
    h = hashlib.sha256()
    with open(local_path, "rb") as f:
        while chunk := f.read(1 << 20):  # 1MB chunks
            h.update(chunk)
    return h.hexdigest()


async def astage_file(
    local_path: Path,
    *,
    config: IsambardConfig | None = None,
    print_fn=print,
) -> StagedRef:
    """Upload a single file to Isambard's staged data area (idempotent).

    The file is stored at {project_dir}/.staged_data/{content_hash}/{filename}.
    A .complete marker makes the upload idempotent: if the marker exists,
    the upload is skipped.

    Args:
        local_path: Local file to stage.
        config: Isambard configuration.
        print_fn: Print function for progress logging.

    Returns:
        StagedRef with the remote path and content hash.
    """
    from .ssh import arun as async_ssh_run
    from .transfer import aupload

    config = _get_config(config)
    local_path = Path(local_path)
    content_hash = compute_file_hash(local_path)

    staged_base = str(PurePosixPath(config.project_dir) / ".staged_data")
    remote_dir = f"{staged_base}/{content_hash}"
    remote_file = f"{remote_dir}/{local_path.name}"
    complete_marker = f"{remote_dir}/.complete"

    # Check if already staged
    check = await async_ssh_run(
        f"test -f {complete_marker}", config=config, check=False,
    )
    if check.returncode == 0:
        print_fn(f"staging: {local_path.name} already staged ({content_hash[:12]}...)")
        return StagedRef(remote_path=remote_file, content_hash=content_hash)

    # Upload the file
    print_fn(f"staging: uploading {local_path.name} ({content_hash[:12]}...)")
    await async_ssh_run(f"mkdir -p {remote_dir}", config=config)
    await aupload(str(local_path), remote_file, config=config)

    # Mark complete
    await async_ssh_run(f"touch {complete_marker}", config=config)
    print_fn(f"staging: {local_path.name} staged successfully")

    return StagedRef(remote_path=remote_file, content_hash=content_hash)


async def astage_files(
    files: dict[str, Path],
    *,
    config: IsambardConfig | None = None,
    print_fn=print,
) -> dict[str, StagedRef]:
    """Stage multiple files to Isambard (each independently content-addressed).

    Args:
        files: Mapping of logical names to local file paths.
        config: Isambard configuration.
        print_fn: Print function for progress logging.

    Returns:
        Dict mapping the same logical names to StagedRef instances.
    """
    config = _get_config(config)
    refs = {}
    for name, local_path in files.items():
        refs[name] = await astage_file(
            local_path, config=config, print_fn=print_fn,
        )
    return refs
