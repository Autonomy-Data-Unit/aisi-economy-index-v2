# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Environment Setup
#
# Bootstrap and maintain the Python environment on Isambard.

# %%
#|default_exp env

# %%
#|export
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import run as ssh_run, _get_config
from isambard_utils.transfer import upload as rsync_upload

# %%
#|export
_DEFAULT_RSYNC_EXCLUDES = [
    ".venv", ".git", "__pycache__", ".cache", "*.pyc",
    "old-data", "old-repo", ".mypy_cache", ".pytest_cache",
    "nbs", ".ipynb_checkpoints",
]

# %%
#|export
def setup(*, config: IsambardConfig | None = None,
          sync_code: bool = True,
          local_project_dir: str | None = None,
          git_url: str | None = None,
          git_branch: str | None = None) -> None:
    """Full environment bootstrap (idempotent).

    Steps:
        1. Create project_dir and logs_dir on remote
        2. Install uv (if missing)
        3. Sync code (rsync local dir or git clone/pull)
        4. Create venv with uv (if missing) and uv sync

    Args:
        config: Isambard configuration.
        sync_code: Whether to sync code to the remote.
        local_project_dir: Local project directory to rsync (default: ".").
        git_url: Git repo URL (alternative to rsync).
        git_branch: Git branch to checkout.
    """
    config = _get_config(config)

    # 1. Create directories
    ssh_run(f"mkdir -p {config.project_dir} {config.logs_dir} {config.hf_cache_dir}",
            config=config)

    # 2. Install uv if missing
    _ensure_uv(config=config)

    # 3. Sync code
    if sync_code:
        if git_url:
            _sync_code_git(git_url=git_url, git_branch=git_branch, config=config)
        else:
            local_dir = local_project_dir or "."
            sync_code_rsync(local_project_dir=local_dir, config=config)

    # 4. Create venv and install deps
    _ensure_venv(config=config)

# %%
#|export
def _ensure_uv(*, config: IsambardConfig) -> None:
    """Install uv on the remote if not already present."""
    result = ssh_run("bash -lc 'which uv'", config=config, check=False)
    if result.returncode == 0:
        return
    # Install uv using the official installer
    ssh_run("curl -LsSf https://astral.sh/uv/install.sh | sh", config=config,
            timeout=120)

# %%
#|export
def _ensure_venv(*, config: IsambardConfig) -> None:
    """Create venv and sync dependencies if needed."""
    # Use bash -l so that module and uv are available (login shell).
    # --no-dev avoids building dev deps that need npm/node (e.g. netrun-ui).
    # After uv sync, reinstall torch from the PyTorch CUDA index because
    # the default PyPI torch for aarch64 is CPU-only.
    cuda_index = "https://download.pytorch.org/whl/cu126"
    script = f"""
cd {config.project_dir}
module load cray-python/3.11.7 2>/dev/null || true
uv sync --no-dev
uv pip install torch --index-url {cuda_index} --reinstall
""".strip()
    ssh_run(f"bash -lc {_shlex_quote(script)}", config=config, timeout=600)

# %%
#|exporti
def _shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)

# %%
#|export
def _sync_code_git(*, git_url: str, git_branch: str | None,
                   config: IsambardConfig) -> None:
    """Clone or pull code from a git repository."""
    # Check if project_dir already has a git repo
    result = ssh_run(f"test -d {config.project_dir}/.git", config=config, check=False)
    if result.returncode == 0:
        # Pull latest
        branch_flag = f"-b {git_branch}" if git_branch else ""
        cmds = f"cd {config.project_dir} && git pull"
        if git_branch:
            cmds = f"cd {config.project_dir} && git checkout {git_branch} && git pull"
        ssh_run(cmds, config=config, timeout=300)
    else:
        # Clone
        branch_flag = f"-b {git_branch}" if git_branch else ""
        ssh_run(f"git clone {branch_flag} {git_url} {config.project_dir}",
                config=config, timeout=300)

# %%
#|export
def sync_code_rsync(*, config: IsambardConfig | None = None,
                    local_project_dir: str = ".",
                    exclude: list[str] | None = None) -> None:
    """rsync local project to remote project_dir.

    Args:
        config: Isambard configuration.
        local_project_dir: Local directory to sync (default: ".").
        exclude: rsync exclude patterns. Defaults to standard excludes.
    """
    config = _get_config(config)
    exclude = exclude or _DEFAULT_RSYNC_EXCLUDES
    # Ensure trailing slash for rsync directory sync
    local = local_project_dir.rstrip("/") + "/"
    rsync_upload(local, config.project_dir, config=config,
                 exclude=exclude, delete=False)

# %%
#|export
def check_setup(*, config: IsambardConfig | None = None) -> dict:
    """Check environment status on Isambard.

    Returns a dict with status booleans:
        - uv_installed: whether uv is available
        - venv_exists: whether .venv exists in project_dir
        - code_synced: whether project_dir contains pyproject.toml
    """
    config = _get_config(config)

    uv_check = ssh_run("which uv", config=config, check=False)
    venv_check = ssh_run(f"test -d {config.project_dir}/.venv", config=config, check=False)
    code_check = ssh_run(f"test -f {config.project_dir}/pyproject.toml", config=config, check=False)

    return {
        "uv_installed": uv_check.returncode == 0,
        "venv_exists": venv_check.returncode == 0,
        "code_synced": code_check.returncode == 0,
    }
