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
from isambard_utils.ssh import run as ssh_run, arun as async_ssh_run, _get_config, _run_sync
from isambard_utils.transfer import upload as rsync_upload, aupload as async_rsync_upload

# %%
#|export
_DEFAULT_RSYNC_EXCLUDES = [
    ".venv", ".git", "__pycache__", ".cache", "*.pyc",
    "old-data", "old-repo", ".mypy_cache", ".pytest_cache",
    "nbs", ".ipynb_checkpoints",
]

# %%
#|export
async def asetup(*, config: IsambardConfig | None = None,
                 sync_code: bool = True,
                 local_project_dir: str | None = None,
                 git_url: str | None = None,
                 git_branch: str | None = None) -> None:
    """Full environment bootstrap (idempotent, async).

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
    await async_ssh_run(f"mkdir -p {config.project_dir} {config.logs_dir} {config.hf_cache_dir}",
                        config=config)

    # 2. Install uv if missing
    await _aensure_uv(config=config)

    # 3. Sync code
    if sync_code:
        if git_url:
            await _async_code_git(git_url=git_url, git_branch=git_branch, config=config)
        else:
            local_dir = local_project_dir or "."
            await async_code_rsync(local_project_dir=local_dir, config=config)

    # 4. Create venv and install deps
    await _aensure_venv(config=config)

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
    _run_sync(asetup(config=config, sync_code=sync_code,
                      local_project_dir=local_project_dir,
                      git_url=git_url, git_branch=git_branch))

# %%
#|export
async def _aensure_uv(*, config: IsambardConfig) -> None:
    """Install uv on the remote if not already present (async)."""
    result = await async_ssh_run("bash -lc 'which uv'", config=config, check=False)
    if result.returncode == 0:
        return
    # Install uv using the official installer
    await async_ssh_run("curl -LsSf https://astral.sh/uv/install.sh | sh", config=config,
                        timeout=120)

# %%
#|export
def _ensure_uv(*, config: IsambardConfig) -> None:
    """Install uv on the remote if not already present."""
    _run_sync(_aensure_uv(config=config))

# %%
#|export
async def _aensure_venv(*, config: IsambardConfig) -> None:
    """Create venv and sync dependencies if needed (async)."""
    cuda_index = "https://download.pytorch.org/whl/cu126"
    script = f"""
cd {config.project_dir}
module load cray-python/3.11.7 2>/dev/null || true
uv sync --no-dev
uv pip install torch --index-url {cuda_index} --reinstall
""".strip()
    await async_ssh_run(f"bash -lc {_shlex_quote(script)}", config=config, timeout=600)

# %%
#|export
def _ensure_venv(*, config: IsambardConfig) -> None:
    """Create venv and sync dependencies if needed."""
    _run_sync(_aensure_venv(config=config))

# %%
#|exporti
def _shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)

# %%
#|export
async def _async_code_git(*, git_url: str, git_branch: str | None,
                          config: IsambardConfig) -> None:
    """Clone or pull code from a git repository (async)."""
    result = await async_ssh_run(f"test -d {config.project_dir}/.git", config=config, check=False)
    if result.returncode == 0:
        # Pull latest
        cmds = f"cd {config.project_dir} && git pull"
        if git_branch:
            cmds = f"cd {config.project_dir} && git checkout {git_branch} && git pull"
        await async_ssh_run(cmds, config=config, timeout=300)
    else:
        # Clone
        branch_flag = f"-b {git_branch}" if git_branch else ""
        await async_ssh_run(f"git clone {branch_flag} {git_url} {config.project_dir}",
                            config=config, timeout=300)

# %%
#|export
def _sync_code_git(*, git_url: str, git_branch: str | None,
                   config: IsambardConfig) -> None:
    """Clone or pull code from a git repository."""
    _run_sync(_async_code_git(git_url=git_url, git_branch=git_branch, config=config))

# %%
#|export
async def async_code_rsync(*, config: IsambardConfig | None = None,
                           local_project_dir: str = ".",
                           exclude: list[str] | None = None) -> None:
    """rsync local project to remote project_dir (async).

    Args:
        config: Isambard configuration.
        local_project_dir: Local directory to sync (default: ".").
        exclude: rsync exclude patterns. Defaults to standard excludes.
    """
    config = _get_config(config)
    exclude = exclude or _DEFAULT_RSYNC_EXCLUDES
    # Ensure trailing slash for rsync directory sync
    local = local_project_dir.rstrip("/") + "/"
    await async_rsync_upload(local, config.project_dir, config=config,
                             exclude=exclude, delete=False)

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
    _run_sync(async_code_rsync(config=config, local_project_dir=local_project_dir,
                                exclude=exclude))

# %%
#|export
async def acheck_setup(*, config: IsambardConfig | None = None) -> dict:
    """Check environment status on Isambard (async).

    Returns a dict with status booleans:
        - uv_installed: whether uv is available
        - venv_exists: whether .venv exists in project_dir
        - code_synced: whether project_dir contains pyproject.toml
    """
    config = _get_config(config)

    import asyncio as _asyncio
    uv_check, venv_check, code_check = await _asyncio.gather(
        async_ssh_run("which uv", config=config, check=False),
        async_ssh_run(f"test -d {config.project_dir}/.venv", config=config, check=False),
        async_ssh_run(f"test -f {config.project_dir}/pyproject.toml", config=config, check=False),
    )

    return {
        "uv_installed": uv_check.returncode == 0,
        "venv_exists": venv_check.returncode == 0,
        "code_synced": code_check.returncode == 0,
    }

# %%
#|export
def check_setup(*, config: IsambardConfig | None = None) -> dict:
    """Check environment status on Isambard.

    Returns a dict with status booleans:
        - uv_installed: whether uv is available
        - venv_exists: whether .venv exists in project_dir
        - code_synced: whether project_dir contains pyproject.toml
    """
    return _run_sync(acheck_setup(config=config))

# %% [markdown]
# ## llm_runner deployment
#
# Deploy the `llm_runner` package as a standalone project on Isambard.
# The runner has its own venv with minimal dependencies (no netrun, no ai_index).

# %%
#|export
_RUNNER_PYPROJECT = """\
[project]
name = "llm-runner"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy", "torch", "sentence-transformers", "transformers", "accelerate"]

[project.optional-dependencies]
api = ["adulib[llm]>=0.1"]
vllm = ["vllm"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
sources = ["src"]
packages = ["src/llm_runner"]
"""

# %%
#|export
async def asetup_runner(*, config: IsambardConfig | None = None,
                         print_fn=print) -> None:
    """Ensure llm_runner is installed on Isambard (idempotent, async).

    Steps:
        1. Create remote runner project directory
        2. Rsync src/llm_runner/ to remote
        3. Upload minimal pyproject.toml (if changed)
        4. uv sync + torch CUDA reinstall

    Args:
        config: Isambard configuration.
        print_fn: Print function for progress logging.
    """
    config = _get_config(config)
    runner_dir = f"{config.project_dir}/llm_runner_env"

    # 1. Create remote directory structure
    await async_ssh_run(f"mkdir -p {runner_dir}/src", config=config)

    # 2. Install uv if needed
    await _aensure_uv(config=config)

    # 3. Rsync llm_runner source code
    import importlib.resources as resources
    import os
    # Find the local src/llm_runner directory relative to the project
    # We look for it relative to the isambard_utils package location
    local_runner_src = os.path.join(os.getcwd(), "src", "llm_runner")
    if os.path.isdir(local_runner_src):
        print_fn("runner setup: syncing llm_runner source...")
        await async_rsync_upload(
            local_runner_src + "/", f"{runner_dir}/src/llm_runner",
            config=config, exclude=["__pycache__", "*.pyc"],
        )
    else:
        raise FileNotFoundError(
            f"Cannot find src/llm_runner at {local_runner_src}. "
            "Run from the project root directory."
        )

    # 4. Upload pyproject.toml (check if changed via hash)
    import hashlib
    pyproject_hash = hashlib.md5(_RUNNER_PYPROJECT.encode()).hexdigest()[:12]
    check = await async_ssh_run(
        f"cat {runner_dir}/.pyproject_hash 2>/dev/null || echo ''",
        config=config, check=False,
    )
    remote_hash = check.stdout.strip() if check.returncode == 0 else ""

    if remote_hash != pyproject_hash:
        print_fn("runner setup: uploading pyproject.toml...")
        from isambard_utils.transfer import aupload_bytes
        await aupload_bytes(_RUNNER_PYPROJECT.encode(), f"{runner_dir}/pyproject.toml",
                            config=config)
        await aupload_bytes(pyproject_hash.encode(), f"{runner_dir}/.pyproject_hash",
                            config=config)

    # 5. Create venv and sync deps
    print_fn("runner setup: installing dependencies...")
    cuda_index = "https://download.pytorch.org/whl/cu126"
    script = f"""
cd {runner_dir}
module load cray-python/3.11.7 2>/dev/null || true
uv sync --no-dev
uv pip install torch --index-url {cuda_index} --reinstall
""".strip()
    await async_ssh_run(f"bash -lc {_shlex_quote(script)}", config=config, timeout=600)
    print_fn("runner setup: done")

# %%
#|export
def setup_runner(*, config: IsambardConfig | None = None,
                  print_fn=print) -> None:
    """Ensure llm_runner is installed on Isambard (idempotent).

    Args:
        config: Isambard configuration.
        print_fn: Print function for progress logging.
    """
    _run_sync(asetup_runner(config=config, print_fn=print_fn))
