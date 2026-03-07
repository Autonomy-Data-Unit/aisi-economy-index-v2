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
# Low-level helpers for bootstrapping the Python environment on Isambard.

# %%
#|default_exp env

# %%
#|export
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import arun as async_ssh_run, _get_config, _run_sync

# %%
#|exporti
def _shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)

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
async def _aensure_venv(*, config: IsambardConfig) -> None:
    """Create venv and sync dependencies if needed (async)."""
    script = f"""
cd {config.project_dir}
export UV_CACHE_DIR={config.project_dir}/.uv_cache
export UV_LINK_MODE=copy
module load cray-python/3.11.7 2>/dev/null || true
uv sync --no-dev --no-install-project
""".strip()
    await async_ssh_run(f"bash -lc {_shlex_quote(script)}", config=config, timeout=1800)

# %%
#|export
async def _afix_lustre_hardlinks(*, config: IsambardConfig) -> None:
    """Reinstall packages that were installed with hardlinks on Lustre.

    uv defaults to hardlink mode, which causes stale inode issues on Lustre:
    compute nodes can see the file via ``ls`` but ``stat``/``open`` fail with
    ENOENT because the inode was replaced on the login node during a reinstall.

    This checks whether torch's init file has link count > 1 (indicating
    hardlinks) and, if so, reinstalls torch, triton and vllm with copy mode
    so every file gets its own inode. Idempotent: once packages are installed
    with copy mode (link count == 1), this is a no-op.
    """
    check_script = f"""
cd {config.project_dir}
site=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
torch_init="$site/torch/__init__.py"
[ -f "$torch_init" ] && stat -c %h "$torch_init" || echo 0
""".strip()
    result = await async_ssh_run(
        f"bash -lc {_shlex_quote(check_script)}", config=config, check=False,
    )
    if result.returncode != 0:
        return
    link_count = result.stdout.strip()
    if not link_count.isdigit() or int(link_count) <= 1:
        return  # Already using copies, nothing to do

    # Reinstall torch, triton, vllm with copy mode to eliminate hardlinks.
    # --no-deps on triton prevents uv from pulling CPU-only torch from PyPI.
    fix_script = f"""
cd {config.project_dir}
export UV_CACHE_DIR={config.project_dir}/.uv_cache
export UV_LINK_MODE=copy
uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --index-url {config.torch_index_url} --quiet
uv pip install --reinstall-package triton --no-deps triton --quiet
""".strip()
    await async_ssh_run(
        f"bash -lc {_shlex_quote(fix_script)}", config=config, timeout=600,
    )

    # Reinstall vllm if present
    vllm_check = await async_ssh_run(
        f"bash -lc {_shlex_quote(f'cd {config.project_dir} && .venv/bin/python -c \"import vllm\"')}",
        config=config, check=False,
    )
    if vllm_check.returncode == 0:
        vllm_script = f"""
cd {config.project_dir}
export UV_CACHE_DIR={config.project_dir}/.uv_cache
export UV_LINK_MODE=copy
uv pip install --reinstall-package vllm vllm --extra-index-url {config.torch_index_url} --quiet
""".strip()
        await async_ssh_run(
            f"bash -lc {_shlex_quote(vllm_script)}", config=config, timeout=1200,
        )

# %%
#|export
async def _aensure_cuda_torch(*, config: IsambardConfig) -> None:
    """Replace CPU-only torch with CUDA build if needed (async).

    PyPI's default torch wheel is CPU-only on ARM64 (GH200).
    This checks the installed torch version string for '+cu' and,
    if absent, reinstalls torch + torchvision from PyTorch's CUDA index.
    If vllm is installed, it's also reinstalled so its compiled extensions
    are ABI-compatible with the new CUDA torch.
    """
    # Check if torch already has CUDA support
    check_script = f"""
cd {config.project_dir}
.venv/bin/python -c "import torch; print(torch.__version__)"
""".strip()
    result = await async_ssh_run(
        f"bash -lc {_shlex_quote(check_script)}", config=config, check=False,
    )
    if result.returncode != 0:
        return  # torch not installed, nothing to do
    if "+cu" in result.stdout.strip():
        return  # Already has CUDA torch

    # Check if vllm is installed (before changing torch)
    has_vllm_script = f"""
cd {config.project_dir}
.venv/bin/python -c "import importlib.util; print('yes' if importlib.util.find_spec('vllm') else 'no')"
""".strip()
    vllm_check = await async_ssh_run(
        f"bash -lc {_shlex_quote(has_vllm_script)}", config=config, check=False,
    )
    has_vllm = vllm_check.returncode == 0 and "yes" in vllm_check.stdout

    # Install CUDA torch + torchvision.
    # UV_LINK_MODE=copy avoids hardlinks on Lustre, which can leave stale
    # inodes on compute nodes after reinstalls.
    install_script = f"""
cd {config.project_dir}
export UV_CACHE_DIR={config.project_dir}/.uv_cache
export UV_LINK_MODE=copy
uv pip install torch torchvision --index-url {config.torch_index_url} --reinstall-package torch --reinstall-package torchvision --quiet
""".strip()
    await async_ssh_run(
        f"bash -lc {_shlex_quote(install_script)}", config=config, timeout=600,
    )

    # Reinstall vllm if present: use CUDA index as extra source so uv resolves
    # torch from cu126 (keeping CUDA) instead of reverting to CPU-only PyPI torch
    if has_vllm:
        vllm_script = f"""
cd {config.project_dir}
export UV_CACHE_DIR={config.project_dir}/.uv_cache
export UV_LINK_MODE=copy
uv pip install vllm --extra-index-url {config.torch_index_url} --reinstall-package vllm --quiet
""".strip()
        await async_ssh_run(
            f"bash -lc {_shlex_quote(vllm_script)}", config=config, timeout=1200,
        )
