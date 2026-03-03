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
module load cray-python/3.11.7 2>/dev/null || true
uv sync --no-dev --no-install-project
""".strip()
    await async_ssh_run(f"bash -lc {_shlex_quote(script)}", config=config, timeout=1800)
