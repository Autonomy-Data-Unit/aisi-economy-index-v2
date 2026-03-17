# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # deploy.deploy_pipeline
#
# Provisions a Hetzner server, sets it up with pyinfra, syncs the repo code,
# and creates the store symlink to the storage box.
#
# Usage:
#     uv run remote-deploy-pipeline
#
# Idempotent: safe to re-run. If the server already exists, it skips provisioning.
# If the local repo is at a different commit than the remote, it re-syncs the code
# (without touching the store symlink or .venv).

# %%
#|default_exp deploy_pipeline

# %%
#|export
import os
import subprocess
import sys

from deploy.config import (
    get_server_ip,
    load_deploy_config,
    run_ssh,
    server_exists,
    ssh_key_exists,
    wait_for_ssh,
)

# %%
#|export
def _ensure_ssh_key(config: dict) -> None:
    """Ensure the SSH key is registered in Hetzner Cloud."""
    key_name = config["server"]["ssh_key_name"]
    if ssh_key_exists(key_name):
        return

    pubkey_path = os.path.expanduser(config["server"]["ssh_pubkey_path"])
    print(f"Uploading SSH key '{key_name}' from {pubkey_path}")
    subprocess.run(
        ["hcloud", "ssh-key", "create", "--name", key_name,
         "--public-key-from-file", pubkey_path],
        check=True,
    )

# %%
#|export
def _ensure_server(config: dict) -> str:
    """Ensure the Hetzner server exists. Returns the server IP."""
    server = config["server"]
    name = server["name"]

    if server_exists(name):
        print(f"Server '{name}' already exists")
        return get_server_ip(name)

    print(f"Creating server '{name}' (type={server['type']}, image={server['image']}, location={server['location']})")
    subprocess.run(
        ["hcloud", "server", "create",
         "--name", name,
         "--type", server["type"],
         "--image", server["image"],
         "--location", server["location"],
         "--ssh-key", server["ssh_key_name"]],
        check=True,
    )
    return get_server_ip(name)

# %%
#|export
def _run_pyinfra_setup(ip: str) -> None:
    """Run the pyinfra server setup script."""
    print("Running pyinfra server setup...")
    subprocess.run(
        ["uv", "run", "pyinfra", ip, "--ssh-user", "root", "scripts/deploy_setup.py", "-y"],
        check=True,
    )

# %%
#|export
def _sync_code(ip: str, repo_path: str) -> None:
    """Rsync the local repo to the remote server, excluding store and transient files."""
    print("Syncing code to remote...")
    subprocess.run(
        ["rsync", "-avz", "--delete",
         "--exclude=store",
         "--exclude=.venv",
         "--exclude=__pycache__",
         "--exclude=*.pyc",
         "--exclude=.nbl",
         "--exclude=.git",
         "--exclude=.cache",
         "--exclude=_dev",
         "./", f"root@{ip}:{repo_path}/"],
        check=True,
    )

# %%
#|export
def _setup_store_symlink(ip: str, config: dict) -> None:
    """Create the store symlink pointing to the storage box."""
    repo_path = config["repo"]["path"]
    storage = config["storage_box"]
    mount_point = storage["mount_point"]
    store_path = storage["store_path"]
    target = f"{mount_point}/{store_path}"

    # Create the directory on the storage box if needed
    run_ssh(ip, f"mkdir -p {target}")

    # Create symlink (only if store is not already a symlink)
    run_ssh(ip, f"cd {repo_path} && (test -L store || ln -s {target} store)")

# %%
#|export
def _install_dependencies(ip: str, repo_path: str) -> None:
    """Run uv sync on the remote server."""
    print("Installing dependencies on remote...")
    run_ssh(ip, f"cd {repo_path} && export PATH=$PATH:/root/.cargo/bin && /root/.local/bin/uv sync --no-dev")

# %%
#|export
def deploy_pipeline() -> None:
    """Provision and deploy the pipeline to a remote Hetzner server."""
    from dotenv import load_dotenv
    load_dotenv()

    config = load_deploy_config()

    if "STORAGE_BOX_PASSWORD" not in os.environ:
        print("Error: STORAGE_BOX_PASSWORD environment variable is required", file=sys.stderr)
        sys.exit(1)

    # 1. Ensure SSH key and server exist
    _ensure_ssh_key(config)
    ip = _ensure_server(config)

    # 2. Wait for SSH
    print(f"Server IP: {ip}")
    wait_for_ssh(ip)

    # 3. Run pyinfra setup (packages, uv, storage box mount)
    _run_pyinfra_setup(ip)

    # 4. Sync code
    _sync_code(ip, config["repo"]["path"])

    # 5. Set up store symlink
    _setup_store_symlink(ip, config)

    # 6. Install dependencies
    _install_dependencies(ip, config["repo"]["path"])

    print(f"Deploy complete. Server: root@{ip}")

# %%
#|export
def main():
    deploy_pipeline()
