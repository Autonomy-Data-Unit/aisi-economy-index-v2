"""Pyinfra deploy script for remote server setup.

Installs system packages, uv, and mounts the Hetzner storage box.

Invoked by: uv run pyinfra root@IP scripts/deploy_setup.py -y
Must be run from the project root.
Requires STORAGE_BOX_PASSWORD environment variable.
"""

import os
import tomllib

from pyinfra.operations import apt, files, server

# Load deploy config
with open("config/deploy.toml", "rb") as f:
    config = tomllib.load(f)

storage_box = config["storage_box"]
storage_username = storage_box["username"]
storage_mount_point = storage_box["mount_point"]
storage_box_password = os.environ["STORAGE_BOX_PASSWORD"]
credentials_path = "/etc/storage-box-credentials.txt"

# --- System packages ---

apt.update(
    name="Update apt cache",
    cache_time=3600,
)

apt.packages(
    name="Install system packages",
    packages=["git", "curl", "cifs-utils", "rsync", "build-essential"],
)

# --- Install uv ---

server.shell(
    name="Install uv (if not present)",
    commands=[
        "command -v /root/.local/bin/uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh",
    ],
)

server.shell(
    name="Install Rust (if not present, needed for netrun-sim)",
    commands=[
        "command -v /root/.cargo/bin/rustc > /dev/null 2>&1 || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)",
    ],
)

# --- Mount storage box ---

server.shell(
    name="Write storage box credentials",
    commands=[
        f"printf 'username={storage_username}\\npassword={storage_box_password}\\n' > {credentials_path}",
        f"chmod 600 {credentials_path}",
    ],
)

files.line(
    name="Add storage box mount to fstab",
    path="/etc/fstab",
    line=f"//{storage_username}.your-storagebox.de/backup {storage_mount_point} cifs vers=3.1.1,seal,credentials={credentials_path} 0 0",
    present=True,
)

server.shell(
    name="Create mount point and mount storage box",
    commands=[
        f"mkdir -p {storage_mount_point}",
        f"mountpoint -q {storage_mount_point} || mount {storage_mount_point}",
    ],
)
