"""Pyinfra deploy script for remote server setup.

Installs system packages and uv.

Invoked by: uv run pyinfra root@IP scripts/deploy_setup.py -y
Must be run from the project root.
"""

from pyinfra.operations import apt, server

# --- System packages ---

apt.update(
    name="Update apt cache",
    cache_time=3600,
)

apt.packages(
    name="Install system packages",
    packages=["git", "curl", "rsync", "build-essential"],
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
