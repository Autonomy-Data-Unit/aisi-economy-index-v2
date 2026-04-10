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

# --- Install Clifton (Isambard VPN/SSH certificate tool) ---

server.shell(
    name="Install Clifton CLI (if not present)",
    commands=[
        "test -x /usr/local/bin/clifton || "
        "(curl -L https://github.com/isambard-sc/clifton/releases/latest/download/clifton-linux-musl-x86_64 -o /usr/local/bin/clifton "
        "&& chmod +x /usr/local/bin/clifton)",
    ],
)

# --- Swap (safety net for DuckDB memory spikes) ---

server.shell(
    name="Create 16GB swap file (if not present)",
    commands=[
        "test -f /swapfile || (fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile)",
        "swapon /swapfile 2>/dev/null || true",
        "grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab",
        "sysctl -w vm.swappiness=100",
        "sysctl -w vm.overcommit_memory=1",
        "grep -q 'vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=100' >> /etc/sysctl.conf",
        "grep -q 'vm.overcommit_memory' /etc/sysctl.conf || echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf",
    ],
)

# --- Generate SSH key for Isambard ---

server.shell(
    name="Generate SSH key (if not present)",
    commands=[
        'test -f /root/.ssh/id_ed25519 || ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N ""',
    ],
)
