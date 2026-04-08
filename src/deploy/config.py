__all__ = [
    'DEPLOY_CONFIG_PATH', 'get_server_ip', 'load_deploy_config', 'run_ssh',
    'server_exists', 'ssh_key_exists', 'wait_for_ssh',
    'volume_exists', 'get_volume_id', 'volume_attached_to',
]

import json
import subprocess
import time
import tomllib
from pathlib import Path

DEPLOY_CONFIG_PATH = Path("config/deploy.toml")


def load_deploy_config() -> dict:
    """Load and validate deploy configuration from config/deploy.toml."""
    with open(DEPLOY_CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    # Validate required fields exist (fail loudly if missing)
    server = config["server"]
    _ = server["name"], server["type"], server["location"], server["image"]
    _ = server["ssh_key_name"], server["ssh_pubkey_path"]

    volume = config["volume"]
    _ = volume["name"], volume["size"], volume["format"], volume["mount_point"]

    repo = config["repo"]
    _ = repo["path"]

    return config


def get_server_ip(server_name: str) -> str:
    """Get the IPv4 address of a Hetzner server by name."""
    result = subprocess.run(
        ["hcloud", "server", "ip", server_name],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def get_server_id(server_name: str) -> int:
    """Get the numeric ID of a Hetzner server by name."""
    result = subprocess.run(
        ["hcloud", "server", "describe", server_name, "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["id"]


def server_exists(server_name: str) -> bool:
    """Check if a Hetzner server exists."""
    result = subprocess.run(
        ["hcloud", "server", "describe", server_name, "-o", "json"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def ssh_key_exists(key_name: str) -> bool:
    """Check if an SSH key exists in Hetzner Cloud."""
    result = subprocess.run(
        ["hcloud", "ssh-key", "describe", key_name, "-o", "json"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def volume_exists(volume_name: str) -> bool:
    """Check if a Hetzner volume exists."""
    result = subprocess.run(
        ["hcloud", "volume", "describe", volume_name, "-o", "json"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def get_volume_id(volume_name: str) -> int:
    """Get the numeric ID of a Hetzner volume by name."""
    result = subprocess.run(
        ["hcloud", "volume", "describe", volume_name, "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["id"]


def volume_attached_to(volume_name: str) -> int | None:
    """Return the server ID the volume is attached to, or None if detached."""
    result = subprocess.run(
        ["hcloud", "volume", "describe", volume_name, "-o", "json"],
        capture_output=True, text=True, check=True,
    )
    server = json.loads(result.stdout)["server"]
    return server


def run_ssh(ip: str, command: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command on the remote server via SSH."""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        f"root@{ip}", command,
    ]
    if capture:
        return subprocess.run(ssh_cmd, capture_output=True, text=True, check=check)
    return subprocess.run(ssh_cmd, check=check)


def wait_for_ssh(ip: str, max_attempts: int = 30, interval: int = 5) -> None:
    """Wait for SSH to become available on the server."""
    for i in range(max_attempts):
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             f"root@{ip}", "echo ok"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        print(f"Waiting for SSH... ({i + 1}/{max_attempts})")
        time.sleep(interval)
    raise RuntimeError(f"SSH not available after {max_attempts * interval}s")
