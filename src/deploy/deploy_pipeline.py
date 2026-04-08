__all__ = ['deploy_pipeline', 'main']

import os
import subprocess

from .config import (
    get_server_id,
    get_server_ip,
    get_volume_id,
    load_deploy_config,
    run_ssh,
    server_exists,
    ssh_key_exists,
    volume_attached_to,
    volume_exists,
    wait_for_ssh,
)


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


def _ensure_volume(config: dict) -> None:
    """Ensure the Hetzner volume exists, is attached to the server, and is mounted."""
    vol = config["volume"]
    vol_name = vol["name"]
    server_name = config["server"]["name"]
    mount_point = vol["mount_point"]
    repo_path = config["repo"]["path"]

    server_id = get_server_id(server_name)

    # Create volume if it doesn't exist
    if not volume_exists(vol_name):
        print(f"Creating volume '{vol_name}' ({vol['size']}GB, {vol['format']})...")
        subprocess.run(
            ["hcloud", "volume", "create",
             "--name", vol_name,
             "--size", str(vol["size"]),
             "--format", vol["format"],
             "--location", config["server"]["location"]],
            check=True,
        )

    # Attach to server if not already attached
    attached_to = volume_attached_to(vol_name)
    if attached_to != server_id:
        if attached_to is not None:
            print(f"Detaching volume '{vol_name}' from server {attached_to}...")
            subprocess.run(
                ["hcloud", "volume", "detach", vol_name],
                check=True,
            )
        print(f"Attaching volume '{vol_name}' to server '{server_name}'...")
        subprocess.run(
            ["hcloud", "volume", "attach", vol_name, "--server", server_name],
            check=True,
        )
    else:
        print(f"Volume '{vol_name}' already attached to '{server_name}'")

    # Mount on the server and symlink store/ to the mount point
    ip = get_server_ip(server_name)
    run_ssh(ip, f"mkdir -p {mount_point}")

    # Mount if not already mounted
    mount_check = run_ssh(ip, f"mountpoint -q {mount_point}", check=False)
    if mount_check.returncode != 0:
        # Find the volume's block device (hcloud volumes appear as /dev/disk/by-id/scsi-0HC_Volume_<id>)
        vol_id = get_volume_id(vol_name)
        dev_path = f"/dev/disk/by-id/scsi-0HC_Volume_{vol_id}"
        print(f"Mounting volume at {mount_point}...")
        run_ssh(ip, f"mount -o discard,defaults {dev_path} {mount_point}")
    else:
        print(f"Volume already mounted at {mount_point}")

    # Ensure the mount persists across reboots via fstab
    vol_id = get_volume_id(vol_name)
    dev_path = f"/dev/disk/by-id/scsi-0HC_Volume_{vol_id}"
    fstab_line = f"{dev_path} {mount_point} {vol['format']} discard,nofail,defaults 0 0"
    run_ssh(ip, f"grep -q 'HC_Volume_{vol_id}' /etc/fstab || echo '{fstab_line}' >> /etc/fstab")

    # Symlink repo store/ -> volume mount point
    store_path = f"{repo_path}/store"
    run_ssh(ip, f"mkdir -p {repo_path}")
    # Remove existing store dir/symlink and create fresh symlink
    run_ssh(ip, f"rm -rf {store_path} && ln -sfn {mount_point} {store_path}")
    print(f"Symlinked {store_path} -> {mount_point}")


def _run_pyinfra_setup(ip: str) -> None:
    """Run the pyinfra server setup script."""
    # Remove any stale host key for this IP (Hetzner recycles IPs across servers)
    subprocess.run(["ssh-keygen", "-R", ip], capture_output=True)
    print("Running pyinfra server setup...")
    subprocess.run(
        ["uv", "run", "pyinfra", ip, "--ssh-user", "root", "src/deploy/deploy_setup.py", "-y"],
        check=True,
    )


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


def _install_dependencies(ip: str, repo_path: str) -> None:
    """Run uv sync on the remote server."""
    print("Installing dependencies on remote...")
    run_ssh(ip, f"cd {repo_path} && export PATH=$PATH:/root/.cargo/bin && /root/.local/bin/uv sync --no-dev")


def deploy_pipeline() -> None:
    """Provision and deploy the pipeline to a remote Hetzner server."""
    from dotenv import load_dotenv
    load_dotenv()

    config = load_deploy_config()

    # 1. Ensure SSH key and server exist
    _ensure_ssh_key(config)
    ip = _ensure_server(config)

    # 2. Wait for SSH
    print(f"Server IP: {ip}")
    wait_for_ssh(ip)

    # 3. Run pyinfra setup (packages, uv)
    _run_pyinfra_setup(ip)

    # 4. Sync code
    _sync_code(ip, config["repo"]["path"])

    # 5. Create, attach, and mount the volume; symlink store/
    _ensure_volume(config)

    # 6. Install dependencies
    _install_dependencies(ip, config["repo"]["path"])

    print(f"Deploy complete. Server: root@{ip}")


def main():
    deploy_pipeline()
