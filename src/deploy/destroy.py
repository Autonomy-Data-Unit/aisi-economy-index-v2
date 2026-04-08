__all__ = ['main']

import subprocess

from .config import load_deploy_config, server_exists, volume_attached_to, volume_exists


def main():
    config = load_deploy_config()
    server_name = config["server"]["name"]
    vol_name = config["volume"]["name"]

    if not server_exists(server_name):
        print(f"Server '{server_name}' does not exist, nothing to destroy.")
        return

    # Detach volume before destroying server
    if volume_exists(vol_name):
        attached_to = volume_attached_to(vol_name)
        if attached_to is not None:
            print(f"Detaching volume '{vol_name}'...")
            subprocess.run(["hcloud", "volume", "detach", vol_name], check=True)

    print(f"Destroying server '{server_name}'...")
    subprocess.run(["hcloud", "server", "delete", server_name], check=True)
    print(f"Server '{server_name}' destroyed.")

    if volume_exists(vol_name):
        print(f"Volume '{vol_name}' preserved (use 'hcloud volume delete {vol_name}' to remove it).")
