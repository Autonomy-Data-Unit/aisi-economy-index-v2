# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # deploy.destroy
#
# Destroys the remote Hetzner server. Does not touch the storage box.
#
# Usage:
#     uv run remote-destroy

# %%
#|default_exp destroy

# %%
#|export
import subprocess
import sys

from deploy.config import load_deploy_config, server_exists

# %%
#|export
def main():
    config = load_deploy_config()
    name = config["server"]["name"]

    if not server_exists(name):
        print(f"Server '{name}' does not exist, nothing to destroy.")
        return

    print(f"Destroying server '{name}'...")
    subprocess.run(
        ["hcloud", "server", "delete", name],
        check=True,
    )
    print(f"Server '{name}' destroyed.")
