# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # deploy.download_store
#
# Download files from the remote store via rsync.
#
# Usage:
#     uv run remote-download-store <remote_rel_store_path> <local_path>
#
# Example:
#     uv run remote-download-store pipeline/baseline/llm_summarise ./local_output/

# %%
#|default_exp download_store

# %%
#|export
import subprocess
import sys

from deploy.config import get_server_ip, load_deploy_config

# %%
#|export
def main():
    if len(sys.argv) != 3:
        print("Usage: remote-download-store <remote_rel_store_path> <local_path>", file=sys.stderr)
        sys.exit(1)

    remote_rel = sys.argv[1]
    local_path = sys.argv[2]

    config = load_deploy_config()
    ip = get_server_ip(config["server"]["name"])
    repo_path = config["repo"]["path"]

    remote_full = f"root@{ip}:{repo_path}/store/{remote_rel}"

    print(f"Downloading {remote_full} -> {local_path}")
    subprocess.run(
        ["rsync", "-avz", remote_full, local_path],
        check=True,
    )
