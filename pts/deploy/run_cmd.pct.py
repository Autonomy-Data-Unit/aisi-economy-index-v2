# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # deploy.run_cmd
#
# Run a command on the remote server with the working directory set to the repo folder.
# Streams output directly.
#
# Usage:
#     uv run remote-run-cmd <command...>
#
# Example:
#     uv run remote-run-cmd uv run run-pipeline

# %%
#|default_exp run_cmd

# %%
#|export
import subprocess
import sys

from deploy.config import get_server_ip, load_deploy_config

# %%
#|export
def main():
    if len(sys.argv) < 2:
        print("Usage: remote-run-cmd <command...>", file=sys.stderr)
        sys.exit(1)

    cmd = " ".join(sys.argv[1:])
    config = load_deploy_config()
    ip = get_server_ip(config["server"]["name"])
    repo_path = config["repo"]["path"]

    # Run via SSH, streaming output directly (no capture)
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-t",
         f"root@{ip}",
         f"cd {repo_path} && {cmd}"],
    )
    sys.exit(result.returncode)
