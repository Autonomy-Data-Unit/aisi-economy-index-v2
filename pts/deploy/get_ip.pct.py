# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # deploy.get_ip
#
# Print the IP address of the remote server.
#
# Usage:
#     uv run remote-ip

# %%
#|default_exp get_ip

# %%
#|export
import sys

from deploy.config import get_server_ip, load_deploy_config

# %%
#|export
def main():
    config = load_deploy_config()
    ip = get_server_ip(config["server"]["name"])
    print(ip)
