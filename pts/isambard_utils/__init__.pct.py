# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # isambard_utils
#
# Isambard HPC support package for the AI Index pipeline. Provides SSH command
# execution, file transfer, Slurm job management, environment bootstrapping,
# and SBATCH script generation.

# %%
#|default_exp __init__

# %%
#|export
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import run as ssh_run, check_connection, check_clifton_auth
from isambard_utils.transfer import upload, download, upload_bytes
from isambard_utils.slurm import submit, status, wait, cancel, job_log, SlurmJob
from isambard_utils.env import setup, sync_code_rsync as sync_code, check_setup
from isambard_utils.sbatch import generate as generate_sbatch, SbatchConfig
