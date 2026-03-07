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
from isambard_utils.ssh import arun as async_ssh_run, acheck_connection, acheck_clifton_auth
from isambard_utils.transfer import upload, download, upload_bytes
from isambard_utils.transfer import aupload, adownload, aupload_bytes
from isambard_utils.transfer import upload_tar_pipe, download_tar_pipe, upload_idempotent, upload_compressed
from isambard_utils.transfer import aupload_tar_pipe, adownload_tar_pipe, aupload_idempotent, aupload_compressed
from isambard_utils.transfer import compute_content_hash
from isambard_utils.slurm import submit, status, wait, cancel, job_log, SlurmJob
from isambard_utils.slurm import asubmit, astatus, await_job, acancel, ajob_log
from isambard_utils.sbatch import generate as generate_sbatch, SbatchConfig
from isambard_utils.models import ensure_model, check_model, set_model_env, load_embedding_model, load_llm, EmbeddingModel, LLM
from isambard_utils.models import aensure_model, acheck_model
from isambard_utils.slurm import job_state, ajob_state
from isambard_utils.orchestrate import run_remote, arun_remote, setup_runner, asetup_runner, TransferMode
from isambard_utils.orchestrate import compute_job_hash
from isambard_utils.orchestrate import clear_job_cache, aclear_job_cache, job_status, ajob_status
