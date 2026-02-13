# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Combine Job Exposure
#
# For each job ad: weighted sum of matched O*NET codes' exposure scores
# to produce per-job multi-dimension AI exposure. Includes selected job
# ad metadata columns.

# %%
#|default_exp nodes.combine_job_exposure
#|export_as_func true

# %%
#|set_func_signature
def combine_job_exposure(job_ads, weighted_codes, exposure_scores, print) -> {"job_exposure_index": dict}:
    """Combine matched O*NET weights with exposure scores to produce per-job AI exposure."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args(combine_job_exposure)

# %%
#|export
print("combine_job_exposure: returning dummy data")
{"job_exposure_index": {"dummy": True}}  #|func_return_line
