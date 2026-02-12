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
def combine_job_exposure(job_ads, weighted_codes, exposure_scores, print) -> {"index": dict}:
    """Combine matched O*NET weights with exposure scores to produce per-job AI exposure."""
    ...

# %%
#|export
raise NotImplementedError("combine_job_exposure not yet implemented")
