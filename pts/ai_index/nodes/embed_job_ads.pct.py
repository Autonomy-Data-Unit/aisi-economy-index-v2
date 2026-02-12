# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embed Job Ads
#
# Embed job ads with BGE-large sentence transformer
# (Nx1024 float16, role + task/skill text).

# %%
#|default_exp nodes.embed_job_ads
#|export_as_func true

# %%
#|set_func_signature
def embed_job_ads(job_ads, print) -> {"job_ad_embeddings": dict}:
    """Embed job ads with BGE-large."""
    ...

# %%
#|export
raise NotImplementedError("embed_job_ads not yet implemented")
