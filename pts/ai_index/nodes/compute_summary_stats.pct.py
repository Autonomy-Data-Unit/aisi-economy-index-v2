# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compute Summary Stats
#
# Compute summary statistics and visualizations of the full index.

# %%
#|default_exp nodes.compute_summary_stats
#|export_as_func true

# %%
#|set_func_signature
def compute_summary_stats(job_exposure_index, print) -> {"summary": dict}:
    """Compute summary statistics and visualizations of the full index."""
    ...

# %%
#|export
print("compute_summary_stats: returning dummy data")
return {"summary": {"dummy": True}}
