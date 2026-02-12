# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Aggregate SOC Exposure
#
# Merge and normalize all exposure score types at SOC level. Compute
# composite vulnerability metric.

# %%
#|default_exp nodes.aggregate_soc_exposure
#|export_as_func true

# %%
#|set_func_signature
def aggregate_soc_exposure(task_scores, presence_scores, felten_scores, print) -> {"exposure_scores": dict}:
    """Merge and normalize all exposure score types at SOC level."""
    ...

# %%
#|export
raise NotImplementedError("aggregate_soc_exposure not yet implemented")
