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

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args(aggregate_soc_exposure)

# %%
#|export
print("aggregate_soc_exposure: returning dummy data")
return {"exposure_scores": {"dummy": True}}
