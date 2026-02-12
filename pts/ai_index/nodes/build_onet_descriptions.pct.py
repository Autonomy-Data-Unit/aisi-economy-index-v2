# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build O*NET Descriptions
#
# Build standard occupation descriptions (894 rows: SOC code, title, description,
# tasks/skills text) from raw O*NET data.

# %%
#|default_exp nodes.build_onet_descriptions
#|export_as_func true

# %%
#|set_func_signature
def build_onet_descriptions(onet_data, print) -> {"descriptions": dict}:
    """Build standard occupation descriptions from raw O*NET data."""
    ...

# %%
#|export
raise NotImplementedError("build_onet_descriptions not yet implemented")
