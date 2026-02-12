# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fetch O*NET
#
# Download and extract O*NET 30.0 database.

# %%
#|default_exp nodes.fetch_onet
#|export_as_func true

# %%
#|set_func_signature
def fetch_onet(print) -> {"onet_tables": dict}:
    """Download and extract O*NET 30.0 database."""
    ...

# %%
#|export
print("fetch_onet: returning dummy data")
return {"onet_tables": {"dummy": True}}
