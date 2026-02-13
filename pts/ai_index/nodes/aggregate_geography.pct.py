# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Aggregate Geography
#
# Aggregate the full index by geographic dimensions from job ad
# location metadata.

# %%
#|default_exp nodes.aggregate_geography
#|export_as_func true

# %%
#|set_func_signature
def aggregate_geography(job_exposure_index, print) -> {"geography_index": dict}:
    """Aggregate the full index by geographic dimensions."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dotenv import load_dotenv; load_dotenv()
from dev_utils import set_node_func_args
set_node_func_args(aggregate_geography)

# %%
#|export
print("aggregate_geography: returning dummy data")
{"geography_index": {"dummy": True}}  #|func_return_line
