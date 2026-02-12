# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build O*NET Evaluation DataFrames
#
# Build evaluation DataFrames for exposure scoring (skills, abilities,
# knowledge, tasks, work context) from raw O*NET data.

# %%
#|default_exp nodes.build_onet_eval_dfs
#|export_as_func true

# %%
#|set_func_signature
def build_onet_eval_dfs(onet_tables, print) -> {"tasks": dict, "skills": dict, "abilities": dict, "knowledge": dict, "work_context": dict}:
    """Build evaluation DataFrames for exposure scoring from raw O*NET data."""
    ...

# %%
#|export
print("build_onet_eval_dfs: returning dummy data")
return {"tasks": {"dummy": True}, "skills": {"dummy": True}, "abilities": {"dummy": True}, "knowledge": {"dummy": True}, "work_context": {"dummy": True}}
