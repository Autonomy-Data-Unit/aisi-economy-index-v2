# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Score Task Exposure
#
# GPT task-level 3-level scoring (0=no change, 1=collaboration,
# 2=independent). 23,851 task-occupation pairs evaluated, aggregated
# to SOC level.

# %%
#|default_exp nodes.score_task_exposure
#|export_as_func true

# %%
#|set_func_signature
def score_task_exposure(tasks, print) -> {"task_scores": dict}:
    """Compute GPT task-level AI exposure scores, aggregated to SOC level."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args
set_node_func_args(score_task_exposure)

# %%
#|export
print("score_task_exposure: returning dummy data")
{"task_scores": {"dummy": True}}  #|func_return_line
