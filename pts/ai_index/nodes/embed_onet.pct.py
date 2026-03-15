# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # embed_onet
#
# Embed O*NET occupations as single rich documents (title + description +
# tasks + skills concatenated). Replaces the v1 node which produced separate
# role and taskskill embeddings.
#
# Handles document-side prompt support from `embed_models.toml`:
# - Fixed prefix (`document_prefix`): prepended to all texts
# - Named prompt (`document_prompt_name`): passed as `prompt_name`

# %%
#|default_exp nodes.embed_onet
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print) -> bool:

# %%
#|export
raise NotImplementedError("embed_onet v2 not yet implemented")
