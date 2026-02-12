# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embed O*NET
#
# Embed O*NET occupations with BGE-large sentence transformer
# (894x1024 float16, role + task descriptions).

# %%
#|default_exp nodes.embed_onet
#|export_as_func true

# %%
#|set_func_signature
def embed_onet(descriptions, print) -> {"onet_embeddings": dict}:
    """Embed O*NET occupations with BGE-large."""
    ...

# %%
#|export
print("embed_onet: returning dummy data")
return {"onet_embeddings": {"dummy": True}}
