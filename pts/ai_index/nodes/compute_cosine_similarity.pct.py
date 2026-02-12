# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compute Cosine Similarity
#
# Top-K candidate O*NET matches per job ad (top-5 role + top-5 task,
# averaged overlaps).

# %%
#|default_exp nodes.compute_cosine_similarity
#|export_as_func true

# %%
#|set_func_signature
def compute_cosine_similarity(onet_descriptions, onet_embeddings, job_ad_embeddings, print) -> {"candidates": dict}:
    """Compute top-K cosine similarity matches between job ads and O*NET occupations."""
    ...

# %%
#|export
print("compute_cosine_similarity: returning dummy data")
return {"candidates": {"dummy": True}}
