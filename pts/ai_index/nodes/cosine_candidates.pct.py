# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # cosine_candidates
#
# Compute cosine similarity between ad embeddings and O*NET embeddings,
# select top-N candidates per ad. Produces a wider candidate set (default
# top-100) for downstream reranking.
#
# Replaces the v1 `cosine_match` node which produced top-10 with dual-channel
# role/taskskill scoring.

# %%
#|default_exp nodes.cosine_candidates
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: "list[int]", onet_done: bool) -> {
    'ad_ids': "list[int]"
}:

# %%
#|export
raise NotImplementedError("cosine_candidates not yet implemented")
