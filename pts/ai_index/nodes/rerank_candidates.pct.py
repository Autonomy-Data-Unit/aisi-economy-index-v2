# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # rerank_candidates
#
# Rerank cosine candidates using Qwen3-Reranker (generative reranker on GPU).
# Takes the top-100 from cosine_candidates and produces a top-10 using
# cross-attention scoring with a domain-specific instruction.
#
# Uses the `arerank()` utility which routes to Isambard via sbatch.

# %%
#|default_exp nodes.rerank_candidates
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: list[int]) -> {
    'ad_ids': list[int]
}:

# %%
#|export
raise NotImplementedError("rerank_candidates not yet implemented")
