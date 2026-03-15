# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # llm_filter_candidates
#
# LLM-based final selection from reranked candidates. Reads the top-10
# from rerank_candidates and uses an LLM to make the final occupation
# assignment.
#
# Replaces v1 which read from cosine_match and used dual-channel scores
# (role_score, taskskill_score, combined_score). This v2 reads from
# rerank_candidates which has a single rerank_score.

# %%
#|default_exp nodes.llm_filter_candidates
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: "list[int]") -> {
    'successful_ad_ids': "list[int]"
}:

# %%
#|export
raise NotImplementedError("llm_filter_candidates v2 not yet implemented")
