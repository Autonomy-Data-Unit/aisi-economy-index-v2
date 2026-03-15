# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # embed_ads
#
# Embed raw job ad text directly using the configured embedding model.
# No LLM summarisation needed. Replaces the v1 node which embedded LLM summaries.
#
# Reads ad texts from Adzuna DuckDB and embeds them, handling the three
# categories of prompt support from `embed_models.toml`:
# 1. Fixed prefix (`query_prefix`): prepended to all texts unconditionally
# 2. Named prompt (`query_prompt_name`): passed as `prompt_name` to encode()
# 3. Custom task instruction (`supports_prompt`): passed as `prompt` if the
#    model supports it and `embed_task_prompt` node var is set

# %%
#|default_exp nodes.embed_ads
#|export_as_func true

# %%
#|set_func_signature
async def main(ctx, print, ad_ids: "np.ndarray") -> {
    'ad_ids': "list[int]"
}:

# %%
#|export
raise NotImplementedError("embed_ads v2 not yet implemented")
