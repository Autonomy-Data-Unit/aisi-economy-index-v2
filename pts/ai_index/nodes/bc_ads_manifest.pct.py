# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Broadcast ads_manifest
#
# Fan-out node: duplicates ads_manifest to embed_job_ads and llm_filter_candidates.

# %%
#|default_exp nodes.bc_ads_manifest
#|export_as_func true

# %%
#|set_func_signature
def main(ads_manifest) -> {"ads_manifest_embed": dict, "ads_manifest_llm": dict}:
    """Broadcast ads_manifest to embedding and LLM filter nodes."""
    ...

# %%
#|export
{"ads_manifest_embed": ads_manifest, "ads_manifest_llm": ads_manifest}  #|func_return_line
