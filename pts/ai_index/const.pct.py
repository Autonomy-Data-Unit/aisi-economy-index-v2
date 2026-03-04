# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
#|default_exp const

# %% [markdown]
# # const

# %%
#|export
from importlib.resources import files
from pathlib import Path

pkg_path = Path(str(files("ai_index")))
assets_path = pkg_path / "assets"
store_path = pkg_path / "store"
inputs_path = store_path / "inputs"

llm_models_config_path = assets_path / "llm_models.toml"
embed_models_config_path = assets_path / "embed_models.toml"
adulib_cache_path = store_path / "adulib_cache"

onet_store_path = inputs_path / "onet"
adzuna_store_path = inputs_path / "adzuna"

pipeline_store_path = store_path / "pipeline"
run_defs_path = assets_path / "run_defs.toml"
