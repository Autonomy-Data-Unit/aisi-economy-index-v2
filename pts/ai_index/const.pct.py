# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
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
llm_models_config_path = assets_path / "llm_models.toml"
embed_models_config_path = assets_path / "embed_models.toml"
adulib_cache_path = store_path / "adulib_cache"
onet_store_path = store_path / "onet"
adzuna_store_path = store_path / "adzuna"
