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
from dotenv import load_dotenv
load_dotenv()

# %%
#|export
from pathlib import Path
import ai_index

# Repo root: const.py lives at src/ai_index/const.py -> ../../
repo_root = Path(ai_index.__file__).parent.parent.parent

config_path = repo_root / "config"
store_path = repo_root / "store"
inputs_path = store_path / "inputs"

llm_models_config_path = config_path / "llm_models.toml"
embed_models_config_path = config_path / "embed_models.toml"
netrun_config_path = config_path / "netrun.json"
run_defs_path = config_path / "run_defs.toml"

adulib_cache_path = store_path / "adulib_cache"
onet_store_path = inputs_path / "onet"
onet_targets_path = inputs_path / "onet_targets.parquet"
adzuna_db_path = inputs_path / "adzuna.duckdb"
adzuna_store_path = inputs_path / "adzuna"  # legacy parquet store
pipeline_store_path = store_path / "pipeline"
onet_exposure_scores_path = store_path / "onet_exposure_scores"
aspectt_vectors_path = inputs_path / "aspectt_vectors"

# %%
#|export
import adulib.caching
adulib.caching.set_default_cache_path(adulib_cache_path)
