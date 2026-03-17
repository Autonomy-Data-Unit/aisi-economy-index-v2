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
rerank_models_config_path = config_path / "rerank_models.toml"
netrun_config_path = config_path / "netrun.json"
run_defs_path = config_path / "run_defs.toml"

adulib_cache_path = store_path / "adulib_cache"
onet_store_path = inputs_path / "onet"
onet_targets_path = inputs_path / "onet_targets.parquet"
adzuna_db_path = inputs_path / "adzuna.duckdb"
adzuna_store_path = inputs_path / "adzuna"  # legacy parquet store
lad22_lookup_path = inputs_path / "lad22_lookup.csv"
pipeline_store_path = store_path / "pipeline"
logs_path = store_path / "logs"
calibration_results_path = store_path / "calibration" / "results"
isambard_config_path = config_path / "isambard.toml"
validation_config_path = config_path / "validation.toml"
validation_results_path = store_path / "validation" / "results"
outputs_path = store_path / "outputs"
onet_exposure_scores_path = outputs_path / "onet_exposure_scores"
aspectt_vectors_path = inputs_path / "aspectt_vectors"

def rel(path: Path) -> Path:
    """Return path relative to repo root, for cleaner log output."""
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path

# %%
#|export
import adulib.caching
adulib.caching.set_default_cache_path(adulib_cache_path)
