# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model config internals

# %%
#|default_exp utils._model_config

# %%
#|export
import tomllib
from pathlib import Path

def _load_model_config(config_path: Path, model_key: str) -> tuple[str, dict]:
    """Load config for a named model key.

    Looks up cfg["models"][model_key], reads its mode, merges
    cfg["defaults"][mode] with the model entry (minus 'mode' key).
    Returns (mode, merged_dict).
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    models = cfg.get("models", {})
    if model_key not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(f"Unknown model key {model_key!r}. Available: {available}")
    entry = dict(models[model_key])
    mode = entry.pop("mode")
    defaults = dict(cfg.get("defaults", {}).get(mode, {}))
    defaults.update(entry)
    return mode, defaults

# %%
#|export
# Keys that belong to run_remote/arun_remote (orchestration), not to the
# underlying llm_runner operation. _split_remote_kwargs() pops these from the
# config dict so they're passed to run_remote, while the rest goes to the
# remote llm_runner CLI as config_dict.
_RUN_REMOTE_KEYS = {
    "setup", "job_name", "time", "transfer_modes",
    "output_transfer", "isambard_config", "print_fn",
}

def _resolve_model_args(config_path, model_key, kwargs):
    """Resolve config + kwargs into (mode, model_name, cfg) tuple."""
    mode, cfg = _load_model_config(config_path, model_key)
    cfg.update(kwargs)
    model_name = cfg.pop("model")
    return mode, model_name, cfg

def _split_remote_kwargs(cfg):
    """Pop run_remote keys from cfg and return them as a separate dict."""
    return {k: cfg.pop(k) for k in list(cfg) if k in _RUN_REMOTE_KEYS}
