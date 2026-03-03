# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # utils

# %%
#|default_exp utils

# %%
#|export
import tomllib
from ai_index.const import llm_models_config_path

# %%
#|export
def _load_llm_config(mode: str, model: str | None) -> dict:
    """Load config: {mode}.defaults -> {mode}."{model}" """
    with open(llm_models_config_path, "rb") as f:
        cfg = tomllib.load(f)
    mode_cfg = cfg.get(mode, {})
    result = dict(mode_cfg.get("defaults", {}))
    if model and model in mode_cfg:
        result.update(mode_cfg[model])
    return result

# %%
#|export
_RUN_REMOTE_KEYS = {
    "setup", "job_name", "time", "transfer_modes",
    "output_transfer", "isambard_config", "print_fn",
}

def llm_generate(
    prompts: list[str],
    *,
    mode: str = "api",
    model: str | None = None,
    **kwargs,
) -> list[str]:
    """Generate LLM responses using api, local, or sbatch mode.

    Config is loaded from llm_models.toml ({mode}.defaults -> {mode}."{model}"),
    then any explicit **kwargs override config values.
    """
    cfg = _load_llm_config(mode, model)
    cfg.update(kwargs)
    if model is None:
        model = cfg.pop("model", "Qwen/Qwen2.5-7B-Instruct")
    else:
        cfg.pop("model", None)

    if mode in ("api", "local"):
        from llm_runner.llm import run_llm_generate
        if mode == "api":
            cfg["backend"] = "api"
        return run_llm_generate(prompts, model_name=model, **cfg)

    elif mode == "sbatch":
        from isambard_utils.orchestrate import run_remote
        # Split cfg into run_remote kwargs vs config_dict (sent to llm_runner on remote)
        remote_kw = {k: cfg.pop(k) for k in list(cfg) if k in _RUN_REMOTE_KEYS}
        cfg["model_name"] = model
        result = run_remote(
            "llm_generate",
            inputs={"prompts": prompts},
            config_dict=cfg,
            required_models=[model],
            **remote_kw,
        )
        return result["responses"]

    else:
        raise ValueError(f"Unknown mode: {mode!r}")
