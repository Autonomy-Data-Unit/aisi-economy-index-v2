# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HuggingFace Model Helpers
#
# Orchestration functions (run locally, SSH to Isambard) for pre-downloading
# HuggingFace models, plus compute-node functions for loading models in SBATCH
# jobs. Compute nodes have no internet — models must be pre-cached on login nodes.

# %%
#|default_exp models

# %%
#|export
import os
from dataclasses import dataclass, field
from typing import Any

from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import run as ssh_run, _get_config

# %% [markdown]
# ## Orchestration (run locally, SSH to Isambard)

# %%
#|export
def _shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)

# %%
#|export
def _hf_cache_dir(config: IsambardConfig) -> str:
    """Return the HF cache directory on the remote."""
    return config.hf_cache_dir

# %%
#|export
def _remote_python(script: str, *, config: IsambardConfig, timeout: int = 600) -> str:
    """Run a Python script in the remote venv and return stdout."""
    cmd = (
        f"cd {config.project_dir} && source .venv/bin/activate && "
        f"python -c {_shlex_quote(script)}"
    )
    result = ssh_run(f"bash -lc {_shlex_quote(cmd)}", config=config, timeout=timeout)
    return result.stdout

# %%
#|export
def ensure_model(model_name: str, *, config: IsambardConfig | None = None,
                 token: str | None = None, timeout: int = 1800) -> str:
    """Pre-download a HuggingFace model to the Isambard cache via the login node.

    Uses huggingface_hub.snapshot_download() in the remote venv. Returns the
    remote snapshot path.

    Args:
        model_name: HuggingFace model ID (e.g. "BAAI/bge-large-en-v1.5").
        config: Isambard configuration.
        token: Optional HuggingFace token for gated models.
        timeout: SSH timeout in seconds (default 30 minutes for large models).
    """
    config = _get_config(config)
    cache_dir = _hf_cache_dir(config)

    token_arg = f", token={token!r}" if token else ""
    script = (
        f"from huggingface_hub import snapshot_download; "
        f"path = snapshot_download({model_name!r}, cache_dir={cache_dir!r}{token_arg}); "
        f"print(path)"
    )
    stdout = _remote_python(script, config=config, timeout=timeout)
    return stdout.strip().split("\n")[-1]

# %%
#|export
def check_model(model_name: str, *, config: IsambardConfig | None = None) -> bool:
    """Check if a HuggingFace model is already cached on Isambard.

    Does a quick SSH `test -d` on the expected snapshot directory.

    Args:
        model_name: HuggingFace model ID.
        config: Isambard configuration.
    """
    config = _get_config(config)
    cache_dir = _hf_cache_dir(config)
    # HF hub stores snapshots under models--org--name
    model_dir = model_name.replace("/", "--")
    path = f"{cache_dir}/models--{model_dir}"
    result = ssh_run(f"test -d {_shlex_quote(path)}", config=config, check=False)
    return result.returncode == 0

# %% [markdown]
# ## Compute-node functions (run on Isambard in SBATCH jobs)
#
# All `torch`/`transformers`/`sentence-transformers` imports are **lazy**
# (inside function bodies) since they're only available on Isambard.

# %%
#|export
_DTYPE_MAP = {
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
}

def _resolve_dtype(dtype: str) -> Any:
    """Resolve a string dtype to a torch dtype. Lazy torch import."""
    import torch
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype {dtype!r}, expected one of {list(mapping)}")
    return mapping[dtype]

# %%
#|export
def set_model_env(hf_cache_dir: str | None = None) -> None:
    """Set environment variables for offline HuggingFace model loading.

    Should be called before loading any model on compute nodes (no internet).

    Args:
        hf_cache_dir: Override HF cache directory. If None, uses HF_HOME if set.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if hf_cache_dir is not None:
        os.environ["HF_HOME"] = hf_cache_dir

# %%
#|export
@dataclass
class EmbeddingModel:
    """Wrapper around a SentenceTransformer model with a convenience encode method."""
    model: Any
    model_name: str
    device: str
    dtype: str

    def encode(self, texts: list[str], batch_size: int = 64, **kwargs) -> Any:
        """Encode texts into embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.
            **kwargs: Additional arguments passed to SentenceTransformer.encode().

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=kwargs.pop("show_progress_bar", len(texts) > batch_size),
            **kwargs,
        )

# %%
#|export
def load_embedding_model(model_name: str = "BAAI/bge-large-en-v1.5", *,
                         device: str = "cuda",
                         dtype: str = "float16") -> EmbeddingModel:
    """Load a SentenceTransformer embedding model.

    Args:
        model_name: HuggingFace model ID.
        device: Device to load onto ("cuda", "cpu").
        dtype: Model precision ("float16", "bfloat16", "float32").

    Returns:
        EmbeddingModel wrapping the loaded SentenceTransformer.
    """
    set_model_env()
    import torch
    from sentence_transformers import SentenceTransformer

    torch_dtype = _resolve_dtype(dtype)
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    return EmbeddingModel(model=model, model_name=model_name, device=device, dtype=dtype)

# %%
#|export
@dataclass
class LLM:
    """Wrapper around a causal LM with tokenizer and a convenience generate method."""
    model: Any
    tokenizer: Any
    model_name: str
    device: str
    dtype: str

    def generate(self, prompts: list[str] | str, *,
                 max_new_tokens: int = 60,
                 system_message: str | None = None,
                 **kwargs) -> list[str]:
        """Generate text completions for one or more prompts.

        Applies the chat template if the tokenizer has one and system_message
        is provided or the tokenizer expects chat format.

        Args:
            prompts: Single prompt string or list of prompts.
            max_new_tokens: Maximum tokens to generate per prompt.
            system_message: Optional system message prepended to each prompt.
            **kwargs: Additional arguments passed to model.generate().

        Returns:
            List of generated response strings (new tokens only).
        """
        import torch

        if isinstance(prompts, str):
            prompts = [prompts]

        # Build chat messages and apply template
        all_texts = []
        for prompt in prompts:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(text)

        # Tokenize with left-padding
        inputs = self.tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.pop("do_sample", False),
                **kwargs,
            )

        # Decode only new tokens
        responses = []
        for output in outputs:
            new_tokens = output[input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(text.strip())

        return responses

# %%
#|export
def load_llm(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", *,
             device: str = "cuda",
             dtype: str = "float16") -> LLM:
    """Load a causal language model with tokenizer.

    Args:
        model_name: HuggingFace model ID.
        device: Device to load onto ("cuda", "cpu").
        dtype: Model precision ("float16", "bfloat16", "float32").

    Returns:
        LLM wrapping the loaded model and tokenizer.
    """
    set_model_env()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _resolve_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    model.eval()

    return LLM(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
