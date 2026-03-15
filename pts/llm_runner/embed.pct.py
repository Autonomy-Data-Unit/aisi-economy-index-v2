# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Embedding
#
# Run embedding inference on a list of texts using SentenceTransformers.

# %%
#|default_exp embed

# %%
#|export
import numpy as np

from llm_runner.models import load_embedding_model

# %%
#|export
def run_embeddings(texts: list[str], *, model_name: str = "BAAI/bge-large-en-v1.5",
                   device: str = "cuda", dtype: str = "float16",
                   batch_size: int = 64,
                   prompt: str | None = None,
                   prompt_name: str | None = None,
                   st_kwargs: dict | None = None) -> np.ndarray:
    """Embed a list of texts using a SentenceTransformer model.

    Args:
        texts: List of strings to embed.
        model_name: HuggingFace model ID.
        device: Device ("cuda", "cpu").
        dtype: Model precision ("float16", "bfloat16", "float32").
        batch_size: Batch size for encoding.
        prompt: Optional instruction string prepended to each text. Used by
            instruction-following embedding models (e.g. Qwen3-Embedding).
        prompt_name: Optional named prompt from the model's config (e.g. "query").
            Mutually exclusive with prompt.
        st_kwargs: Extra kwargs passed to the SentenceTransformer constructor.

    Returns:
        numpy array of shape (len(texts), embedding_dim) with the specified dtype.
    """
    model = load_embedding_model(
        model_name, device=device, dtype=dtype,
        offline=(device != "cpu"),
        st_kwargs=st_kwargs,
    )
    encode_kwargs = {}
    if prompt is not None:
        encode_kwargs["prompt"] = prompt
    if prompt_name is not None:
        encode_kwargs["prompt_name"] = prompt_name
    embeddings = model.encode(texts, batch_size=batch_size, **encode_kwargs)

    # Cast to target dtype
    target_np_dtype = {"float16": np.float16, "bfloat16": np.float32, "float32": np.float32}[dtype]
    return embeddings.astype(target_np_dtype)
