# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM Generate
#
# Run text generation on a list of prompts using transformers, vLLM, or API backends.

# %%
#|default_exp llm

# %%
#|export
from llm_runner.models import load_llm

# %%
#|export
def run_llm_generate(prompts: list[str], *, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                     device: str = "cuda", dtype: str = "float16",
                     backend: str = "transformers", max_new_tokens: int = 60,
                     batch_size: int = 128,
                     system_message: str | None = None,
                     json_schema: dict | None = None) -> list[str]:
    """Generate text completions for a list of prompts.

    Args:
        prompts: List of prompt strings.
        model_name: HuggingFace model ID (or API model name for api backend).
        device: Device ("cuda", "cpu"). Ignored for api backend.
        dtype: Model precision. Ignored for api backend.
        backend: "transformers", "vllm", or "api".
        max_new_tokens: Maximum tokens to generate per prompt.
        batch_size: Number of prompts per batch (transformers backend only).
        system_message: Optional system message prepended to each prompt.
        json_schema: Optional JSON schema dict to constrain output to valid JSON.

    Returns:
        List of generated response strings, one per prompt.
    """
    llm = load_llm(model_name, device=device, dtype=dtype, backend=backend)

    # For api and vllm backends, send all prompts at once (they handle batching internally)
    if backend in ("api", "vllm"):
        return llm.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            system_message=system_message,
            json_schema=json_schema,
        )

    # For transformers backend, batch manually
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        responses = llm.generate(
            batch,
            max_new_tokens=max_new_tokens,
            system_message=system_message,
            json_schema=json_schema,
        )
        all_responses.extend(responses)

    return all_responses
