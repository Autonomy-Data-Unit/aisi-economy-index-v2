# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reranking
#
# Run cross-encoder reranking: given queries and candidate documents, score
# each (query, document) pair and return top-K indices and scores per query.
#
# Supports two backends:
# - `cross-encoder`: Classic cross-encoder via sentence-transformers CrossEncoder
# - `vllm`: Generative reranker via vLLM (e.g. Qwen3-Reranker). Scores pairs by
#   extracting yes/no logprobs from the model's output.

# %%
#|default_exp rerank

# %%
#|export
import math

import numpy as np

# %%
#|export
def _rerank_cross_encoder(
    queries: list[str],
    documents: list[str],
    top_k: int,
    *,
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
) -> dict:
    """Rerank using a classic cross-encoder (sentence-transformers)."""
    from llm_runner.models import load_rerank_model

    model = load_rerank_model(model_name, device=device, dtype=dtype)
    n_queries = len(queries)
    n_docs = len(documents)
    top_k = min(top_k, n_docs)

    all_indices = np.zeros((n_queries, top_k), dtype=np.int64)
    all_scores = np.zeros((n_queries, top_k), dtype=np.float32)

    for qi in range(n_queries):
        pairs = [(queries[qi], doc) for doc in documents]
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = model.predict(batch, batch_size=batch_size)
            scores.extend(batch_scores)

        scores = np.array(scores, dtype=np.float32)
        top_idx = np.argsort(-scores)[:top_k]
        all_indices[qi] = top_idx
        all_scores[qi] = scores[top_idx]

        if (qi + 1) % 100 == 0 or qi == n_queries - 1:
            print(f"  rerank [cross-encoder]: {qi + 1}/{n_queries} queries done")

    return {"indices": all_indices, "scores": all_scores}

# %%
#|export
def _init_vllm_reranker(model_name, tokenizer, tensor_parallel_size, dtype, vllm_prompt_style, max_model_len):
    """Initialise vLLM engine and scoring parameters for a reranker model."""
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        dtype="half" if dtype == "float16" else dtype,
    )

    if vllm_prompt_style == "qwen":
        true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=20,
            allowed_token_ids=[true_token, false_token],
        )
    elif vllm_prompt_style == "bge-gemma":
        true_token = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        false_token = None
        suffix_tokens = []
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=20,
        )
    else:
        raise ValueError(f"Unknown vllm_prompt_style: {vllm_prompt_style!r}")

    return engine, sampling_params, true_token, false_token, suffix_tokens


def _build_vllm_prompt(query, doc, tokenizer, instruction, vllm_prompt_style, suffix_tokens, max_model_len):
    """Build a single tokenised prompt for a (query, doc) pair."""
    from vllm.inputs.data import TokensPrompt

    if vllm_prompt_style == "qwen":
        messages = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"},
        ]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        max_prompt_len = max_model_len - len(suffix_tokens)
        token_ids = token_ids[:max_prompt_len] + suffix_tokens
    elif vllm_prompt_style == "bge-gemma":
        text = f"A: {query}\nB: {doc}\n{instruction}"
        token_ids = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)
        token_ids = token_ids[:max_model_len]

    return TokensPrompt(prompt_token_ids=token_ids)


def _extract_vllm_score(output, true_token, false_token, vllm_prompt_style):
    """Extract a relevance score from a vLLM generation output."""
    logprobs = output.outputs[0].logprobs[-1]

    if vllm_prompt_style == "qwen":
        true_logit = logprobs[true_token].logprob if true_token in logprobs else -10
        false_logit = logprobs[false_token].logprob if false_token in logprobs else -10
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        return true_score / (true_score + false_score)
    elif vllm_prompt_style == "bge-gemma":
        return math.exp(logprobs[true_token].logprob) if true_token in logprobs else 0.0


def _rerank_vllm(
    queries: list[str],
    documents: list[str],
    top_k: int,
    *,
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    tensor_parallel_size: int = 1,
    vllm_prompt_style: str = "qwen",
    max_model_len: int = 8192,
) -> dict:
    """Rerank using a generative reranker via vLLM.

    Supports multiple prompt styles:
    - "qwen": Qwen3-Reranker (chat template, yes/no logprob ratio)
    - "bge-gemma": BGE Gemma reranker (raw text A:/B: format, Yes logprob)
    """
    from llm_runner.models import set_model_env
    set_model_env()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    engine, sampling_params, true_token, false_token, suffix_tokens = _init_vllm_reranker(
        model_name, tokenizer, tensor_parallel_size, dtype, vllm_prompt_style, max_model_len,
    )

    n_queries = len(queries)
    n_docs = len(documents)
    top_k = min(top_k, n_docs)

    all_indices = np.zeros((n_queries, top_k), dtype=np.int64)
    all_scores = np.zeros((n_queries, top_k), dtype=np.float32)

    for qi in range(n_queries):
        token_prompts = [
            _build_vllm_prompt(queries[qi], doc, tokenizer, instruction, vllm_prompt_style, suffix_tokens, max_model_len)
            for doc in documents
        ]

        scores = []
        for i in range(0, len(token_prompts), batch_size):
            batch = token_prompts[i:i + batch_size]
            outputs = engine.generate(batch, sampling_params, use_tqdm=False)
            for output in outputs:
                scores.append(_extract_vllm_score(output, true_token, false_token, vllm_prompt_style))

        scores = np.array(scores, dtype=np.float32)
        top_idx = np.argsort(-scores)[:top_k]
        all_indices[qi] = top_idx
        all_scores[qi] = scores[top_idx]

        if (qi + 1) % 10 == 0 or qi == n_queries - 1:
            print(f"  rerank [vllm/{vllm_prompt_style}]: {qi + 1}/{n_queries} queries done")

    return {"indices": all_indices, "scores": all_scores}

# %%
#|export
def run_rerank(
    queries: list[str],
    documents: list[str],
    top_k: int = 10,
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cuda",
    dtype: str = "float16",
    batch_size: int = 64,
    backend: str = "cross-encoder",
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    tensor_parallel_size: int = 1,
    vllm_prompt_style: str = "qwen",
    max_model_len: int = 8192,
) -> dict:
    """Rerank documents for each query using a cross-encoder or generative reranker.

    Each query is scored against all documents. Returns the top-K document
    indices and scores per query, sorted by descending score.

    Args:
        queries: List of query texts (e.g. job ad texts).
        documents: List of document texts (e.g. O*NET occupation descriptions).
        top_k: Number of top documents to return per query.
        model_name: HuggingFace model ID.
        device: Device ("cuda", "cpu").
        dtype: Model precision ("float16", "bfloat16", "float32").
        batch_size: Batch size for scoring pairs.
        backend: "cross-encoder" for sentence-transformers CrossEncoder,
                 "vllm" for generative reranker (e.g. Qwen3-Reranker).
        instruction: Task instruction for generative rerankers (vllm backend only).
        tensor_parallel_size: Number of GPUs for vLLM (vllm backend only).

    Returns:
        Dict with:
        - "indices": np.ndarray of shape (n_queries, top_k), int64
        - "scores": np.ndarray of shape (n_queries, top_k), float32
    """
    if backend == "cross-encoder":
        return _rerank_cross_encoder(
            queries, documents, top_k,
            model_name=model_name, device=device, dtype=dtype, batch_size=batch_size,
        )
    elif backend == "vllm":
        return _rerank_vllm(
            queries, documents, top_k,
            model_name=model_name, device=device, dtype=dtype, batch_size=batch_size,
            instruction=instruction, tensor_parallel_size=tensor_parallel_size,
            vllm_prompt_style=vllm_prompt_style, max_model_len=max_model_len,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'cross-encoder' or 'vllm'.")

# %%
#|export
def _rerank_pairs_cross_encoder(
    items,
    *,
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
) -> list[list[float]]:
    """Score pre-paired (query, documents) items using a cross-encoder.

    Flattens all pairs into a single batch for efficient scoring, then splits
    scores back by group boundaries.
    """
    from llm_runner.models import load_rerank_model

    model = load_rerank_model(model_name, device=device, dtype=dtype)

    # Flatten all (query, doc) pairs, tracking group boundaries
    pairs = []
    boundaries = [0]
    for item in items:
        query, docs = item[0], item[1]
        for doc in docs:
            pairs.append((query, doc))
        boundaries.append(len(pairs))

    print(f"  rerank_pairs [cross-encoder]: {len(items)} items, {len(pairs)} total pairs")

    # Score all pairs in batches
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_scores = model.predict(batch, batch_size=batch_size)
        all_scores.extend(batch_scores)

    # Split back by group
    return [
        [float(s) for s in all_scores[boundaries[i]:boundaries[i + 1]]]
        for i in range(len(items))
    ]

# %%
#|export
def _rerank_pairs_vllm(
    items,
    *,
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    tensor_parallel_size: int = 1,
    vllm_prompt_style: str = "qwen",
    max_model_len: int = 8192,
) -> list[list[float]]:
    """Score pre-paired (query, documents) items using vLLM.

    Flattens all pairs into a single batch for efficient scoring, then splits
    scores back by group boundaries.
    """
    from llm_runner.models import set_model_env
    set_model_env()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    engine, sampling_params, true_token, false_token, suffix_tokens = _init_vllm_reranker(
        model_name, tokenizer, tensor_parallel_size, dtype, vllm_prompt_style, max_model_len,
    )

    # Build all prompts, tracking group boundaries
    all_prompts = []
    boundaries = [0]
    for item in items:
        query, docs = item[0], item[1]
        for doc in docs:
            all_prompts.append(
                _build_vllm_prompt(query, doc, tokenizer, instruction, vllm_prompt_style, suffix_tokens, max_model_len)
            )
        boundaries.append(len(all_prompts))

    print(f"  rerank_pairs [vllm/{vllm_prompt_style}]: {len(items)} items, {len(all_prompts)} total pairs")

    # Score all pairs in batches
    all_scores = []
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i:i + batch_size]
        outputs = engine.generate(batch, sampling_params, use_tqdm=False)
        for output in outputs:
            all_scores.append(_extract_vllm_score(output, true_token, false_token, vllm_prompt_style))

    # Split back by group
    return [
        all_scores[boundaries[i]:boundaries[i + 1]]
        for i in range(len(items))
    ]

# %%
#|export
def run_rerank_pairs(
    items,
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cuda",
    dtype: str = "float16",
    batch_size: int = 64,
    backend: str = "cross-encoder",
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    tensor_parallel_size: int = 1,
    vllm_prompt_style: str = "qwen",
    max_model_len: int = 8192,
) -> list[list[float]]:
    """Score grouped (query, documents) pairs efficiently in a single batch.

    Unlike run_rerank which computes the full cross-product of queries x documents,
    this function scores only the specified per-query candidate documents. This avoids
    redundant pairs and allows batching all items into a single GPU call.

    Args:
        items: List of (query, documents) pairs. Each element is a tuple/list where
               item[0] is the query text and item[1] is a list of document texts.
               JSON-safe (tuples round-trip to lists).
        model_name: HuggingFace model ID.
        device: Device ("cuda", "cpu").
        dtype: Model precision ("float16", "bfloat16", "float32").
        batch_size: Batch size for scoring pairs.
        backend: "cross-encoder" or "vllm".
        instruction: Task instruction for vllm backend.
        tensor_parallel_size: Number of GPUs for vLLM.
        vllm_prompt_style: "qwen" (chat template, yes/no ratio) or
                           "bge-gemma" (raw A:/B: text, Yes logprob).
        max_model_len: Maximum sequence length for vLLM.

    Returns:
        list[list[float]]: Scores per item, matching document order in each item.
    """
    if backend == "cross-encoder":
        return _rerank_pairs_cross_encoder(
            items, model_name=model_name, device=device, dtype=dtype, batch_size=batch_size,
        )
    elif backend == "vllm":
        return _rerank_pairs_vllm(
            items, model_name=model_name, device=device, dtype=dtype, batch_size=batch_size,
            instruction=instruction, tensor_parallel_size=tensor_parallel_size,
            vllm_prompt_style=vllm_prompt_style, max_model_len=max_model_len,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'cross-encoder' or 'vllm'.")
