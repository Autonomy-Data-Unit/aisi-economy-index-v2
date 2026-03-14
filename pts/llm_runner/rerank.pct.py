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

# %%
#|default_exp rerank

# %%
#|export
import numpy as np

from llm_runner.models import load_rerank_model

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
) -> dict:
    """Rerank documents for each query using a cross-encoder model.

    Each query is scored against all documents. Returns the top-K document
    indices and scores per query, sorted by descending score.

    Args:
        queries: List of query texts (e.g. job ad texts).
        documents: List of document texts (e.g. O*NET occupation descriptions).
        top_k: Number of top documents to return per query.
        model_name: HuggingFace cross-encoder model ID.
        device: Device ("cuda", "cpu").
        dtype: Model precision ("float16", "bfloat16", "float32").
        batch_size: Batch size for scoring pairs.

    Returns:
        Dict with:
        - "indices": np.ndarray of shape (n_queries, top_k), int64
        - "scores": np.ndarray of shape (n_queries, top_k), float32
    """
    model = load_rerank_model(model_name, device=device, dtype=dtype)

    n_queries = len(queries)
    n_docs = len(documents)
    top_k = min(top_k, n_docs)

    # Build all (query, document) pairs
    # Process one query at a time to keep memory bounded
    all_indices = np.zeros((n_queries, top_k), dtype=np.int64)
    all_scores = np.zeros((n_queries, top_k), dtype=np.float32)

    for qi in range(n_queries):
        pairs = [(queries[qi], doc) for doc in documents]

        # Score in batches
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
            print(f"  rerank: {qi + 1}/{n_queries} queries done")

    return {"indices": all_indices, "scores": all_scores}
