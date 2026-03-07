# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Batch processor
#
# Generic chunked batch processing with DuckDB-backed resume and retry.
# Used by pipeline nodes that process large sets of IDs through LLMs,
# embeddings, or cosine similarity — any operation that can fail per-item
# and benefits from incremental persistence.

# %%
#|default_exp utils.batch

# %%
#|export
import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ai_index.utils.result_store import ResultStore


async def run_batched(
    all_ids: list,
    store: ResultStore,
    work_fn: Callable,
    *,
    batch_size: int,
    max_concurrent: int = 1,
    max_retries: int = 0,
    resume: bool = True,
    node_name: str = "batch",
    print_fn: Callable = print,
) -> dict:
    """Process IDs in batches with resume, concurrency, and retry support.

    Args:
        all_ids: Complete list of IDs to process.
        store: ResultStore for incremental persistence.
        work_fn: ``async (chunk_ids: list) -> pd.DataFrame`` — processes a
            chunk of IDs and returns a DataFrame matching the store schema.
            Must include the store's id and error columns.
        batch_size: Number of IDs per chunk.
        max_concurrent: Max concurrent work_fn calls.
        max_retries: Number of retry rounds for failed IDs (0 = no retries).
        resume: If True, skip IDs already in the store.
        node_name: Label for log messages.
        print_fn: Print function for progress logging.

    Returns:
        Summary dict with keys: n_total, n_success, n_failed, failed_ids,
        db_path.
    """
    # Determine remaining IDs
    if resume:
        done_ids = store.done_ids()
        remaining_ids = [i for i in all_ids if i not in done_ids]
        n_skipped = len(all_ids) - len(remaining_ids)
        if n_skipped:
            print_fn(f"{node_name}: resuming — {n_skipped}/{len(all_ids)} already done, {len(remaining_ids)} remaining")
    else:
        store.clear()
        remaining_ids = list(all_ids)

    n_total = len(all_ids)
    print_fn(f"{node_name}: {n_total} total, {len(remaining_ids)} to process (batch_size={batch_size}, max_concurrent={max_concurrent})")

    # Process chunks
    sem = asyncio.Semaphore(max_concurrent)
    chunks = [
        remaining_ids[i : i + batch_size]
        for i in range(0, len(remaining_ids), batch_size)
    ]
    n_chunks = len(chunks)

    async def _process(chunk_ids, chunk_num):
        async with sem:
            print_fn(f"{node_name}: chunk {chunk_num}/{n_chunks} ({len(chunk_ids)} items)")
            df = await work_fn(chunk_ids)
            n_ok = int((df[store.error_col].isna()).sum())
            n_err = len(df) - n_ok
            print_fn(f"{node_name}: chunk {chunk_num} done — {n_ok} ok, {n_err} failed")
            return df

    for coro in asyncio.as_completed([
        _process(chunk_ids, i + 1)
        for i, chunk_ids in enumerate(chunks)
    ]):
        chunk_df = await coro
        store.insert(chunk_df)

    # Retry failed IDs
    for retry_num in range(1, max_retries + 1):
        retry_ids = store.failed_ids()
        if not retry_ids:
            print_fn(f"{node_name}: no failures to retry")
            break

        print_fn(f"{node_name}: retry {retry_num}/{max_retries} — {len(retry_ids)} failed")
        store.delete_ids(retry_ids)

        retry_chunks = [
            retry_ids[i : i + batch_size]
            for i in range(0, len(retry_ids), batch_size)
        ]

        async def _retry(chunk_ids):
            async with sem:
                return await work_fn(chunk_ids)

        for coro in asyncio.as_completed([
            _retry(chunk_ids) for chunk_ids in retry_chunks
        ]):
            chunk_df = await coro
            store.insert(chunk_df)

        retry_ok = len([i for i in retry_ids if i not in set(store.failed_ids())])
        retry_err = len(retry_ids) - retry_ok
        print_fn(f"{node_name}: retry {retry_num} done — {retry_ok} recovered, {retry_err} still failed")

    # Summary scoped to all_ids
    all_ids_set = set(all_ids)
    done = store.done_ids() & all_ids_set
    failed = [i for i in store.failed_ids() if i in all_ids_set]
    n_success = len(done) - len(failed)
    n_failed = len(failed)

    print_fn(f"{node_name}: {n_success} succeeded, {n_failed} failed out of {n_total}")
    if failed:
        print_fn(f"{node_name}: failed IDs: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    return {
        "db_path": store.db_path,
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "failed_ids": failed,
    }
