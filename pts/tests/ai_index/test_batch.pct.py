# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tests: ResultStore and run_batched

# %%
#|default_exp ai_index.test_batch

# %%
#|export
import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_index.utils.result_store import ResultStore
from ai_index.utils.batch import run_batched

# %% [markdown]
# ## ResultStore tests

# %%
#|export
class TestResultStore:
    def _make_store(self, tmp_path, columns=None):
        if columns is None:
            columns = {"id": "BIGINT NOT NULL", "data": "VARCHAR NOT NULL", "error": "VARCHAR"}
        return ResultStore(tmp_path / "test.duckdb", columns)

    def test_insert_and_done_ids(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({"id": [1, 2, 3], "data": ["a", "b", "c"], "error": [None, None, None]})
        store.insert(df)
        assert store.done_ids() == {1, 2, 3}
        store.close()

    def test_failed_ids(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({"id": [1, 2, 3], "data": ["a", "b", "c"], "error": [None, "bad", None]})
        store.insert(df)
        assert store.failed_ids() == [2]
        store.close()

    def test_counts(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({"id": [1, 2, 3], "data": ["a", "b", "c"], "error": [None, "bad", None]})
        store.insert(df)
        assert store.counts() == (2, 1)
        store.close()

    def test_delete_ids(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({"id": [1, 2, 3], "data": ["a", "b", "c"], "error": [None, None, None]})
        store.insert(df)
        store.delete_ids([2])
        assert store.done_ids() == {1, 3}
        store.close()

    def test_clear(self, tmp_path):
        store = self._make_store(tmp_path)
        df = pd.DataFrame({"id": [1, 2], "data": ["a", "b"], "error": [None, None]})
        store.insert(df)
        store.clear()
        assert store.done_ids() == set()
        store.close()

    def test_context_manager(self, tmp_path):
        with self._make_store(tmp_path) as store:
            df = pd.DataFrame({"id": [1], "data": ["a"], "error": [None]})
            store.insert(df)
            assert store.done_ids() == {1}

    def test_custom_columns_embedding(self, tmp_path):
        columns = {"id": "BIGINT NOT NULL", "embedding": "FLOAT[]", "error": "VARCHAR"}
        store = ResultStore(tmp_path / "embed.duckdb", columns)
        df = pd.DataFrame({
            "id": [1, 2],
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "error": [None, None],
        })
        store.insert(df)
        assert store.done_ids() == {1, 2}
        assert store.counts() == (2, 0)
        store.close()

    def test_custom_columns_cosine(self, tmp_path):
        columns = {
            "id": "BIGINT NOT NULL",
            "indices": "INT[]",
            "scores": "FLOAT[]",
            "error": "VARCHAR",
        }
        store = ResultStore(tmp_path / "cosine.duckdb", columns)
        df = pd.DataFrame({
            "id": [1],
            "indices": [[10, 20, 30]],
            "scores": [[0.9, 0.8, 0.7]],
            "error": [None],
        })
        store.insert(df)
        assert store.done_ids() == {1}
        store.close()

    def test_resume_across_connections(self, tmp_path):
        db_path = tmp_path / "resume.duckdb"
        columns = {"id": "BIGINT NOT NULL", "data": "VARCHAR NOT NULL", "error": "VARCHAR"}
        store1 = ResultStore(db_path, columns)
        store1.insert(pd.DataFrame({"id": [1, 2], "data": ["a", "b"], "error": [None, None]}))
        store1.close()

        store2 = ResultStore(db_path, columns)
        assert store2.done_ids() == {1, 2}
        store2.close()

# %% [markdown]
# ## run_batched tests

# %%
#|export
class TestRunBatched:
    def _make_store(self, tmp_path, columns=None):
        if columns is None:
            columns = {"id": "BIGINT NOT NULL", "data": "VARCHAR NOT NULL", "error": "VARCHAR"}
        return ResultStore(tmp_path / "test.duckdb", columns)

    @pytest.mark.asyncio
    async def test_processes_all_ids(self, tmp_path):
        store = self._make_store(tmp_path)

        async def work_fn(chunk_ids):
            return pd.DataFrame({
                "id": chunk_ids,
                "data": [f"result_{i}" for i in chunk_ids],
                "error": [None] * len(chunk_ids),
            })

        result = await run_batched(
            [1, 2, 3, 4, 5], store, work_fn,
            batch_size=2, node_name="test",
        )
        assert result["n_total"] == 5
        assert result["n_success"] == 5
        assert result["n_failed"] == 0
        store.close()

    @pytest.mark.asyncio
    async def test_resume_skips_done(self, tmp_path):
        store = self._make_store(tmp_path)
        store.insert(pd.DataFrame({
            "id": [1, 2], "data": ["a", "b"], "error": [None, None],
        }))

        calls = []
        async def work_fn(chunk_ids):
            calls.append(chunk_ids)
            return pd.DataFrame({
                "id": chunk_ids,
                "data": [f"result_{i}" for i in chunk_ids],
                "error": [None] * len(chunk_ids),
            })

        result = await run_batched(
            [1, 2, 3, 4], store, work_fn,
            batch_size=10, resume=True, node_name="test",
        )
        assert result["n_success"] == 4
        # Only IDs 3, 4 should have been processed
        assert len(calls) == 1
        assert set(calls[0]) == {3, 4}
        store.close()

    @pytest.mark.asyncio
    async def test_resume_false_clears(self, tmp_path):
        store = self._make_store(tmp_path)
        store.insert(pd.DataFrame({
            "id": [1, 2], "data": ["a", "b"], "error": [None, None],
        }))

        async def work_fn(chunk_ids):
            return pd.DataFrame({
                "id": chunk_ids,
                "data": [f"result_{i}" for i in chunk_ids],
                "error": [None] * len(chunk_ids),
            })

        result = await run_batched(
            [1, 2], store, work_fn,
            batch_size=10, resume=False, node_name="test",
        )
        assert result["n_success"] == 2
        store.close()

    @pytest.mark.asyncio
    async def test_retries_failed(self, tmp_path):
        store = self._make_store(tmp_path)
        attempt = [0]

        async def work_fn(chunk_ids):
            attempt[0] += 1
            errors = []
            for i in chunk_ids:
                # Fail id=2 on first attempt, succeed on retry
                if i == 2 and attempt[0] == 1:
                    errors.append("transient error")
                else:
                    errors.append(None)
            return pd.DataFrame({
                "id": chunk_ids,
                "data": [f"result_{i}" for i in chunk_ids],
                "error": errors,
            })

        result = await run_batched(
            [1, 2, 3], store, work_fn,
            batch_size=10, max_retries=1, node_name="test",
        )
        assert result["n_success"] == 3
        assert result["n_failed"] == 0
        store.close()

    @pytest.mark.asyncio
    async def test_scoped_to_all_ids(self, tmp_path):
        """Stale rows from other runs don't inflate counts."""
        store = self._make_store(tmp_path)
        # Insert stale rows from a "previous run"
        store.insert(pd.DataFrame({
            "id": [100, 200], "data": ["old", "old"], "error": [None, None],
        }))

        async def work_fn(chunk_ids):
            return pd.DataFrame({
                "id": chunk_ids,
                "data": [f"result_{i}" for i in chunk_ids],
                "error": [None] * len(chunk_ids),
            })

        result = await run_batched(
            [1, 2, 3], store, work_fn,
            batch_size=10, node_name="test",
        )
        assert result["n_total"] == 3
        assert result["n_success"] == 3
        store.close()

    @pytest.mark.asyncio
    async def test_concurrent_batches(self, tmp_path):
        store = self._make_store(tmp_path)
        max_concurrent = [0]
        current = [0]

        async def work_fn(chunk_ids):
            current[0] += 1
            max_concurrent[0] = max(max_concurrent[0], current[0])
            await asyncio.sleep(0.01)
            current[0] -= 1
            return pd.DataFrame({
                "id": chunk_ids,
                "data": [f"r_{i}" for i in chunk_ids],
                "error": [None] * len(chunk_ids),
            })

        await run_batched(
            list(range(10)), store, work_fn,
            batch_size=2, max_concurrent=3, node_name="test",
        )
        assert max_concurrent[0] <= 3
        assert store.counts() == (10, 0)
        store.close()
