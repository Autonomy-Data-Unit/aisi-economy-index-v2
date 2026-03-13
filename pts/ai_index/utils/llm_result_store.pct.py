# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM result store

# %%
#|default_exp utils.llm_result_store

# %%
#|export
from pathlib import Path

import duckdb
import pandas as pd


class LLMResultStore:
    """DuckDB store for LLM JSON responses.

    Fixed schema: ``id`` (BIGINT), ``data`` (VARCHAR), ``error`` (VARCHAR).

    - ``data``: raw LLM response string (always stored, even on failure)
    - ``error``: NULL if valid, error message string if validation failed

    Reusable across any node that calls an LLM and stores JSON results.
    Use the Pydantic model only at call-time (for ``json_schema``) and
    at read-time (for validation/parsing). The store itself is schema-free.

    Usage::

        with LLMResultStore(db_path) as store:
            remaining = [i for i in all_ids if i not in store.done_ids()]
            ...
            store.insert(chunk_df)  # DataFrame with id, data, error columns
            ...
            n_ok, n_err = store.counts()
    """

    def __init__(self, db_path: str | Path, table: str = "results",
                 memory_limit: str | None = None):
        self.conn = duckdb.connect(str(db_path))
        if memory_limit is not None:
            self.conn.execute(f"SET memory_limit = '{memory_limit}'")
        self.table = table
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id BIGINT NOT NULL,
                data VARCHAR NOT NULL,
                error VARCHAR
            )
        """)

    def done_ids(self) -> set:
        """Return set of IDs already in the store."""
        return set(
            self.conn.execute(
                f"SELECT id FROM {self.table}"
            ).fetchnumpy()["id"].tolist()
        )

    def failed_ids(self) -> list:
        """Return list of IDs where error IS NOT NULL."""
        return self.conn.execute(
            f"SELECT id FROM {self.table} WHERE error IS NOT NULL"
        ).fetchnumpy()["id"].tolist()

    def insert(self, df: pd.DataFrame):
        """Insert a DataFrame with columns: id, data, error."""
        self.conn.execute(
            f"INSERT INTO {self.table} BY NAME SELECT * FROM df"
        )

    def delete_ids(self, ids: list):
        """Delete rows by ID."""
        self.conn.execute(
            f"CREATE OR REPLACE TEMP TABLE _del_ids (id BIGINT)"
        )
        self.conn.executemany(
            "INSERT INTO _del_ids VALUES (?)", [(i,) for i in ids]
        )
        self.conn.execute(
            f"DELETE FROM {self.table} WHERE id IN (SELECT id FROM _del_ids)"
        )

    def counts(self) -> tuple[int, int]:
        """Return (n_success, n_failed)."""
        row = self.conn.execute(f"""
            SELECT
                COUNT(*) FILTER (WHERE error IS NULL),
                COUNT(*) FILTER (WHERE error IS NOT NULL)
            FROM {self.table}
        """).fetchone()
        return row[0], row[1]

    def clear(self):
        """Delete all rows."""
        self.conn.execute(f"DELETE FROM {self.table}")

    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
