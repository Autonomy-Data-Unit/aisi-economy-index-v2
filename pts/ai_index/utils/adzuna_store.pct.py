# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DuckDB Adzuna store

# %%
#|default_exp utils.adzuna_store

# %%
#|export
import duckdb
import pyarrow as pa

from ai_index.const import adzuna_db_path

# Canonical schema for the ads table. All string-like columns use VARCHAR.
# year/month are partition columns added during ingest.
_ADS_SCHEMA = [
    ("id", "BIGINT NOT NULL"),
    ("year", "INTEGER NOT NULL"),
    ("month", "INTEGER NOT NULL"),
    ("date_created", "VARCHAR"),
    ("title", "VARCHAR"),
    ("description", "VARCHAR"),
    ("location_raw", "VARCHAR"),
    ("geo_lat", "VARCHAR"),
    ("geo_lng", "VARCHAR"),
    ("LAD22CD", "VARCHAR"),
    ("LAD22NM", "VARCHAR"),
    ("LAD24CD", "VARCHAR"),
    ("category_id", "BIGINT"),
    ("category_name", "VARCHAR"),
    ("company_raw", "VARCHAR"),
    ("company_id", "DOUBLE"),
    ("normalised_company", "VARCHAR"),
    ("salary_min", "DOUBLE"),
    ("salary_max", "DOUBLE"),
    ("salary_currency", "VARCHAR"),
    ("salary_predicted", "DOUBLE"),
    ("salary_raw", "VARCHAR"),
    ("contract_type", "VARCHAR"),
    ("contract_time", "VARCHAR"),
    ("soc2020_major_group", "VARCHAR"),
    ("soc2020_submajor_group", "VARCHAR"),
    ("soc2020_minor_group", "VARCHAR"),
    ("soc2020", "VARCHAR"),
    ("sic_section", "VARCHAR"),
    ("SIC1", "VARCHAR"),
    ("SIC2", "VARCHAR"),
    ("SIC3", "VARCHAR"),
    ("SIC4", "VARCHAR"),
]

# Columns that may be numeric (double) in older parquets and need CAST to VARCHAR
_CAST_IF_NUMERIC = {
    "geo_lat", "geo_lng",
    "soc2020_major_group", "soc2020_submajor_group",
    "soc2020_minor_group", "soc2020",
}


def get_adzuna_conn(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection to the Adzuna database."""
    return duckdb.connect(str(adzuna_db_path), read_only=read_only)


def ensure_ads_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ads table and indexes if they don't exist."""
    cols = ", ".join(f"{name} {dtype}" for name, dtype in _ADS_SCHEMA)
    conn.execute(f"CREATE TABLE IF NOT EXISTS ads ({cols})")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ads_id ON ads(id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ads_year_month ON ads(year, month)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)


def build_insert_from_parquet(
    parquet_path: str,
    year: int,
    month: int,
    source_schema: list[str] | None = None,
) -> str:
    """Build an INSERT INTO ads SELECT ... FROM read_parquet(...) statement.

    Handles schema differences: casts numeric columns to VARCHAR,
    fills missing columns with NULL, and adds year/month.

    Args:
        parquet_path: Path to the parquet file.
        year: Year partition value.
        month: Month partition value.
        source_schema: Column names in the source parquet. If None,
            reads from the file metadata.
    """
    if source_schema is None:
        import pyarrow.parquet as pq
        source_schema_obj = pq.read_schema(parquet_path)
        source_cols = set(source_schema_obj.names)
        # Build a map of source column types for cast decisions
        source_types = {f.name: f.type for f in source_schema_obj}
    else:
        source_cols = set(source_schema)
        source_types = {}

    select_exprs = []
    for col_name, _ in _ADS_SCHEMA:
        if col_name == "year":
            select_exprs.append(f"{year} AS year")
        elif col_name == "month":
            select_exprs.append(f"{month} AS month")
        elif col_name not in source_cols:
            select_exprs.append(f"NULL AS {col_name}")
        elif col_name in _CAST_IF_NUMERIC and source_types:
            # Check if source type is numeric (double/float)
            src_type = source_types.get(col_name)
            if src_type is not None and (pa.types.is_floating(src_type) or pa.types.is_integer(src_type)):
                select_exprs.append(f"CAST({col_name} AS VARCHAR) AS {col_name}")
            else:
                select_exprs.append(col_name)
        else:
            select_exprs.append(col_name)

    select_clause = ", ".join(select_exprs)
    return f"INSERT INTO ads SELECT {select_clause} FROM read_parquet('{parquet_path}')"


def get_ads_by_id(
    ids: list[int],
    columns: list[str] | None = None,
) -> pa.Table:
    """Retrieve job ad rows by ID from the Adzuna DuckDB store.

    Args:
        ids: List of job ad IDs to retrieve.
        columns: Columns to include in the result. If None, all columns
            are returned. The ``id`` column is always included.

    Returns:
        A pyarrow Table with matching rows.
    """
    if columns is not None and "id" not in columns:
        columns = ["id"] + list(columns)

    col_clause = ", ".join(columns) if columns else "*"
    placeholders = ", ".join(["?"] * len(ids))

    conn = get_adzuna_conn(read_only=True)
    try:
        result = conn.execute(
            f"SELECT {col_clause} FROM ads WHERE id IN ({placeholders})", ids
        ).fetch_arrow_table()
    finally:
        conn.close()
    return result

def get_all_ad_ids():
    conn = get_adzuna_conn(read_only=True)
    all_ids = conn.execute("SELECT id FROM ads").fetchnumpy()["id"].tolist()
    conn.close()
    return all_ids
