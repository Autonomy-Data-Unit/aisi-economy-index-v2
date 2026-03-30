# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.aggregate_geo
#
# Aggregate ad-level AI exposure scores by Local Authority District (LAD22CD).
#
# Joins `ad_exposure.parquet` (from `compute_job_ad_exposure`) with the Adzuna
# ads table to get each ad's LAD22CD, then computes per-LAD mean scores.
#
# The entire join + aggregation runs inside DuckDB's columnar engine, so no
# pandas intermediate is needed for the full dataset. Only the ~373-row
# result set is materialized in memory.
#
# Ads without a LAD22CD (~27-34%) are excluded from geographic aggregation
# and reported in the coverage summary.
#
# Node variables:
# - `run_name` (global): Pipeline run name

# %%
#|default_exp aggregate_geo
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, ad_ids: list[int]) -> None:
    """Aggregate ad-level AI exposure scores by Local Authority District (LAD22CD)."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import *
run_name = 'test_local'
set_node_func_args('aggregate_geo', run_name=run_name)
show_node_vars('aggregate_geo', run_name=run_name)

# %% [markdown]
#
# # Function body

# %% [markdown]
# ## Setup

# %%
#|export
import duckdb

from ai_index import const

# %%
#|export
run_name = ctx.vars["run_name"]
duckdb_memory_limit = ctx.vars["duckdb_memory_limit"]

output_dir = const.outputs_path / run_name
output_dir.mkdir(parents=True, exist_ok=True)

ad_exposure_path = const.pipeline_store_path / run_name / "compute_job_ad_exposure" / "ad_exposure.parquet"

# %% [markdown]
# ## Discover score columns and aggregate via DuckDB
#
# DuckDB handles the entire join + aggregation in its columnar engine.
# Only the ~373-row GROUP BY result is materialized in memory. The parquet
# file and adzuna database are streamed through without loading into pandas.

# %%
#|export
conn = duckdb.connect()
conn.execute(f"SET memory_limit = '{duckdb_memory_limit}'")

# Discover score columns from parquet schema
all_cols = [row[0] for row in conn.execute(
    f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{ad_exposure_path}'))"
).fetchall()]
score_cols = [c for c in all_cols if c not in ("ad_id", "n_matches", "error")]
print(f"aggregate_geo: {len(score_cols)} score columns: {score_cols}")

# Attach adzuna database (read-only to avoid lock contention)
conn.execute(f"ATTACH '{const.adzuna_db_path}' AS adzuna (READ_ONLY)")

# Build aggregation SQL. Only averages over ads with actual scores (n_matches > 0).
# LAD22NM comes from the ONS lookup table (complete coverage) rather than the ads
# table (which has gaps).
agg_parts = []
for col in score_cols:
    agg_parts.append(
        f'AVG(CASE WHEN e.n_matches > 0 THEN e."{col}" ELSE NULL END) AS "{col}"'
    )
agg_sql = ",\n    ".join(agg_parts)

sql = f"""
SELECT
    agg.LAD22CD,
    lad.LAD22NM,
    agg.n_ads,
    agg.n_ads_with_scores,
    {", ".join(f'agg."{col}"' for col in score_cols)}
FROM (
    SELECT
        a.LAD22CD,
        COUNT(*) AS n_ads,
        SUM(CASE WHEN e.n_matches > 0 THEN 1 ELSE 0 END)::INTEGER AS n_ads_with_scores,
        {agg_sql}
    FROM read_parquet('{ad_exposure_path}') e
    JOIN adzuna.ads a ON e.ad_id = a.id
    WHERE a.LAD22CD IS NOT NULL
    GROUP BY a.LAD22CD
) agg
LEFT JOIN read_csv('{const.lad22_lookup_path}', header=true) lad
    ON agg.LAD22CD = lad.LAD22CD
ORDER BY agg.LAD22CD
"""

result_df = conn.execute(sql).fetchdf()
conn.close()

# Coverage stats (derived from result, no second scan)
n_total = len(ad_ids)
n_with_lad = int(result_df["n_ads"].sum())
n_without_lad = n_total - n_with_lad

# %% [markdown]
# ## Write output

# %%
#|export
output_path = output_dir / "geo_lad.csv"
result_df.to_csv(output_path, index=False)

print(f"aggregate_geo: {len(result_df)} LADs")
print(f"  coverage: {n_with_lad}/{n_total} ads have LAD22CD ({n_with_lad / n_total * 100:.1f}%)")
print(f"  {n_without_lad} ads excluded (no LAD22CD)")
print(f"  output: {const.rel(output_path)}")

# %% [markdown]
# ## Sample output

# %%
print(f"\nScore columns: {score_cols}")
print(f"\nTop 10 LADs by ad count:")
result_df.sort_values("n_ads", ascending=False).head(10)
