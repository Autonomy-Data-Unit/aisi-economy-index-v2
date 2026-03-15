# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tests: cosine_candidates node
#
# Tests the cosine similarity candidate selection node, verifying:
# - Correct top-K selection and ranking
# - Cosine scores are correct
# - Output parquet format
# - Handles topk larger than number of occupations

# %%
#|default_exp ai_index.test_cosine_candidates

# %%
#|export
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pandas as pd
import pytest

# %% [markdown]
# ## Helpers

# %%
#|export
# 3 occupations with known embeddings (4-dim, unit vectors in different directions)
ONET_CODES = ["11-1011.00", "15-1252.00", "29-1141.00"]
ONET_TITLES = {
    "11-1011.00": "Chief Executives",
    "15-1252.00": "Software Developers",
    "29-1141.00": "Registered Nurses",
}
ONET_TARGETS_DF = pd.DataFrame({
    "O*NET-SOC Code": ONET_CODES,
    "Title": [ONET_TITLES[c] for c in ONET_CODES],
})

# O*NET embeddings: each occupation points in a distinct direction
ONET_EMBEDDINGS = np.array([
    [1.0, 0.0, 0.0, 0.0],  # Chief Executives -> x-axis
    [0.0, 1.0, 0.0, 0.0],  # Software Developers -> y-axis
    [0.0, 0.0, 1.0, 0.0],  # Registered Nurses -> z-axis
], dtype=np.float32)

# Ad embeddings: each ad is close to one occupation
AD_EMBEDDINGS = {
    100: np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),  # -> Chief Executives
    200: np.array([0.1, 0.8, 0.2, 0.0], dtype=np.float32),  # -> Software Developers
    300: np.array([0.0, 0.1, 0.9, 0.1], dtype=np.float32),  # -> Registered Nurses
}


def _setup_embed_db(tmp_path, ad_embeddings):
    """Create a mock embeddings.duckdb with the given ad embeddings."""
    db_path = tmp_path / "pipeline" / "test_run" / "embed_ads" / "embeddings.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE results (id BIGINT NOT NULL, embedding BLOB NOT NULL, error VARCHAR)")
    for ad_id, emb in ad_embeddings.items():
        conn.execute("INSERT INTO results VALUES (?, ?, NULL)", [ad_id, emb.tobytes()])
    conn.close()
    return db_path


def _setup_onet_dir(tmp_path, onet_codes, onet_embeddings):
    """Create mock embed_onet output directory."""
    onet_dir = tmp_path / "pipeline" / "test_run" / "embed_onet"
    onet_dir.mkdir(parents=True, exist_ok=True)
    with open(onet_dir / "onet_codes.json", "w") as f:
        json.dump(onet_codes, f)
    np.save(onet_dir / "onet_embeddings.npy", onet_embeddings)


def _run_node(tmp_path, ad_ids=None, cosine_topk=3, ad_embeddings=None,
              onet_codes=None, onet_embeddings=None):
    """Run cosine_candidates.main with mocks. Returns (result, candidates_df)."""
    from ai_index.nodes.cosine_candidates import main

    if ad_ids is None:
        ad_ids = [100, 200, 300]
    if ad_embeddings is None:
        ad_embeddings = AD_EMBEDDINGS
    if onet_codes is None:
        onet_codes = ONET_CODES
    if onet_embeddings is None:
        onet_embeddings = ONET_EMBEDDINGS

    _setup_embed_db(tmp_path, ad_embeddings)
    _setup_onet_dir(tmp_path, onet_codes, onet_embeddings)

    ctx = MagicMock()
    ctx.vars = {
        "run_name": "test_run",
        "cosine_topk": cosine_topk,
    }

    with patch("ai_index.const.pipeline_store_path", tmp_path / "pipeline"), \
         patch("ai_index.const.onet_targets_path", tmp_path / "onet_targets.parquet"), \
         patch("ai_index.const.rel", lambda p: p):
        # Write onet targets parquet
        ONET_TARGETS_DF.to_parquet(tmp_path / "onet_targets.parquet")

        result = asyncio.run(main(ctx, print, ad_ids, True))

    # Read the output parquet
    output_path = tmp_path / "pipeline" / "test_run" / "cosine_candidates" / "candidates.parquet"
    candidates_df = pd.read_parquet(output_path)

    return result, candidates_df

# %% [markdown]
# ## Test candidate selection

# %%
#|export
class TestCandidateSelection:
    """Test that the correct top-K candidates are selected."""

    def test_top1_matches_closest_occupation(self, tmp_path):
        """Each ad's top-1 should be the most similar occupation."""
        result, df = _run_node(tmp_path)

        # Ad 100 (mostly x-axis) -> Chief Executives (x-axis)
        ad100 = df[df["ad_id"] == 100].sort_values("rank")
        assert ad100.iloc[0]["onet_code"] == "11-1011.00"

        # Ad 200 (mostly y-axis) -> Software Developers (y-axis)
        ad200 = df[df["ad_id"] == 200].sort_values("rank")
        assert ad200.iloc[0]["onet_code"] == "15-1252.00"

        # Ad 300 (mostly z-axis) -> Registered Nurses (z-axis)
        ad300 = df[df["ad_id"] == 300].sort_values("rank")
        assert ad300.iloc[0]["onet_code"] == "29-1141.00"

    def test_topk_returns_correct_count(self, tmp_path):
        """Should return exactly topk candidates per ad."""
        result, df = _run_node(tmp_path, cosine_topk=2)

        for ad_id in [100, 200, 300]:
            assert len(df[df["ad_id"] == ad_id]) == 2

    def test_topk_clamped_to_n_occupations(self, tmp_path):
        """topk larger than n_occupations should return all occupations."""
        result, df = _run_node(tmp_path, cosine_topk=100)

        for ad_id in [100, 200, 300]:
            assert len(df[df["ad_id"] == ad_id]) == 3  # only 3 occupations

    def test_ranks_are_zero_indexed_and_contiguous(self, tmp_path):
        """Ranks should be 0, 1, 2, ... for each ad."""
        result, df = _run_node(tmp_path, cosine_topk=3)

        for ad_id in [100, 200, 300]:
            ranks = df[df["ad_id"] == ad_id]["rank"].tolist()
            assert ranks == [0, 1, 2]

# %% [markdown]
# ## Test cosine scores

# %%
#|export
class TestCosineScores:
    """Test that cosine scores are computed correctly."""

    def test_scores_are_descending(self, tmp_path):
        """Scores should decrease with rank for each ad."""
        result, df = _run_node(tmp_path)

        for ad_id in [100, 200, 300]:
            scores = df[df["ad_id"] == ad_id].sort_values("rank")["cosine_score"].tolist()
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]

    def test_scores_are_valid_cosine(self, tmp_path):
        """All scores should be in [-1, 1]."""
        result, df = _run_node(tmp_path)

        assert df["cosine_score"].min() >= -1.0
        assert df["cosine_score"].max() <= 1.0

    def test_identical_vectors_score_one(self, tmp_path):
        """An ad identical to an occupation should get cosine=1.0."""
        ad_embeddings = {100: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
        result, df = _run_node(tmp_path, ad_ids=[100], ad_embeddings=ad_embeddings)

        top = df[df["ad_id"] == 100].sort_values("rank").iloc[0]
        assert top["onet_code"] == "11-1011.00"
        assert abs(top["cosine_score"] - 1.0) < 1e-5

# %% [markdown]
# ## Test output format

# %%
#|export
class TestOutput:
    """Test output format and return values."""

    def test_returns_ad_ids(self, tmp_path):
        """Node should return the ad_ids list."""
        result, _ = _run_node(tmp_path)
        assert result == [100, 200, 300]

    def test_output_columns(self, tmp_path):
        """Output parquet should have the expected columns."""
        _, df = _run_node(tmp_path)
        assert set(df.columns) == {"ad_id", "rank", "onet_code", "onet_title", "cosine_score"}

    def test_includes_onet_titles(self, tmp_path):
        """Output should include human-readable O*NET titles."""
        _, df = _run_node(tmp_path)

        top = df[(df["ad_id"] == 100) & (df["rank"] == 0)].iloc[0]
        assert top["onet_title"] == "Chief Executives"
