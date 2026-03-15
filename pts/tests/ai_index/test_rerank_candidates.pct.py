# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tests: rerank_candidates node
#
# Tests the reranking node, verifying:
# - Passthrough mode (no rerank_model) uses cosine scores
# - Reranking mode filters to per-ad candidates and takes top-K
# - Output parquet format

# %%
#|default_exp ai_index.test_rerank_candidates

# %%
#|export
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

# %% [markdown]
# ## Helpers

# %%
#|export
SAMPLE_CANDIDATES = pd.DataFrame({
    "ad_id": [100, 100, 100, 200, 200, 200],
    "rank": [0, 1, 2, 0, 1, 2],
    "onet_code": ["11-1011.00", "15-1252.00", "29-1141.00",
                  "15-1252.00", "11-1011.00", "29-1141.00"],
    "onet_title": ["Chief Executives", "Software Developers", "Registered Nurses",
                   "Software Developers", "Chief Executives", "Registered Nurses"],
    "cosine_score": [0.9, 0.8, 0.7, 0.85, 0.75, 0.65],
})

ONET_TARGETS = pd.DataFrame({
    "O*NET-SOC Code": ["11-1011.00", "15-1252.00", "29-1141.00"],
    "Title": ["Chief Executives", "Software Developers", "Registered Nurses"],
    "Description": [
        "Determine and formulate policies.",
        "Research, design, and develop software.",
        "Assess patient health problems.",
    ],
})

SAMPLE_ADS_TABLE = pa.table({
    "id": [100, 200],
    "title": ["CEO role", "Python developer"],
    "description": ["Leading a company.", "Building software."],
})


def _setup_candidates(tmp_path, candidates_df=None):
    """Write filtered_matches.parquet (input from llm_filter_candidates)."""
    if candidates_df is None:
        candidates_df = SAMPLE_CANDIDATES
    filter_dir = tmp_path / "pipeline" / "test_run" / "llm_filter_candidates"
    filter_dir.mkdir(parents=True, exist_ok=True)
    candidates_df.to_parquet(filter_dir / "filtered_matches.parquet")


def _run_node(tmp_path, ad_ids=None, rerank_model="test-reranker",
              fake_rerank_pairs_fn=None):
    """Run rerank_candidates.main with mocks."""
    from ai_index.nodes.rerank_candidates import main

    if ad_ids is None:
        ad_ids = [100, 200]

    _setup_candidates(tmp_path)

    ctx = MagicMock()
    ctx.vars = {
        "run_name": "test_run",
        "rerank_model": rerank_model,
        "sbatch_time": "00:10:00",
        "chunk_size": 500,
    }

    async def _default_rerank_pairs(items, *, model, **kwargs):
        # Return descending scores for each item's documents (reverses order vs cosine)
        return [
            [float(v) for v in np.linspace(1.0, 0.0, len(item[1]))]
            for item in items
        ]

    rerank_pairs_fn = fake_rerank_pairs_fn or _default_rerank_pairs

    with patch("ai_index.const.pipeline_store_path", tmp_path / "pipeline"), \
         patch("ai_index.const.onet_targets_path", tmp_path / "onet_targets.parquet"), \
         patch("ai_index.const.rel", lambda p: p), \
         patch("ai_index.utils.rerank.arerank_pairs", side_effect=rerank_pairs_fn), \
         patch("ai_index.utils.arerank_pairs", side_effect=rerank_pairs_fn), \
         patch("ai_index.utils.adzuna_store.get_ads_by_id", return_value=SAMPLE_ADS_TABLE), \
         patch("ai_index.utils.get_ads_by_id", return_value=SAMPLE_ADS_TABLE):
        ONET_TARGETS.to_parquet(tmp_path / "onet_targets.parquet")
        result = asyncio.run(main(ctx, print, ad_ids))

    output_path = tmp_path / "pipeline" / "test_run" / "rerank_candidates" / "reranked_matches.parquet"
    reranked_df = pd.read_parquet(output_path)

    return result, reranked_df

# %% [markdown]
# ## Test reranking

# %%
#|export
class TestReranking:
    """Test actual reranking with a mock reranker."""

    def test_reranking_scores_all_candidates(self, tmp_path):
        """Should score all filtered candidates per ad (no narrowing)."""
        result, df = _run_node(tmp_path, rerank_model="test-reranker")

        # 3 candidates per ad in sample data, all should be scored
        for ad_id in [100, 200]:
            assert len(df[df["ad_id"] == ad_id]) == 3

    def test_reranking_preserves_candidate_set(self, tmp_path):
        """All scored candidates should match the filtered input set."""
        result, df = _run_node(tmp_path, rerank_model="test-reranker")

        for ad_id in [100, 200]:
            reranked_codes = set(df[df["ad_id"] == ad_id]["onet_code"])
            input_codes = set(SAMPLE_CANDIDATES[SAMPLE_CANDIDATES["ad_id"] == ad_id]["onet_code"])
            assert reranked_codes == input_codes

# %% [markdown]
# ## Test output format

# %%
#|export
class TestOutput:
    """Test output format and return values."""

    def test_returns_ad_ids(self, tmp_path):
        """Node should return the ad_ids list."""
        result, _ = _run_node(tmp_path)
        assert result == [100, 200]

    def test_output_columns(self, tmp_path):
        """Output parquet should have the expected columns."""
        _, df = _run_node(tmp_path)
        assert set(df.columns) == {"ad_id", "rank", "onet_code", "onet_title", "rerank_score"}

    def test_ranks_are_contiguous(self, tmp_path):
        """Ranks should be 0, 1, 2 for each ad."""
        _, df = _run_node(tmp_path)
        for ad_id in [100, 200]:
            ranks = sorted(df[df["ad_id"] == ad_id]["rank"].tolist())
            assert ranks == [0, 1, 2]

    def test_all_candidates_have_scores(self, tmp_path):
        """Every candidate should have a rerank_score assigned."""
        _, df = _run_node(tmp_path, rerank_model="test-reranker")
        assert df["rerank_score"].notna().all()
        assert len(df) == 6  # 3 candidates per ad x 2 ads
