# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tests: llm_filter_candidates node (v2)
#
# Tests the LLM negative selection node, verifying:
# - FilterResponseModel validation
# - Prompt construction from raw ad text (no LLM summaries)
# - Drop logic (1-based indices, keeps at least 1)
# - Output format (filtered_matches.parquet with cosine_score)

# %%
#|default_exp ai_index.test_llm_filter_candidates

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

from ai_index.nodes.llm_filter_candidates import FilterResponseModel

# %% [markdown]
# ## FilterResponseModel tests

# %%
#|export
class TestFilterResponseModel:
    """Test the pydantic model for LLM filter responses."""

    def test_valid_keep(self):
        """Valid keep indices should parse."""
        resp = FilterResponseModel.model_validate_json('{"keep": [1, 3]}')
        assert resp.keep == [1, 3]

    def test_rejects_empty_keep(self):
        """Empty keep list should fail (must keep at least 1)."""
        with pytest.raises(Exception):
            FilterResponseModel.model_validate_json('{"keep": []}')

    def test_rejects_zero_index(self):
        """Keep indices must be 1-based (positive)."""
        with pytest.raises(Exception):
            FilterResponseModel.model_validate_json('{"keep": [0]}')

    def test_rejects_negative_index(self):
        """Negative indices should fail."""
        with pytest.raises(Exception):
            FilterResponseModel.model_validate_json('{"keep": [-1]}')

    def test_rejects_missing_keep(self):
        """Missing keep field should fail."""
        with pytest.raises(Exception):
            FilterResponseModel.model_validate_json('{"drop": [1]}')

# %% [markdown]
# ## Helpers for full node tests

# %%
#|export
SAMPLE_MATCHES = pd.DataFrame({
    "ad_id": [100, 100, 100, 200, 200, 200],
    "rank": [0, 1, 2, 0, 1, 2],
    "onet_code": ["11-1011.00", "15-1252.00", "29-1141.00",
                  "15-1252.00", "11-1011.00", "29-1141.00"],
    "onet_title": ["Chief Executives", "Software Developers", "Registered Nurses",
                   "Software Developers", "Chief Executives", "Registered Nurses"],
    "cosine_score": [0.95, 0.85, 0.75, 0.90, 0.80, 0.70],
})

# Mock ads for the Adzuna connection
MOCK_ADS = {
    100: {"title": "CEO Role", "category_name": "Executive Jobs", "description": "Leading a company at the highest level."},
    200: {"title": "Python Developer", "category_name": "IT Jobs", "description": "Building data pipelines and APIs."},
}


def _setup_matches(tmp_path, matches_df=None):
    """Write candidates.parquet (cosine candidates input)."""
    if matches_df is None:
        matches_df = SAMPLE_MATCHES
    match_dir = tmp_path / "pipeline" / "test_run" / "cosine_candidates"
    match_dir.mkdir(parents=True, exist_ok=True)
    matches_df.to_parquet(match_dir / "candidates.parquet")


def _make_ctx(tmp_path, llm_responses=None):
    """Build mock ctx and set up all mocks for a node run."""
    ctx = MagicMock()
    ctx.vars = {
        "run_name": "test_run",
        "llm_model": "test-llm",
        "sbatch_cache": False,
        "sbatch_time": "00:10:00",
        "llm_batch_size": 100,
        "llm_max_new_tokens": 200,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_concurrent_chunks": 1,
        "filter_resume": False,
        "filter_max_retries": 0,
        "filter_raise_on_failure": False,
        "duckdb_memory_limit": "256MB",
        "system_prompt": "llm_filter/v2/system",
        "user_prompt": "llm_filter/v2/user",
    }
    return ctx


def _mock_ads_conn():
    """Create a mock Adzuna connection that returns sample ads."""
    conn = MagicMock()

    def _execute(sql, *args):
        result = MagicMock()
        if "SELECT a.id" in sql:
            rows = [(ad_id, ad["title"], ad["category_name"], ad["description"])
                    for ad_id, ad in MOCK_ADS.items()]
            result.fetchall.return_value = rows
        return result

    conn.execute = _execute
    conn.close = MagicMock()
    return conn


def _run_node(tmp_path, ad_ids=None, llm_responses=None, matches_df=None):
    """Run llm_filter_candidates.main with mocks."""
    from ai_index.nodes.llm_filter_candidates import main

    if ad_ids is None:
        ad_ids = [100, 200]
    if llm_responses is None:
        # Default: keep candidates 1 and 3 for each ad
        llm_responses = ['{"keep": [1, 3]}'] * len(ad_ids)

    _setup_matches(tmp_path, matches_df)
    ctx = _make_ctx(tmp_path)
    mock_conn = _mock_ads_conn()

    async def _mock_allm_generate(prompts, **kwargs):
        return llm_responses[:len(prompts)]

    with patch("ai_index.const.pipeline_store_path", tmp_path / "pipeline"), \
         patch("ai_index.const.rel", lambda p: p), \
         patch("ai_index.utils.allm_generate", side_effect=_mock_allm_generate), \
         patch("ai_index.utils.llm.allm_generate", side_effect=_mock_allm_generate), \
         patch("ai_index.utils.load_prompt", return_value="mock prompt {n_candidates} {job_ad_title} {job_sector_category} {full_ad_excerpt} {candidates_str}"), \
         patch("ai_index.utils.prompts.load_prompt", return_value="mock prompt {n_candidates} {job_ad_title} {job_sector_category} {full_ad_excerpt} {candidates_str}"), \
         patch("ai_index.utils.is_reasoning_model", return_value=False), \
         patch("ai_index.utils.llm.is_reasoning_model", return_value=False), \
         patch("ai_index.utils.uses_structured_output", return_value=True), \
         patch("ai_index.utils.llm.uses_structured_output", return_value=True), \
         patch("ai_index.utils.get_adzuna_conn", return_value=mock_conn), \
         patch("ai_index.utils.adzuna_store.get_adzuna_conn", return_value=mock_conn):
        result = asyncio.run(main(ctx, print, ad_ids))

    # Read outputs
    output_dir = tmp_path / "pipeline" / "test_run" / "llm_filter_candidates"
    filtered_path = output_dir / "filtered_matches.parquet"
    filtered_df = pd.read_parquet(filtered_path) if filtered_path.exists() else pd.DataFrame()

    return result, filtered_df

# %% [markdown]
# ## Test drop logic

# %%
#|export
class TestKeepLogic:
    """Test that keep indices correctly filter candidates."""

    def test_keep_two_candidates(self, tmp_path):
        """Keeping candidates 1 and 3 should exclude candidate 2."""
        result, df = _run_node(tmp_path, llm_responses=['{"keep": [1, 3]}', '{"keep": [1, 3]}'])

        ad100 = df[df["ad_id"] == 100]
        kept_codes = ad100["onet_code"].tolist()
        assert "11-1011.00" in kept_codes  # candidate 1 kept
        assert "15-1252.00" not in kept_codes  # candidate 2 dropped
        assert "29-1141.00" in kept_codes  # candidate 3 kept

    def test_keep_one_candidate(self, tmp_path):
        """Keeping only candidate 2."""
        result, df = _run_node(tmp_path, llm_responses=['{"keep": [2]}', '{"keep": [2]}'])

        ad100 = df[df["ad_id"] == 100]
        assert len(ad100) == 1
        assert ad100.iloc[0]["onet_code"] == "15-1252.00"

    def test_keep_all(self, tmp_path):
        """Keeping all candidates."""
        result, df = _run_node(tmp_path, llm_responses=['{"keep": [1, 2, 3]}', '{"keep": [1, 2, 3]}'])

        for ad_id in [100, 200]:
            assert len(df[df["ad_id"] == ad_id]) == 3

    def test_ranks_are_reassigned(self, tmp_path):
        """Kept candidates should have contiguous ranks starting from 0."""
        result, df = _run_node(tmp_path, llm_responses=['{"keep": [1, 3]}', '{"keep": [1, 3]}'])

        for ad_id in [100, 200]:
            ranks = df[df["ad_id"] == ad_id].sort_values("rank")["rank"].tolist()
            assert ranks == list(range(len(ranks)))

# %% [markdown]
# ## Test output format

# %%
#|export
class TestOutput:
    """Test output files and format."""

    def test_returns_successful_ad_ids(self, tmp_path):
        """Should return IDs of successfully processed ads."""
        result, _ = _run_node(tmp_path)
        assert set(result) == {100, 200}

    def test_output_columns(self, tmp_path):
        """Output should have cosine_score instead of combined_score."""
        _, df = _run_node(tmp_path)
        assert set(df.columns) == {"ad_id", "rank", "onet_code", "onet_title", "cosine_score"}

    def test_preserves_cosine_scores(self, tmp_path):
        """Kept candidates should retain their original cosine scores."""
        _, df = _run_node(tmp_path, llm_responses=['{"keep": [1, 2, 3]}', '{"keep": [1, 2, 3]}'])

        ad100 = df[df["ad_id"] == 100].sort_values("rank")
        assert abs(ad100.iloc[0]["cosine_score"] - 0.95) < 1e-5
        assert abs(ad100.iloc[1]["cosine_score"] - 0.85) < 1e-5

    def test_writes_filter_results_db(self, tmp_path):
        """Should write filter_results.duckdb."""
        _run_node(tmp_path)
        db_path = tmp_path / "pipeline" / "test_run" / "llm_filter_candidates" / "filter_results.duckdb"
        assert db_path.exists()

    def test_writes_meta_json(self, tmp_path):
        """Should write filter_meta.json."""
        _run_node(tmp_path)
        meta_path = tmp_path / "pipeline" / "test_run" / "llm_filter_candidates" / "filter_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "failed_ids" in meta

# %% [markdown]
# ## Test error handling

# %%
#|export
class TestErrorHandling:
    """Test handling of invalid LLM responses."""

    def test_invalid_json_is_error(self, tmp_path):
        """Invalid JSON should be recorded as error, not crash."""
        result, df = _run_node(tmp_path, llm_responses=['not json', '{"keep": [1, 3]}'])

        # Ad 100 should fail, ad 200 should succeed
        assert 200 in result
        ad200 = df[df["ad_id"] == 200]
        assert len(ad200) == 2  # kept 2 of 3

    def test_out_of_range_index_is_error(self, tmp_path):
        """Keep index > n_candidates should be recorded as error."""
        result, df = _run_node(tmp_path, llm_responses=['{"keep": [99]}', '{"keep": [2, 3]}'])

        # Ad 100 should fail (index 99 out of range), ad 200 succeeds
        assert 200 in result
