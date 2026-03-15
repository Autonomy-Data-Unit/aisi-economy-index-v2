# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tests: embed_ads node (v2)
#
# Tests the raw-text embedding node, verifying:
# - Text construction from ad title + description
# - Prompt support (prefix, prompt_name, custom instruction)
# - Output format (DuckDB with embedding BLOBs)

# %%
#|default_exp ai_index.test_embed_ads

# %%
#|export
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

# %% [markdown]
# ## Helpers

# %%
#|export
def _make_ads_table(ads: list[dict]) -> pa.Table:
    """Build a pyarrow Table matching get_ads_by_id output."""
    return pa.table({
        "id": [a["id"] for a in ads],
        "title": [a["title"] for a in ads],
        "description": [a["description"] for a in ads],
    })


def _fake_embed(texts, *, model, **kwargs):
    """Return deterministic fake embeddings (4-dim)."""
    return np.array([[float(len(t) % 10)] * 4 for t in texts], dtype=np.float32)


SAMPLE_ADS = [
    {"id": 100, "title": "Registered Nurse", "description": "Caring for patients in a hospital ward."},
    {"id": 200, "title": "Software Developer", "description": "Building Python applications for data pipelines."},
    {"id": 300, "title": "HGV Driver", "description": "Driving waste collection routes across London."},
]


def _run_node(tmp_path, ads, embedding_model="text-embedding-3-large",
              embed_task_prompt="", model_cfg_override=None):
    """Run embed_ads.main with mocks. Returns (result, embed_calls).

    Patches get_ads_by_id, aembed, _load_model_config, and const at their
    source modules so they resolve correctly inside export_as_func.
    """
    import asyncio
    from ai_index.nodes.embed_ads import main

    ads_table = _make_ads_table(ads)
    ad_ids = np.array([a["id"] for a in ads])

    ctx = MagicMock()
    ctx.vars = {
        "run_name": "test_run",
        "embedding_model": embedding_model,
        "sbatch_cache": False,
        "sbatch_time": "00:10:00",
        "embed_task_prompt": embed_task_prompt,
        "duckdb_memory_limit": "256MB",
    }

    embed_calls = []

    async def _capture_embed(texts, *, model, **kwargs):
        embed_calls.append({"texts": list(texts), "kwargs": kwargs})
        return _fake_embed(texts, model=model, **kwargs)

    if model_cfg_override is None:
        model_cfg_override = ("api", {"model": "text-embedding-3-large"})

    # Patch at the source modules where the function imports from
    mock_const = MagicMock()
    mock_const.pipeline_store_path = tmp_path / "pipeline"
    mock_const.embed_models_config_path = Path("/fake/embed_models.toml")
    mock_const.rel = lambda p: p

    with patch("ai_index.const.pipeline_store_path", tmp_path / "pipeline"), \
         patch("ai_index.const.embed_models_config_path", Path("/fake")), \
         patch("ai_index.const.rel", lambda p: p), \
         patch("ai_index.utils.get_ads_by_id", return_value=ads_table), \
         patch("ai_index.utils.adzuna_store.get_ads_by_id", return_value=ads_table), \
         patch("ai_index.utils.aembed", side_effect=_capture_embed), \
         patch("ai_index.utils.embed.aembed", side_effect=_capture_embed), \
         patch("ai_index.utils._model_config._load_model_config", return_value=model_cfg_override):
        result = asyncio.run(main(ctx, print, ad_ids))

    return result, embed_calls

# %% [markdown]
# ## Test text construction

# %%
#|export
class TestTextConstruction:
    """Test that ad texts are built correctly from title + description."""

    def test_basic_text_format(self, tmp_path):
        """Text should be 'title. description'."""
        result, embed_calls = _run_node(tmp_path, SAMPLE_ADS)

        assert len(embed_calls) == 1
        texts = embed_calls[0]["texts"]
        assert texts[0] == "Registered Nurse. Caring for patients in a hospital ward."
        assert texts[1] == "Software Developer. Building Python applications for data pipelines."
        assert texts[2] == "HGV Driver. Driving waste collection routes across London."

    def test_missing_description(self, tmp_path):
        """Ads with None description should still work."""
        ads = [{"id": 100, "title": "Nurse", "description": None}]
        result, embed_calls = _run_node(tmp_path, ads)

        assert embed_calls[0]["texts"][0] == "Nurse. "

# %% [markdown]
# ## Test prompt support

# %%
#|export
class TestPromptSupport:
    """Test the three categories of prompt support."""

    def test_query_prefix(self, tmp_path):
        """Models with query_prefix should have it prepended to texts."""
        model_cfg = ("sbatch", {"model": "intfloat/e5-large-v2", "query_prefix": "query: "})
        result, embed_calls = _run_node(
            tmp_path, SAMPLE_ADS[:1],
            embedding_model="e5-large-sbatch",
            model_cfg_override=model_cfg,
        )
        assert embed_calls[0]["texts"][0].startswith("query: Registered Nurse.")

    def test_prompt_name(self, tmp_path):
        """Models with query_prompt_name should pass it to aembed."""
        model_cfg = ("sbatch", {"model": "Snowflake/snowflake-arctic-embed-l-v2.0",
                                "query_prompt_name": "query"})
        result, embed_calls = _run_node(
            tmp_path, SAMPLE_ADS[:1],
            embedding_model="arctic-embed-l-sbatch",
            model_cfg_override=model_cfg,
        )
        assert embed_calls[0]["kwargs"]["prompt_name"] == "query"

    def test_supports_prompt_with_instruction(self, tmp_path):
        """Models with supports_prompt=true should use embed_task_prompt."""
        model_cfg = ("sbatch", {"model": "Qwen/Qwen3-Embedding-8B",
                                "supports_prompt": True, "query_prompt_name": "query"})
        task_prompt = "Instruct: Classify by occupation\nQuery: "
        result, embed_calls = _run_node(
            tmp_path, SAMPLE_ADS[:1],
            embedding_model="qwen3-embed-8b-sbatch",
            embed_task_prompt=task_prompt,
            model_cfg_override=model_cfg,
        )
        # Custom prompt should override prompt_name
        assert embed_calls[0]["kwargs"]["prompt"] == task_prompt
        assert "prompt_name" not in embed_calls[0]["kwargs"]

    def test_supports_prompt_without_instruction_falls_back_to_prompt_name(self, tmp_path):
        """If supports_prompt=true but no embed_task_prompt, use prompt_name."""
        model_cfg = ("sbatch", {"model": "Qwen/Qwen3-Embedding-8B",
                                "supports_prompt": True, "query_prompt_name": "query"})
        result, embed_calls = _run_node(
            tmp_path, SAMPLE_ADS[:1],
            embedding_model="qwen3-embed-8b-sbatch",
            embed_task_prompt="",
            model_cfg_override=model_cfg,
        )
        assert embed_calls[0]["kwargs"]["prompt_name"] == "query"
        assert "prompt" not in embed_calls[0]["kwargs"]

    def test_no_prompt_support(self, tmp_path):
        """Models without any prompt config should pass no prompt kwargs."""
        model_cfg = ("api", {"model": "text-embedding-3-large"})
        result, embed_calls = _run_node(
            tmp_path, SAMPLE_ADS[:1],
            model_cfg_override=model_cfg,
        )
        assert "prompt" not in embed_calls[0]["kwargs"]
        assert "prompt_name" not in embed_calls[0]["kwargs"]

# %% [markdown]
# ## Test output format

# %%
#|export
class TestOutput:
    """Test output storage and return values."""

    def test_returns_ad_ids(self, tmp_path):
        """Node should return the list of embedded ad IDs."""
        result, _ = _run_node(tmp_path, SAMPLE_ADS)
        # export_as_func returns the raw value; netrun wraps it in {"ad_ids": ...}
        assert result == [100, 200, 300]

    def test_writes_embeddings_db(self, tmp_path):
        """Node should write embeddings.duckdb with correct schema."""
        import duckdb
        _run_node(tmp_path, SAMPLE_ADS)

        db_path = tmp_path / "pipeline" / "test_run" / "embed_ads" / "embeddings.duckdb"
        assert db_path.exists()

        conn = duckdb.connect(str(db_path), read_only=True)
        rows = conn.execute("SELECT id, embedding FROM results WHERE error IS NULL ORDER BY id").fetchall()
        conn.close()

        assert len(rows) == 3
        assert rows[0][0] == 100
        emb = np.frombuffer(rows[0][1], dtype=np.float32)
        assert emb.shape == (4,)

    def test_writes_meta_json(self, tmp_path):
        """Node should write embed_meta.json."""
        _run_node(tmp_path, SAMPLE_ADS[:1])

        meta_path = tmp_path / "pipeline" / "test_run" / "embed_ads" / "embed_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["n_total"] == 1
        assert meta["n_embedded"] == 1
