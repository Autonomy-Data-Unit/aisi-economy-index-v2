# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tests: embed_onet node (v2)
#
# Tests the O*NET embedding node, verifying:
# - Rich text construction (title + description + tasks/skills)
# - Document-side prompt support (prefix, prompt_name)
# - Skip logic when output files already exist
# - Output format (npy embeddings + JSON codes + meta)

# %%
#|default_exp ai_index.test_embed_onet

# %%
#|export
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# %% [markdown]
# ## Helpers

# %%
#|export
SAMPLE_ONET = pd.DataFrame({
    "O*NET-SOC Code": ["11-1011.00", "15-1252.00", "29-1141.00"],
    "Title": ["Chief Executives", "Software Developers", "Registered Nurses"],
    "Description": [
        "Determine and formulate policies.",
        "Research, design, and develop software.",
        "Assess patient health problems.",
    ],
    "Job Role Description": [
        "Chief Executives - Determine and formulate policies.",
        "Software Developers - Research, design, and develop software.",
        "Registered Nurses - Assess patient health problems.",
    ],
    "Work Activities/Tasks/Skills": [
        "Making Decisions, Establishing Relationships",
        "Programming, Systems Analysis, Complex Problem Solving",
        "Active Listening, Critical Thinking, Monitoring",
    ],
})


def _fake_embed(texts, *, model, **kwargs):
    """Return deterministic fake embeddings (4-dim)."""
    return np.array([[float(len(t) % 10)] * 4 for t in texts], dtype=np.float32)


def _run_node(tmp_path, onet_df=None, embedding_model="text-embedding-3-large",
              model_cfg_override=None, pre_existing=False):
    """Run embed_onet.main with mocks. Returns (result, embed_calls)."""
    from ai_index.nodes.embed_onet import main

    if onet_df is None:
        onet_df = SAMPLE_ONET

    ctx = MagicMock()
    ctx.vars = {
        "run_name": "test_run",
        "embedding_model": embedding_model,
        "sbatch_cache": False,
        "sbatch_time": "00:10:00",
    }

    embed_calls = []

    async def _capture_embed(texts, *, model, **kwargs):
        embed_calls.append({"texts": list(texts), "kwargs": kwargs})
        return _fake_embed(texts, model=model, **kwargs)

    if model_cfg_override is None:
        model_cfg_override = ("api", {"model": "text-embedding-3-large"})

    output_dir = tmp_path / "pipeline" / "test_run" / "embed_onet"
    if pre_existing:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "onet_embeddings.npy", np.zeros((3, 4)))
        with open(output_dir / "onet_codes.json", "w") as f:
            json.dump(["a", "b", "c"], f)

    with patch("ai_index.const.pipeline_store_path", tmp_path / "pipeline"), \
         patch("ai_index.const.onet_targets_path", tmp_path / "onet_targets.parquet"), \
         patch("ai_index.const.embed_models_config_path", Path("/fake")), \
         patch("ai_index.const.rel", lambda p: p), \
         patch("ai_index.utils.aembed", side_effect=_capture_embed), \
         patch("ai_index.utils.embed.aembed", side_effect=_capture_embed), \
         patch("ai_index.utils._model_config._load_model_config", return_value=model_cfg_override):
        # Write sample parquet for the node to read
        parquet_path = tmp_path / "onet_targets.parquet"
        onet_df.to_parquet(parquet_path)

        result = asyncio.run(main(ctx, print))

    return result, embed_calls

# %% [markdown]
# ## Test text construction

# %%
#|export
class TestTextConstruction:
    """Test that O*NET texts are built as single rich documents."""

    def test_rich_document_format(self, tmp_path):
        """Each text should contain title, description, and tasks/skills."""
        result, embed_calls = _run_node(tmp_path)

        assert len(embed_calls) == 1
        texts = embed_calls[0]["texts"]
        assert len(texts) == 3

        # Check first text has all components
        assert "Chief Executives" in texts[0]
        assert "Determine and formulate policies." in texts[0]
        assert "Making Decisions, Establishing Relationships" in texts[0]
        assert "Key tasks and skills:" in texts[0]

    def test_text_structure(self, tmp_path):
        """Text should follow: Title\\n\\nDescription\\n\\nKey tasks..."""
        result, embed_calls = _run_node(tmp_path)
        text = embed_calls[0]["texts"][1]

        parts = text.split("\n\n")
        assert parts[0] == "Software Developers"
        assert parts[1] == "Research, design, and develop software."
        assert parts[2].startswith("Key tasks and skills:")

# %% [markdown]
# ## Test prompt support

# %%
#|export
class TestPromptSupport:
    """Test document-side prompt support."""

    def test_document_prefix(self, tmp_path):
        """Models with document_prefix should have it prepended."""
        model_cfg = ("sbatch", {"model": "intfloat/e5-large-v2",
                                "document_prefix": "passage: "})
        result, embed_calls = _run_node(tmp_path, model_cfg_override=model_cfg)

        assert embed_calls[0]["texts"][0].startswith("passage: Chief Executives")

    def test_document_prompt_name(self, tmp_path):
        """Models with document_prompt_name should pass it to aembed."""
        model_cfg = ("sbatch", {"model": "google/EmbeddingGemma-300M",
                                "document_prompt_name": "document"})
        result, embed_calls = _run_node(tmp_path, model_cfg_override=model_cfg)

        assert embed_calls[0]["kwargs"]["prompt_name"] == "document"

    def test_no_prompt(self, tmp_path):
        """Models without prompt config should pass no prompt kwargs."""
        model_cfg = ("api", {"model": "text-embedding-3-large"})
        result, embed_calls = _run_node(tmp_path, model_cfg_override=model_cfg)

        assert "prompt" not in embed_calls[0]["kwargs"]
        assert "prompt_name" not in embed_calls[0]["kwargs"]

# %% [markdown]
# ## Test skip logic

# %%
#|export
class TestSkipLogic:
    """Test that existing embeddings are skipped."""

    def test_skips_when_files_exist(self, tmp_path):
        """Should return True without embedding when output files exist."""
        result, embed_calls = _run_node(tmp_path, pre_existing=True)

        assert result is True
        assert len(embed_calls) == 0

# %% [markdown]
# ## Test output format

# %%
#|export
class TestOutput:
    """Test output files and return value."""

    def test_returns_true(self, tmp_path):
        """Node should return True on success."""
        result, _ = _run_node(tmp_path)
        assert result is True

    def test_writes_embeddings_npy(self, tmp_path):
        """Should write onet_embeddings.npy with correct shape."""
        _run_node(tmp_path)

        npy_path = tmp_path / "pipeline" / "test_run" / "embed_onet" / "onet_embeddings.npy"
        assert npy_path.exists()
        embeddings = np.load(npy_path)
        assert embeddings.shape == (3, 4)
        assert embeddings.dtype == np.float32

    def test_writes_codes_json(self, tmp_path):
        """Should write onet_codes.json with correct codes."""
        _run_node(tmp_path)

        codes_path = tmp_path / "pipeline" / "test_run" / "embed_onet" / "onet_codes.json"
        assert codes_path.exists()
        codes = json.loads(codes_path.read_text())
        assert codes == ["11-1011.00", "15-1252.00", "29-1141.00"]

    def test_writes_meta_json(self, tmp_path):
        """Should write embed_meta.json with stats."""
        _run_node(tmp_path)

        meta_path = tmp_path / "pipeline" / "test_run" / "embed_onet" / "embed_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["n_occupations"] == 3
        assert meta["embedding_dim"] == 4
