"""Unit tests for the llm_runner package."""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    """Test llm_runner.serialization serialize/deserialize round-trip."""

    def test_numpy_roundtrip(self):
        from llm_runner.serialization import serialize, deserialize
        data = {"embeddings": np.random.randn(10, 64).astype(np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            serialize(data, Path(tmp))
            result = deserialize(Path(tmp))
        np.testing.assert_array_equal(result["embeddings"], data["embeddings"])

    def test_json_roundtrip(self):
        from llm_runner.serialization import serialize, deserialize
        data = {
            "texts": ["hello", "world"],
            "config": {"k": 5, "nested": [1, 2, 3]},
            "count": 42,
            "flag": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            serialize(data, Path(tmp))
            result = deserialize(Path(tmp))
        assert result == data

    def test_mixed_roundtrip(self):
        from llm_runner.serialization import serialize, deserialize
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        data = {
            "embeddings": arr,
            "labels": ["a", "b", "c"],
            "meta": {"version": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            serialize(data, Path(tmp))
            result = deserialize(Path(tmp))
        np.testing.assert_array_equal(result["embeddings"], arr)
        assert result["labels"] == ["a", "b", "c"]
        assert result["meta"] == {"version": 1}

    def test_pickle_fallback(self):
        """Non-JSON-serializable objects should fall back to pickle."""
        from llm_runner.serialization import serialize, deserialize
        data = {"obj": set([1, 2, 3])}
        with tempfile.TemporaryDirectory() as tmp:
            serialize(data, Path(tmp))
            result = deserialize(Path(tmp))
        assert result["obj"] == {1, 2, 3}

    def test_manifest_structure(self):
        from llm_runner.serialization import serialize
        data = {"arr": np.zeros(5), "text": "hello", "obj": set([1])}
        with tempfile.TemporaryDirectory() as tmp:
            serialize(data, Path(tmp))
            with open(Path(tmp) / "_manifest.json") as f:
                manifest = json.load(f)
        assert manifest["arr"]["type"] == "npy"
        assert manifest["text"]["type"] == "json"
        assert manifest["obj"]["type"] == "pkl"

    def test_empty_dict(self):
        from llm_runner.serialization import serialize, deserialize
        with tempfile.TemporaryDirectory() as tmp:
            serialize({}, Path(tmp))
            result = deserialize(Path(tmp))
        assert result == {}


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

class TestContentHash:
    """Test isambard_utils.transfer.compute_content_hash."""

    def test_deterministic(self):
        from isambard_utils.transfer import compute_content_hash
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("hello")
            (Path(tmp) / "b.bin").write_bytes(b"\x00\x01\x02")
            h1 = compute_content_hash(tmp)
            h2 = compute_content_hash(tmp)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_different_content_different_hash(self):
        from isambard_utils.transfer import compute_content_hash
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            (Path(t1) / "f.txt").write_text("aaa")
            (Path(t2) / "f.txt").write_text("bbb")
            assert compute_content_hash(t1) != compute_content_hash(t2)

    def test_same_content_same_hash(self):
        from isambard_utils.transfer import compute_content_hash
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            (Path(t1) / "f.txt").write_text("same")
            (Path(t2) / "f.txt").write_text("same")
            assert compute_content_hash(t1) == compute_content_hash(t2)


# ---------------------------------------------------------------------------
# Cosine top-K (CPU)
# ---------------------------------------------------------------------------

class TestCosineTopK:
    """Test llm_runner.cosine.run_cosine_topk on CPU."""

    def test_basic_topk(self):
        from llm_runner.cosine import run_cosine_topk
        # Create simple embeddings where we know the answer
        A = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = run_cosine_topk(A, B, k=2, device="cpu")
        assert result["indices"].shape == (2, 2)
        assert result["scores"].shape == (2, 2)
        # Row 0 of A is [1,0,0], should match B[0] first
        assert result["indices"][0, 0] == 0
        # Row 1 of A is [0,1,0], should match B[1] first
        assert result["indices"][1, 0] == 1

    def test_k_larger_than_candidates(self):
        from llm_runner.cosine import run_cosine_topk
        A = np.random.randn(5, 10).astype(np.float32)
        B = np.random.randn(3, 10).astype(np.float32)
        result = run_cosine_topk(A, B, k=10, device="cpu")
        # k should be clamped to B.shape[0]
        assert result["indices"].shape == (5, 3)
        assert result["scores"].shape == (5, 3)

    def test_scores_sorted_descending(self):
        from llm_runner.cosine import run_cosine_topk
        A = np.random.randn(3, 8).astype(np.float32)
        B = np.random.randn(10, 8).astype(np.float32)
        result = run_cosine_topk(A, B, k=5, device="cpu")
        for i in range(3):
            scores = result["scores"][i]
            assert np.all(scores[:-1] >= scores[1:]), "Scores not sorted descending"


# ---------------------------------------------------------------------------
# ApiLLM with mocked adulib
# ---------------------------------------------------------------------------

class TestApiLLM:
    """Test ApiLLM.generate() with mocked adulib."""

    def test_generate_single_prompt(self):
        from llm_runner.models import ApiLLM

        mock_async_single = AsyncMock(return_value=("test response", False, {}))
        mock_batch_executor = AsyncMock(return_value=["test response"])

        with patch.dict("sys.modules", {
            "adulib": MagicMock(),
            "adulib.llm": MagicMock(async_single=mock_async_single),
            "adulib.asynchronous": MagicMock(batch_executor=mock_batch_executor),
        }):
            llm = ApiLLM(model_name="test-model")
            results = llm.generate("hello", max_new_tokens=100, system_message="be helpful")

        assert results == ["test response"]

    def test_generate_multiple_prompts(self):
        from llm_runner.models import ApiLLM

        responses = ["resp1", "resp2", "resp3"]
        mock_batch_executor = AsyncMock(return_value=responses)

        with patch.dict("sys.modules", {
            "adulib": MagicMock(),
            "adulib.llm": MagicMock(async_single=AsyncMock()),
            "adulib.asynchronous": MagicMock(batch_executor=mock_batch_executor),
        }):
            llm = ApiLLM(model_name="test-model", max_concurrent=64)
            results = llm.generate(["a", "b", "c"])

        assert len(results) == 3


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    """Test the llm_runner CLI entry point."""

    def test_cosine_topk_via_cli(self):
        from llm_runner.serialization import serialize, deserialize
        from llm_runner.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            inputs_dir = Path(tmp) / "inputs"
            outputs_dir = Path(tmp) / "outputs"

            # Prepare inputs
            A = np.random.randn(4, 8).astype(np.float32)
            B = np.random.randn(10, 8).astype(np.float32)
            serialize({"A": A, "B": B}, inputs_dir)

            # Run CLI
            config = json.dumps({"k": 3, "device": "cpu"})
            main(["cosine_topk", "--inputs-dir", str(inputs_dir),
                  "--outputs-dir", str(outputs_dir), "--config", config])

            # Check outputs
            result = deserialize(outputs_dir)
            assert "indices" in result
            assert "scores" in result
            assert result["indices"].shape == (4, 3)

            # Check status.json
            with open(outputs_dir / "status.json") as f:
                status = json.load(f)
            assert status["state"] == "COMPLETED"
