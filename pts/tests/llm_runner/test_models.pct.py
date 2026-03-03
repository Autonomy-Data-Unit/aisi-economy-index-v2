# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test: llm_runner ApiLLM with mocked adulib

# %%
#|default_exp llm_runner.test_models

# %%
#|export
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# %%
#|export
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
