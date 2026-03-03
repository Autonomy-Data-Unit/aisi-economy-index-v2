# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test: llm_runner CLI entry point

# %%
#|default_exp llm_runner.test_cli

# %%
#|export
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# %%
#|export
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
