# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test: isambard_utils content hash

# %%
#|default_exp isambard_utils.test_transfer

# %%
#|export
import tempfile
from pathlib import Path

import pytest

# %%
#|export
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
