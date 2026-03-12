# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test: Orchestrate (content-addressed jobs)
#
# Unit tests for `compute_job_hash` and the idempotency state machine.
# All tests are mocked, no Isambard connection required.

# %%
#|default_exp isambard_utils.test_orchestrate

# %%
#|export
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# %% [markdown]
# ## TestComputeJobHash

# %%
#|export
class TestComputeJobHash:
    """Test isambard_utils.orchestrate.compute_job_hash."""

    def test_deterministic(self):
        from isambard_utils.orchestrate import compute_job_hash
        inputs = {"texts": ["hello", "world"]}
        config = {"model": "test-model", "batch_size": 32}
        h1 = compute_job_hash("embed", inputs, config)
        h2 = compute_job_hash("embed", inputs, config)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_different_inputs_different_hash(self):
        from isambard_utils.orchestrate import compute_job_hash
        config = {"model": "test-model"}
        h1 = compute_job_hash("embed", {"texts": ["a"]}, config)
        h2 = compute_job_hash("embed", {"texts": ["b"]}, config)
        assert h1 != h2

    def test_different_config_different_hash(self):
        from isambard_utils.orchestrate import compute_job_hash
        inputs = {"texts": ["hello"]}
        h1 = compute_job_hash("embed", inputs, {"model": "model-a"})
        h2 = compute_job_hash("embed", inputs, {"model": "model-b"})
        assert h1 != h2

    def test_different_operation_different_hash(self):
        from isambard_utils.orchestrate import compute_job_hash
        inputs = {"texts": ["hello"]}
        config = {"model": "test-model"}
        h1 = compute_job_hash("embed", inputs, config)
        h2 = compute_job_hash("llm_generate", inputs, config)
        assert h1 != h2

    def test_config_key_order_irrelevant(self):
        from isambard_utils.orchestrate import compute_job_hash
        inputs = {"texts": ["hello"]}
        h1 = compute_job_hash("embed", inputs, {"a": 1, "b": 2})
        h2 = compute_job_hash("embed", inputs, {"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_excludes_orchestration_params(self):
        from isambard_utils.orchestrate import compute_job_hash
        inputs = {"texts": ["hello"]}
        config_base = {"model": "test-model"}
        config_with_orch = {"model": "test-model", "setup": True, "job_name": "foo",
                            "time": "01:00:00", "print_fn": print, "cache": True,
                            "upload_timeout": 300}
        h1 = compute_job_hash("embed", inputs, config_base)
        h2 = compute_job_hash("embed", inputs, config_with_orch)
        assert h1 == h2

    def test_numpy_content_matters(self):
        from isambard_utils.orchestrate import compute_job_hash
        config = {"model": "test-model"}
        h1 = compute_job_hash("embed", {"A": np.array([1.0, 2.0])}, config)
        h2 = compute_job_hash("embed", {"A": np.array([1.0, 3.0])}, config)
        assert h1 != h2

    def test_ndarray_vs_list_same_hash(self):
        from isambard_utils.orchestrate import compute_job_hash
        config = {"model": "test-model"}
        h1 = compute_job_hash("embed", {"texts": np.array(["a", "b"])}, config)
        h2 = compute_job_hash("embed", {"texts": ["a", "b"]}, config)
        assert h1 == h2

    def test_unsupported_type_raises(self):
        from isambard_utils.orchestrate import compute_job_hash
        config = {"model": "test-model"}
        with pytest.raises(TypeError, match="Cannot hash"):
            compute_job_hash("embed", {"obj": object()}, config)

# %% [markdown]
# ## TestIdempotencyStateMachine

# %%
#|export
def _make_mock_config():
    """Create a mock IsambardConfig for testing."""
    cfg = MagicMock()
    cfg.project_dir = "/projects/test"
    cfg.ssh_host = "test-host"
    cfg.ssh_user = "test-user"
    cfg.hf_cache_dir = "/projects/test/hf_cache"
    cfg.logs_dir = "/projects/test/logs"
    return cfg

# %%
#|export
class TestIdempotencyStateMachine:
    """Test the idempotency state machine in arun_remote."""

    @pytest.mark.asyncio
    async def test_cache_miss_submits_job(self):
        from isambard_utils.orchestrate import arun_remote, _JOB_LOCKS
        _JOB_LOCKS.clear()

        mock_config = _make_mock_config()
        mock_job = MagicMock()
        mock_job.job_id = "12345"

        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate.asetup_runner", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock, return_value=None) as mock_read, \
             patch("isambard_utils.orchestrate._write_remote_status", new_callable=AsyncMock) as mock_write, \
             patch("isambard_utils.orchestrate._upload_inputs", new_callable=AsyncMock, return_value={"texts": "/remote/path"}) as mock_upload, \
             patch("isambard_utils.orchestrate._submit_job", new_callable=AsyncMock, return_value=mock_job) as mock_submit, \
             patch("isambard_utils.orchestrate._poll_job", new_callable=AsyncMock) as mock_poll, \
             patch("isambard_utils.orchestrate._download_outputs", new_callable=AsyncMock, return_value={"embeddings": [1, 2, 3]}) as mock_download:

            result = await arun_remote(
                "embed", {"texts": ["hello"]}, {"model": "test"},
                job_name="embed", setup=True, cache=True,
            )

            assert result == {"embeddings": [1, 2, 3]}
            mock_upload.assert_called_once()
            mock_submit.assert_called_once()
            mock_poll.assert_called_once()
            mock_download.assert_called_once()

            # Should have written uploading, submitted, completed states
            states_written = [call.args[1]["state"] for call in mock_write.call_args_list]
            assert "uploading" in states_written
            assert "submitted" in states_written
            assert "completed" in states_written

    @pytest.mark.asyncio
    async def test_cache_hit_completed_returns_immediately(self):
        from isambard_utils.orchestrate import arun_remote, _JOB_LOCKS
        _JOB_LOCKS.clear()

        mock_config = _make_mock_config()

        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate.asetup_runner", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock, return_value={"state": "completed", "job_id": "99"}), \
             patch("isambard_utils.orchestrate._submit_job", new_callable=AsyncMock) as mock_submit, \
             patch("isambard_utils.orchestrate._upload_inputs", new_callable=AsyncMock) as mock_upload, \
             patch("isambard_utils.orchestrate._download_outputs", new_callable=AsyncMock, return_value={"result": "cached"}):

            result = await arun_remote(
                "embed", {"texts": ["hello"]}, {"model": "test"},
                job_name="embed", cache=True,
            )

            assert result == {"result": "cached"}
            mock_submit.assert_not_called()
            mock_upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_running_attaches(self):
        from isambard_utils.orchestrate import arun_remote, _JOB_LOCKS
        _JOB_LOCKS.clear()

        mock_config = _make_mock_config()

        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate.asetup_runner", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock, return_value={"state": "submitted", "job_id": "555"}), \
             patch("isambard_utils.orchestrate._check_job_alive", new_callable=AsyncMock, return_value="RUNNING"), \
             patch("isambard_utils.orchestrate._write_remote_status", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._submit_job", new_callable=AsyncMock) as mock_submit, \
             patch("isambard_utils.orchestrate._poll_job", new_callable=AsyncMock) as mock_poll, \
             patch("isambard_utils.orchestrate._download_outputs", new_callable=AsyncMock, return_value={"result": "attached"}):

            result = await arun_remote(
                "embed", {"texts": ["hello"]}, {"model": "test"},
                job_name="embed", cache=True,
            )

            assert result == {"result": "attached"}
            mock_submit.assert_not_called()
            mock_poll.assert_called_once()
            assert mock_poll.call_args[0][0] == "555"
            assert mock_poll.call_args.kwargs["config"] == mock_config

    @pytest.mark.asyncio
    async def test_cache_hit_stale_resubmits(self):
        from isambard_utils.orchestrate import arun_remote, _JOB_LOCKS
        _JOB_LOCKS.clear()

        mock_config = _make_mock_config()
        mock_job = MagicMock()
        mock_job.job_id = "777"

        # First read returns "submitted" with dead job, second read returns "failed"
        read_side_effects = [
            {"state": "submitted", "job_id": "666"},
            {"state": "failed", "job_id": "666"},
        ]

        mock_ssh = AsyncMock()
        mock_ssh.return_value = MagicMock(returncode=0, stdout='{"texts": "/remote/path"}')

        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate.asetup_runner", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock, side_effect=read_side_effects), \
             patch("isambard_utils.orchestrate._check_job_alive", new_callable=AsyncMock, return_value="FAILED"), \
             patch("isambard_utils.orchestrate._write_remote_status", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate.async_ssh_run", new_callable=AsyncMock, return_value=MagicMock(returncode=0, stdout='{"texts": "/remote/path"}')), \
             patch("isambard_utils.orchestrate._submit_job", new_callable=AsyncMock, return_value=mock_job) as mock_submit, \
             patch("isambard_utils.orchestrate._poll_job", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._download_outputs", new_callable=AsyncMock, return_value={"result": "resubmitted"}):

            result = await arun_remote(
                "embed", {"texts": ["hello"]}, {"model": "test"},
                job_name="embed", cache=True,
            )

            assert result == {"result": "resubmitted"}
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_failed_resubmits(self):
        from isambard_utils.orchestrate import arun_remote, _JOB_LOCKS
        _JOB_LOCKS.clear()

        mock_config = _make_mock_config()
        mock_job = MagicMock()
        mock_job.job_id = "888"

        mock_ssh = MagicMock(returncode=0, stdout='{"texts": "/remote/path"}')

        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate.asetup_runner", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock, return_value={"state": "failed", "job_id": "old"}), \
             patch("isambard_utils.orchestrate._write_remote_status", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate.async_ssh_run", new_callable=AsyncMock, return_value=mock_ssh), \
             patch("isambard_utils.orchestrate._submit_job", new_callable=AsyncMock, return_value=mock_job) as mock_submit, \
             patch("isambard_utils.orchestrate._poll_job", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._download_outputs", new_callable=AsyncMock, return_value={"result": "resubmitted"}):

            result = await arun_remote(
                "embed", {"texts": ["hello"]}, {"model": "test"},
                job_name="embed", cache=True,
            )

            assert result == {"result": "resubmitted"}
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_name_includes_hash_prefix(self):
        from isambard_utils.orchestrate import arun_remote, compute_job_hash, _JOB_LOCKS
        _JOB_LOCKS.clear()

        mock_config = _make_mock_config()
        mock_job = MagicMock()
        mock_job.job_id = "123"

        inputs = {"texts": ["hello"]}
        config = {"model": "test"}
        expected_hash = compute_job_hash("embed", inputs, config)

        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate.asetup_runner", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock, return_value=None), \
             patch("isambard_utils.orchestrate._write_remote_status", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._upload_inputs", new_callable=AsyncMock, return_value={}), \
             patch("isambard_utils.orchestrate._submit_job", new_callable=AsyncMock, return_value=mock_job) as mock_submit, \
             patch("isambard_utils.orchestrate._poll_job", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._download_outputs", new_callable=AsyncMock, return_value={}):

            await arun_remote(
                "embed", inputs, config,
                job_name="embed", cache=True,
            )

            submit_call = mock_submit.call_args
            assert submit_call.kwargs["slurm_job_name"] == f"embed_{expected_hash[:8]}"

    @pytest.mark.asyncio
    async def test_concurrent_same_hash_uses_lock(self):
        from isambard_utils.orchestrate import arun_remote, _JOB_LOCKS
        _JOB_LOCKS.clear()

        mock_config = _make_mock_config()
        mock_job = MagicMock()
        mock_job.job_id = "111"

        call_order = []

        async def mock_upload(*args, **kwargs):
            call_order.append("upload_start")
            await asyncio.sleep(0.05)
            call_order.append("upload_end")
            return {"texts": "/remote/path"}

        # First call: cache miss -> upload. Second call: sees completed.
        read_calls = [0]
        async def mock_read(*args, **kwargs):
            read_calls[0] += 1
            if read_calls[0] <= 1:
                return None  # cache miss for first call
            return {"state": "completed", "job_id": "111"}  # cache hit for second

        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate.asetup_runner", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._read_remote_status", side_effect=mock_read), \
             patch("isambard_utils.orchestrate._write_remote_status", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._upload_inputs", side_effect=mock_upload), \
             patch("isambard_utils.orchestrate._submit_job", new_callable=AsyncMock, return_value=mock_job), \
             patch("isambard_utils.orchestrate._poll_job", new_callable=AsyncMock), \
             patch("isambard_utils.orchestrate._download_outputs", new_callable=AsyncMock, return_value={"result": "ok"}):

            inputs = {"texts": ["hello"]}
            config = {"model": "test"}

            r1, r2 = await asyncio.gather(
                arun_remote("embed", inputs, config, job_name="embed", cache=True),
                arun_remote("embed", inputs, config, job_name="embed", cache=True),
            )

            # Both should succeed
            assert r1 == {"result": "ok"}
            assert r2 == {"result": "ok"}

# %% [markdown]
# ## TestJobStatus

# %%
#|export
class TestJobStatus:
    """Test job_status / ajob_status."""

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_hash(self):
        from isambard_utils.orchestrate import ajob_status

        mock_config = _make_mock_config()
        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock, return_value=None):
            result = await ajob_status("nonexistent_hash", config=mock_config)
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_completed_status(self):
        from isambard_utils.orchestrate import ajob_status

        mock_config = _make_mock_config()
        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock,
                   return_value={"state": "completed", "job_id": "42"}):
            result = await ajob_status("abc123", config=mock_config)
            assert result["state"] == "completed"
            assert result["hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_detects_stale_running_job(self):
        from isambard_utils.orchestrate import ajob_status

        mock_config = _make_mock_config()
        with patch("isambard_utils.orchestrate._get_config", return_value=mock_config), \
             patch("isambard_utils.orchestrate._read_remote_status", new_callable=AsyncMock,
                   return_value={"state": "submitted", "job_id": "99"}), \
             patch("isambard_utils.orchestrate._check_job_alive", new_callable=AsyncMock, return_value="FAILED"):
            result = await ajob_status("abc123", config=mock_config)
            assert result["state"] == "failed"
            assert result["slurm_final_state"] == "FAILED"
