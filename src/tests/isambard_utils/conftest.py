import pytest
from isambard_utils.config import IsambardConfig


@pytest.fixture
def cfg():
    """Provide IsambardConfig from env, skip if unavailable."""
    try:
        config = IsambardConfig.from_env()
    except (ValueError, Exception) as e:
        pytest.skip(f"Isambard not available: {e}")
    from isambard_utils.ssh import check_connection
    if not check_connection(config):
        pytest.skip("Cannot connect to Isambard (SSH failed)")
    return config
