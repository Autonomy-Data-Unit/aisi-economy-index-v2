# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Configuration
#
# Pydantic models for Isambard HPC settings.

# %%
#|default_exp config

# %%
#|export
import os
from pydantic import BaseModel, model_validator

# %%
#|export
class IsambardConfig(BaseModel):
    """Configuration for Isambard HPC cluster access."""

    ssh_host: str
    ssh_user: str | None = None
    project_dir: str = "/projects/a5u/adu_dev/aisi-economy-index-v2"
    hf_cache_dir: str = "{project_dir}/hf_cache"
    logs_dir: str = "{project_dir}/logs"
    partition: str = "workq"
    default_gpus: int = 1
    default_cpus: int = 16
    default_mem: str = "80G"
    default_time: str = "12:00:00"
    cuda_module: str = "cudatoolkit/24.11_12.6"
    python_version: str = "3.12"

    @model_validator(mode="after")
    def _interpolate_paths(self):
        """Interpolate {project_dir} in path fields."""
        self.hf_cache_dir = self.hf_cache_dir.format(project_dir=self.project_dir)
        self.logs_dir = self.logs_dir.format(project_dir=self.project_dir)
        return self

    @classmethod
    def from_env(cls, **overrides) -> "IsambardConfig":
        """Load config with ssh_host from ISAMBARD_HOST env var (.env file)."""
        ssh_host = os.environ.get("ISAMBARD_HOST")
        if not ssh_host:
            raise ValueError("ISAMBARD_HOST not set. Check your .env file.")
        return cls(ssh_host=ssh_host, **overrides)
