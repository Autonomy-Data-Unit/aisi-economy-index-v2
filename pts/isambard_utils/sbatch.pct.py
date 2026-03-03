# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SBATCH Script Generation
#
# Generate SBATCH scripts from configuration for Isambard Slurm jobs.

# %%
#|default_exp sbatch

# %%
#|export
from dataclasses import dataclass, field
from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import _get_config

# %%
#|export
@dataclass
class SbatchConfig:
    """Configuration for generating an SBATCH script."""
    job_name: str
    partition: str = "workq"
    gpus: int = 1
    cpus_per_task: int = 16
    mem: str = "80G"
    time: str = "12:00:00"
    array: str | None = None
    modules: list[str] = field(default_factory=lambda: ["cudatoolkit/24.11_12.6"])
    env_vars: dict[str, str] = field(default_factory=dict)
    pre_commands: list[str] = field(default_factory=list)
    python_script: str | None = None
    python_command: str | None = None
    command: str | None = None

# %%
#|export
def generate(sbatch_config: SbatchConfig, *,
             isambard_config: IsambardConfig | None = None) -> str:
    """Generate a complete SBATCH script string.

    Args:
        sbatch_config: Job-specific configuration.
        isambard_config: Isambard cluster configuration.
    """
    isambard_config = _get_config(isambard_config)
    sc = sbatch_config
    ic = isambard_config

    lines = ["#!/bin/bash"]

    # SBATCH directives
    lines.append(f"#SBATCH --job-name={sc.job_name}")
    lines.append(f"#SBATCH --partition={sc.partition}")
    lines.append(f"#SBATCH --output={ic.logs_dir}/%x_%j.out")
    lines.append(f"#SBATCH --error={ic.logs_dir}/%x_%j.err")
    lines.append(f"#SBATCH --gpus={sc.gpus}")
    lines.append(f"#SBATCH --cpus-per-task={sc.cpus_per_task}")
    lines.append(f"#SBATCH --mem={sc.mem}")
    lines.append(f"#SBATCH --time={sc.time}")
    lines.append(f"#SBATCH --ntasks=1")
    if sc.array:
        lines.append(f"#SBATCH --array={sc.array}")
    lines.append("")

    # Shell setup
    lines.append("set -euo pipefail")
    lines.append(f"cd {ic.project_dir}")
    lines.append("")

    # Module loading
    lines.append("module purge")
    for mod in sc.modules:
        lines.append(f"module load {mod}")
    lines.append("")

    # Activate venv
    lines.append("source .venv/bin/activate")
    lines.append("")

    # Environment variables
    lines.append(f'export HF_HUB_CACHE="{ic.hf_cache_dir}"')
    lines.append('export HF_HUB_DISABLE_TELEMETRY=1')
    lines.append('export TOKENIZERS_PARALLELISM=false')
    for key, val in sc.env_vars.items():
        lines.append(f'export {key}="{val}"')
    lines.append("")

    # Pre-commands
    for cmd in sc.pre_commands:
        lines.append(cmd)
    if sc.pre_commands:
        lines.append("")

    # Main execution
    if sc.python_script:
        lines.append(f'srun python "{sc.python_script}"')
    elif sc.python_command:
        lines.append(f'srun python -c {_shell_quote(sc.python_command)}')
    elif sc.command:
        lines.append(f'srun {sc.command}')
    else:
        lines.append("# No command specified")

    lines.append("")
    return "\n".join(lines)

# %%
#|export
def _shell_quote(s: str) -> str:
    """Quote a string for safe shell use."""
    import shlex
    return shlex.quote(s)
