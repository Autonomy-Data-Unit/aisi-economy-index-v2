# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # validation
#
# Model sensitivity analysis tools for the job-ad-to-O*NET matching pipeline.

# %%
#|default_exp __init__

# %%
#|export
import sys

from validation.run_validation import run_validation
from validation.run_all import plan_runs
from validation.analyze import analyze

# %%
#|export
def analyze_main():
    """CLI entry point for analyze-validation."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) != 1:
        print(
            "Usage: uv run analyze-validation <run_def_name>",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        analyze(args[0])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
