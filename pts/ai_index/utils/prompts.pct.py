# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prompt library
#
# Load prompt templates from `prompt_library/` by path key.

# %%
#|default_exp utils.prompts

# %%
#|export
from pathlib import Path

from ai_index import const


def load_prompt(prompt_path: str) -> str:
    """Load a prompt template from the prompt library.

    Args:
        prompt_path: Slash-separated path relative to ``prompt_library/``,
            e.g. ``"llm_summarise/main/system"`` resolves to
            ``prompt_library/llm_summarise/main/system.md``.

    Returns:
        The prompt text (with leading/trailing whitespace stripped).

    Raises:
        FileNotFoundError: If the resolved ``.md`` file does not exist.
    """
    file_path = const.repo_root / "prompt_library" / f"{prompt_path}.md"
    if not file_path.is_file():
        raise FileNotFoundError(f"Prompt not found: {file_path}")
    return file_path.read_text().strip()
