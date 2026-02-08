# AGENT_INSTRUCTIONS.md - nblite for LLM Agents

This document provides guidance for LLM agents contributing to nblite projects.

---

## Overview

**nblite** is a notebook-driven Python development tool that enables *literate programming*. In nblite projects, source code lives in notebooks (`.ipynb` or `.pct.py` files), which are then exported to Python modules.

**Critical rules**:

1. **Never modify the exported Python module directly.** Those files are auto-generated and will be overwritten by `nbl export`.

2. **If you edit `.pct.py` files, always run `nbl export --reverse` afterward.** Otherwise, when `nbl export` runs (which typically goes `nbs->pts->lib`), your changes in `pts` will be overwritten by the older `nbs` versions.

---

## Project Structure

nblite projects typically have these directories:

```
project/
├── nblite.toml          # Configuration file
├── nbs/                  # Jupyter notebooks (.ipynb) - often the source of truth
│   ├── mod/             # Module notebooks
│   │   ├── core.ipynb
│   │   ├── utils.ipynb
│   │   ├── _utils/      # Submodules (underscore = internal)
│   │   └── cmds/
│   └── tests/           # Test notebooks
├── pts/                  # Percent notebooks (.pct.py) - plaintext format
│   ├── mod/
│   │   ├── core.pct.py
│   │   ├── utils.pct.py
│   │   ├── _utils/
│   │   └── cmds/
│   └── tests/           # Test notebooks (plaintext)
└── src/                  # Auto-generated code (DO NOT EDIT)
    ├── my_package/      # Python module
    │   ├── __init__.py
    │   ├── core.py
    │   └── utils.py
    └── tests/           # Test files
```

But the actual folder structure may vary. See the folder structure of the project for yourself, and also inspect the `nblite.toml` to see how the project is structured.

### Understanding `nblite.toml`

The `nblite.toml` file defines code locations and export pipelines:

```toml
export_pipeline = """
nbs -> pts
nbs -> lib
nbs_test -> lib_test
nbs_test -> pts_test
"""

[cl.lib]
path="my_package"

[cl.nbs]
format="ipynb"
path="nbs/mod"

[cl.pts]
format="percent"
path="pts/mod"

[cl.lib_test]
path="src/tests"
format="module"

[cl.nbs_test]
path="nbs/tests"
format="ipynb"

[cl.pts_test]
path="pts/tests"
format="percent"
```

The `export_pipeline` defines the flow: notebooks export to percent format, which exports to the Python module.

---

## Where to Make Changes

### Preferred: Edit `.pct.py` files (percent notebooks)

Percent notebooks are plaintext Python files that use `# %%` to delimit cells. They are much easier to edit than `.ipynb` files.

After editing a `.pct.py` file, run:
```bash
nbl export --reverse
```

This syncs your changes back to the `.ipynb` notebooks. If you skip this step, your changes will be lost when someone runs `nbl export` (which exports nbs->pts->lib).

### Alternative: Edit `.ipynb` files

If the project only has `.ipynb` notebooks (no `pts` folder), edit those directly. After editing, run:
```bash
nbl export
```

### Never Edit: The Python module

Files in the module directory (e.g., `my_package/`) are auto-generated. Any changes will be overwritten by `nbl export`.

---

## Notebook Cell Types and Directives

### Cell Delimiters (in .pct.py files)

```python
# %%
# Code cell

# %% [markdown]
# # Markdown cell
```

### Essential Directives

Place directives at the top of cells (the "topmatter"):

```python
# %%
#|default_exp core       # Sets the export target module (once per notebook)
```

```python
# %%
#|export                 # Export this cell to the module
def my_function():
    pass
```

```python
# %%
#|exporti                # Export but exclude from __all__ (internal)
def _internal_helper():
    pass
```

```python
# %%
#|hide                   # Hide from documentation (used for setup/testing cells)
import nblite; nblite.nbl_export()
```

### Function Export Mode

Regular notebooks export each `#|export` cell as top-level module code (functions, classes, constants). **Function notebooks** are different: the entire notebook becomes a single callable function.

This is useful for:
- **CLI commands** - Each command is a notebook that exports as one function
- **Complex workflows** - Multi-step processes that benefit from notebook-style development
- **Scripts** - Operations that need to be callable programmatically

In function export mode:
- All `#|export` cells become the **function body** (indented inside the function)
- `#|top_export` cells stay at **module level** (imports, constants). Cells exported using this directive get exported to the top of the Python file, outside of the function declaration. This is useful if types need to be imported for the function signature, or if you want to define extra helper functions that you also want to be re-usable from outside the submodule.
- `#|set_func_signature` defines the function's name, parameters, and docstring
- `#|func_return` adds a `return` statement to the first line of the cell
- `#|return_line` is an inline directive that adds `return` to a specific line (e.g., `result  #|return_line`)

Example:

```python
# %%
#|default_exp cmds._my_command
#|export_as_func true

# %%
#|top_export             # Code placed at module level, outside the function
from pathlib import Path
import subprocess

# %%
#|set_func_signature     # Define the function signature
def my_command(arg1: str, arg2: int = 10) -> dict:
    """Docstring goes here."""
    ...

# %%
#|export                 # Function body cells
result = do_something(arg1)

# %%
#|export
more_processing()

# %%
#|func_return            # Prepend 'return' to first line
result;

# %%
#|export
# Alternative: use inline directive for specific line
if success:
    result  #|return_line
else:
    None  #|return_line
```

### Execution Control Directives

```python
# %%
#|eval: false            # Skip this cell during nbl fill/test

# %%
#|skip_evals             # Skip this and all following cells

# %%
#|skip_evals_stop        # Resume execution
```

---

## Literate Programming Style

nblite encourages combining production code with documentation and examples. Structure notebooks like this:

### 1. Header and Setup

```python
# %% [markdown]
# # Module Name
# Brief description of what this module does.

# %%
#|default_exp my_module

# %%
#|hide
import nblite; nblite.nbl_export()
```

### 2. Imports (exported)

```python
# %%
#|export
from pathlib import Path
from typing import Optional
```

### 3. Function/Class with Documentation

```python
# %% [markdown]
# ## `process_data`
#
# This function processes input data and returns results.

# %%
#|export
def process_data(input_path: str) -> dict:
    """
    Process data from the given path.

    Args:
        input_path: Path to input file.

    Returns:
        Dictionary with processed results.
    """
    path = Path(input_path)
    content = path.read_text()
    return {"lines": len(content.splitlines())}
```

### 4. Example Usage (not exported)

```python
# %% [markdown]
# ### Example

# %%
# Create a test file
test_file = Path("/tmp/test.txt")
test_file.write_text("line 1\nline 2\nline 3")

# Process it
result = process_data("/tmp/test.txt")
print(result)  # {'lines': 3}
```

### 5. More Functions...

Continue alternating between exported code and examples/documentation.

---

## CLI Commands Reference

### `nbl export`

Export notebooks to modules:

```bash
nbl export                              # Use config pipeline
nbl export --pipeline "pts->lib" # Custom pipeline
nbl export --pipeline "pts->nbs" # Reverse: sync pts back to nbs
nbl export --dry-run                    # Preview without writing
```

### `nbl fill`

Execute notebooks and save outputs:

```bash
nbl fill                    # Fill all notebooks
nbl fill path/to/nb.ipynb   # Fill specific notebook
nbl fill --timeout 60       # Set cell timeout (seconds)
nbl fill --workers 4        # Parallel workers
```

### `nbl test`

Test notebooks execute without errors (dry run):

```bash
nbl test                    # Test all notebooks
nbl test path/to/nb.ipynb   # Test specific notebook
```

### `nbl new`

Create a new notebook:

```bash
nbl new nbs/mod/my_module.ipynb              # Create ipynb
nbl new pts/mod/my_module.pct.py             # Create pct.py
nbl new pts/mod/cmd.pct.py --template script # Function export template
```

### `nbl clean`

Remove outputs and metadata from notebooks:

```bash
nbl clean                   # Clean all notebooks
nbl clean path/to/nb.ipynb  # Clean specific notebook
```

---

## Typical Workflow for LLM Agents

### Adding a New Feature

1. **Create the notebook** (if it doesn't exist):
   ```bash
   nbl new pts/mod/new_feature.pct.py
   ```

2. **Edit the `.pct.py` file** with your implementation, following literate programming style

3. **Sync back to ipynb** (if the project uses nbs as source of truth):
   ```bash
   nbl export --pipeline "pts->nbs"
   ```

4. **Export to module**:
   ```bash
   nbl export
   ```

5. **Test the notebooks execute**:
   ```bash
   nbl test
   ```

### Modifying Existing Code

1. **Find the source notebook** - look in `pts/` or `nbs/`, not the module directory

2. **Edit the `.pct.py` file** (preferred) or `.ipynb` file

3. **Sync and export**:
   ```bash
   nbl export --reverse  # If editing pts
   nbl export                                # Generate module
   ```

4. **Test**:
   ```bash
   nbl test
   ```

### Debugging

1. **Run the notebook** to see actual outputs:
   ```bash
   nbl fill path/to/notebook.ipynb
   ```

2. **Check for errors** in non-exported cells (example code, tests)

---

## Import Conventions

nblite automatically converts absolute imports to relative imports during export. Write imports naturally:

```python
# %%
#|export
from my_package.utils import helper  # Will become: from .utils import helper
```

For modules in subdirectories, the relative import depth is calculated automatically:

```python
# In my_package/submodule/feature.py
from my_package.core import something  # Becomes: from ..core import something
```

---

## `__init__.py` Management

nblite auto-generates `__init__.py` files for packages that contain non-underscore-prefixed modules. However, packages with only underscore-prefixed modules (e.g., `_context.py`, `_net.py`) do **not** get an auto-generated `__init__.py`.

In this project, **`__init__.py` files are usually manually managed** — they contain explicit re-exports to control the public API of each package. When adding a new module to a package, you must manually update its `__init__.py` to re-export any new public symbols. The `nbl export` command will not overwrite manually managed `__init__.py` files.

---

## Common Mistakes to Avoid

1. **Editing the Python module directly** - Changes will be overwritten. This should only be done if you want to quickly test something, or add a debug `print` statement, and if it is not a concern that it may get overwritten.

2. **Forgetting `nbl export --reverse`** - If you edit `.pct.py` files, sync them back to `.ipynb` before committing, otherwise your changes get lost

3. **Missing `#|export` directive** - Code without this directive won't appear in the module

4. **Putting `#|default_exp` in wrong place** - It should be in a cell near the top of the notebook, once per notebook

5. **Not writing example cells** - Literate programming means showing how to use your code, not just the code itself

---

## Quick Reference: Directives

| Directive | Description |
|-----------|-------------|
| `#|default_exp mod` | Set default export module path |
| `#|export` | Export cell to module |
| `#|exporti` | Export cell but exclude from `__all__` |
| `#|export_to mod` | Export cell to specific module |
| `#|hide` | Hide cell from documentation |
| `#|hide_input` | Hide input, show output in docs |
| `#|hide_output` | Show input, hide output in docs |
| `#|eval: false` | Skip cell during execution |
| `#|skip_evals` | Skip this and following cells |
| `#|skip_evals_stop` | Resume cell execution |
| `#|export_as_func true` | Export notebook as callable function |
| `#|set_func_signature` | Define function signature |
| `#|top_export` | Code at module level (for function export) |
| `#|func_return` | Prepend 'return' to first line |

---

## Summary

1. **Edit `.pct.py` files** when available (easier than `.ipynb`)
2. **Never edit the Python module** - it's auto-generated
3. **Run `nbl export --pipeline "pts->nbs"`** after editing `.pct.py` files
4. **Run `nbl export`** to generate the Python module
5. **Follow literate programming style**: exported code + markdown documentation + example cells
6. **Run `nbl test`** to verify notebooks execute correctly
