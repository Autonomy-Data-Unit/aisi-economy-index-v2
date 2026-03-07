# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Serialization
#
# Shared data format for transferring inputs/outputs between local and remote.
# Supports numpy arrays (.npy) and JSON-serializable objects (.json).
# A `_manifest.json` tracks keys and their types.

# %%
#|default_exp serialization

# %%
#|export
import json
from pathlib import Path

import numpy as np

# %%
#|export
def serialize(data: dict, directory: Path) -> None:
    """Serialize named inputs/outputs to a directory.

    Args:
        data: Dict mapping names to values. Supported types:
            - numpy.ndarray -> <key>.npy
            - JSON-serializable (list, dict, str, int, float, bool, None) -> <key>.json
        directory: Target directory (created if needed).

    Raises:
        TypeError: If a value is not a numpy array and not JSON-serializable.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.save(directory / f"{key}.npy", value)
            manifest[key] = {"type": "npy"}
        elif _is_json_serializable(value):
            with open(directory / f"{key}.json", "w") as f:
                json.dump(value, f)
            manifest[key] = {"type": "json"}
        else:
            raise TypeError(
                f"Cannot serialize key {key!r}: unsupported type {type(value).__name__}. "
                f"Supported types: numpy.ndarray, and JSON-serializable "
                f"(list, dict, str, int, float, bool, None)."
            )

    with open(directory / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

# %%
#|export
def deserialize(directory: Path) -> dict:
    """Deserialize from directory back to dict.

    Args:
        directory: Directory containing serialized data and _manifest.json.

    Returns:
        Dict mapping names to deserialized values.

    Raises:
        ValueError: If manifest contains unsupported type (e.g. "pkl").
    """
    directory = Path(directory)

    with open(directory / "_manifest.json") as f:
        manifest = json.load(f)

    result = {}
    for key, info in manifest.items():
        typ = info["type"]
        if typ == "npy":
            result[key] = np.load(directory / f"{key}.npy")
        elif typ == "json":
            with open(directory / f"{key}.json") as f:
                result[key] = json.load(f)
        else:
            raise ValueError(
                f"Unsupported type {typ!r} for key {key!r}. "
                f"Only 'npy' and 'json' are supported."
            )

    return result

# %%
#|export
def _is_json_serializable(value) -> bool:
    """Check if a value is JSON-serializable without actually serializing it."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_serializable(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_json_serializable(v)
            for k, v in value.items()
        )
    return False
