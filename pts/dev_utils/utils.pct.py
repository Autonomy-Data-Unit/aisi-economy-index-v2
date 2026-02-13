# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dev Utils
#
# Helpers for developing pipeline node notebooks interactively. The main entry
# point is `set_node_func_args`, which populates the notebook's global namespace
# with the input data a node would receive when running inside the Net.

# %%
#|default_exp utils

# %%
#|export
import asyncio
import builtins
import inspect
import os
import sys
from collections import namedtuple
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

from netrun.core import Net, NetConfig

# %% [markdown]
# ## Config loading

# %%
#|export
def _load_net_config() -> NetConfig:
    """Load the pipeline NetConfig from ai_index assets."""
    config_path = resources.files("ai_index.assets") / "netrun.json"
    config = NetConfig.from_file(str(config_path))
    config.project_root_override = str(Path.cwd())
    return config


def _resolve_node_vars(config: NetConfig) -> dict:
    """Resolve node variables from config into a plain dict (with type casting)."""
    resolved = {}
    for key, var in config.node_vars.items():
        val = var.value
        # Handle $env references if not already resolved by NetConfig
        if isinstance(val, dict) and "$env" in val:
            val = os.environ.get(val["$env"], val.get("default", ""))
        # Cast to declared type
        try:
            if var.type == "int":
                val = int(val)
            elif var.type == "float":
                val = float(val)
            elif var.type == "bool":
                val = str(val).lower() in ("true", "1", "yes") if not isinstance(val, bool) else val
        except (ValueError, TypeError):
            pass
        resolved[key] = val
    return resolved

# %% [markdown]
# ## Node name resolution

# %%
#|export
def _resolve_node_name(config: NetConfig, bare_name: str) -> str:
    """Resolve a bare node name to its (possibly prefixed) name in the graph.

    With subgraphs, node names are prefixed (e.g., ``embed_onet`` becomes
    ``matching.embed_onet``). This helper finds the full name so that
    ``set_node_func_args`` works without changes to individual node notebooks.

    Args:
        config: The resolved NetConfig.
        bare_name: The unqualified node name (e.g. "embed_onet").

    Returns:
        The full node name (e.g. "matching.embed_onet"), or ``bare_name``
        unchanged if it already matches a node directly.

    Raises:
        ValueError: If the name cannot be found, or matches multiple nodes.
    """
    # Resolve the graph to get all flattened node names
    resolved = config.graph.resolve(config)
    all_names = [n.name for n in resolved.nodes]

    # Exact match — return as-is
    if bare_name in all_names:
        return bare_name

    # Look for suffix match (e.g. "embed_onet" matches "matching.embed_onet")
    matches = [n for n in all_names if n.endswith(f".{bare_name}")]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(
            f"Ambiguous node name '{bare_name}' — matches multiple nodes: {matches}"
        )

    raise ValueError(
        f"Node '{bare_name}' not found in graph. "
        f"Available nodes: {sorted(all_names)}"
    )

# %% [markdown]
# ## Salvo retrieval

# %%
#|export
async def _get_input_salvo(config: NetConfig, node_name: str) -> dict[str, list]:
    """Get input salvo for a node — from cache if available, otherwise by running upstream.

    Returns:
        dict mapping port_name -> list of packet values.
    """
    async with Net(config) as net:
        cached = net.get_cached_input_salvos(node_name)
        if cached:
            print(f"set_node_func_args: using cached inputs for '{node_name}' ({len(cached)} cached run(s))")
            return cached[-1]  # already dict[str, list[Any]]

        print(f"set_node_func_args: no cache for '{node_name}', running upstream nodes...")
        salvos = await net.run_to_targets(node_name)
        if not salvos:
            raise RuntimeError(
                f"No input salvos produced for node '{node_name}'. "
                f"Check that the node exists and has connected upstream nodes."
            )
        return salvos[0].packets  # extract dict from TargetInputSalvo


def _run_async(coro):
    """Run an async coroutine, handling Jupyter's already-running event loop."""
    try:
        asyncio.get_running_loop()
        # Inside Jupyter or another async context — patch to allow nested run()
        import nest_asyncio
        nest_asyncio.apply()
    except RuntimeError:
        pass
    return asyncio.run(coro)

# %% [markdown]
# ## Main entry point

# %%
#|export
_SPECIAL_PARAMS = frozenset({"ctx", "print"})


def set_node_func_args(func, *, node_name=None, return_args=False):
    """Populate the caller's namespace with the inputs a pipeline node would receive.

    Loads input data for a node from the netrun cache (or by running upstream
    nodes via ``Net.run_to_targets``) and injects the values into the caller's
    global namespace so that subsequent notebook cells can use them directly.

    Args:
        func: The node function defined via ``#|set_func_signature``. Used to
            infer the node name (from ``func.__name__``) and to determine which
            parameters are input ports vs special (``ctx``, ``print``).
        node_name: Explicit node name override. If None, uses ``func.__name__``.
        return_args: If True, return the arguments as a namedtuple instead of
            setting them in the caller's globals.

    Returns:
        If ``return_args`` is True, a namedtuple whose fields are the function
        parameter names. Otherwise ``None`` (values are set in caller's globals).

    Example::

        from dev_utils import set_node_func_args
        set_node_func_args(llm_filter_candidates)
        # => candidates, job_ads, ctx, print are now available as globals
    """
    name = node_name or func.__name__
    config = _load_net_config()

    # Resolve bare name to prefixed name (e.g. "embed_onet" -> "matching.embed_onet")
    name = _resolve_node_name(config, name)

    # Retrieve input salvo (cached or computed)
    salvo = _run_async(_get_input_salvo(config, name))

    # Extract port values — each port typically has one packet
    args = {}
    for port_name, values in salvo.items():
        args[port_name] = values[0] if len(values) == 1 else values

    # Add special parameters based on what the function signature expects
    sig = inspect.signature(func)
    if "ctx" in sig.parameters:
        args["ctx"] = SimpleNamespace(vars=_resolve_node_vars(config))
    if "print" in sig.parameters:
        args["print"] = builtins.print

    if return_args:
        NodeArgs = namedtuple("NodeArgs", list(args.keys()))
        return NodeArgs(**args)

    # Set in caller's globals
    caller_globals = sys._getframe(1).f_globals
    caller_globals.update(args)
